import json
import sys

import numpy as np
import pandas as pd
import pytest

from hyperweave.baselines.phase2 import run as run_phase2
from hyperweave.baselines.run import run as run_phase1
from hyperweave.config import Config
from hyperweave.eval import hypervolume
from hyperweave.kernel.runner import bench, compile_kernel, infer, write_inputs
from hyperweave.models import (HyperNet, NeuronSpec, bake, load_hypernet, quantize,
                               read_int8, sample_preference, save_hypernet, write_int8)
from hyperweave.registry import Registry

pytestmark = pytest.mark.phase2


def _model(seed=11):
    return HyperNet(NeuronSpec(5, 64, 3), seed=seed)


def test_hypernet_init_variance():
    model = _model()
    generated = model.generate((.45, .45, .10))
    assert abs(np.var(generated["w1"]) - 1 / 5) / (1 / 5) < .2


def test_shared_config_schema():
    for model in ("b1", "b2", "b2b", "b3", "b4", "b5", "b5b"):
        config = Config(model=model, preference=(.2, .3, .5), context=(1, 2))
        assert Config(**config.as_dict()).as_dict() == config.as_dict()
    with pytest.raises(ValueError):
        Config(model="unknown")
    with pytest.raises(ValueError):
        Config(preference=(.2, .3, .6))


def test_hypernet_shapes():
    for d, m, h in ((3, 7, 2), (5, 9, 4), (43, 16, 4)):
        model = HyperNet(NeuronSpec(d, m, h))
        generated = model.generate((.4, .4, .2))
        assert generated["raw"].shape == (m, h * d + 2 * h + 3)
        assert generated["w1"].shape == (m, h, d)


def test_gate_exact_zeros():
    model = _model()
    x = np.random.default_rng(1).normal(size=(20, 5))
    y = (x[:, 0] > 0).astype(int)
    for _ in range(10):
        model.train_step(x, y, (.05, .05, .9), lr=.03)
    assert np.any(model.generate((.05, .05, .9))["gate"] == 0)


def test_gate_prunes_outgoing():
    model = _model()
    preference = (.10, .10, .80)
    generated = model.generate(preference)
    baked = bake(model, preference)
    assert np.array_equal(baked.source_indices, np.flatnonzero(generated["gate"] > 0))
    assert baked.macs == len(baked.source_indices) * model.spec.macs_per_neuron
    assert np.all(generated["gate"][~baked.gate_mask] == 0)


def test_lambda_changes_cost():
    low, high = [], []
    x = np.random.default_rng(9).normal(size=(32, 5))
    y = (x[:, 0] > 0).astype(int)
    for seed in (11, 23, 37):
        model = _model(seed)
        model.fit(x, y, epochs=2, batch_size=16, seed=seed)
        low.append(model.expected_macs(model.generate((.8, .1, .1))))
        high.append(model.expected_macs(model.generate((.1, .1, .8))))
    assert np.mean(high) < np.mean(low)


def test_context_changes_params():
    model = _model()
    first = model.generate((.4, .4, .2), (0, 0))["raw"]
    second = model.generate((.4, .4, .2), (2, 2))["raw"]
    assert not np.array_equal(first, second)


def test_generation_once_per_batch():
    model = _model()
    x = np.random.default_rng(1).normal(size=(100, 5))
    model.train_step(x, (x[:, 0] > 0).astype(int), (.4, .4, .2))
    assert model.forward_count == 1


def test_bake_equivalence_float():
    model = _model()
    x = np.random.default_rng(3).normal(size=(20, 5))
    for preference in ((.8, .1, .1), (.25, .10, .65), (.1, .1, .8)):
        generated = model.generate(preference, (1, 2))
        baked = bake(model, preference, (1, 2))
        np.testing.assert_allclose(baked.logits(x), model.logits_from(x, generated), atol=1e-5)


def test_bake_drops_neurons():
    model = HyperNet(NeuronSpec(43, 16, 4))
    counts = [bake(model, ((1 - cost) * .7, (1 - cost) * .3, cost)).macs
              for cost in (.1, .6, .65)]
    assert counts[0] > counts[1] > counts[2]


def test_int8_gap(tmp_path):
    model = _model()
    x = np.random.default_rng(3).normal(size=(300, 5))
    y = (x[:, 0] > 0).astype(int)
    for i, preference in enumerate(((.8, .1, .1), (.28, .12, .6))):
        baked = bake(model, preference)
        int8 = quantize(baked, x)
        path = tmp_path / f"point{i}.hwi"
        write_int8(int8, path, {"point": i})
        assert np.isfinite(int8.logits(x)).all()
        assert path.stat().st_size == int8.model_bytes
        assert np.isfinite(np.mean((baked.logits(x) > 0) == y) - np.mean((int8.logits(x) > 0) == y))


def test_kernel_vs_python(tmp_path):
    model = _model()
    x = np.random.default_rng(2).normal(size=(100, 5))
    int8 = quantize(bake(model, (.8, .1, .1)), x)
    path = tmp_path / "model.hwi"
    write_int8(int8, path, {"test": True})
    loaded = read_int8(path)
    input_path = tmp_path / "inputs.i8"
    write_inputs(input_path, int8.quantize_inputs(x))
    binary = compile_kernel(tmp_path / "infer")
    np.testing.assert_array_equal(int8.logits_int(int8.quantize_inputs(x)),
                                  loaded.logits_int(loaded.quantize_inputs(x)))
    np.testing.assert_array_equal(int8.logits_int(int8.quantize_inputs(x)),
                                  infer(binary, path, input_path, len(x)))


def test_kernel_timing_protocol(tmp_path):
    model = _model()
    x = np.random.default_rng(2).normal(size=(100, 5))
    int8 = quantize(bake(model, (.8, .1, .1)), x)
    path = tmp_path / "model.hwi"
    write_int8(int8, path, {})
    input_path = tmp_path / "inputs.i8"
    write_inputs(input_path, int8.quantize_inputs(x))
    result = bench(compile_kernel(tmp_path / "infer"), path, input_path, warmup=5, iterations=100)
    assert result["warmup"] == 5 and result["iterations"] >= 100 and result["repeats"] == 5
    assert result["p99_ns_per_flow"] >= result["median_ns_per_flow"] > 0
    assert result["repeat_median_variance"] >= 0
    assert isinstance(result["pinned"], bool)
    if sys.platform == "linux":
        assert result["pinned"] and result["core"] >= 0
    else:
        assert result["pinned"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="physical CPU core pinning unavailable on this host")
def test_kernel_core_pinned(tmp_path):
    model = _model()
    x = np.random.default_rng(2).normal(size=(10, 5))
    int8 = quantize(bake(model, (.8, .1, .1)), x)
    path = tmp_path / "model.hwi"
    write_int8(int8, path, {})
    input_path = tmp_path / "inputs.i8"
    write_inputs(input_path, int8.quantize_inputs(x))
    result = bench(compile_kernel(tmp_path / "infer"), path, input_path, warmup=5, iterations=100)
    assert result["pinned"] and result["core"] >= 0


def test_phn_hvi_runs():
    model = _model()
    x = np.random.default_rng(3).normal(size=(32, 5))
    history = model.fit(x, (x[:, 0] > 0).astype(int), epochs=2,
                        batch_size=16, seed=11, mode="phn_hvi")
    assert len(history) == model.optimizer_steps == 4
    assert model.forward_count == 3 * model.optimizer_steps
    assert all(np.isfinite(result["loss"]) for result in history)


def test_hypervolume_correct():
    points = [(0.2, 0.5), (0.5, 0.2)]
    assert hypervolume(points, (1, 1)) == pytest.approx(.55)
    assert hypervolume([(a, b, .5) for a, b in points], (1, 1, 1)) == pytest.approx(.275)


def test_hypernet_no_divergence():
    x = np.random.default_rng(3).normal(size=(32, 5))
    y = (x[:, 0] > 0).astype(int)
    for seed in (11, 23, 37):
        history = _model(seed).fit(x, y, epochs=3, batch_size=16, seed=seed)
        assert all(np.isfinite(r["loss"]) and abs(r["loss"]) < 10 for r in history)


def test_heldout_lambda_split():
    rng = np.random.default_rng(7)
    heldout = ((.2, .3, .5), (.7, .2, .1))
    for _ in range(100):
        preference = sample_preference(rng, heldout=heldout)
        assert all(np.linalg.norm(preference - vector) >= .05 for vector in heldout)


def test_hypernet_checkpoint_roundtrip(tmp_path):
    model = _model()
    path = tmp_path / "hypernet.npz"
    save_hypernet(model, path)
    loaded = load_hypernet(path)
    x = np.random.default_rng(2).normal(size=(12, 5))
    np.testing.assert_array_equal(model.logits(x, (.5, .3, .2)), loaded.logits(x, (.5, .3, .2)))


def test_checkpoint_to_baked_mvp_cycle(tmp_path):
    rng = np.random.default_rng(4)
    features = [f"f{i}" for i in range(43)]
    frames = []
    for campaign in range(4):
        x = rng.normal(size=(300, 43))
        frame = pd.DataFrame(x, columns=features)
        frame["Label"] = (x[:, 0] > 0).astype(int)
        frame["campaign"] = f"c{campaign}"
        frame["time"] = campaign * 300 + np.arange(300)
        frames.append(frame)
    csv = tmp_path / "flows.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv, index=False)
    phase1 = tmp_path / "phase1"
    run_phase1(csv, features, "nf_unsw_nb15_v2", "campaign", "time", phase1, epochs=1, target_fpr=.01)
    phase2 = tmp_path / "phase2"
    report = run_phase2(csv, features, "nf_unsw_nb15_v2", "campaign", "time",
                        phase1 / "registry.sqlite", "nf_unsw_nb15_v2_b2_seed11", phase2,
                        epochs=1, preferences=((.8, .1, .1), (.28, .12, .6)))
    registry = Registry(phase2 / "registry.sqlite")
    assert registry.read(report["fixed_run_id"])["metrics"] == report["fixed"]
    assert len(report["b3_run_ids"]) == len(report["b2b_run_ids"]) == 2
    for run_id in report["b3_run_ids"] + report["b2b_run_ids"]:
        record = registry.read(run_id)
        metrics = record["metrics"]
        assert metrics["threshold_source"] == "a_val"
        assert np.isfinite(metrics["int8_detection_gap"])
        assert metrics["int8_model_bytes"] > 0
        assert metrics["latency"]["median_ns_per_flow"] > 0
        assert json.loads((phase2 / f"{run_id}.hwi.json").read_text())["dataset_hash"] == record["dataset_hash"]
    if report["timing_comparable"]:
        assert np.isfinite(report["b3_hv"]) and np.isfinite(report["b2b_hv"])
    else:
        assert report["b3_hv"] is report["b2b_hv"] is None
