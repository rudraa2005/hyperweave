import numpy as np
import pytest

from hyperweave.data.evasion import constrained_nf_evasion, nf_mutable_mask
from hyperweave.data.shifts import SHIFT_FAMILIES, apply_nf_shift
from hyperweave.eval.robustness import robustness_report

pytestmark = pytest.mark.phase4

FEATURES = ["IPV4_SRC_ADDR", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS", "SRC_TO_DST_SECOND_BYTES", "PROTOCOL"]


def flows(n=100):
    rng = np.random.default_rng(19)
    out = np.zeros((n, len(FEATURES)))
    out[:, 0] = 12345
    out[:, 2] = rng.integers(1, 20, n)
    out[:, 1] = out[:, 2] * rng.integers(10, 100, n)
    out[:, 4] = rng.integers(1, 20, n)
    out[:, 3] = out[:, 4] * rng.integers(10, 100, n)
    out[:, 5] = rng.integers(1, 100, n)
    out[:, 6] = out[:, 1] / out[:, 5]
    out[:, 7] = 6
    return out


def test_stochastic_shift_families_preserve_core_fields():
    x = flows()
    assert not SHIFT_FAMILIES["train"] & SHIFT_FAMILIES["eval"]
    for family in SHIFT_FAMILIES["train"] | SHIFT_FAMILIES["eval"]:
        first = apply_nf_shift(x, FEATURES, family, severity=2, seed=9)
        np.testing.assert_array_equal(first, apply_nf_shift(x, FEATURES, family, severity=2, seed=9))
        np.testing.assert_array_equal(first[:, [0, 7]], x[:, [0, 7]])
        assert np.isfinite(first).all() and (first >= 0).all()
        assert (first[:, 1] >= first[:, 2]).all()
        assert (first[:, 3] >= first[:, 4]).all()
    assert not np.array_equal(apply_nf_shift(x, FEATURES, "count_rescale", seed=9),
                              apply_nf_shift(x, FEATURES, "count_rescale", seed=10))
    with pytest.raises(ValueError, match="immutable"):
        apply_nf_shift(x, FEATURES, "count_rescale", columns=[0])


def test_constrained_evasion_preserves_immutables_and_budget():
    x = flows()
    reference = np.median(x, axis=0)
    mask = nf_mutable_mask(FEATURES)
    for kind in ("noise", "jitter", "padding", "toward_benign"):
        zero = constrained_nf_evasion(x, reference, FEATURES, 0, kind)
        np.testing.assert_array_equal(zero, x)
        changed = constrained_nf_evasion(x, reference, FEATURES, .2, kind)
        np.testing.assert_array_equal(changed[:, ~mask], x[:, ~mask])
        assert np.all(np.abs(changed - x) <= .2 * np.maximum(np.abs(x), 1) + 1e-8)
        assert (changed[:, 1] >= changed[:, 2]).all()
        assert (changed[:, 3] >= changed[:, 4]).all()


def test_offline_robustness_report_uses_fixed_threshold():
    class Detector:
        def logits(self, x):
            return x[:, 1]
    x = flows(100)
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    report = robustness_report(Detector(), FEATURES, x, y, x, y,
                               np.zeros(len(FEATURES)), np.ones(len(FEATURES)), 300.,
                               severities=(1,), epsilons=(0., .2))
    assert report["threshold_source"] == "a_val"
    assert report["attack_rows"] == 50
    assert set(report["shifts"]) == SHIFT_FAMILIES["eval"]
    assert report["evasion_detection_rate"]["noise"]["0.0"] == report["baseline"]["detection_rate"]
    transformed = robustness_report(Detector(), FEATURES, x, y, x, y,
                                    np.zeros(len(FEATURES)), np.ones(len(FEATURES)), 600.,
                                    severities=(1,), epsilons=(0.,), transform=lambda raw: raw * 2)
    assert transformed["baseline"] == report["baseline"]


def test_hypernet_training_receives_shift_and_evasion_batches():
    from hyperweave.models import HyperNet, NeuronSpec
    x = np.random.default_rng(1).normal(size=(24, 3))
    y = np.tile([0, 1], 12)
    calls = []
    def shift(batch, family, columns, seed):
        calls.append(("shift", family))
        return batch + 1
    def evasion(batch, labels, seed):
        calls.append(("evasion", int(labels.sum())))
        return batch + labels[:, None]
    model = HyperNet(NeuronSpec(3, 4, 2))
    model.fit(x, y, epochs=1, batch_size=8, shift_transform=shift,
              evasion_transform=evasion)
    assert len(calls) == 6
    assert {value for kind, value in calls if kind == "shift"} == SHIFT_FAMILIES["train"]
    assert all(value > 0 for kind, value in calls if kind == "evasion")


def test_autoencoder_requires_optional_torch_or_trains(tmp_path):
    from hyperweave.models.noise_envelope import NoiseEnvelope
    x = np.random.default_rng(4).normal(size=(80, 3)).astype(np.float32)
    model = NoiseEnvelope(3, hidden=6, latent=2).fit(x[:60], epochs=2, device="cpu")
    model.calibrate(x[60:], .1)
    path = tmp_path / "envelope.npz"
    model.save(path)
    loaded = NoiseEnvelope.load(path)
    np.testing.assert_allclose(model.reconstruction_error(x[60:]), loaded.reconstruction_error(x[60:]))
    assert model.device in {"cpu", "cpu_numpy"}


def test_line_rate_gate_refuses_unpinned_or_over_budget_measurements():
    from hyperweave.kernel.runner import enforce_latency_budget
    measurement = {"pinned": False, "median_ns_per_flow": 100, "p99_ns_per_flow": 150}
    with pytest.raises(RuntimeError, match="pinned"):
        enforce_latency_budget(measurement, median_ns=200, p99_ns=300)
    measurement["pinned"] = True
    assert enforce_latency_budget(measurement, median_ns=200, p99_ns=300)
    with pytest.raises(RuntimeError, match="exceeded"):
        enforce_latency_budget(measurement, median_ns=90, p99_ns=300)
