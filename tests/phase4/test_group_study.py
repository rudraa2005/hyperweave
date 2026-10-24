import ipaddress

import numpy as np
import pandas as pd
import pytest

from hyperweave.data.group_split import (endpoint_group_hash, file_sha256,
                                         partition_for_group, sample_group_disjoint,
                                         study_dataset_hash)
from hyperweave.data.normalization import FlowNormalizer
from hyperweave.eval.phase4_report import bootstrap_seed_interval

pytestmark = pytest.mark.phase4


def test_endpoint_pair_hash_is_direction_independent():
    a = np.array([1, 7, 800], dtype=np.uint64)
    b = np.array([9, 2, 801], dtype=np.uint64)
    np.testing.assert_array_equal(endpoint_group_hash(a, b), endpoint_group_hash(b, a))
    assert np.all((partition_for_group(endpoint_group_hash(a, b)) >= 0) &
                  (partition_for_group(endpoint_group_hash(a, b)) < 4))


def test_streaming_group_split_has_no_pair_or_feature_overlap(tmp_path):
    features = [line.strip() for line in open("configs/nf_v2_features.txt")]
    chosen = {}
    for i in range(1, 20000):
        src, dst = i, i + 100000
        group = endpoint_group_hash(np.array([src]), np.array([dst]))
        part = int(partition_for_group(group)[0])
        chosen.setdefault(part, (src, dst))
        if len(chosen) == 4:
            break
    assert len(chosen) == 4
    files = []
    for dataset, parts in (("unsw", (0, 1)), ("cse", (2, 3))):
        rows = []
        for part in parts:
            src, dst = chosen[part]
            for label in (0, 1):
                for offset in range(4):
                    row = {feature: 0 for feature in features}
                    row["IPV4_SRC_ADDR"] = str(ipaddress.IPv4Address(src))
                    row["IPV4_DST_ADDR"] = str(ipaddress.IPv4Address(dst))
                    row["IN_BYTES"] = 100 + label * 100 + offset
                    row["IN_PKTS"] = 2
                    row["Label"] = label
                    rows.append(row)
        path = tmp_path / f"{dataset}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        files.append(path)
    manifest = sample_group_disjoint(*files, features, tmp_path / "split", limit=4,
                                     chunksize=3, min_val_benign=2)
    groups, hashes = [], []
    for part in ("a_train", "a_val", "b", "sealed"):
        folder = "private" if part == "sealed" else "public"
        with np.load(tmp_path / "split" / folder / f"{part}.npz") as data:
            groups.append(set(map(int, data["group"])))
            hashes.append(set(map(tuple, data["x"])))
            assert set(data["y"]) == {0, 1}
    assert all(not groups[i] & groups[j] and not hashes[i] & hashes[j]
               for i in range(4) for j in range(i + 1, 4))
    assert manifest["partitions"]["sealed"]["dataset"] == "nf_cse_cic_ids2018_v2"
    for part in ("a_train", "a_val", "b", "sealed"):
        folder = "private" if part == "sealed" else "public"
        assert manifest["partitions"][part]["sha256"] == file_sha256(
            tmp_path / "split" / folder / f"{part}.npz")
    repeat = sample_group_disjoint(*files, features, tmp_path / "repeat", limit=4,
                                   chunksize=3, min_val_benign=2)
    assert study_dataset_hash(repeat) == study_dataset_hash(manifest)


def test_normalizer_handles_extreme_finite_rates_without_leaking_eval(tmp_path):
    features = ["IN_BYTES", "SRC_TO_DST_SECOND_BYTES", "IPV4_SRC_ADDR"]
    train = np.array([[10., 1e2, 100.], [20., 1e4, 200.], [30., 1e6, 300.]])
    eval_rows = np.array([[1e7, 8.8e209, 400.]])
    normalizer = FlowNormalizer(features).fit(train)
    before = normalizer.center.copy()
    assert np.isfinite(normalizer.transform(eval_rows)).all()
    np.testing.assert_array_equal(before, normalizer.center)
    path = tmp_path / "normalizer.npz"
    normalizer.save(path)
    np.testing.assert_allclose(normalizer.transform(train), FlowNormalizer.load(path).transform(train))


def test_seed_bootstrap_is_deterministic_and_requires_1000_resamples():
    a = bootstrap_seed_interval([.1, .2, .3], resamples=1000)
    assert a == bootstrap_seed_interval([.1, .2, .3], resamples=1000)
    assert 0 <= a[0] <= a[1] <= 1
    with pytest.raises(ValueError):
        bootstrap_seed_interval([.1, .2, .3], resamples=999)


def test_fixed_functional_bias_is_folded_into_bit_exact_int8_kernel(tmp_path):
    from hyperweave.baselines.phase4 import _functional_int8_metrics
    from hyperweave.kernel.runner import compile_kernel
    from hyperweave.models import FunctionalNet
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 5))
    y = np.tile([0, 1], 40)
    model = FunctionalNet(5, 4, 2, seed=3)
    model.bias = .3
    binary = compile_kernel(tmp_path / "infer")
    metrics = _functional_int8_metrics(model, x, x, y, .1, tmp_path, "fixed", binary, "fixture")
    assert metrics["kernel_bit_exact"]
    assert metrics["int8_model_bytes"] > 0
    assert metrics["latency"]["batch_rows"] == len(x)
    assert np.isfinite(metrics["int8_detection_gap"])
