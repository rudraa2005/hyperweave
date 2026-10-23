import numpy as np
import pandas as pd
import pytest

from hyperweave.data import SHIFT_FAMILIES, deduplicate, load_nf_csv, split_by_campaign, validate_disjoint, validate_chronological
from hyperweave.data.evasion import projected_gradient, toward_benign
from hyperweave.data.fashion import corrupt, fixed_partition
from hyperweave.eval import score_binary, threshold_at_fpr
from hyperweave.models import AffineMLP, FunctionalNet, affine_macs, functional_macs, match_affine_width

pytestmark = pytest.mark.phase1


def test_fmnist_corruptions_deterministic():
    x = np.random.default_rng(1).integers(0, 255, (10, 28, 28), dtype=np.uint8)
    for family in ("noise", "blur", "brightness", "occlusion"):
        for severity in (1, 2, 3):
            np.testing.assert_array_equal(corrupt(x, family, severity, 5), corrupt(x, family, severity, 5))
    a, b, c = fixed_partition(100)
    assert len(set(a) | set(b) | set(c)) == 100


def test_mac_counter_affine():
    assert affine_macs(43, 16) == 43 * 16 + 16
    assert AffineMLP(43, 16).macs == affine_macs(43, 16)


def test_mac_counter_functional():
    assert functional_macs(43, 7, 4) == 7 * (43 * 4 + 4) + 7
    assert FunctionalNet(43, 7, 4).macs == functional_macs(43, 7, 4)


def test_mac_matching():
    for d, m, h in ((43, 8, 2), (43, 16, 4), (43, 32, 8)):
        target = functional_macs(d, m, h)
        assert abs(affine_macs(d, match_affine_width(d, m, h)) - target) / target < .02


def _fixture():
    features = [f"f{i}" for i in range(43)]
    rows = []
    for campaign in range(6):
        for j in range(4):
            row = {f: float(campaign * 10 + j + i) for i, f in enumerate(features)}
            row.update(Label=j % 2, campaign=f"c{campaign}", time=campaign * 100 + j)
            rows.append(row)
    return pd.DataFrame(rows), features


def test_split_no_campaign_overlap():
    df, features = _fixture()
    splits = split_by_campaign(df, "campaign", "time")
    validate_disjoint(splits, features, "campaign")
    validate_chronological(splits, "time")
    assert all(len(v) for v in splits.values())
    overlapping = {name: part.copy() for name, part in splits.items()}
    overlapping["b"].loc[:, "time"] = 0
    with pytest.raises(ValueError, match="time windows overlap"):
        validate_chronological(overlapping, "time")


def test_dedup():
    df, features = _fixture()
    duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = deduplicate(duplicate, features)
    assert len(cleaned) == len(df)
    validate_disjoint(split_by_campaign(cleaned, "campaign", "time"), features, "campaign")
    conflict = duplicate.copy()
    conflict.loc[len(conflict) - 1, "Label"] = 1 - conflict.loc[0, "Label"]
    with pytest.raises(ValueError, match="conflicting labels"):
        deduplicate(conflict, features)


def test_schema_alignment(tmp_path):
    df, features = _fixture()
    p = tmp_path / "flows.csv"
    df.to_csv(p, index=False)
    a = load_nf_csv(p, "nf_unsw_nb15_v2", features, campaign_col="campaign", time_col="time")
    b = load_nf_csv(p, "nf_cse_cic_ids2018_v2", features, campaign_col="campaign", time_col="time")
    assert a.columns.tolist() == b.columns.tolist()
    with pytest.raises(ValueError):
        load_nf_csv(p, "nf_unsw_nb15_v2", features[:-1], campaign_col="campaign", time_col="time")


def test_threshold_calibration():
    benign = np.arange(10000, dtype=float)
    threshold = threshold_at_fpr(benign, .001)
    assert abs(np.mean(benign > threshold) - .001) <= .0002
    metrics = score_binary(np.r_[np.zeros(10000), np.ones(10)], np.r_[benign, np.full(10, 10001)], threshold)
    assert metrics["detection_rate"] == 1


def test_shift_family_disjoint():
    assert not SHIFT_FAMILIES["train"] & SHIFT_FAMILIES["eval"]


def test_evasion_constraints():
    x = np.array([[100., 3., 7.]])
    mutable = [True, True, False]
    kw = dict(bytes_col=0, packets_col=1, min_packet_size=10)
    attack = projected_gradient(x, lambda z: np.ones_like(z), 50, mutable, **kw)
    assert attack[0, 2] == x[0, 2]
    assert attack[0, 0] >= attack[0, 1] * 10


def test_evasion_monotone():
    x = np.array([[100., 2., 7.], [90., 2., 7.]])
    benign = np.array([20., 2., 7.])
    rates = []
    for epsilon in (0, .25, .5, .75, 1):
        attack = toward_benign(x, benign, epsilon, [True, False, False], bytes_col=0, packets_col=1, min_packet_size=10)
        rates.append(np.mean(attack[:, 0] > 50))
        if epsilon == 0:
            np.testing.assert_array_equal(attack, x)
    assert rates == sorted(rates, reverse=True)


def test_affine_is_not_functional():
    # ReLU kink: finite differences on either side of zero are distinct.
    model = FunctionalNet(1, 1, 2)
    model.w1[:] = [[[1.], [-1.]]]
    model.b1[:] = 0
    model.w2[:] = [[1., 1.]]
    model.b2[:] = 0
    model.out[:] = 1
    model.bias = 0
    x = np.array([[-1.], [0.], [1.]])
    y = model.logits(x)
    assert not np.isclose(y[0] + y[2], 2 * y[1])


def test_baseline_runner_writes_auditable_records(tmp_path):
    from hyperweave.baselines.run import run
    from hyperweave.registry import Registry

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
    out = tmp_path / "runs"
    records = run(csv, features, "nf_unsw_nb15_v2", "campaign", "time", out, epochs=1, target_fpr=.01)
    assert len(records) == 2
    registry = Registry(out / "registry.sqlite")
    for record in records:
        assert registry.read(record["run_id"]) == record
        assert record["metrics"]["threshold_source"] == "a_val"
        assert record["metrics"]["macs"] > 0
        assert (out / (record["run_id"] + ".npz")).exists()
    with pytest.raises(FileExistsError):
        run(csv, features, "nf_unsw_nb15_v2", "campaign", "time", out, epochs=1, target_fpr=.01)
