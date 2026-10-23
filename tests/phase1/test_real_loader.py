import ipaddress
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyperweave.data import MissingSplitMetadataError, iter_nf_csv_chunks, load_nf_csv
from hyperweave.data.audit_nf import audit
from hyperweave.registry import Registry

pytestmark = pytest.mark.phase1


def _fixture(tmp_path, with_metadata=False):
    features = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"] + [f"f{i}" for i in range(41)]
    rows = []
    for i in range(12):
        row = {name: float(i + j) for j, name in enumerate(features[2:])}
        row.update(IPV4_SRC_ADDR=f"10.0.0.{i+1}", IPV4_DST_ADDR="192.0.2.1",
                   Label=i % 2, Attack="Benign" if i % 2 == 0 else "Attack")
        if with_metadata:
            row.update(campaign=f"c{i//3}", timestamp=i)
        rows.append(row)
    path = tmp_path / "nf.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, features


def test_ipv4_encoding_and_canonical_order(tmp_path):
    path, features = _fixture(tmp_path, with_metadata=True)
    loaded = load_nf_csv(path, "nf_unsw_nb15_v2", features,
                         campaign_col="campaign", time_col="timestamp")
    assert loaded.columns.tolist() == features + ["Label", "campaign", "timestamp"]
    assert loaded.loc[0, "IPV4_SRC_ADDR"] == int(ipaddress.IPv4Address("10.0.0.1"))
    assert loaded.loc[0, "IPV4_DST_ADDR"] == int(ipaddress.IPv4Address("192.0.2.1"))
    assert all(np.issubdtype(loaded[name].dtype, np.number) for name in features)


def test_missing_campaign_time_fails_before_full_read(tmp_path):
    path, features = _fixture(tmp_path)
    with pytest.raises(MissingSplitMetadataError, match="do not use a random-flow"):
        load_nf_csv(path, "nf_unsw_nb15_v2", features,
                    campaign_col="campaign", time_col="timestamp")
    chunks = list(iter_nf_csv_chunks(path, "nf_unsw_nb15_v2", features, chunksize=5))
    assert [len(chunk) for chunk in chunks] == [5, 5, 2]


def test_real_csv_audit_smoke_has_no_performance_claim(tmp_path):
    path, features = _fixture(tmp_path)
    report = audit(path, "nf_unsw_nb15_v2", features, tmp_path / "out",
                   chunksize=5, smoke_train=True, sample_per_class=4)
    assert report["rows"] == 12 and report["labels"] == {0: 6, 1: 6}
    assert report["campaign_time_split_ready"] is False
    registry = Registry(tmp_path / "out" / "registry.sqlite")
    assert len(registry.all_runs()) == 2
    assert all(run["metrics"]["finite_logits"] for run in registry.all_runs())
    assert all(run["metrics"]["detection_rate"] is None for run in registry.all_runs())


def test_official_manifest_has_43_unique_features():
    path = Path(__file__).resolve().parents[2] / "configs" / "nf_v2_features.txt"
    features = path.read_text().splitlines()
    assert len(features) == len(set(features)) == 43
    assert {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "IN_BYTES", "OUT_PKTS"} <= set(features)
