import json
import sqlite3

import numpy as np
import pytest

from hyperweave.config import Config
from hyperweave.data.group_split import study_dataset_hash
from hyperweave.eval.phase4_report import build_report
from hyperweave.registry import Registry

pytestmark = pytest.mark.phase4


def test_registry_derived_scoreboard_enforces_equal_budgets_and_seed_intervals(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    manifest = {"source_sha256": {"a": "hash-a", "b": "hash-b"},
                "partitions": {"a_val": {"benign": 6000}}}
    dataset_hash = study_dataset_hash(manifest)
    sealed = {}
    for condition in ("b1", "b2", "b2b", "b3", "b4", "b5", "b5b"):
        run_ids = []
        for seed in (11, 23, 37):
            for i in range(3 if condition in {"b1", "b2"} else 4):
                run_id = (f"group_{condition}_w{(4, 8, 16)[i]}_seed{seed}"
                          if condition in {"b1", "b2"} else f"{condition}_seed{seed}_trial{i}")
                detection = .1 + .01 * i + .001 * seed
                registry.write({"run_id": run_id, "config": Config(model=condition, seed=seed).as_dict(),
                                "seed": seed, "dataset_hash": dataset_hash, "git_commit": "fixture",
                                "checkpoint_path": "fixture.npz", "status": "completed",
                                "metrics": {"detection_rate": detection, "fpr": .02,
                                            "pr_auc": .4, "macs": 700 + i * 100,
                                            "training_seconds": 1.0}})
                run_ids.append(run_id)
                if condition in {"b3", "b4", "b5", "b5b"}:
                    registry.append_event("budget_reservation", {"condition": condition, "cost": 1})
        sealed[condition] = {"run_id": run_ids[0], "original": {"detection_rate": .2, "fpr": .1,
                                                                "pr_auc": .3}}
    registry.append_event("sealed_final", {"results": sealed})
    report = build_report(registry, manifest, sealed, tmp_path)
    assert report["budget"] == {name: 12 for name in ("b3", "b4", "b5", "b5b")}
    assert report["scoreboard"]["b2"]["attempts"] == 9
    assert report["scoreboard"]["b3"]["attempts"] == 12
    assert len(report["analysis"]["c1_matched_mac_pairs"]) == 9
    assert report["scoreboard"]["b3"]["seed_bootstrap_detection_95"] is not None
    assert report["scoreboard"]["b3"]["research_hv_3d"] is None
    assert len(report["hypervolume_protocol"]["sha256"]) == 64
    assert all(value.startswith("inconclusive") for value in report["claim_verdicts"].values())
    with pytest.raises(ValueError, match="sealed results differ"):
        build_report(registry, manifest, {**sealed, "b3": None}, tmp_path)
    with pytest.raises(ValueError, match="split manifest"):
        build_report(registry, {**manifest, "different": True}, sealed, tmp_path)

    # Every deployed point has same-host, same-core pinned timing: 3D HV is reportable.
    with sqlite3.connect(registry.path) as db:
        for run_id, payload in db.execute("SELECT run_id, record FROM runs").fetchall():
            record = json.loads(payload)
            record["metrics"]["latency"] = {"pinned": True, "hostname": "fixture-host",
                                             "core": 2, "median_ns_per_flow": 1000.0}
            db.execute("UPDATE runs SET record=? WHERE run_id=?",
                       (json.dumps(record, sort_keys=True), run_id))
    pinned_report = build_report(registry, manifest, sealed, tmp_path)
    assert pinned_report["scoreboard"]["b3"]["research_hv_3d"] > 0
    assert pinned_report["latency_note"].startswith("Pinned")
    with sqlite3.connect(registry.path) as db:
        payload = json.loads(db.execute("SELECT record FROM runs WHERE run_id=?",
                                        ("b3_seed11_trial0",)).fetchone()[0])
        payload["metrics"]["latency"]["hostname"] = "other-host"
        db.execute("UPDATE runs SET record=? WHERE run_id=?",
                   (json.dumps(payload, sort_keys=True), "b3_seed11_trial0"))
    with pytest.raises(ValueError, match="different hosts, cores, or batch sizes"):
        build_report(registry, manifest, sealed, tmp_path)
    payload["metrics"]["latency"]["hostname"] = "fixture-host"
    with sqlite3.connect(registry.path) as db:
        db.execute("UPDATE runs SET record=? WHERE run_id=?",
                   (json.dumps(payload, sort_keys=True), "b3_seed11_trial0"))

    registry.write({"run_id": "extra_b3", "config": Config(model="b3", seed=11).as_dict(),
                    "seed": 11, "dataset_hash": dataset_hash, "git_commit": "fixture",
                    "checkpoint_path": "fixture.npz", "status": "failed", "metrics": {}})
    with pytest.raises(ValueError, match="12-run seed protocol"):
        build_report(registry, manifest, sealed, tmp_path)


def test_report_refuses_missing_final_sealed_event(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    registry.write({"run_id": "only", "config": Config().as_dict(), "seed": 11,
                    "dataset_hash": "same", "git_commit": "fixture", "metrics": {},
                    "checkpoint_path": "fixture.npz"})
    with pytest.raises(ValueError, match="exactly once"):
        build_report(registry, {}, {}, tmp_path)


def test_sealed_reopen_is_refused_before_file_access(tmp_path):
    from hyperweave.baselines.phase4 import final_sealed_evaluation
    registry = Registry(tmp_path / "registry.sqlite")
    registry.append_event("sealed_open_attempt", {"selected_run_ids": {}})
    with pytest.raises(RuntimeError, match="already performed"):
        final_sealed_evaluation(registry, tmp_path / "missing_split", tmp_path, None)


def test_phase4_preflight_leaves_output_unused_when_split_missing(tmp_path):
    from hyperweave.baselines.phase4 import run

    output = tmp_path / "study"
    with pytest.raises(FileNotFoundError, match="split manifest is missing"):
        run(tmp_path / "missing_split", output)
    assert not output.exists()


def test_deployment_selection_uses_int8_detection_instead_of_float(tmp_path):
    from hyperweave.baselines.phase4 import _selected
    registry = Registry(tmp_path / "registry.sqlite")
    for run_id, float_rate, int8_rate in (("float_winner", .9, 0.), ("int8_winner", .7, .6)):
        registry.write({"run_id": run_id, "config": Config(model="b3").as_dict(),
                        "seed": 11, "dataset_hash": "same", "git_commit": "fixture",
                        "checkpoint_path": "fixture.npz", "status": "completed",
                        "metrics": {"detection_rate": float_rate, "fpr": .1, "macs": 100,
                                    "int8": {"detection_rate": int8_rate, "fpr": .1}}})
    assert _selected(registry, "b3")["run_id"] == "int8_winner"
