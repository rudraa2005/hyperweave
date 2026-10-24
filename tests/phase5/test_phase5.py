import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from hyperweave.config import Config
from hyperweave.data.group_split import file_sha256, study_dataset_hash
from hyperweave.data.normalization import FlowNormalizer
from hyperweave.demo.phase5 import DemoState, STATIC, make_handler
from hyperweave.models import HyperNet, NeuronSpec, save_hypernet
from hyperweave.registry import Registry

pytestmark = pytest.mark.phase5


def _fixture(tmp_path):
    study, split = tmp_path / "study", tmp_path / "split"
    study.mkdir()
    (split / "public").mkdir(parents=True)
    features = Path("configs/nf_v2_features.txt").read_text().splitlines()
    rng = np.random.default_rng(5)
    for name in ("a_train", "a_val", "b"):
        x = rng.uniform(1, 10, size=(120, len(features)))
        x[:, features.index("IN_BYTES")] = 100
        x[:, features.index("IN_PKTS")] = 2
        x[:, features.index("OUT_BYTES")] = 90
        x[:, features.index("OUT_PKTS")] = 2
        y = np.tile([0, 1], 60).astype(np.uint8)
        np.savez(split / "public" / f"{name}.npz", x=x, y=y)
    manifest = {"kind": "endpoint_pair_group_disjoint_v1", "partitions": {
        name: {"sha256": file_sha256(split / "public" / f"{name}.npz")}
        for name in ("a_train", "a_val", "b")}}
    (split / "manifest.json").write_text(json.dumps(manifest))
    dataset_hash = study_dataset_hash(manifest)
    with np.load(split / "public" / "a_train.npz") as data:
        FlowNormalizer(features).fit(data["x"]).save(study / "normalizer.npz")
    (study / "artifacts").mkdir()
    checkpoint = study / "artifacts" / "candidate.npz"
    save_hypernet(HyperNet(NeuronSpec(43, 8, 2)), checkpoint)
    record = {"run_id": "candidate", "config": Config(model="b3", width=8,
              functional_hidden=2).as_dict(), "seed": 11, "dataset_hash": dataset_hash,
              "git_commit": "fixture-commit", "metrics": {"detection_rate": .5},
              "checkpoint_path": "artifacts/candidate.npz", "status": "completed"}
    Registry(study / "registry.sqlite").write(record)
    (study / "report.json").write_text(json.dumps({
        "kind": "real_nf_v2_endpoint_group_alternate_study", "dataset_hash": dataset_hash,
        "selected_runs": {"b3": "candidate"}, "scoreboard": {}, "partitions": {},
    }))
    return study, split


def test_slider_generates_int8_without_training_and_evasion_is_constrained(tmp_path):
    state = DemoState(*_fixture(tmp_path))
    before = state.hypernet.optimizer_steps
    result = state.generate(.2, 1, 2)
    assert state.hypernet.optimizer_steps == before
    assert result["optimizer_steps_during_generation"] == 0
    assert result["macs"] >= 0 and result["int8_model_bytes"] > 0
    assert result["latency"]["median_ns_per_flow"] > 0
    assert result["latency"]["batch_rows"] == 120
    artifact = state.generated_dir / "generation000001_cost0200_load1_alerts2.hwi"
    assert artifact.exists()
    proof = result["export"]
    verification = result["kernel_verification"]
    assert proof["sha256"] and proof["manifest_sha256"]
    assert proof["active_neurons"] + proof["pruned_neurons"] == proof["total_neurons"]
    assert proof["dimensions"]["neurons_after_pruning"] == proof["active_neurons"]
    assert verification["verified"] and verification["rows_compared"] == len(state.b_x)
    assert verification["mismatched_rows"] == verification["maximum_absolute_logit_difference"] == 0
    assert all(pair["c"] == pair["python"] for pair in verification["sample_integer_logits"])
    artifact_handler = object.__new__(make_handler(state))
    artifact_handler.path = f"/api/artifact?name={proof['filename']}"
    artifact_handler.wfile = BytesIO()
    artifact_handler.send_response = lambda code: None
    artifact_handler.send_header = lambda key, value: None
    artifact_handler.end_headers = lambda: None
    artifact_handler.do_GET()
    assert artifact_handler.wfile.getvalue() == artifact.read_bytes()
    event = state.demo_registry.events("model_generation")[0]
    assert result["registry_event_id"] == event["event_id"]
    assert event["payload"]["result"]["int8"] == result["int8"]
    assert event["payload"]["artifact"] == str(artifact)
    traffic = state.traffic(5)
    assert len(traffic["sample"]) == 5
    assert state.demo_registry.events("traffic_replay")[0]["payload"]["result"]["sample"] == traffic["sample"]
    attacked = state.evasion("padding", .1)
    assert attacked["feature_space_proxy"] and attacked["immutable_fields_preserved"]
    assert attacked["attack_rows"] == 60
    assert attacked["constraint_checks"] == {"immutable_fields_preserved": True,
        "epsilon_bound_preserved": True, "nonnegative_features": True,
        "all_changes_within_mutable_features": True}
    assert attacked["changed_flows"] > 0 and attacked["feature_changes"]
    assert len(attacked["examples"]) == min(attacked["changed_flows"], 12)
    assert attacked["detected_to_missed"] <= attacked["original_detected"]
    assert attacked["candidate_detected"] == attacked["original_detected"] - attacked["detected_to_missed"] + attacked["missed_to_detected"]
    unchanged = state.evasion("padding", 0)
    assert unchanged["changed_flows"] == unchanged["changed_feature_count"] == 0
    assert unchanged["original_detection_rate"] == unchanged["evasion_detection_rate"]
    assert state.demo_registry.events("evasion_measurement")[0]["payload"]["result"]["evasion_detection_rate"] == attacked["evasion_detection_rate"]
    replay = state.autopilot(window=20)
    assert replay["windows"] == 6 and replay["temporal_order_verified"] is False
    assert state.demo_registry.events("autopilot_measurement")[0]["payload"]["result"]["windows"] == replay["windows"]
    if not replay["equal_detection_within_tolerance"]:
        assert replay["cost_reduction_at_equal_detection"] is None


def test_human_gate_rejection_prevents_execution_and_approval_records_simulation(tmp_path):
    state = DemoState(*_fixture(tmp_path))
    rejected = state.decide(False)
    assert rejected["outcome"]["executed"] is False
    assert not state.demo_registry.all_runs()
    approved = state.decide(True)
    assert approved["outcome"]["executed"] is True
    assert approved["production_action"] is False
    assert state.demo_registry.read(approved["outcome"]["run_id"])["status"] == "simulated"


def test_domain_analysis_uses_generated_parameters_and_actual_domain_rows(tmp_path):
    state = DemoState(*_fixture(tmp_path))
    measured = state.domain_analysis()
    assert measured["source_domain"]["name"] == "UNSW-NB15 A-train"
    assert measured["target_domain"]["name"] == "CSE-CIC-IDS2018 B-evaluation"
    assert len(measured["feature_points"]) == 41
    assert len(measured["source_logit_curve"]) == len(measured["target_logit_curve"]) == 41
    assert measured["math"]["network"].startswith("f(x)")

    same_context = state.domain_analysis(*measured["source_context"])
    assert same_context["same_context"]
    assert same_context["parameter_delta_l2"] == pytest.approx(0, abs=1e-12)

    changed_context = state.domain_analysis(2, 2)
    assert changed_context["target_context"] == [2, 2]
    assert changed_context["parameter_delta_l2"] > 0
    assert changed_context["target_context_mode"] == "user_selected_context_scenario"
    assert changed_context["registry_event_id"]


def test_five_views_limits_and_offline_assets():
    html = (STATIC / "index.html").read_text()
    for view in ("traffic", "posture", "evasion", "notebook", "scoreboard"):
        assert f'id="{view}"' in html
    assert 'id="limits-button"' in html and 'id="limits"' in html
    assert "https://" not in html and "http://" not in html
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert all(path in js for path in ("/api/generate", "/api/evasion", "/api/autopilot", "/api/decision", "/api/domain/compare"))
    assert "B HV · 3D research" in js and "r.latency_note" in js


def test_http_routes_in_process_without_network(tmp_path):
    state = DemoState(*_fixture(tmp_path))
    handler = object.__new__(make_handler(state))
    handler.path = "/api/overview"
    handler.wfile = BytesIO()
    handler.send_response = lambda code: None
    handler.send_header = lambda key, value: None
    handler.end_headers = lambda: None
    handler.do_GET()
    assert json.loads(handler.wfile.getvalue())["agent_mode"].startswith("recorded scripted")
    handler.path = "/"
    handler.wfile = BytesIO()
    handler.do_GET()
    assert b"Traffic console" in handler.wfile.getvalue()
    handler.path = "/api/live/status"
    handler.wfile = BytesIO()
    handler.do_GET()
    status = json.loads(handler.wfile.getvalue())
    assert status["running"] is False
    assert status["ground_truth_available"] is False
    handler.path = "/api/live/interfaces"
    handler.wfile = BytesIO()
    handler.do_GET()
    assert "interfaces" in json.loads(handler.wfile.getvalue())
    handler.path = "/api/domain/compare?load=2&alerts=1&feature=IN_BYTES"
    handler.wfile = BytesIO()
    handler.do_GET()
    comparison = json.loads(handler.wfile.getvalue())
    assert comparison["target_context"] == [2, 1]
    assert comparison["target_context_mode"] == "user_selected_context_scenario"


def test_demo_refuses_mismatched_split_and_tampered_public_partition(tmp_path):
    study, split = _fixture(tmp_path)
    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["kind"] = "different_protocol"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="split does not match"):
        DemoState(study, split)
    manifest["kind"] = "endpoint_pair_group_disjoint_v1"
    manifest_path.write_text(json.dumps(manifest))
    with (split / "public" / "b.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="partition hash mismatch"):
        DemoState(study, split)
