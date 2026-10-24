import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from hyperweave.config import Config
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
    with np.load(split / "public" / "a_train.npz") as data:
        FlowNormalizer(features).fit(data["x"]).save(study / "normalizer.npz")
    (study / "artifacts").mkdir()
    checkpoint = study / "artifacts" / "candidate.npz"
    save_hypernet(HyperNet(NeuronSpec(43, 8, 2)), checkpoint)
    record = {"run_id": "candidate", "config": Config(model="b3", width=8,
              functional_hidden=2).as_dict(), "seed": 11, "dataset_hash": "fixture-hash",
              "git_commit": "fixture-commit", "metrics": {"detection_rate": .5},
              "checkpoint_path": "artifacts/candidate.npz", "status": "completed"}
    Registry(study / "registry.sqlite").write(record)
    (study / "report.json").write_text(json.dumps({
        "kind": "real_nf_v2_endpoint_group_alternate_study", "dataset_hash": "fixture-hash",
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
    assert (state.generated_dir / "cost0200_load1_alerts2.hwi").exists()
    assert len(state.traffic(5)["sample"]) == 5
    attacked = state.evasion("padding", .1)
    assert attacked["feature_space_proxy"] and attacked["immutable_fields_preserved"]
    assert attacked["attack_rows"] == 60
    replay = state.autopilot(window=20)
    assert replay["windows"] == 6 and replay["temporal_order_verified"] is False
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


def test_five_views_limits_and_offline_assets():
    html = (STATIC / "index.html").read_text()
    for view in ("traffic", "posture", "evasion", "notebook", "scoreboard"):
        assert f'id="{view}"' in html
    assert 'id="limits-button"' in html and 'id="limits"' in html
    assert "https://" not in html and "http://" not in html
    js = (STATIC / "app.js").read_text()
    assert all(path in js for path in ("/api/generate", "/api/evasion", "/api/autopilot", "/api/decision"))


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
