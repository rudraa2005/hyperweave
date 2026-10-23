import ast
import json
from pathlib import Path

import numpy as np
import pytest

from hyperweave.agents.budget import BudgetLedger
from hyperweave.agents.cycle import AgentCycle, HumanDecision, ReplayClient, ScriptedClient
from hyperweave.agents.experiment import ExperimentHarness, synthetic_fixture
from hyperweave.agents.proposals import REQUIRED, SCHEMA_PATH, validate_proposal
from hyperweave.agents.surrogate import Surrogate, prediction_calibration
from hyperweave.agents.tools import (AdversarialValidatorTools, AgentFilesystem, CoordinatorTools,
    ExperimentalScientistTools, NeuronArchitectTools, OptimizationScientistTools, TrainingStrategistTools)
from hyperweave.baselines.nsga2 import NSGA2Search
from hyperweave.baselines.phase3 import run as run_phase3, scripted_responses, run_condition
from hyperweave.config import Config, SEARCH_OPTIONS
from hyperweave.eval.sealed import SealedValidator
from hyperweave.models import NeuronSpec
from hyperweave.registry import Registry

pytestmark = pytest.mark.phase3


def _proposal(width=8):
    return {"hypothesis": "A narrower model may lower cost", "config_diff": {"width": width},
            "predicted_effect": .01, "cost_estimate": 1,
            "success_criterion": "A-val threshold and B detection logged",
            "action": "run_experiment", "rationale": "Bounded architecture test",
            "validation_plan": "Request one coarse validator verdict"}


def _setup(tmp_path, *, condition="b5", blind=False, effect=.03):
    registry = Registry(tmp_path / "registry.sqlite")
    experiment = ExperimentHarness(registry, tmp_path / "artifacts", *synthetic_fixture(samples=240))
    validator = SealedValidator(tmp_path / "private" / "checks.sqlite", lambda _: effect, query_limit=10)
    responses = scripted_responses(count=1, validation_calls=1)
    responses["Experimental Scientist"] = [_proposal()]
    client = ScriptedClient(responses, model_version="mock-model-1", temperature=0.0)
    cycle = AgentCycle(registry, experiment, validator, client, condition=condition, blind=blind,
                       budget_limit=1)
    base = Config(model=condition, epochs=1, batch_size=64, target_fpr=.01)
    return registry, experiment, validator, cycle, base


def test_proposal_schema_valid():
    base = Config()
    assert validate_proposal(_proposal(), base).config_diff == {"width": 8}
    schema = json.loads(SCHEMA_PATH.read_text())
    assert set(schema["required"]) == REQUIRED
    for key in REQUIRED:
        invalid = _proposal()
        del invalid[key]
        with pytest.raises(ValueError):
            validate_proposal(invalid, base)
    with pytest.raises(TypeError):
        validate_proposal(dict(_proposal(), predicted_effect="better"), base)
    with pytest.raises(TypeError):
        validate_proposal(dict(_proposal(), config_diff={"seed": 99}), base)
    with pytest.raises(ValueError):
        validate_proposal(dict(_proposal(), config_diff={"context": [True, 0]}), base)
    with pytest.raises(ValueError, match="shared search"):
        validate_proposal(dict(_proposal(), config_diff={"width": 7}), base)


def test_invalid_proposal_costs_nothing(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path)
    cycle.client.responses["Experimental Scientist"] = [dict(_proposal(), predicted_effect="unknown")]
    outcome = cycle.run("invalid1", base, HumanDecision(True, "test_human"))
    assert outcome["status"] == "invalid" and not outcome["executed"]
    assert cycle.budget.remaining("b5") == 1
    assert registry.all_runs() == []


def test_budget_enforced(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    ledger = BudgetLedger(registry, limits={"b5": 1})
    assert not ledger.reserve("b5", "too_large", 2, approved=True, actor="human")
    assert ledger.remaining("b5") == 1
    assert registry.events("budget_refusal")
    with pytest.raises(ValueError, match="already decided"):
        ledger.reserve("b5", "too_large", 1, approved=True, actor="human")
    assert ledger.reserve("b5", "valid", 1, approved=True, actor="human")
    assert ledger.remaining("b5") == 0
    with pytest.raises(PermissionError):
        ledger.start("not_approved")
    ledger.release_unstarted("valid")
    assert ledger.remaining("b5") == 1


def test_validator_query_budget(tmp_path):
    validator = SealedValidator(tmp_path / "private.sqlite", lambda _: .03, query_limit=10)
    for i in range(10):
        assert validator.query(f"r{i}")["verdict"] == "accept"
    with pytest.raises(RuntimeError, match="budget"):
        validator.query("r10")
    assert validator.count() == 10


def test_validator_coarse_output(tmp_path):
    for effect, expected in ((.03, ("accept", "better")), (-.03, ("reject", "worse")),
                             (0, ("inconclusive", "similar")), (None, ("inconclusive", "unknown"))):
        validator = SealedValidator(tmp_path / f"v{effect}.sqlite", lambda _, e=effect: e)
        reply = validator.query("candidate")
        assert set(reply) == {"verdict", "bucket"}
        assert (reply["verdict"], reply["bucket"]) == expected
        assert not any(isinstance(v, (int, float)) for v in reply.values())
    checks = SealedValidator(tmp_path / "checks.sqlite", {"shift": lambda _: -.03,
                                                            "latency": lambda _: .04}, query_limit=10)
    tools = AdversarialValidatorTools(checks.query)
    assert tools.shift_test("r") == {"verdict": "reject", "bucket": "worse"}
    assert tools.evasion_test("r") == {"verdict": "inconclusive", "bucket": "unknown"}
    assert tools.latency_profile("r") == {"verdict": "accept", "bucket": "better"}
    assert tools.regression_check("r") == {"verdict": "inconclusive", "bucket": "unknown"}
    assert checks.count() == 4


def test_agents_cannot_read_sealed(tmp_path):
    public = tmp_path / "public"
    private = public / "private"
    private.mkdir(parents=True)
    (private / "secret.txt").write_text("hidden")
    (public / "visible.txt").write_text("okay")
    fs = AgentFilesystem(public, denied_roots=(private,))
    assert fs.read_text(public / "visible.txt") == "okay"
    with pytest.raises(PermissionError):
        fs.read_text(private / "secret.txt")
    with pytest.raises(PermissionError):
        fs.read_text(tmp_path / "elsewhere.txt")
    root = Path(__file__).resolve().parents[2] / "hyperweave" / "agents"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "eval.sealed" not in ast.unparse(node)


def test_tool_typing(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    ledger = BudgetLedger(registry, limits={"b5": 1})
    with pytest.raises(TypeError):
        CoordinatorTools(registry, ledger).remaining(3)
    with pytest.raises(TypeError):
        CoordinatorTools(registry, ledger).approve("b5", "p1", 1, decision=True)
    architect = NeuronArchitectTools()
    with pytest.raises(TypeError):
        architect.profile(NeuronSpec(3, 2, 2), [1, 0])
    with pytest.raises(TypeError):
        architect.parameter_count({"inputs": 3})
    trainer = TrainingStrategistTools(registry, lambda *_: None, ledger)
    with pytest.raises(TypeError):
        trainer.edit_config({}, {})
    with pytest.raises(PermissionError):
        trainer.launch("not_approved", Config(model="b5"))
    with pytest.raises(TypeError):
        OptimizationScientistTools(registry).frontier(3)
    with pytest.raises(TypeError):
        OptimizationScientistTools(registry).hypervolume_evaluator("bad")
    with pytest.raises(TypeError):
        ExperimentalScientistTools(registry, Surrogate(registry)).compare("r1")
    with pytest.raises(TypeError):
        AdversarialValidatorTools(lambda _: {}).validate(3)
    with pytest.raises(TypeError):
        AdversarialValidatorTools(lambda *_: {}).evasion_test(3)


def test_registry_logs_prompt_and_model(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path)
    cycle.run("p1", base, HumanDecision(True, "test_human"))
    calls = registry.events("agent_call")
    assert {event["payload"]["role"] for event in calls} == {
        "Neuron Architect", "Training Strategist", "Optimization Scientist",
        "Experimental Scientist", "Coordinator", "Adversarial Validator"}
    assert all({"prompt", "model_version", "temperature", "tool_results", "response"}
               <= event["payload"].keys() for event in calls)
    assert all(event["payload"]["model_version"] == "mock-model-1" for event in calls)
    assert registry.events("frontier_snapshot")


def test_replay_deterministic(tmp_path):
    first, _, _, cycle, base = _setup(tmp_path / "first")
    outcome = cycle.run("p1", base, HumanDecision(True, "test_human"))
    assert outcome["status"] == "completed"
    replay_client = ReplayClient.from_events(first.events())
    second = Registry(tmp_path / "second" / "registry.sqlite")
    experiment = ExperimentHarness(second, tmp_path / "second" / "artifacts", *synthetic_fixture(samples=240))
    validator = SealedValidator(tmp_path / "second" / "private" / "checks.sqlite", lambda _: .03)
    replay = AgentCycle(second, experiment, validator, replay_client, condition="b5", budget_limit=1)
    assert replay.run("p1", base, HumanDecision(True, "test_human")) == outcome
    assert second.events() == first.events()
    assert second.all_runs() == first.all_runs()


def test_surrogate_trains(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    surrogate = Surrogate(registry)
    assert surrogate.predict(Config())["source"] == "sparse_fallback"
    for i, width in enumerate((4, 8, 16)):
        config = Config(model="b5", width=width)
        registry.write({"run_id": f"r{i}", "config": config.as_dict(), "seed": 11,
                        "dataset_hash": "synthetic", "git_commit": "test", "checkpoint_path": f"r{i}.npz",
                        "metrics": {"proxy_hv_gain_per_cost": .01 * (i + 1)}})
    prediction = surrogate.predict(Config(model="b5", width=12))
    assert prediction["source"] == "ridge_proxy" and prediction["samples"] == 3
    assert np.isfinite(prediction["prediction"]) and prediction["uncertainty"] > 0
    for i, width in enumerate((4, 8, 16)):
        config = Config(model="b5", width=width)
        registry.write({"run_id": f"real{i}", "config": config.as_dict(), "seed": 11,
                        "dataset_hash": "research", "git_commit": "test", "checkpoint_path": f"real{i}.npz",
                        "metrics": {"hv_gain_per_cost": .02 * (i + 1)}})
    assert surrogate.predict(Config(model="b5", width=12))["source"] == "ridge_hv"


def test_prediction_logged_for_calibration(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path)
    outcome = cycle.run("p1", base, HumanDecision(True, "test_human"))
    record = registry.read(outcome["run_id"])
    assert record["predicted_effect"] == .01
    assert prediction_calibration(registry)["count"] == 1
    assert np.isfinite(prediction_calibration(registry)["mae"])


def test_b4_same_config_space(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    experiment = ExperimentHarness(registry, tmp_path / "artifacts", *synthetic_fixture(samples=200))
    search = NSGA2Search(registry, experiment, Config(model="b4", epochs=1, target_fpr=.01),
                         seeds=(11, 23), trials_per_seed=2)
    records = search.run()
    assert len(records) == 4 and search.budget.remaining("b4") == 0
    assert all(Config(**record["config"]).model == "b4" for record in records)
    for record in records:
        config = Config(**record["config"])
        assert all((tuple(getattr(config, key)) if isinstance(getattr(config, key), list)
                    else getattr(config, key)) in options for key, options in SEARCH_OPTIONS.items())
    assert len(registry.events("search_decision")) == 4
    research = NSGA2Search(Registry(tmp_path / "research.sqlite"),
                           ExperimentHarness(Registry(tmp_path / "research.sqlite"), tmp_path / "research_artifacts",
                                             *synthetic_fixture(samples=200)),
                           Config(model="b4", epochs=1, target_fpr=.01),
                           seeds=(11, 23), trials_per_seed=2, objective_mode="research")
    with pytest.raises(ValueError, match="pinned-core"):
        research.run()


def test_b5b_blind(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path, condition="b5b", blind=True)
    outcome = cycle.run("p1", base, HumanDecision(True, "test_human"))
    assert outcome["status"] == "completed"
    forbidden = {"metrics", "detection_rate", "pr_auc", "macs", "hypervolume", "hv_contribution"}
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert all(not forbidden & keys(event["payload"]["tool_results"])
               for event in registry.events("agent_call"))


def test_b5_feedback_visible_b5b_hidden(tmp_path):
    for condition in ("b5", "b5b"):
        folder = tmp_path / condition
        registry = Registry(folder / "registry.sqlite")
        experiment = ExperimentHarness(registry, folder / "artifacts", *synthetic_fixture(samples=200))
        validator = SealedValidator(folder / "private.sqlite", lambda _: .03)
        outcomes, remaining = run_condition(registry, experiment, validator, condition,
                                            seeds=(11,), trials_per_seed=2)
        assert len(outcomes) == 2 and remaining == 0
        strategist_calls = [e["payload"] for e in registry.events("agent_call")
                            if e["payload"]["role"] == "Training Strategist"]
        prior = strategist_calls[1]["tool_results"]["prior_diagnostics"]
        if condition == "b5":
            assert "detection_rate" in prior
        else:
            assert prior == {"status": "completed"}


def test_agent_decision_recorded(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path)
    outcome = cycle.run("p1", base, HumanDecision(True, "test_human"))
    assert outcome["executed"]
    assert registry.read(outcome["run_id"])["config"]["width"] == 8
    assert registry.events("proposal")[0]["payload"]["proposal"]["config_diff"] == {"width": 8}


def test_human_rejection_prevents_execution(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path)
    outcome = cycle.run("p1", base, HumanDecision(False, "test_human"))
    assert outcome["status"] == "rejected_by_human" and not outcome["executed"]
    assert registry.all_runs() == [] and cycle.budget.remaining("b5") == 1


def test_simulated_action_labelled(tmp_path):
    registry, _, _, cycle, base = _setup(tmp_path)
    cycle.client.responses["Experimental Scientist"] = [dict(_proposal(), action="simulate_response")]
    outcome = cycle.run("p1", base, HumanDecision(True, "test_human"), request_validation=False)
    record = registry.read(outcome["run_id"])
    assert record["status"] == "simulated" and record["action_result"]["simulated"] is True


def test_failed_experiment_recorded(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    experiment = ExperimentHarness(registry, tmp_path / "artifacts", *synthetic_fixture(samples=100))
    bad = Config(model="b5", input_dim=44, target_fpr=.01)
    with pytest.raises(RuntimeError, match="input dimension"):
        experiment("failed1", bad, predicted_effect=.01)
    record = registry.read("failed1")
    assert record["status"] == "failed" and record["predicted_effect"] == .01


def test_phase3_full_budgeted_workflow(tmp_path):
    report = run_phase3(tmp_path / "full")
    assert report["api_used"] is False and report["validator_effects"] == "scripted"
    assert report["b4_runs"] == report["b5_runs"] == report["b5b_runs"] == 12
    assert report["remaining"] == {"b4": 0, "b5": 0, "b5b": 0}
    assert report["validator_queries"] == 6
    registry = Registry(tmp_path / "full" / "registry.sqlite")
    assert len(registry.all_runs()) == 36
    seeds = {model: {r["seed"] for r in registry.all_runs() if r["config"]["model"] == model}
             for model in ("b4", "b5", "b5b")}
    assert all(value == {11, 23, 37} for value in seeds.values())
    assert all(r["metrics"]["threshold_source"] == "a_val" for r in registry.all_runs())
    selected = report["selected_outcome"]
    chosen = registry.read(selected["run_id"])
    assert selected["approved"] and selected["status"] == "completed"
    assert selected["detection_rate"] == chosen["metrics"]["detection_rate"]
    assert selected["model_size_bytes"] == chosen["metrics"]["model_size_bytes"]
    assert selected["latency_ns_per_flow"] is None
