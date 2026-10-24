"""Offline scripted Phase 3 research-loop integration and budget smoke run."""
import argparse
import json
from pathlib import Path

from hyperweave.agents.cycle import AgentCycle, HumanDecision, ScriptedClient
from hyperweave.agents.experiment import ExperimentHarness, synthetic_fixture
from hyperweave.baselines.nsga2 import NSGA2Search
from hyperweave.config import Config
from hyperweave.eval.sealed import SealedValidator
from hyperweave.registry import Registry


SEEDS = (11, 23, 37)
TRIALS_PER_SEED = 4


def scripted_responses(*, count=12, validation_calls=3):
    proposals = []
    for i in range(count):
        proposals.append({"hypothesis": "Changing neuron width may improve the debug cost-quality proxy",
                          "config_diff": {"width": [4, 8, 12, 16][i % 4], "functional_hidden": [2, 3, 4, 2][i % 4]},
                          "predicted_effect": 0.01, "cost_estimate": 1,
                          "success_criterion": "B detection and MACs logged under the fixed A-val threshold",
                          "action": "run_experiment", "rationale": "Compare a bounded architecture change",
                          "validation_plan": "Use at most one coarse validator query per seed"})
    return {"Neuron Architect": [{"note": "profile reviewed"} for _ in range(count)],
            "Training Strategist": [{"config_diff": {}} for _ in range(count)],
            "Optimization Scientist": [{"note": "frontier reviewed"} for _ in range(count)],
            "Experimental Scientist": proposals,
            "Coordinator": [{"recommendation": "seek human decision"} for _ in range(count)],
            "Adversarial Validator": [{"assessment": "coarse verdict recorded"} for _ in range(validation_calls)]}


def run_condition(registry, experiment, validator, condition, *, seeds=SEEDS, trials_per_seed=TRIALS_PER_SEED,
                  target_fpr=.01, shift_columns=(0,)):
    count = len(seeds) * trials_per_seed
    client = ScriptedClient(scripted_responses(count=count, validation_calls=len(seeds)))
    cycle = AgentCycle(registry, experiment, validator, client, condition=condition,
                       blind=(condition == "b5b"), budget_limit=count)
    outcomes = []
    for seed in seeds:
        for trial in range(trials_per_seed):
            base = Config(model=condition, seed=seed, input_dim=43, width=16,
                          functional_hidden=4, epochs=1, batch_size=128, target_fpr=target_fpr,
                          shift_columns=shift_columns)
            outcomes.append(cycle.run(f"{condition}_seed{seed}_trial{trial}", base,
                                      HumanDecision(True, "scripted_human_approval"),
                                      request_validation=(trial == 0)))
    return outcomes, cycle.budget.remaining(condition)


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh output directory for immutable Phase 3 replay")
    output.mkdir(parents=True, exist_ok=True)
    registry = Registry(output / "registry.sqlite")
    train, val, b = synthetic_fixture()
    experiment = ExperimentHarness(registry, output / "artifacts", train, val, b)
    # Scripted effects test the validator contract; they are not real sealed evaluation.
    validator = SealedValidator(output / "private" / "validator.sqlite", lambda _: .03, query_limit=10)
    base_b4 = Config(model="b4", seed=11, epochs=1, batch_size=128, target_fpr=.01)
    b4 = NSGA2Search(registry, experiment, base_b4)
    b4_records = b4.run()
    b5_outcomes, b5_remaining = run_condition(registry, experiment, validator, "b5")
    b5b_outcomes, b5b_remaining = run_condition(registry, experiment, validator, "b5b")
    rows = registry.all_runs()
    outcomes = [event["payload"] for event in registry.events("cycle_outcome")]
    chosen = next(row for row in rows if row["run_id"] == "b5_b5_seed11_trial0")
    chosen_proposal_id = "b5_seed11_trial0"
    decision = next(event["payload"] for event in registry.events("human_decision")
                    if event["payload"]["proposal_id"] == chosen_proposal_id)
    verdict = next(event["payload"] for event in registry.events("validator_verdict")
                   if event["payload"]["run_id"] == chosen["run_id"])
    measured = chosen["metrics"]
    report = {"kind": "synthetic_scripted_phase3_debug", "api_used": False,
              "validator_effects": "scripted", "dataset_hash": chosen["dataset_hash"],
              "seeds": list(SEEDS), "run_budget_per_condition": 12,
              "b4_runs": sum(row["config"]["model"] == "b4" for row in rows),
              "b5_runs": sum(row["config"]["model"] == "b5" for row in rows),
              "b5b_runs": sum(row["config"]["model"] == "b5b" for row in rows),
              "remaining": {"b4": b4.budget.remaining("b4"), "b5": b5_remaining, "b5b": b5b_remaining},
              "validator_queries": len(registry.events("validator_verdict")),
              "outcomes": {"b5": [item for item in outcomes if item.get("run_id", "").startswith("b5_")],
                           "b5b": [item for item in outcomes if item.get("run_id", "").startswith("b5b_")]},
              "selected_outcome": {"run_id": chosen["run_id"], "status": chosen["status"],
                                   "action": chosen["proposal"]["action"],
                                   "approved": decision["approved"], "approval_actor": decision["actor"],
                                   "traffic_processed": measured["traffic_processed"],
                                   "detection_rate": measured["detection_rate"], "fpr": measured["fpr"],
                                   "pr_auc": measured["pr_auc"], "macs": measured["macs"],
                                   "model_size_bytes": measured["model_size_bytes"],
                                   "latency_ns_per_flow": measured["latency"]["median_ns_per_flow"],
                                   "validator": {"verdict": verdict["verdict"], "bucket": verdict["bucket"]}}}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({key: report[key] for key in ("kind", "api_used", "validator_effects", "seeds",
                                                    "run_budget_per_condition", "b4_runs", "b5_runs",
                                                    "b5b_runs", "remaining", "validator_queries",
                                                    "selected_outcome")}, indent=2))


if __name__ == "__main__":
    main()
