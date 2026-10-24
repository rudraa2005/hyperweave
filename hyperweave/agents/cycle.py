"""Six-role proposal, human decision, execution, validation workflow."""
import copy

import numpy as np

from hyperweave.config import Config, SEARCH_OPTIONS
from hyperweave.models import NeuronSpec

from .budget import BudgetLedger, HumanDecision
from .proposals import validate_proposal
from .surrogate import Surrogate
from .tools import (AdversarialValidatorTools, CoordinatorTools, ExperimentalScientistTools,
                    NeuronArchitectTools, OptimizationScientistTools, TrainingStrategistTools)


ROLES = ("Neuron Architect", "Training Strategist", "Optimization Scientist",
         "Experimental Scientist", "Coordinator", "Adversarial Validator")


def _role_config_diff(reply, role, allowed):
    if not isinstance(reply, dict):
        raise TypeError(f"{role} response must be an object")
    diff = reply.get("config_diff", {})
    if not isinstance(diff, dict) or not set(diff) <= allowed:
        raise ValueError(f"{role} proposed unsupported config settings")
    for key, value in diff.items():
        comparable = tuple(value) if isinstance(value, list) else value
        if comparable not in SEARCH_OPTIONS[key]:
            raise ValueError(f"{role} config diff is outside the shared search space")
    return diff


class ScriptedClient:
    def __init__(self, responses, *, model_version="scripted-v1", temperature=0.0):
        self.responses = {key: list(values) for key, values in responses.items()}
        self.model_version = model_version
        self.temperature = temperature

    def complete(self, role, prompt):
        if role not in self.responses or not self.responses[role]:
            raise RuntimeError(f"no scripted response for {role}")
        return copy.deepcopy(self.responses[role].pop(0))


class ReplayClient(ScriptedClient):
    @classmethod
    def from_events(cls, events):
        calls = [event["payload"] for event in events if event["kind"] == "agent_call"]
        if not calls:
            raise ValueError("no agent calls to replay")
        responses = {}
        for call in calls:
            responses.setdefault(call["role"], []).append(call["response"])
        versions = {(call["model_version"], call["temperature"]) for call in calls}
        if len(versions) != 1:
            raise ValueError("replay needs one fixed model/version setting")
        version, temperature = versions.pop()
        return cls(responses, model_version=version, temperature=temperature)


class AgentCaller:
    def __init__(self, registry, client):
        self.registry = registry
        self.client = client

    def call(self, role, prompt, tool_results, *, proposal_id=None, condition=None):
        if role not in ROLES or not isinstance(prompt, str) or not isinstance(tool_results, dict):
            raise TypeError("typed role, prompt and tool results required")
        response = self.client.complete(role, prompt)
        self.registry.append_event("agent_call", {"role": role, "prompt": prompt,
            "proposal_id": proposal_id, "condition": condition,
            "model_version": self.client.model_version, "temperature": self.client.temperature,
            "tool_results": tool_results, "response": response})
        return response


class AgentCycle:
    def __init__(self, registry, experiment, validator, client, *, condition="b5", blind=False,
                 budget_limit=12, bake_action=None):
        if condition not in {"b5", "b5b"}:
            raise ValueError("agent cycle condition must be b5 or b5b")
        if condition == "b5b" and not blind:
            raise ValueError("B5b requires evidence-blind agent tools")
        self.registry = registry
        self.experiment = experiment
        self.validator = validator
        self.client = client
        self.condition = condition
        self.blind = blind
        self.bake_action = bake_action
        self.budget = BudgetLedger(registry, limits={condition: budget_limit})
        self.caller = AgentCaller(registry, client)

    def run(self, proposal_id: str, base: Config, human: HumanDecision, *, request_validation=True):
        if not isinstance(proposal_id, str) or not proposal_id or not isinstance(base, Config) or not isinstance(human, HumanDecision):
            raise TypeError("proposal ID, Config and HumanDecision required")
        if base.model != self.condition:
            raise ValueError("base model condition mismatch")
        architect = NeuronArchitectTools()
        spec = NeuronSpec(base.input_dim, base.width, base.functional_hidden)
        profile = architect.profile(spec, np.ones(spec.neurons, dtype=bool), blind=self.blind)
        architect_reply = self.caller.call("Neuron Architect", "Review architecture and deployment profile",
                                           {"profile": profile}, proposal_id=proposal_id,
                                           condition=self.condition)
        architect_diff = _role_config_diff(architect_reply, "Neuron Architect",
                                          {"width", "functional_hidden"})
        working = Config(**dict(base.as_dict(), **architect_diff))
        if architect_diff:
            self.registry.append_event("role_config_applied", {"role": "Neuron Architect",
                                                                "proposal_id": proposal_id,
                                                                "config_diff": architect_diff})
        prior = [r for r in self.registry.all_runs() if r["config"]["model"] == self.condition]
        diagnostics = ({"status": "missing"} if not prior else TrainingStrategistTools(
            self.registry, lambda *_: None, self.budget, blind=self.blind).diagnostics(prior[-1]["run_id"]))
        strategist_reply = self.caller.call("Training Strategist", "Review training settings",
                                            {"config": working.as_dict(), "prior_diagnostics": diagnostics},
                                            proposal_id=proposal_id, condition=self.condition)
        strategist_diff = _role_config_diff(strategist_reply, "Training Strategist",
                                            {"learning_rate", "epochs", "batch_size", "loss_mode",
                                             "dirichlet_concentration", "shift_columns"})
        working = Config(**dict(working.as_dict(), **strategist_diff))
        if strategist_diff:
            self.registry.append_event("role_config_applied", {"role": "Training Strategist",
                                                                "proposal_id": proposal_id,
                                                                "config_diff": strategist_diff})
        optimizer = OptimizationScientistTools(self.registry, blind=self.blind)
        frontier = optimizer.frontier(self.condition)
        ranked_candidates = optimizer.rank_candidates(self.condition)
        self.registry.append_event("frontier_snapshot", {"condition": self.condition,
                                                        "proposal_id": proposal_id, "frontier": frontier})
        optimizer_reply = self.caller.call("Optimization Scientist", "Review current Pareto evidence",
                                           {"frontier": frontier, "ranked_candidates": ranked_candidates},
                                           proposal_id=proposal_id, condition=self.condition)
        if not isinstance(optimizer_reply, dict):
            raise TypeError("Optimization Scientist response must be an object")
        preferred_run_id = optimizer_reply.get("preferred_run_id")
        if preferred_run_id is not None:
            preferred = self.registry.read(preferred_run_id) if isinstance(preferred_run_id, str) else None
            if self.blind or preferred is None or preferred["config"]["model"] != self.condition or \
                    preferred.get("status") != "completed":
                raise ValueError("Optimization Scientist preferred run is unavailable")
        experimental = ExperimentalScientistTools(self.registry,
                                                  Surrogate(self.registry, condition=self.condition),
                                                  blind=self.blind)
        predicted = experimental.predict(working)
        candidate = self.caller.call("Experimental Scientist", "Propose one bounded experiment",
                                     {"surrogate": predicted, "budget": self.budget.remaining(self.condition),
                                      "optimization_advice": {"frontier": frontier,
                                                              "ranked_candidates": ranked_candidates,
                                                              "preferred_run_id": preferred_run_id}},
                                     proposal_id=proposal_id, condition=self.condition)
        try:
            proposal = validate_proposal(candidate, working)
        except (TypeError, ValueError) as exc:
            self.registry.append_event("proposal_rejected", {"proposal_id": proposal_id, "reason": str(exc)})
            return {"proposal_id": proposal_id, "status": "invalid", "approved": False, "executed": False}
        self.registry.append_event("proposal", {"proposal_id": proposal_id, "proposal": proposal.as_dict(),
                                                "condition": self.condition, "blind": self.blind})
        if (not self.blind and isinstance(predicted.get("prediction"), (int, float)) and
                predicted.get("source") not in {"sparse_fallback", None} and
                abs(proposal.predicted_effect - predicted["prediction"]) > predicted.get("uncertainty", 0)):
            self.registry.append_event("surrogate_override", {"proposal_id": proposal_id,
                "surrogate_prediction": predicted["prediction"],
                "agent_prediction": proposal.predicted_effect, "reason": proposal.rationale})
        coordinator_reply = self.caller.call("Coordinator", "Recommend an action subject to human approval",
                                             {"proposal": proposal.as_dict(),
                                              "remaining_runs": self.budget.remaining(self.condition)},
                                             proposal_id=proposal_id, condition=self.condition)
        if not isinstance(coordinator_reply, dict) or coordinator_reply.get("recommendation") not in {
                "seek human decision", "reject"}:
            raise ValueError("Coordinator must recommend human review or rejection")
        if coordinator_reply["recommendation"] == "reject":
            outcome = {"proposal_id": proposal_id, "status": "rejected_by_coordinator",
                       "approved": False, "executed": False}
            self.registry.append_event("coordinator_rejection", {"proposal_id": proposal_id,
                                                                   "reason": coordinator_reply.get("reason", "unspecified")})
            self.registry.append_event("cycle_outcome", outcome)
            return outcome
        coordinator = CoordinatorTools(self.registry, self.budget)
        approved = coordinator.approve(self.condition, proposal_id, proposal.cost_estimate,
                                      decision=human)
        if not approved:
            status = "rejected_by_human" if not human.approved else "over_budget"
            outcome = {"proposal_id": proposal_id, "status": status, "approved": False, "executed": False}
            self.registry.append_event("cycle_outcome", outcome)
            return outcome
        run_id = f"{self.condition}_{proposal_id}"
        config = Config(**dict(working.as_dict(), **proposal.config_diff))
        if proposal.action == "run_experiment":
            launcher = lambda pid, cfg: self.experiment(run_id, cfg, proposal=proposal.as_dict(),
                                                        predicted_effect=proposal.predicted_effect)
            trainer = TrainingStrategistTools(self.registry, launcher, self.budget, blind=self.blind)
            try:
                record = trainer.launch(proposal_id, config)
            except Exception as exc:
                outcome = {"proposal_id": proposal_id, "run_id": run_id, "status": "failed",
                           "approved": True, "executed": True, "error": str(exc)}
                self.registry.append_event("cycle_outcome", outcome)
                return outcome
        else:
            self.budget.start(proposal_id)
            try:
                if proposal.action == "bake_model":
                    if self.bake_action is None:
                        raise RuntimeError("bake action is not configured")
                    action_result = self.bake_action(config)
                else:
                    action_result = {"simulated": True, "description": "Controlled response simulation only"}
                record = {"run_id": run_id, "config": config.as_dict(), "seed": config.seed,
                          "dataset_hash": self.experiment.dataset_hash, "git_commit": self.experiment.git_commit,
                          "metrics": {}, "checkpoint_path": "", "status": "simulated" if proposal.action == "simulate_response" else "completed",
                          "action_result": action_result, "proposal": proposal.as_dict(),
                          "predicted_effect": proposal.predicted_effect}
                self.registry.write(record)
                self.budget.finish(proposal_id, success=True)
            except Exception as exc:
                self.budget.finish(proposal_id, success=False)
                outcome = {"proposal_id": proposal_id, "run_id": run_id, "status": "failed",
                           "approved": True, "executed": True, "error": str(exc)}
                self.registry.append_event("cycle_outcome", outcome)
                return outcome
        verdict = None
        if self.validator is not None and request_validation:
            try:
                validator_tool = AdversarialValidatorTools(self.validator.query)
                verdict = validator_tool.validate(run_id)
                self.caller.call("Adversarial Validator", "Assess coarse sealed verdict only",
                                 {"validator_reply": verdict}, proposal_id=proposal_id,
                                 condition=self.condition)
                self.registry.append_event("validator_verdict", {"run_id": run_id, **verdict})
            except RuntimeError:
                self.registry.append_event("validator_budget_exhausted", {"run_id": run_id})
        outcome = {"proposal_id": proposal_id, "run_id": run_id,
                   "status": record["status"], "approved": True, "executed": True,
                   "validator": verdict}
        self.registry.append_event("cycle_outcome", outcome)
        return outcome
