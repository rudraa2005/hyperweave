"""Typed capability views for the six research agents."""
from pathlib import Path

import numpy as np

from hyperweave.config import Config
from hyperweave.eval.hypervolume import hypervolume, nondominated, normalize_objectives
from hyperweave.models import NeuronSpec
from hyperweave.models.baselines import affine_macs, functional_macs

from .budget import HumanDecision


class AgentFilesystem:
    """Read-only allowlist. Agent tools never expose arbitrary filesystem access."""
    def __init__(self, public_root, denied_roots=()):
        self.public_root = Path(public_root).resolve()
        self.denied_roots = tuple(Path(root).resolve() for root in denied_roots)

    def read_text(self, path):
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be text")
        target = Path(path).resolve()
        if not target.is_relative_to(self.public_root) or any(target.is_relative_to(root) for root in self.denied_roots):
            raise PermissionError("path outside agent public workspace")
        return target.read_text()


class CoordinatorTools:
    def __init__(self, registry, budget):
        self.registry = registry
        self.budget = budget

    def approve(self, condition: str, proposal_id: str, cost: int, *, decision: HumanDecision):
        if not isinstance(decision, HumanDecision):
            raise TypeError("approval requires a trusted HumanDecision")
        return self.budget.reserve(condition, proposal_id, cost,
                                   approved=decision.approved, actor=decision.actor)

    def remaining(self, condition: str):
        if not isinstance(condition, str):
            raise TypeError("condition must be text")
        return self.budget.remaining(condition)

    def run_state(self, run_id: str):
        if not isinstance(run_id, str):
            raise TypeError("run ID must be text")
        record = self.registry.read(run_id)
        return {"status": record.get("status", "completed")} if record else {"status": "missing"}

    def rollback(self, proposal_id: str):
        if not isinstance(proposal_id, str):
            raise TypeError("proposal ID must be text")
        self.budget.release_unstarted(proposal_id)
        return {"rolled_back": True}


class NeuronArchitectTools:
    def architecture(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("config must be Config")
        return {"inputs": config.input_dim, "neurons": config.width,
                "hidden": config.functional_hidden}

    def parameter_count(self, spec: NeuronSpec):
        if not isinstance(spec, NeuronSpec):
            raise TypeError("spec must be NeuronSpec")
        d, m, h = spec.inputs, spec.neurons, spec.hidden
        return m * (h * d + 2 * h + 2) + 1

    def profile(self, spec: NeuronSpec, gate_mask, *, blind=False):
        if not isinstance(spec, NeuronSpec):
            raise TypeError("spec must be NeuronSpec")
        mask = np.asarray(gate_mask)
        if mask.shape != (spec.neurons,) or mask.dtype != bool:
            raise TypeError("gate_mask must be boolean per neuron")
        if not isinstance(blind, bool):
            raise TypeError("blind must be boolean")
        if blind:
            return {"withheld": True}
        survivors = int(mask.sum())
        n, h, d = survivors, spec.hidden, spec.inputs
        int8_bytes = 48 + n * h * d + 4 * n * h + n * h + 4 * n + n + 4
        return {"macs": functional_macs(d, n, h), "gate_sparsity": 1 - survivors / spec.neurons,
                "int8_bytes": int8_bytes}


class TrainingStrategistTools:
    def __init__(self, registry, launcher, budget, *, blind=False):
        self.registry, self.launcher, self.budget, self.blind = registry, launcher, budget, blind

    def edit_config(self, base: Config, diff: dict):
        if not isinstance(base, Config) or not isinstance(diff, dict):
            raise TypeError("expected Config and config diff")
        return Config(**dict(base.as_dict(), **diff))

    def launch(self, proposal_id: str, config: Config):
        if not isinstance(proposal_id, str) or not isinstance(config, Config):
            raise TypeError("expected proposal ID and Config")
        self.budget.start(proposal_id)
        try:
            result = self.launcher(proposal_id, config)
        except Exception:
            self.budget.finish(proposal_id, success=False)
            raise
        self.budget.finish(proposal_id, success=True)
        return result

    def diagnostics(self, run_id: str):
        if not isinstance(run_id, str):
            raise TypeError("run ID must be text")
        record = self.registry.read(run_id)
        if record is None:
            return {"status": "missing"}
        return {"status": record.get("status", "completed")} if self.blind else record["metrics"]


class OptimizationScientistTools:
    def __init__(self, registry, *, blind=False):
        self.registry, self.blind = registry, blind

    def _points(self, condition):
        rows = [r for r in self.registry.all_runs() if r["config"]["model"] == condition and r.get("status", "completed") == "completed"]
        pinned = bool(rows) and all(r.get("metrics", {}).get("latency", {}).get("pinned") for r in rows)
        if pinned:
            points = [normalize_objectives(r["metrics"]["detection_rate"], r["metrics"]["macs"],
                                           r["metrics"]["latency"]["median_ns_per_flow"]) for r in rows]
        else:
            points = [(1 - r["metrics"]["detection_rate"], r["metrics"]["macs"] / 100000,
                       r["metrics"].get("params", 0) * 8 / 1000000) for r in rows]
        return rows, points, pinned

    def frontier(self, condition: str):
        if not isinstance(condition, str):
            raise TypeError("condition must be text")
        rows, points, pinned = self._points(condition)
        if self.blind:
            return {"results_withheld": True, "run_count": len(rows)}
        front = nondominated(points) if points else np.empty((0, 3))
        volume = hypervolume(front, (1, 1, 1))
        return {"points": front.tolist(), "hypervolume": volume if pinned else None,
                "debug_proxy_hv": None if pinned else volume, "timing_comparable": pinned}

    def rank_candidates(self, condition: str):
        if not isinstance(condition, str):
            raise TypeError("condition must be text")
        rows, points, pinned = self._points(condition)
        if self.blind:
            return {"results_withheld": True, "run_count": len(rows)}
        total = hypervolume(points, (1, 1, 1))
        ranked = []
        for index, row in enumerate(rows):
            contribution = total - hypervolume(points[:index] + points[index + 1:], (1, 1, 1))
            ranked.append({"run_id": row["run_id"],
                           "hv_contribution" if pinned else "debug_proxy_contribution": contribution})
        field = "hv_contribution" if pinned else "debug_proxy_contribution"
        return sorted(ranked, key=lambda r: (-r[field], r["run_id"]))

    def hypervolume_evaluator(self, points):
        if self.blind:
            return {"results_withheld": True}
        array = np.asarray(points)
        if array.ndim != 2 or array.shape[1] != 3 or not np.issubdtype(array.dtype, np.number):
            raise TypeError("points must be a numeric 3D objective matrix")
        return {"hypervolume": hypervolume(array, (1, 1, 1)), "reference": [1, 1, 1]}


class ExperimentalScientistTools:
    def __init__(self, registry, surrogate, *, blind=False):
        self.registry, self.surrogate, self.blind = registry, surrogate, blind
        self.queue_items = []

    def plan(self, hypothesis: str, config_diff: dict):
        if not isinstance(hypothesis, str) or not isinstance(config_diff, dict):
            raise TypeError("expected hypothesis and config diff")
        return {"hypothesis": hypothesis, "config_diff": config_diff}

    def queue(self, proposal_id: str):
        if not isinstance(proposal_id, str):
            raise TypeError("proposal ID must be text")
        self.queue_items.append(proposal_id)
        return {"position": len(self.queue_items)}

    def budget_estimate(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("config must be Config")
        return {"runs": 1}

    def compare(self, run_ids: list[str]):
        if not isinstance(run_ids, list) or not all(isinstance(v, str) for v in run_ids):
            raise TypeError("run IDs must be a text list")
        if self.blind:
            return {"results_withheld": True, "run_count": len(run_ids)}
        return {run_id: self.registry.read(run_id)["metrics"] for run_id in run_ids if self.registry.read(run_id)}

    def predict(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("config must be Config")
        return {"results_withheld": True} if self.blind else self.surrogate.predict(config)


class AdversarialValidatorTools:
    def __init__(self, query_fn):
        self._query_fn = query_fn

    def validate(self, candidate_run_id: str):
        if not isinstance(candidate_run_id, str):
            raise TypeError("run ID must be text")
        return self._query_fn(candidate_run_id)

    def shift_test(self, candidate_run_id: str):
        if not isinstance(candidate_run_id, str):
            raise TypeError("run ID must be text")
        return self._query_fn(candidate_run_id, check="shift")

    def evasion_test(self, candidate_run_id: str):
        if not isinstance(candidate_run_id, str):
            raise TypeError("run ID must be text")
        return self._query_fn(candidate_run_id, check="evasion")

    def latency_profile(self, candidate_run_id: str):
        if not isinstance(candidate_run_id, str):
            raise TypeError("run ID must be text")
        return self._query_fn(candidate_run_id, check="latency")

    def regression_check(self, candidate_run_id: str):
        if not isinstance(candidate_run_id, str):
            raise TypeError("run ID must be text")
        return self._query_fn(candidate_run_id, check="regression")
