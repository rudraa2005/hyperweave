"""Budgeted NSGA-II search over the shared Config schema."""
import numpy as np

from hyperweave.agents.budget import BudgetLedger
from hyperweave.config import Config, SEARCH_OPTIONS


def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_fronts(objectives):
    remaining = set(range(len(objectives)))
    fronts = []
    while remaining:
        front = [i for i in sorted(remaining) if not any(_dominates(objectives[j], objectives[i])
                                                         for j in remaining if j != i)]
        fronts.append(front)
        remaining -= set(front)
    return fronts


def crowding_distance(objectives, front):
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for axis in range(len(objectives[0])):
        order = sorted(front, key=lambda i: objectives[i][axis])
        distance[order[0]] = distance[order[-1]] = float("inf")
        span = objectives[order[-1]][axis] - objectives[order[0]][axis]
        if span > 0:
            for position in range(1, len(order) - 1):
                distance[order[position]] += (objectives[order[position + 1]][axis] -
                                              objectives[order[position - 1]][axis]) / span
    return distance


def _objective(record, mode):
    if record.get("status") != "completed":
        return (float("inf"),) * 3
    metrics = record["metrics"]
    deployed = metrics.get("int8", metrics)
    if mode == "research":
        latency = metrics.get("latency", {})
        if not latency.get("pinned") or latency.get("median_ns_per_flow") is None:
            raise ValueError("research NSGA-II requires pinned-core latency")
        third = latency["median_ns_per_flow"] / 1000000
    else:
        third = metrics["params"] * 8 / 1000000
    return (1 - deployed["detection_rate"], metrics["macs"] / 100000, third)


class NSGA2Search:
    def __init__(self, registry, experiment, base: Config, *, seeds=(11, 23, 37), trials_per_seed=4,
                 objective_mode="debug_proxy"):
        if base.model != "b4" or len(seeds) < 1 or trials_per_seed < 2:
            raise ValueError("NSGA-II requires B4 and at least two candidates per generation")
        if objective_mode not in {"debug_proxy", "research"}:
            raise ValueError("unknown NSGA-II objective mode")
        self.registry, self.experiment, self.base = registry, experiment, base
        self.seeds, self.trials_per_seed = tuple(seeds), trials_per_seed
        self.objective_mode = objective_mode
        self.budget = BudgetLedger(registry, limits={"b4": len(seeds) * trials_per_seed})

    def _sample(self, rng, seed):
        diff = {key: options[int(rng.integers(len(options)))] for key, options in SEARCH_OPTIONS.items()}
        return Config(**dict(self.base.as_dict(), seed=seed, **diff))

    def _offspring(self, rng, population, seed):
        objectives = [_objective(record, self.objective_mode) for _, record in population]
        fronts = non_dominated_fronts(objectives)
        rank = {index: number for number, front in enumerate(fronts) for index in front}
        crowding = {index: value for front in fronts for index, value in crowding_distance(objectives, front).items()}
        def tournament():
            i, j = rng.choice(len(population), 2, replace=False)
            return population[min((int(i), int(j)), key=lambda k: (rank[k], -crowding[k], k))][0]
        children = []
        for _ in range(self.trials_per_seed):
            a, b = tournament(), tournament()
            diff = {}
            for key, options in SEARCH_OPTIONS.items():
                if rng.random() < .25:
                    diff[key] = options[int(rng.integers(len(options)))]
                else:
                    diff[key] = getattr(a if rng.random() < .5 else b, key)
            children.append(Config(**dict(self.base.as_dict(), seed=seed, **diff)))
        return children

    def _select(self, combined):
        objectives = [_objective(record, self.objective_mode) for _, record in combined]
        selected = []
        for front in non_dominated_fronts(objectives):
            distances = crowding_distance(objectives, front)
            order = sorted(front, key=lambda i: (-distances[i], i))
            selected.extend(combined[i] for i in order[:self.trials_per_seed - len(selected)])
            if len(selected) == self.trials_per_seed:
                break
        return selected

    def run(self):
        rng = np.random.default_rng(991)
        population = []
        all_records = []
        for generation, seed in enumerate(self.seeds):
            candidates = ([self._sample(rng, seed) for _ in range(self.trials_per_seed)] if generation == 0
                          else self._offspring(rng, population, seed))
            offspring = []
            for trial, config in enumerate(candidates):
                run_id = f"b4_seed{seed}_trial{trial}"
                proposal_id = f"nsga2_{run_id}"
                self.registry.append_event("search_decision", {"method": "nsga2", "objective_mode": self.objective_mode,
                                     "run_id": run_id,
                                     "generation": generation, "config": config.as_dict()})
                if not self.budget.reserve("b4", proposal_id, 1, approved=True, actor="predeclared_research_protocol"):
                    raise RuntimeError("B4 budget unexpectedly exhausted")
                self.budget.start(proposal_id)
                try:
                    record = self.experiment(run_id, config, proposal={"method": "nsga2"})
                except RuntimeError:
                    record = self.registry.read(run_id)
                    self.budget.finish(proposal_id, success=False)
                else:
                    self.budget.finish(proposal_id, success=True)
                offspring.append((config, record))
                all_records.append(record)
            population = self._select(population + offspring)
        return all_records
