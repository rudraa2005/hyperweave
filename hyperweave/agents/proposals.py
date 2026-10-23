"""Runtime validation matching configs/proposal.schema.json."""
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from hyperweave.config import Config, SEARCH_OPTIONS


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "proposal.schema.json"
ALLOWED_DIFF = frozenset(SEARCH_OPTIONS)
REQUIRED = frozenset({"hypothesis", "config_diff", "predicted_effect", "cost_estimate",
                      "success_criterion", "action", "rationale", "validation_plan"})
ACTIONS = frozenset({"run_experiment", "bake_model", "simulate_response"})


@dataclass(frozen=True)
class Proposal:
    hypothesis: str
    config_diff: dict
    predicted_effect: float
    cost_estimate: int
    success_criterion: str
    action: str
    rationale: str
    validation_plan: str

    def as_dict(self):
        return asdict(self)


def validate_proposal(value, base_config: Config) -> Proposal:
    if not isinstance(value, dict) or set(value) != REQUIRED:
        raise ValueError(f"proposal fields must be exactly {sorted(REQUIRED)}")
    for key in ("hypothesis", "success_criterion", "rationale", "validation_plan"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise TypeError(f"{key} must be nonempty text")
    if not isinstance(value["config_diff"], dict) or not set(value["config_diff"]) <= ALLOWED_DIFF:
        raise TypeError("config_diff contains unsupported settings")
    effect = value["predicted_effect"]
    if isinstance(effect, bool) or not isinstance(effect, (int, float)) or not isfinite(effect):
        raise TypeError("predicted_effect must be finite numeric")
    cost = value["cost_estimate"]
    if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
        raise TypeError("cost_estimate must be a positive integer")
    if value["action"] not in ACTIONS:
        raise ValueError("unsupported action")
    changed = value["config_diff"]
    for name in ("width", "functional_hidden", "epochs", "batch_size"):
        if name in changed and (isinstance(changed[name], bool) or not isinstance(changed[name], int)):
            raise TypeError(f"{name} must be integer")
    if "learning_rate" in changed and (isinstance(changed["learning_rate"], bool) or not isinstance(changed["learning_rate"], (int, float))):
        raise TypeError("learning_rate must be numeric")
    for name, length in (("preference", 3), ("context", 2), ("dirichlet_concentration", 3)):
        if name in changed and (not isinstance(changed[name], list) or len(changed[name]) != length):
            raise TypeError(f"{name} must be a {length}-element array")
    if "shift_columns" in changed and not isinstance(changed["shift_columns"], list):
        raise TypeError("shift_columns must be an array")
    for name, changed_value in changed.items():
        comparable = tuple(changed_value) if isinstance(changed_value, list) else changed_value
        if comparable not in SEARCH_OPTIONS[name]:
            raise ValueError(f"{name} is outside the shared search options")
    Config(**dict(base_config.as_dict(), **changed))
    # Keep the shipped JSON Schema auditable alongside runtime checks.
    schema = json.loads(SCHEMA_PATH.read_text())
    if set(schema["required"]) != REQUIRED or set(schema["properties"]) != REQUIRED:
        raise RuntimeError("proposal schema and runtime validation drifted")
    return Proposal(**value)
