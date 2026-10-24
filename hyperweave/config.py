from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal


SEARCH_OPTIONS = {
    "width": (4, 8, 12, 16, 24, 32),
    "functional_hidden": (2, 3, 4, 6),
    "learning_rate": (.005, .01, .02),
    "epochs": (1, 2, 3),
    "batch_size": (64, 128, 256),
    "preference": ((1 / 3, 1 / 3, 1 / 3), (.8, .1, .1), (.28, .12, .6), (.25, .10, .65)),
    "context": ((0, 0), (1, 1), (2, 2)),
    "loss_mode": ("linear", "phn_hvi"),
    "dirichlet_concentration": ((1., 1., 1.), (.5, .5, .5)),
    # Complete byte/packet pairs plus timing and derived rates; never IP fields.
    "shift_columns": ((6, 8, 10, 14, 15, 22, 28),
                      (6, 7, 8, 9, 10, 14, 15, 22, 23, 24, 25, 26, 27, 28, 29,
                       30, 31, 32, 33, 34)),
}


@dataclass(frozen=True)
class Config:
    seed: int = 11
    dataset: Literal["fashion_mnist", "nf_unsw_nb15_v2", "nf_cse_cic_ids2018_v2"] = "nf_unsw_nb15_v2"
    model: Literal["b1", "b2", "b2b", "b3", "b4", "b5", "b5b"] = "b1"
    input_dim: int = 43
    width: int = 16
    functional_hidden: int = 4
    learning_rate: float = 0.01
    epochs: int = 20
    batch_size: int = 256
    target_fpr: float = 0.001
    loss_mode: Literal["linear", "phn_hvi"] = "linear"
    dirichlet_concentration: tuple[float, float, float] = (1.0, 1.0, 1.0)
    preference: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    context: tuple[int, int] = (0, 0)
    shift_columns: tuple[int, ...] = (0,)

    def __post_init__(self):
        if self.dataset not in {"fashion_mnist", "nf_unsw_nb15_v2", "nf_cse_cic_ids2018_v2"}:
            raise ValueError("unknown dataset")
        if self.model not in {"b1", "b2", "b2b", "b3", "b4", "b5", "b5b"}:
            raise ValueError("unknown model condition")
        if self.loss_mode not in {"linear", "phn_hvi"}:
            raise ValueError("unknown loss mode")
        for name in ("seed", "input_dim", "width", "functional_hidden", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be integer")
        if self.seed < 0 or self.input_dim < 1 or self.width < 1 or self.functional_hidden < 1:
            raise ValueError("invalid size or seed")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or not isfinite(self.learning_rate):
            raise TypeError("learning_rate must be finite numeric")
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("invalid training config")
        if isinstance(self.target_fpr, bool) or not isinstance(self.target_fpr, (int, float)) or not isfinite(self.target_fpr) or not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1")
        if len(self.dirichlet_concentration) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0 for v in self.dirichlet_concentration):
            raise ValueError("Dirichlet concentration must contain three positive values")
        if len(self.preference) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0 for v in self.preference) or abs(sum(self.preference) - 1) > 1e-8:
            raise ValueError("preference must be a three-objective simplex vector")
        if len(self.context) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1, 2) for v in self.context):
            raise ValueError("context must contain two bins in {0,1,2}")
        if not self.shift_columns or any(not isinstance(v, int) or v < 0 or v >= self.input_dim for v in self.shift_columns):
            raise ValueError("shift_columns must be valid feature indices")

    def as_dict(self):
        result = asdict(self)
        result["dirichlet_concentration"] = list(self.dirichlet_concentration)
        result["preference"] = list(self.preference)
        result["context"] = list(self.context)
        result["shift_columns"] = list(self.shift_columns)
        return result

