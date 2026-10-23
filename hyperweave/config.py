from dataclasses import asdict, dataclass
from typing import Literal


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

    def __post_init__(self):
        if self.seed < 0 or self.input_dim < 1 or self.width < 1 or self.functional_hidden < 1:
            raise ValueError("invalid size or seed")
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("invalid training config")
        if not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1")

    def as_dict(self):
        return asdict(self)

