import numpy as np


SHIFT_FAMILIES = {
    "train": frozenset({"count_rescale", "timing_jitter"}),
    "eval": frozenset({"feature_dropout"}),
}


def apply_shift(x, family: str, columns: list[int], severity: int, seed=11):
    if severity not in (1, 2, 3):
        raise ValueError("invalid severity")
    out = np.asarray(x, float).copy()
    if family == "count_rescale":
        out[:, columns] *= 1 + .1 * severity
    elif family == "timing_jitter":
        rng = np.random.default_rng(seed)
        out[:, columns] = np.maximum(0, out[:, columns] + rng.normal(0, severity, (len(out), len(columns))))
    elif family == "feature_dropout":
        out[:, columns] = 0
    else:
        raise ValueError("unknown shift family")
    return out

