"""Train-only, heavy-tail-safe NF-v2 feature normalization."""
import numpy as np

from .shifts import SHIFTABLE_FEATURES


LOG_FEATURES = SHIFTABLE_FEATURES | frozenset({"LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER"})


class FlowNormalizer:
    def __init__(self, features):
        self.features = tuple(features)
        self.log_mask = np.array([name in LOG_FEATURES for name in features], bool)
        self.center = self.scale = None

    def _log(self, raw):
        x = np.asarray(raw, float).copy()
        if x.ndim != 2 or x.shape[1] != len(self.features) or not np.isfinite(x).all():
            raise ValueError("invalid flow matrix")
        if np.any(x[:, self.log_mask] < 0):
            raise ValueError("log features must be nonnegative")
        x[:, self.log_mask] = np.log1p(x[:, self.log_mask])
        return x

    def fit(self, train_raw):
        x = self._log(train_raw)
        self.center = np.median(x, axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-6] = 1
        return self

    def transform(self, raw):
        if self.center is None:
            raise RuntimeError("normalizer not fitted")
        return np.clip((self._log(raw) - self.center) / self.scale, -20, 20)

    def inverse(self, normalized):
        if self.center is None:
            raise RuntimeError("normalizer not fitted")
        x = np.asarray(normalized, float) * self.scale + self.center
        x[:, self.log_mask] = np.expm1(x[:, self.log_mask])
        return np.maximum(x, 0)

    def save(self, path):
        if self.center is None:
            raise RuntimeError("normalizer not fitted")
        np.savez(path, features=np.array(self.features), log_mask=self.log_mask,
                 center=self.center, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(list(data["features"]))
            model.log_mask = data["log_mask"].copy()
            model.center = data["center"].copy()
            model.scale = data["scale"].copy()
        return model
