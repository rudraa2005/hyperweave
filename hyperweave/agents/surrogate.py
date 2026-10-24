"""Small, uncertainty-aware predictor trained only on past registry runs."""
import numpy as np
from sklearn.linear_model import Ridge

from hyperweave.config import Config


def _features(config: Config):
    return np.array([config.width / 32, config.functional_hidden / 8,
                     np.log10(config.learning_rate) + 4, config.epochs / 20,
                     *config.preference, *(v / 2 for v in config.context)], float)


class Surrogate:
    def __init__(self, registry, *, condition=None):
        self.registry = registry
        self.condition = condition

    def predict(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("config must be Config")
        all_rows = [row for row in self.registry.all_runs()
                    if self.condition is None or row["config"]["model"] == self.condition]
        real_rows = [r for r in all_rows if isinstance(r.get("metrics", {}).get("hv_gain_per_cost"), (int, float))]
        rows = (real_rows if real_rows else [r for r in all_rows if isinstance(
            r.get("metrics", {}).get("proxy_hv_gain_per_cost"), (int, float))])
        if len(rows) < 3:
            return {"prediction": 0.0, "uncertainty": 1.0, "source": "sparse_fallback", "samples": len(rows)}
        x = np.vstack([_features(Config(**r["config"])) for r in rows])
        y = np.asarray([r["metrics"].get("hv_gain_per_cost", r["metrics"].get("proxy_hv_gain_per_cost"))
                        for r in rows], float)
        model = Ridge(alpha=1.0).fit(x, y)
        query = _features(config)
        prediction = float(model.predict(query[None, :])[0])
        residual = float(np.sqrt(np.mean((model.predict(x) - y) ** 2)))
        distance = float(np.min(np.linalg.norm(x - query, axis=1)))
        source = "ridge_hv" if real_rows else "ridge_proxy"
        return {"prediction": prediction, "uncertainty": max(1e-6, residual + .05 * distance),
                "source": source, "samples": len(rows)}


def prediction_calibration(registry):
    pairs = []
    for run in registry.all_runs():
        prediction = run.get("predicted_effect")
        metrics = run.get("metrics", {})
        actual = metrics.get("hv_gain_per_cost", metrics.get("proxy_hv_gain_per_cost"))
        if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
            pairs.append((prediction, actual))
    if not pairs:
        return {"count": 0, "mae": None}
    return {"count": len(pairs), "mae": float(np.mean([abs(a - b) for a, b in pairs]))}
