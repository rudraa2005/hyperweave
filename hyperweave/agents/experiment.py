"""Deterministic local experiment executor for Phase 3 integration tests."""
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from hyperweave.config import Config
from hyperweave.eval import score_binary, threshold_at_fpr
from hyperweave.models import HyperNet, NeuronSpec, bake, save_hypernet


class ExperimentHarness:
    def __init__(self, registry, artifact_root, train, val, shift_val, *, dataset_version="synthetic-phase3-v1"):
        self.registry = registry
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.train, self.val, self.shift_val = train, val, shift_val
        self.dataset_version = dataset_version
        digest = hashlib.sha256(dataset_version.encode())
        for x, y in (train, val, shift_val):
            digest.update(np.ascontiguousarray(x).tobytes())
            digest.update(np.ascontiguousarray(y).tobytes())
        self.dataset_hash = digest.hexdigest()
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        self.git_commit = result.stdout.strip() if result.returncode == 0 else "unversioned"

    def __call__(self, run_id: str, config: Config, *, proposal=None, predicted_effect=None):
        if not isinstance(run_id, str) or not isinstance(config, Config):
            raise TypeError("run ID and Config required")
        path = self.artifact_root / f"{run_id}.npz"
        if self.registry.read(run_id) or path.exists():
            raise FileExistsError("experiment run already exists")
        record = {"run_id": run_id, "config": config.as_dict(), "seed": config.seed,
                  "dataset_hash": self.dataset_hash, "git_commit": self.git_commit,
                  "checkpoint_path": f"artifacts/{run_id}.npz", "metrics": {},
                  "status": "failed", "dataset_version": self.dataset_version,
                  "proposal": proposal, "predicted_effect": predicted_effect}
        try:
            train_x, train_y = self.train
            val_x, val_y = self.val
            b_x, b_y = self.shift_val
            if train_x.shape[1] != config.input_dim:
                raise ValueError("input dimension does not match dataset")
            model = HyperNet(NeuronSpec(config.input_dim, config.width, config.functional_hidden), seed=config.seed)
            model.fit(train_x, train_y, epochs=config.epochs, lr=config.learning_rate,
                      batch_size=config.batch_size, seed=config.seed, mode=config.loss_mode,
                      concentration=config.dirichlet_concentration,
                      shift_columns=config.shift_columns)
            baked = bake(model, config.preference, config.context)
            val_scores = baked.logits(val_x)
            threshold = threshold_at_fpr(val_scores[val_y == 0], config.target_fpr)
            metrics = score_binary(b_y, baked.logits(b_x), threshold)
            metrics.update(macs=baked.macs, params=baked.params, gate_mask=baked.gate_mask.tolist(),
                           threshold=threshold,
                           threshold_source="a_val", traffic_processed=len(b_x),
                           latency={"pinned": False, "median_ns_per_flow": None},
                           proxy_hv_gain_per_cost=float(metrics["detection_rate"] *
                                                        max(0, 1 - baked.macs / 100000)))
            save_hypernet(model, path)
            metrics["model_size_bytes"] = path.stat().st_size
            record["metrics"] = metrics
            record["status"] = "completed"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        self.registry.write(record)
        if record["status"] == "failed":
            raise RuntimeError(record["error"])
        return record


def synthetic_fixture(seed=2026, *, dim=43, samples=300):
    """Chronological campaign-like debug partitions, never research evidence."""
    rng = np.random.default_rng(seed)
    parts = []
    for campaign in range(3):
        x = rng.normal(size=(samples, dim))
        y = (x[:, 0] + .3 * x[:, 1] > 0).astype(int)
        parts.append((x, y))
    return tuple(parts)
