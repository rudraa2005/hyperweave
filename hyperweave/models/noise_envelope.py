"""Offline benign-flow autoencoder; never imported by the int8 kernel.

Torch is optional until this workflow is run. CUDA or MPS is selected when
available; CPU remains a reproducible fallback for development.
"""
from pathlib import Path
import warnings

import numpy as np


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    return requested


class NoiseEnvelope:
    """Train on A-train benign flows, calibrate only on A-val benign flows."""

    def __init__(self, input_dim, hidden=32, latent=8, seed=11):
        if min(input_dim, hidden, latent) < 1:
            raise ValueError("all dimensions must be positive")
        self.input_dim, self.hidden, self.latent, self.seed = input_dim, hidden, latent, seed
        self.center = self.scale = self.threshold = self.weights = None
        self.device = None
        self.converged = None

    def _network(self, torch):
        return torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.hidden), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden, self.latent), torch.nn.ReLU(),
                                   torch.nn.Linear(self.latent, self.hidden), torch.nn.ReLU(),
                                   torch.nn.Linear(self.hidden, self.input_dim))

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError("invalid finite flow matrix")
        return np.clip((x - self.center) / self.scale, -10, 10).astype(np.float32)

    def fit(self, benign_train, *, epochs=10, batch_size=1024, lr=1e-3, device="auto"):
        torch = _torch()
        if epochs < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("invalid training parameters")
        benign_train = np.asarray(benign_train, dtype=np.float32)
        if benign_train.ndim != 2 or benign_train.shape[1] != self.input_dim or len(benign_train) < 2:
            raise ValueError("need at least two benign training flows")
        if not np.isfinite(benign_train).all():
            raise ValueError("nonfinite training flow")
        self.center = np.median(benign_train, axis=0)
        self.scale = np.maximum(np.std(benign_train, axis=0), 1e-6)
        x = self._normalize(benign_train)
        if torch is None:
            if device not in {"auto", "cpu"}:
                raise RuntimeError(f"{device} acceleration requires PyTorch and an available device")
            from sklearn.neural_network import MLPRegressor
            net = MLPRegressor(hidden_layer_sizes=(self.hidden, self.latent, self.hidden),
                               activation="relu", solver="adam", random_state=self.seed,
                               max_iter=epochs, batch_size=min(batch_size, len(x)),
                               learning_rate_init=lr, shuffle=True)
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.fit(x, x)
            self.weights = [value for pair in zip(net.coefs_, net.intercepts_)
                            for value in (pair[0].T.astype(np.float32), pair[1].astype(np.float32))]
            self.device = "cpu_numpy"
            self.converged = bool(net.n_iter_ < epochs)
            return self
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = _device(torch, device)
        net = self._network(torch).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        generator = np.random.default_rng(self.seed)
        for _ in range(epochs):
            for ids in np.array_split(generator.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
                batch = torch.from_numpy(x[ids]).to(self.device)
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(net(batch), batch)
                loss.backward()
                optimizer.step()
        self.weights = [value.detach().cpu().numpy().copy() for value in net.state_dict().values()]
        return self

    def reconstruction_error(self, x, *, batch_size=4096):
        if self.weights is None:
            raise RuntimeError("autoencoder has not been trained")
        normalized = self._normalize(x)
        if self.device == "cpu_numpy" or _torch() is None:
            out = normalized
            for i in range(0, len(self.weights), 2):
                out = out @ self.weights[i].T + self.weights[i + 1]
                if i < len(self.weights) - 2:
                    out = np.maximum(out, 0)
            return np.mean((out - normalized) ** 2, axis=1)
        torch = _torch()
        device = _device(torch, self.device or "cpu")
        net = self._network(torch).to(device)
        state = dict(zip(net.state_dict(), (torch.from_numpy(v) for v in self.weights)))
        net.load_state_dict(state)
        net.eval()
        errors = []
        with torch.no_grad():
            for batch in np.array_split(normalized, max(1, int(np.ceil(len(normalized) / batch_size)))):
                tensor = torch.from_numpy(batch).to(device)
                errors.append(torch.mean((net(tensor) - tensor) ** 2, dim=1).cpu().numpy())
        return np.concatenate(errors)

    def calibrate(self, benign_val, target_fpr=.001):
        from hyperweave.eval.metrics import threshold_at_fpr
        self.threshold = threshold_at_fpr(self.reconstruction_error(benign_val), target_fpr)
        return self.threshold

    def anomaly_flags(self, x):
        if self.threshold is None:
            raise RuntimeError("autoencoder threshold has not been calibrated")
        return self.reconstruction_error(x) > self.threshold

    def save(self, path):
        if self.weights is None:
            raise RuntimeError("autoencoder has not been trained")
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        np.savez(path, center=self.center, scale=self.scale, threshold=np.nan if self.threshold is None else self.threshold,
                 input_dim=self.input_dim, hidden=self.hidden, latent=self.latent, seed=self.seed,
                 backend=self.device, converged=-1 if self.converged is None else int(self.converged),
                 **{f"weight_{i}": v for i, v in enumerate(self.weights)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(int(data["input_dim"]), int(data["hidden"]), int(data["latent"]), int(data["seed"]))
            model.center, model.scale = data["center"].copy(), data["scale"].copy()
            model.threshold = None if np.isnan(data["threshold"]) else float(data["threshold"])
            model.weights = [data[f"weight_{i}"].copy() for i in range(8)]
            model.device = str(data["backend"]) if "backend" in data else "cpu"
            model.converged = (None if "converged" not in data or int(data["converged"]) == -1
                               else bool(int(data["converged"])))
        return model
