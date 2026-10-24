"""Preference and context conditioned functional neurons with L0-style gates."""
from dataclasses import dataclass

import numpy as np

from hyperweave.data.shifts import apply_shift
from hyperweave.eval.hypervolume import hypervolume

from .baselines import _sigmoid


@dataclass(frozen=True)
class NeuronSpec:
    inputs: int
    neurons: int
    hidden: int

    @property
    def parameter_count(self):
        return self.hidden * self.inputs + 2 * self.hidden + 3

    @property
    def macs_per_neuron(self):
        return self.inputs * self.hidden + self.hidden + 1


class HyperNet:
    """Linear H_phi([e_j, lambda, c, 1]) -> [theta_j, s_j].

    The last-layer columns are initialized at the target fan-in scale. The
    learned embeddings and projection matrix are updated through task, shift,
    and expected-cost gradients. Lambda/context are shared by the whole batch.
    """

    gamma = -0.1
    zeta = 1.1
    temperature = 2 / 3

    def __init__(self, spec: NeuronSpec, embedding_dim=16, seed=11):
        if min(spec.inputs, spec.neurons, spec.hidden, embedding_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.spec = spec
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(seed)
        self.embeddings = rng.normal(size=(spec.neurons, embedding_dim))
        self.projection = np.zeros((embedding_dim + 6, spec.parameter_count))
        h, d, m = spec.hidden, spec.inputs, spec.neurons
        sizes = [1 / np.sqrt(d)] * (h * d) + [0.02] * h + [1 / np.sqrt(h)] * h + [0.02, 1 / np.sqrt(m), 0.2]
        for col, scale in enumerate(sizes):
            self.projection[:embedding_dim, col] = rng.normal(0, scale / np.sqrt(embedding_dim), embedding_dim)
        # Conditional effects are nonzero at initialization, including context.
        self.projection[embedding_dim:embedding_dim + 5, :-1] = rng.normal(
            0, .005, (5, spec.parameter_count - 1))
        self.projection[-1, -1] = 1.5
        self.projection[embedding_dim + 2, -1] = -5.0  # cost preference
        self.projection[embedding_dim + 3:embedding_dim + 5, -1] = [.1, -.1]
        self.forward_count = 0
        self.optimizer_steps = 0

    def _condition(self, preference, context):
        lam = np.asarray(preference, float)
        c = np.asarray(context, float)
        if lam.shape != (3,) or np.any(lam < 0) or not np.isclose(lam.sum(), 1):
            raise ValueError("lambda must be a 3-objective simplex vector")
        if c.shape != (2,) or np.any(c < 0) or np.any(c > 2) or np.any(c != np.floor(c)):
            raise ValueError("context must be two discrete bins in {0,1,2}")
        tail = np.r_[lam, c / 2, 1.0]
        return np.column_stack([self.embeddings, np.tile(tail, (self.spec.neurons, 1))])

    def _unpack(self, raw):
        h, d = self.spec.hidden, self.spec.inputs
        p = 0
        w1 = raw[:, p:p + h * d].reshape(-1, h, d); p += h * d
        b1 = raw[:, p:p + h]; p += h
        w2 = raw[:, p:p + h]; p += h
        b2 = raw[:, p]; p += 1
        out = raw[:, p]; p += 1
        logit = raw[:, p]
        return w1, b1, w2, b2, out, logit

    def generate(self, preference, context=(0, 0)):
        self.forward_count += 1
        z = self._condition(preference, context)
        raw = z @ self.projection
        w1, b1, w2, b2, out, logit = self._unpack(raw)
        soft = _sigmoid(logit / self.temperature)
        stretched = soft * (self.zeta - self.gamma) + self.gamma
        gate = np.clip(stretched, 0, 1)
        expected_open = _sigmoid(logit - self.temperature * np.log(-self.gamma / self.zeta))
        return {"z": z, "raw": raw, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
                "out": out, "logit": logit, "soft": soft, "gate": gate,
                "expected_open": expected_open}

    def expected_macs(self, generated):
        return float(generated["expected_open"].sum() * self.spec.macs_per_neuron)

    def logits_from(self, x, generated):
        x = np.asarray(x, float)
        z = np.einsum("bd,mhd->bmh", x, generated["w1"]) + generated["b1"]
        feat = np.einsum("bmh,mh->bm", np.maximum(z, 0), generated["w2"]) + generated["b2"]
        return feat @ (generated["out"] * generated["gate"])

    def logits(self, x, preference, context=(0, 0)):
        return self.logits_from(x, self.generate(preference, context))

    def _task_grad(self, x, y, generated):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.einsum("bd,mhd->bmh", x, generated["w1"]) + generated["b1"]
        a = np.maximum(z, 0)
        feat = np.einsum("bmh,mh->bm", a, generated["w2"]) + generated["b2"]
        effective_out = generated["out"] * generated["gate"]
        logits = feat @ effective_out
        prob = _sigmoid(logits)
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
        delta = (prob - y) / len(x)
        gout = (feat.T @ delta) * generated["gate"]
        ggate = (feat.T @ delta) * generated["out"]
        dfeat = delta[:, None] * effective_out
        gw2 = np.einsum("bm,bmh->mh", dfeat, a)
        gb2 = dfeat.sum(axis=0)
        dz = dfeat[:, :, None] * generated["w2"][None, :, :] * (z > 0)
        gw1 = np.einsum("bmh,bd->mhd", dz, x)
        gb1 = dz.sum(axis=0)
        active = (generated["gate"] > 0) & (generated["gate"] < 1)
        gate_derivative = active * (self.zeta - self.gamma) / self.temperature * generated["soft"] * (1 - generated["soft"])
        gs = ggate * gate_derivative
        grad = np.column_stack([gw1.reshape(len(gout), -1), gb1, gw2, gb2, gout, gs])
        return loss, grad

    def train_step(self, x, y, preference, context=(0, 0), *, shift_x=None, lr=.01, mode="linear"):
        if mode not in {"linear", "phn_hvi"}:
            raise ValueError("unknown loss mode")
        if mode == "phn_hvi":
            lam = np.asarray(preference, float)
            preferences = (lam, np.roll(lam, 1), np.roll(lam, 2))
            return self.train_hvi_step(x, y, preferences, context, shift_x=shift_x, lr=lr)
        generated = self.generate(preference, context)
        task_loss, task_grad = self._task_grad(x, y, generated)
        shift_loss, shift_grad = self._task_grad(x if shift_x is None else shift_x, y, generated)
        cost = self.expected_macs(generated) / (self.spec.neurons * self.spec.macs_per_neuron)
        cost_grad = np.zeros_like(task_grad)
        p = generated["expected_open"]
        cost_grad[:, -1] = p * (1 - p) / self.spec.neurons
        lam = np.asarray(preference, float)
        objectives = np.array([task_loss, shift_loss, cost])
        weights = lam
        loss = float(lam @ objectives)
        grad = weights[0] * task_grad + weights[1] * shift_grad + weights[2] * cost_grad
        norm = np.linalg.norm(grad)
        if norm > 10:
            grad *= 10 / norm
        old_projection = self.projection.copy()
        projection_grad = generated["z"].T @ grad
        embedding_grad = grad @ old_projection[:self.embedding_dim].T
        self.projection -= lr * projection_grad
        self.embeddings -= lr * embedding_grad
        self.optimizer_steps += 1
        if not np.isfinite(loss) or not np.isfinite(self.projection).all():
            raise FloatingPointError("hypernetwork diverged")
        return {"loss": loss, "task": task_loss, "shift": shift_loss, "expected_macs": self.expected_macs(generated)}

    def train_hvi_step(self, x, y, preferences, context=(0, 0), *, shift_x=None, lr=.01):
        """Multi-sample hypervolume-gradient step over three generated solutions."""
        if len(preferences) < 2:
            raise ValueError("PHN-HVI requires multiple preferences")
        generated_set, objective_rows, gradient_rows = [], [], []
        for preference in preferences:
            generated = self.generate(preference, context)
            task_loss, task_grad = self._task_grad(x, y, generated)
            shift_loss, shift_grad = self._task_grad(x if shift_x is None else shift_x, y, generated)
            cost = self.expected_macs(generated) / (self.spec.neurons * self.spec.macs_per_neuron)
            cost_grad = np.zeros_like(task_grad)
            p = generated["expected_open"]
            cost_grad[:, -1] = p * (1 - p) / self.spec.neurons
            generated_set.append(generated)
            objective_rows.append([task_loss, shift_loss, cost])
            gradient_rows.append((task_grad, shift_grad, cost_grad))
        points = np.asarray(objective_rows, float)
        reference = (2., 2., 1.1)
        volume = hypervolume(points, reference)
        eps = 1e-4
        objective_grad = np.zeros_like(points)
        for i in range(len(points)):
            for j in range(3):
                plus = points.copy(); plus[i, j] += eps
                minus = points.copy(); minus[i, j] -= eps
                objective_grad[i, j] = -(hypervolume(plus, reference) - hypervolume(minus, reference)) / (2 * eps)
        projection_grad = np.zeros_like(self.projection)
        embedding_grad = np.zeros_like(self.embeddings)
        for generated, grads, weights in zip(generated_set, gradient_rows, objective_grad):
            raw_grad = sum(weight * component for weight, component in zip(weights, grads))
            projection_grad += generated["z"].T @ raw_grad
            embedding_grad += raw_grad @ self.projection[:self.embedding_dim].T
        norm = np.sqrt(np.sum(projection_grad ** 2) + np.sum(embedding_grad ** 2))
        if norm > 10:
            projection_grad *= 10 / norm
            embedding_grad *= 10 / norm
        self.projection -= lr * projection_grad
        self.embeddings -= lr * embedding_grad
        self.optimizer_steps += 1
        if not np.isfinite(volume) or not np.isfinite(self.projection).all():
            raise FloatingPointError("PHN-HVI diverged")
        return {"loss": -float(volume), "task": float(points[:, 0].mean()),
                "shift": float(points[:, 1].mean()),
                "expected_macs": float(points[:, 2].mean() * self.spec.neurons * self.spec.macs_per_neuron)}

    def fit(self, x, y, *, epochs=3, batch_size=128, lr=.01, seed=11, mode="linear",
            heldout=(), concentration=(1., 1., 1.), shift_columns=(0,), shift_transform=None,
            evasion_transform=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
                lam = sample_preference(rng, concentration=concentration, heldout=heldout)
                context = tuple(int(v) for v in rng.integers(0, 3, 2))
                family = ("count_rescale", "timing_jitter", "feature_dropout")[self.optimizer_steps % 3]
                transform = shift_transform or (lambda batch, name, columns, shift_seed:
                                                apply_shift(batch, name, columns, 1, seed=shift_seed))
                shift = transform(x[ids], family, list(shift_columns), seed + self.optimizer_steps)
                if evasion_transform is not None:
                    shift = evasion_transform(shift, y[ids], seed + self.optimizer_steps)
                if mode == "phn_hvi":
                    preferences = [lam] + [sample_preference(rng, concentration=concentration, heldout=heldout)
                                           for _ in range(2)]
                    history.append(self.train_hvi_step(x[ids], y[ids], preferences, context,
                                                       shift_x=shift, lr=lr))
                else:
                    history.append(self.train_step(x[ids], y[ids], lam, context,
                                                   shift_x=shift, lr=lr, mode=mode))
        return history


def sample_preference(rng, concentration=(1., 1., 1.), heldout=(), tolerance=.05):
    for _ in range(10000):
        lam = rng.dirichlet(concentration)
        if all(np.linalg.norm(lam - np.asarray(v)) >= tolerance for v in heldout):
            return lam
    raise ValueError("held-out preference exclusion exhausted")
