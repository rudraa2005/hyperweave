import numpy as np


def affine_macs(d: int, width: int, out: int = 1) -> int:
    return d * width + width * out


def functional_macs(d: int, neurons: int, h: int, out: int = 1) -> int:
    return neurons * (d * h + h) + neurons * out


def match_affine_width(d: int, neurons: int, h: int, out: int = 1) -> int:
    target = functional_macs(d, neurons, h, out)
    return max(1, round(target / (d + out)))


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


class AffineMLP:
    """One-hidden-layer ReLU binary classifier, with analytic MAC count."""

    def __init__(self, d: int, width: int, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1 / np.sqrt(d), (d, width))
        self.b1 = np.zeros(width)
        self.w2 = rng.normal(0, 1 / np.sqrt(width), width)
        self.b2 = 0.0

    @property
    def macs(self):
        return affine_macs(*self.w1.shape)

    @property
    def params(self):
        return self.w1.size + self.b1.size + self.w2.size + 1

    def logits(self, x):
        return np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2

    def fit(self, x, y, *, epochs=20, lr=0.01, batch_size=256, seed=11):
        x, y = np.asarray(x, float), np.asarray(y, float)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
                xb, yb = x[ids], y[ids]
                z = xb @ self.w1 + self.b1
                a = np.maximum(z, 0)
                delta = (_sigmoid(a @ self.w2 + self.b2) - yb) / len(ids)
                gw2 = a.T @ delta
                gb2 = delta.sum()
                dz = delta[:, None] * self.w2[None, :] * (z > 0)
                gw1, gb1 = xb.T @ dz, dz.sum(axis=0)
                self.w1 -= lr * gw1
                self.b1 -= lr * gb1
                self.w2 -= lr * gw2
                self.b2 -= lr * gb2
        return self


class FunctionalNet:
    """Each hidden neuron is a tiny independent ReLU MLP g(x; theta)."""

    def __init__(self, d: int, neurons: int, h: int, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1 / np.sqrt(d), (neurons, h, d))
        self.b1 = np.zeros((neurons, h))
        self.w2 = rng.normal(0, 1 / np.sqrt(h), (neurons, h))
        self.b2 = np.zeros(neurons)
        self.out = rng.normal(0, 1 / np.sqrt(neurons), neurons)
        self.bias = 0.0

    @property
    def macs(self):
        m, h, d = self.w1.shape
        return functional_macs(d, m, h)

    @property
    def params(self):
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size + self.out.size + 1

    def features(self, x):
        z = np.einsum("bd,mhd->bmh", x, self.w1) + self.b1
        return np.einsum("bmh,mh->bm", np.maximum(z, 0), self.w2) + self.b2

    def logits(self, x):
        return self.features(np.asarray(x, float)) @ self.out + self.bias

    def fit(self, x, y, *, epochs=20, lr=0.01, batch_size=256, seed=11):
        x, y = np.asarray(x, float), np.asarray(y, float)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
                xb, yb = x[ids], y[ids]
                z = np.einsum("bd,mhd->bmh", xb, self.w1) + self.b1
                a = np.maximum(z, 0)
                feat = np.einsum("bmh,mh->bm", a, self.w2) + self.b2
                delta = (_sigmoid(feat @ self.out + self.bias) - yb) / len(ids)
                gout, gbias = feat.T @ delta, delta.sum()
                dfeat = delta[:, None] * self.out[None, :]
                gw2 = np.einsum("bm,bmh->mh", dfeat, a)
                gb2 = dfeat.sum(axis=0)
                dz = dfeat[:, :, None] * self.w2[None, :, :] * (z > 0)
                gw1 = np.einsum("bmh,bd->mhd", dz, xb)
                gb1 = dz.sum(axis=0)
                self.w1 -= lr * gw1
                self.b1 -= lr * gb1
                self.w2 -= lr * gw2
                self.b2 -= lr * gb2
                self.out -= lr * gout
                self.bias -= lr * gbias
        return self
