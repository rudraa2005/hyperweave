"""Prune generated neurons and export a fixed int8 inference graph."""
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"HWF2"
SHIFT = 20


def _scale(values):
    return max(float(np.max(np.abs(values))) / 127, 1e-8) if np.size(values) else 1e-8


def _quantize(values, scale):
    return np.clip(np.rint(np.asarray(values) / scale), -128, 127).astype(np.int8)


def _requantize(values, multiplier, *, nonnegative=False):
    v = np.asarray(values, dtype=np.int64) * int(multiplier)
    rounded = np.where(v >= 0, (v + (1 << (SHIFT - 1))) >> SHIFT,
                       -((-v + (1 << (SHIFT - 1))) >> SHIFT))
    return np.clip(rounded, 0 if nonnegative else -128, 127).astype(np.int8)


@dataclass
class BakedModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    out: np.ndarray
    gate_mask: np.ndarray
    source_indices: np.ndarray
    preference: np.ndarray
    context: tuple[int, int]

    @property
    def macs(self):
        n, h, d = self.w1.shape
        return n * (d * h + h + 1)

    @property
    def params(self):
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size + self.out.size

    def logits(self, x):
        x = np.asarray(x, float)
        z = np.einsum("bd,mhd->bmh", x, self.w1) + self.b1
        feat = np.einsum("bmh,mh->bm", np.maximum(z, 0), self.w2) + self.b2
        return feat @ self.out


def bake(hypernet, preference, context=(0, 0)):
    generated = hypernet.generate(preference, context)
    mask = generated["gate"] > 0
    return BakedModel(generated["w1"][mask].copy(), generated["b1"][mask].copy(),
                      generated["w2"][mask].copy(), generated["b2"][mask].copy(),
                      (generated["out"] * generated["gate"])[mask].copy(), mask.copy(),
                      np.flatnonzero(mask), np.asarray(preference, float), tuple(context))


@dataclass
class Int8Model:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    out: np.ndarray
    bias: int
    scales: tuple[float, float, float, float, float, float]
    multiplier_a: int
    multiplier_f: int

    @property
    def model_bytes(self):
        return 4 + 3 * 4 + 6 * 4 + 2 * 4 + self.w1.nbytes + self.b1.nbytes + self.w2.nbytes + self.b2.nbytes + self.out.nbytes + 4

    def quantize_inputs(self, x):
        return _quantize(np.asarray(x, float), self.scales[0])

    def logits_int(self, inputs_q):
        x = np.asarray(inputs_q, np.int8)
        z = np.einsum("bd,mhd->bmh", x.astype(np.int32), self.w1.astype(np.int32)) + self.b1
        a = _requantize(np.maximum(z, 0), self.multiplier_a, nonnegative=True)
        f = np.einsum("bmh,mh->bm", a.astype(np.int32), self.w2.astype(np.int32)) + self.b2
        f_q = _requantize(f, self.multiplier_f)
        return np.asarray(f_q.astype(np.int32) @ self.out.astype(np.int32) + self.bias, np.int32)

    def logits(self, x):
        return self.logits_int(self.quantize_inputs(x)).astype(float) * self.scales[4] * self.scales[5]


def quantize(baked: BakedModel, calibration_x):
    x = np.asarray(calibration_x, float)
    if len(x) == 0:
        raise ValueError("calibration set cannot be empty")
    x_scale = _scale(x)
    w1_scale = _scale(baked.w1)
    z = np.einsum("bd,mhd->bmh", x, baked.w1) + baked.b1
    a = np.maximum(z, 0)
    a_scale = _scale(a)
    w2_scale = _scale(baked.w2)
    f = np.einsum("bmh,mh->bm", a, baked.w2) + baked.b2
    f_scale = _scale(f)
    out_scale = _scale(baked.out)
    scales = tuple(float(np.float32(s)) for s in (x_scale, w1_scale, a_scale, w2_scale, f_scale, out_scale))
    b1q = np.rint(baked.b1 / (scales[0] * scales[1])).astype(np.int32)
    b2q = np.rint(baked.b2 / (scales[2] * scales[3])).astype(np.int32)
    ma = round(scales[0] * scales[1] / scales[2] * (1 << SHIFT))
    mf = round(scales[2] * scales[3] / scales[4] * (1 << SHIFT))
    return Int8Model(_quantize(baked.w1, scales[1]), b1q, _quantize(baked.w2, scales[3]),
                     b2q, _quantize(baked.out, scales[5]), 0, scales, ma, mf)


def write_int8(model: Int8Model, path, manifest: dict):
    path = Path(path)
    n, h, d = model.w1.shape
    with path.open("wb") as out:
        out.write(struct.pack("<4sIII6fii", MAGIC, n, h, d, *model.scales,
                              model.multiplier_a, model.multiplier_f))
        for arr in (model.w1, model.b1, model.w2, model.b2, model.out):
            out.write(np.asarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
        out.write(struct.pack("<i", model.bias))
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return sidecar


def read_int8(path):
    with Path(path).open("rb") as source:
        header = source.read(struct.calcsize("<4sIII6fii"))
        magic, n, h, d, *tail = struct.unpack("<4sIII6fii", header)
        if magic != MAGIC:
            raise ValueError("not a HyperWeave int8 model")
        scales = tuple(tail[:6]); ma, mf = tail[6:]
        def take(count, dtype):
            size = np.dtype(dtype).itemsize * count
            data = source.read(size)
            if len(data) != size:
                raise ValueError("truncated int8 model")
            return np.frombuffer(data, dtype=dtype).copy()
        w1 = take(n * h * d, np.int8).reshape(n, h, d)
        b1 = take(n * h, "<i4").reshape(n, h)
        w2 = take(n * h, np.int8).reshape(n, h)
        b2 = take(n, "<i4")
        out = take(n, np.int8)
        bias = int(take(1, "<i4")[0])
        if source.read(1):
            raise ValueError("trailing model bytes")
    return Int8Model(w1, b1, w2, b2, out, bias, scales, ma, mf)


def save_hypernet(model, path):
    np.savez(path, embeddings=model.embeddings, projection=model.projection,
             spec=np.array([model.spec.inputs, model.spec.neurons, model.spec.hidden]),
             embedding_dim=model.embedding_dim)


def load_hypernet(path):
    from .hypernet import HyperNet, NeuronSpec
    with np.load(path) as data:
        d, m, h = map(int, data["spec"])
        model = HyperNet(NeuronSpec(d, m, h), int(data["embedding_dim"]))
        model.embeddings = data["embeddings"].copy()
        model.projection = data["projection"].copy()
    return model
