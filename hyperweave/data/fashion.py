import gzip
import struct
from pathlib import Path

import numpy as np


def read_idx_images(path):
    opener = gzip.open if str(path).endswith(".gz") else Path.open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("not IDX image data")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n * rows * cols:
        raise ValueError("truncated IDX file")
    return data.reshape(n, rows, cols).copy()


def fixed_partition(n, seed=11):
    ids = np.random.default_rng(seed).permutation(n)
    a, b = int(n * .8), int(n * .9)
    return ids[:a], ids[a:b], ids[b:]


def corrupt(images, family, severity, seed=11):
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2 or 3")
    x = np.asarray(images, np.uint8)
    rng = np.random.default_rng(seed)
    if family == "noise":
        out = x.astype(float) + rng.normal(0, 8 * severity, x.shape)
    elif family == "brightness":
        out = x.astype(float) + 20 * severity
    elif family == "occlusion":
        out = x.copy()
        side = min(x.shape[-2:]) * severity // 6
        side = max(1, side)
        out[..., :side, :side] = 0
    elif family == "blur":
        out = x.astype(float)
        for _ in range(severity):
            p = np.pad(out, [(0, 0)] * (out.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
            out = sum(p[..., i:i+x.shape[-2], j:j+x.shape[-1]] for i in range(3) for j in range(3)) / 9
    else:
        raise ValueError("unknown corruption")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

