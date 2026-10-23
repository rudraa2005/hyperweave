"""Exact dominated hypervolume for small 2D/3D frontiers."""
import numpy as np


REFERENCE = (1.0, 1.0, 1.0)
DENOMINATORS = (1.0, 100000.0, 1000000.0)


def normalize_objectives(detection_rate, macs, latency_ns_per_flow):
    values = np.array([1 - detection_rate, macs, latency_ns_per_flow], float)
    return tuple(values / DENOMINATORS)


def nondominated(points):
    p = np.asarray(points, float)
    if p.ndim != 2:
        raise ValueError("points must be two-dimensional")
    keep = []
    for i in range(len(p)):
        if not any(np.all(p[j] <= p[i]) and np.any(p[j] < p[i]) for j in range(len(p)) if j != i):
            keep.append(i)
    return p[keep]


def hypervolume(points, reference):
    p = np.asarray(points, float)
    ref = np.asarray(reference, float)
    if ref.ndim != 1 or len(ref) not in (2, 3):
        raise ValueError("reference must be 2D or 3D")
    if p.size == 0:
        return 0.0
    if p.ndim != 2 or p.shape[1] != len(ref) or not np.isfinite(p).all():
        raise ValueError("invalid points")
    p = nondominated(p[np.all(p < ref, axis=1)])
    if len(p) == 0:
        return 0.0
    return float(_hv(p, ref))


def _hv(points, ref):
    if len(ref) == 1:
        return max(0.0, ref[0] - points[:, 0].min())
    stops = sorted(set(points[:, 0].tolist() + [float(ref[0])]))
    area = 0.0
    for left, right in zip(stops, stops[1:]):
        active = points[points[:, 0] <= left, 1:]
        if len(active):
            area += (right - left) * _hv(active, ref[1:])
    return area
