import numpy as np

from .shifts import BYTE_PACKET_PAIRS, DERIVED_RATE_FEATURES, TIMING_FEATURES


def project_valid(x, original, mutable, *, bytes_col, packets_col, min_packet_size=1):
    """Project a feature-space candidate onto simple flow constraints."""
    out = np.asarray(x, float).copy()
    original = np.asarray(original, float)
    mask = np.asarray(mutable, bool)
    out[..., ~mask] = original[..., ~mask]
    out[..., mask] = np.maximum(out[..., mask], 0)
    if mask[bytes_col]:
        out[..., bytes_col] = np.maximum(out[..., bytes_col], out[..., packets_col] * min_packet_size)
    elif mask[packets_col]:
        out[..., packets_col] = np.minimum(out[..., packets_col], out[..., bytes_col] / min_packet_size)
    if np.any(out[..., bytes_col] < out[..., packets_col] * min_packet_size):
        raise ValueError("immutable input violates flow consistency")
    return out


def toward_benign(x, benign, epsilon, mutable, **constraints):
    if epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon outside [0, 1]")
    x = np.asarray(x, float)
    candidate = x + epsilon * (np.asarray(benign, float) - x)
    return project_valid(candidate, x, mutable, **constraints)


def projected_gradient(x, gradient_fn, epsilon, mutable, steps=10, **constraints):
    """White-box score minimization under an L-infinity box and flow projection."""
    original = np.asarray(x, float)
    out = original.copy()
    for _ in range(steps):
        grad = np.asarray(gradient_fn(out), float)
        if grad.shape != out.shape:
            raise ValueError("gradient shape mismatch")
        out = np.clip(out - epsilon / steps * np.sign(grad), original - epsilon, original + epsilon)
        out = project_valid(out, original, mutable, **constraints)
    return out


def nf_mutable_mask(features):
    """Only traffic volume, timing, and derived rates are attacker mutable."""
    names = set(TIMING_FEATURES) | set(DERIVED_RATE_FEATURES)
    names.update(name for pair in BYTE_PACKET_PAIRS for name in pair)
    return np.array([name in names for name in features], dtype=bool)


def constrained_nf_evasion(x, benign_reference, features, epsilon, kind, seed=11):
    """Feature-space proxy for bounded noise, jitter, padding, and camouflage.

    epsilon is a fractional per-feature L-infinity bound relative to the
    original value (with a unit floor). Byte/packet pair relations are checked
    after projection. No immutable endpoint or protocol feature can change.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon outside [0, 1]")
    original = np.asarray(x, float)
    if original.ndim != 2 or original.shape[1] != len(features) or not np.isfinite(original).all():
        raise ValueError("invalid flow matrix")
    reference = np.asarray(benign_reference, float)
    if reference.shape not in {(len(features),), original.shape}:
        raise ValueError("benign reference shape mismatch")
    if kind not in {"noise", "jitter", "padding", "toward_benign"}:
        raise ValueError("unknown evasion kind")
    if epsilon == 0:
        return original.copy()
    rng = np.random.default_rng(seed)
    mask = nf_mutable_mask(features)
    idx = {name: i for i, name in enumerate(features)}
    candidate = original.copy()
    selected = mask.copy()
    if kind == "jitter":
        selected &= np.array([name in TIMING_FEATURES for name in features])
    elif kind == "padding":
        selected &= np.array([name in {pair[0] for pair in BYTE_PACKET_PAIRS} for name in features])
    bound = epsilon * np.maximum(np.abs(original), 1)
    if kind == "toward_benign":
        candidate[:, selected] += np.clip(reference[..., selected] - original[:, selected],
                                          -bound[:, selected], bound[:, selected])
    elif kind == "padding":
        candidate[:, selected] += rng.uniform(0, 1, candidate[:, selected].shape) * bound[:, selected]
    else:
        candidate[:, selected] += rng.uniform(-1, 1, candidate[:, selected].shape) * bound[:, selected]
    candidate[:, selected] = np.maximum(0, candidate[:, selected])
    for byte_name, packet_name in BYTE_PACKET_PAIRS:
        if byte_name not in idx or packet_name not in idx:
            continue
        b, p = idx[byte_name], idx[packet_name]
        # If the source relation is valid, preserve it without exceeding the box.
        valid_source = original[:, b] >= original[:, p]
        need = valid_source & (candidate[:, b] < candidate[:, p])
        candidate[need, p] = np.minimum(candidate[need, p], candidate[need, b])
        candidate[need, b] = np.maximum(candidate[need, b], candidate[need, p])
    if not np.array_equal(candidate[:, ~mask], original[:, ~mask]):
        raise AssertionError("immutable flow feature changed")
    # Subtracting ~1e13-valued counters loses sub-unit precision in float64.
    tolerance = 4 * np.spacing(np.maximum(np.abs(original), 1)) + 1e-8
    if np.any(np.abs(candidate - original) > bound + tolerance):
        raise AssertionError("evasion exceeded epsilon bound")
    return candidate

