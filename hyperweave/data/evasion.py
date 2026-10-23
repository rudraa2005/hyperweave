import numpy as np


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

