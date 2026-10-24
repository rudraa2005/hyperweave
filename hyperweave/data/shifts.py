import numpy as np


SHIFT_FAMILIES = {
    "train": frozenset({"count_rescale", "timing_jitter", "feature_dropout"}),
    "eval": frozenset({"count_spike", "timing_drift", "feature_outage"}),
}

BYTE_PACKET_PAIRS = (("IN_BYTES", "IN_PKTS"), ("OUT_BYTES", "OUT_PKTS"),
                     ("RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS"),
                     ("RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS"))
TIMING_FEATURES = ("FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT")
DERIVED_RATE_FEATURES = ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
                         "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT")
PACKET_BUCKET_FEATURES = ("NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
                          "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
                          "NUM_PKTS_1024_TO_1514_BYTES")
SHIFTABLE_FEATURES = frozenset(TIMING_FEATURES + DERIVED_RATE_FEATURES +
                               PACKET_BUCKET_FEATURES +
                               tuple(name for pair in BYTE_PACKET_PAIRS for name in pair))


def apply_shift(x, family: str, columns: list[int], severity: int, seed=11):
    if severity not in (1, 2, 3):
        raise ValueError("invalid severity")
    out = np.asarray(x, float).copy()
    if family in {"count_rescale", "count_spike"}:
        out[:, columns] *= 1 + .1 * severity
    elif family in {"timing_jitter", "timing_drift"}:
        rng = np.random.default_rng(seed)
        out[:, columns] = np.maximum(0, out[:, columns] + rng.normal(0, severity, (len(out), len(columns))))
    elif family in {"feature_dropout", "feature_outage"}:
        if family == "feature_dropout":
            rng = np.random.default_rng(seed)
            selected = rng.random((len(out), len(columns))) < min(.1 * severity, .5)
            out[:, columns] = np.where(selected, 0, out[:, columns])
        else:
            out[:, columns] = 0
    else:
        raise ValueError("unknown shift family")
    return out


def apply_nf_shift(x, features, family, severity=1, seed=11, columns=None):
    """Shift raw NF-v2 feature groups without changing identity/protocol fields."""
    if severity not in (1, 2, 3):
        raise ValueError("invalid severity")
    out = np.asarray(x, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] != len(features) or not np.isfinite(out).all():
        raise ValueError("expected finite 2D flow matrix matching feature manifest")
    index = {name: i for i, name in enumerate(features)}
    allowed = set(features) & SHIFTABLE_FEATURES if columns is None else {features[i] for i in columns}
    if not allowed <= SHIFTABLE_FEATURES:
        raise ValueError("shift columns include immutable feature")
    rng = np.random.default_rng(seed)
    if family in {"count_rescale", "count_spike"}:
        multiplier = (rng.uniform(1 - .08 * severity, 1 + .08 * severity, len(out))
                      if family == "count_rescale" else np.full(len(out), 1 + .3 * severity))
        primary_changed = set()
        for byte_name, packet_name in BYTE_PACKET_PAIRS:
            if byte_name not in index or packet_name not in index or not {byte_name, packet_name} <= allowed:
                continue
            b, p = index[byte_name], index[packet_name]
            out[:, b] = np.maximum(0, out[:, b] * multiplier)
            out[:, p] = np.maximum(0, np.rint(out[:, p] * multiplier))
            out[:, b] = np.maximum(out[:, b], out[:, p])
            primary_changed.add(byte_name)
        for name in PACKET_BUCKET_FEATURES:
            if name in index and name in allowed:
                out[:, index[name]] = np.maximum(0, np.rint(out[:, index[name]] * multiplier))
        for name in DERIVED_RATE_FEATURES:
            source = "IN_BYTES" if name.startswith("SRC_TO_DST") else "OUT_BYTES"
            if source in primary_changed and name in index and name in allowed:
                out[:, index[name]] *= multiplier
    elif family in {"timing_jitter", "timing_drift"}:
        factor = (rng.lognormal(0, .05 * severity, len(out))
                  if family == "timing_jitter" else np.full(len(out), 1 + .15 * severity))
        for name in TIMING_FEATURES:
            if name in index and name in allowed:
                out[:, index[name]] = np.maximum(0, out[:, index[name]] * factor)
        for name in DERIVED_RATE_FEATURES:
            if name in index and name in allowed:
                out[:, index[name]] /= factor
    elif family in {"feature_dropout", "feature_outage"}:
        selected_columns = [index[name] for name in DERIVED_RATE_FEATURES if name in index and name in allowed]
        if selected_columns:
            if family == "feature_dropout":
                mask = rng.random((len(out), len(selected_columns))) < .1 * severity
                out[:, selected_columns] = np.where(mask, 0, out[:, selected_columns])
            else:
                out[:, selected_columns] = 0
    else:
        raise ValueError("unknown shift family")
    return out

