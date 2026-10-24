"""Offline shift/evasion diagnostics for a baked detector.

Only the detector's fixed-threshold scores are used at deployment. These
diagnostics are evaluated in batches and never called by kernel/infer.c.
"""
import numpy as np

from hyperweave.data.evasion import constrained_nf_evasion
from hyperweave.data.shifts import SHIFT_FAMILIES, apply_nf_shift
from hyperweave.eval.metrics import score_binary


def robustness_report(model, features, train_raw, train_y, eval_raw, eval_y,
                      center, scale, threshold, *, seed=11, severities=(1, 2, 3),
                      epsilons=(0., .1, .2, .3), transform=None):
    """Score B shift families and constrained evasion at the A-val threshold."""
    train_raw, eval_raw = np.asarray(train_raw, float), np.asarray(eval_raw, float)
    train_y, eval_y = np.asarray(train_y), np.asarray(eval_y)
    center, scale = np.asarray(center, float), np.asarray(scale, float)
    if train_raw.shape[1:] != (len(features),) or eval_raw.shape[1:] != (len(features),):
        raise ValueError("feature manifest mismatch")
    if not np.any(train_y == 0) or not np.any(eval_y == 1) or not np.any(eval_y == 0):
        raise ValueError("need train benign and both evaluation classes")
    score = lambda raw: model.logits(transform(raw) if transform is not None else (raw - center) / scale)
    baseline = score_binary(eval_y, score(eval_raw), threshold)
    shifts = {}
    for family in sorted(SHIFT_FAMILIES["eval"]):
        shifts[family] = {}
        for severity in severities:
            shifted = apply_nf_shift(eval_raw, features, family, severity, seed=seed + severity)
            shifts[family][str(severity)] = score_binary(eval_y, score(shifted), threshold)
    benign_reference = np.median(train_raw[train_y == 0], axis=0)
    attack = eval_raw[eval_y == 1]
    evasion = {}
    for kind in ("noise", "jitter", "padding", "toward_benign"):
        evasion[kind] = {}
        for epsilon in epsilons:
            candidate = constrained_nf_evasion(attack, benign_reference, features, epsilon, kind,
                                               seed=seed + round(epsilon * 1000))
            evasion[kind][str(epsilon)] = float(np.mean(score(candidate) > threshold))
    return {"threshold_source": "a_val", "baseline": baseline, "shifts": shifts,
            "evasion_detection_rate": evasion, "attack_rows": int(len(attack)),
            "feature_space_proxy": True}


def reconstruction_report(envelope, eval_raw, eval_y):
    """Batch-only AE diagnostic; threshold must have been calibrated on A-val."""
    errors = envelope.reconstruction_error(eval_raw)
    labels = np.asarray(eval_y)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("both classes needed")
    return {"threshold_source": "a_val_benign", "threshold": envelope.threshold,
            "benign_flag_rate": float(np.mean(errors[labels == 0] > envelope.threshold)),
            "attack_flag_rate": float(np.mean(errors[labels == 1] > envelope.threshold)),
            "mean_benign_error": float(np.mean(errors[labels == 0])),
            "mean_attack_error": float(np.mean(errors[labels == 1])),
            "deployment_path": "offline_only"}
