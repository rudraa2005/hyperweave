import numpy as np
from sklearn.metrics import average_precision_score


def threshold_at_fpr(benign_scores, target_fpr):
    scores = np.sort(np.asarray(benign_scores, float))
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("calibration requires finite benign scores")
    if not 0 < target_fpr < 1:
        raise ValueError("invalid FPR")
    allowed = int(np.floor(target_fpr * len(scores)))
    # Strict > threshold. Ties may make the achieved FPR lower than target.
    return float(scores[-allowed - 1] if allowed < len(scores) else scores[0])


def score_binary(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores, float)
    if not set(np.unique(y)) <= {0, 1} or len(y) != len(scores):
        raise ValueError("invalid labels or scores")
    benign, attack = y == 0, y == 1
    if not benign.any() or not attack.any():
        raise ValueError("both classes needed")
    return {"fpr": float(np.mean(scores[benign] > threshold)),
            "detection_rate": float(np.mean(scores[attack] > threshold)),
            "pr_auc": float(average_precision_score(y, scores))}
