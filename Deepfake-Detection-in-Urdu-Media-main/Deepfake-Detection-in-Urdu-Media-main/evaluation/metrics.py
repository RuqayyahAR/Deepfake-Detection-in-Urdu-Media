"""
Accuracy metrics for a binary deepfake detector.
Convention: label 1 = FAKE (positive), label 0 = REAL. score: higher = more likely fake.

AUC  : probability a random fake is scored above a random real (threshold-free).
EER  : equal error rate, the point where false-accept and false-reject rates are equal.
       Lower is better. The threshold at EER is returned so it can be reported.
CI   : bootstrap confidence interval (resampling videos, not frames).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y, s):
    return float(roc_auc_score(y, s))


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return {"eer": float((fpr[i] + fnr[i]) / 2.0), "threshold": float(thr[i])}


def bootstrap_ci(y, s, fn, n=2000, seed=0, alpha=0.05):
    """Percentile CI for fn(y, s). Resamples whole videos (each row = one video)."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:  # need both classes to compute the metric
            continue
        vals.append(fn(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def report(y, s, n_boot=2000):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        raise ValueError("need both REAL and FAKE videos to compute metrics")
    e = eer(y, s)
    thr = e["threshold"] if np.isfinite(e["threshold"]) else None
    a_lo, a_hi = bootstrap_ci(y, s, auc, n_boot)
    e_lo, e_hi = bootstrap_ci(y, s, lambda yy, ss: eer(yy, ss)["eer"], n_boot)
    return {
        "n_videos": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "auc": round(auc(y, s), 4),
        "auc_95ci": [round(a_lo, 4), round(a_hi, 4)],
        "eer": round(e["eer"], 4),
        "eer_95ci": [round(e_lo, 4), round(e_hi, 4)],
        "eer_threshold": None if thr is None else round(thr, 4),
    }
