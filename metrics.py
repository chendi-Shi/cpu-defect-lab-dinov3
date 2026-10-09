"""Exact binary AUROC with tied scores, without sklearn."""
import numpy as np


def auroc(labels, scores):
    y = np.asarray(labels).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError('Labels/scores must match and scores must be finite')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be binary')
    y = y.astype(np.uint8)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(s, kind='stable')
    sorted_s = s[order]
    # Average ranks make ties count as half a win.
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float((ranks[y[order] == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def threshold_stats(labels, scores, threshold):
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(scores) > threshold
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
