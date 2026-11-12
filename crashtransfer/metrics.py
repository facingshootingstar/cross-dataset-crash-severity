"""Binary and three-class metrics, reliability bins and the percentile bootstrap."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from .schema import SEED


def ece(y, p, w=None, bins=15):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    p = np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot, e = w.sum(), 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += w[m].sum() / tot * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return e


def reliability(y, p, w=None, bins=15):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    p = np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        m = b == i
        if m.any():
            rows.append(
                dict(
                    bin=i,
                    lo=i / bins,
                    hi=(i + 1) / bins,
                    n=int(m.sum()),
                    weight=float(w[m].sum()),
                    mean_pred=float(np.average(p[m], weights=w[m])),
                    frac_pos=float(np.average(y[m], weights=w[m])),
                )
            )
    return pd.DataFrame(rows)


def binary_metrics(y, p, w=None):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return dict(
        auprc=average_precision_score(y, p, sample_weight=w),
        auroc=roc_auc_score(y, p, sample_weight=w),
        logloss=log_loss(y, p, sample_weight=w, labels=[0, 1]),
        brier=brier_score_loss(y, p, sample_weight=w),
        ece=ece(y, p, w),
        prevalence=float(np.average(y, weights=w)),
    )


def three_class_metrics(y3, P, w=None):
    """y3 in {1, 2, 3}; columns of P are Fatal, Serious, Slight."""
    pred = P.argmax(1) + 1
    rec = {}
    for c, name in [(1, "recall_fatal"), (2, "recall_serious")]:
        m = y3 == c
        rec[name] = (
            float(np.average(pred[m] == c, weights=None if w is None else np.asarray(w)[m])) if m.any() else np.nan
        )
    return dict(macro_f1=f1_score(y3, pred, average="macro", sample_weight=w, labels=[1, 2, 3]), **rec)


def boot(fn, y, p, w, mode, B=200, seed=SEED):
    """95% percentile interval. mode 'weighted' draws rows with probability proportional to w and scores them unweighted."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pr = None if mode == "uniform" else np.asarray(w, float) / np.sum(w)
    out = []
    for _ in range(B):
        i = rng.choice(n, n, replace=True, p=pr)
        try:
            out.append(fn(y[i], p[i], None))
        except ValueError:
            continue
    d = pd.DataFrame(out)
    return {k: (float(d[k].quantile(0.025)), float(d[k].quantile(0.975))) for k in d.columns}


def delta_metrics(y, P, w):
    """Column 1 minus column 0 of P, for paired bootstrap intervals."""
    a, b = binary_metrics(y, P[:, 0], w), binary_metrics(y, P[:, 1], w)
    return {f"d_{k}": b[k] - a[k] for k in ["auprc", "auroc", "ece"]}
