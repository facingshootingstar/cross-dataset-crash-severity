"""Pieces shared by the transfer analyses: importance weights and few-shot samples."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import design
from .models import Model, norm_w
from .schema import SEED


def draw(d, n, rng):
    """n rows; CRSS rows are drawn proportional to WEIGHT so that the sample represents the weighted population."""
    p = d.weight.values / d.weight.sum() if d.source.iloc[0] == "CRSS" else None
    return d.iloc[rng.choice(len(d), n, replace=p is not None, p=p)]


def domain_classifier(Xa, Xb, n):
    Xd = pd.concat([Xa, Xb], ignore_index=True)
    yd = np.r_[np.zeros(n), np.ones(n)]
    Xi, Xo, yi, yo = train_test_split(Xd, yd, test_size=0.3, stratify=yd, random_state=SEED)
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=SEED, n_jobs=12, verbose=-1
    ).fit(Xi, yi)
    return clf, float(np.mean((clf.predict_proba(Xo)[:, 1] > 0.5) != yo))


def importance_weights(src_tr, tgt_tr, Xs):
    """Density-ratio weights for the source rows and the proxy A-distance of the domain classifier."""
    rng = np.random.default_rng(SEED)
    n = min(len(src_tr), len(tgt_tr), 150_000)
    a, b = draw(src_tr, n, rng), draw(tgt_tr, n, rng)
    clf, eps = domain_classifier(design(a)[0], design(b)[0], n)
    pr = np.clip(clf.predict_proba(Xs)[:, 1], 1e-3, 1 - 1e-3)
    w = pr / (1 - pr)
    lo, hi = np.quantile(w, [0.01, 0.99])
    w = np.clip(w, lo, hi)
    w = w / w.mean()
    return w, dict(
        domain_classifier_test_error=eps,
        proxy_a_distance=2 * (1 - 2 * eps),
        ess_fraction=float(w.sum() ** 2 / (w**2).sum() / len(w)),
        n_per_domain=int(n),
    )


def few_shot_models(prim, s_tr, Xs_tr, t_tr, frac, seed):
    """Target-only and source + target (mix) models on a fraction of the target training years; 15% of the sample is held out for early stopping."""
    rng = np.random.default_rng(seed)
    n = int(round(frac * len(t_tr)))
    samp = t_tr.iloc[rng.choice(len(t_tr), n, replace=False)].reset_index(drop=True)
    Xsamp, _ = design(samp)
    i_fit, i_es = train_test_split(np.arange(n), test_size=0.15, stratify=samp.y_ksi.values, random_state=seed)
    fit, es = samp.iloc[i_fit].reset_index(drop=True), samp.iloc[i_es].reset_index(drop=True)
    Xfit, Xes = Xsamp.iloc[i_fit].reset_index(drop=True), Xsamp.iloc[i_es].reset_index(drop=True)
    only = Model(prim, "ksi", seed=seed).fit(Xfit, fit, Xes, es, w_tr=fit.weight.values, w_va=es.weight.values)
    mix = pd.concat(
        [s_tr.assign(_w=norm_w(s_tr.weight.values)), fit.assign(_w=norm_w(fit.weight.values))], ignore_index=True
    )
    mixed = Model(prim, "ksi", seed=seed).fit(
        pd.concat([Xs_tr, Xfit], ignore_index=True), mix, Xes, es, w_tr=mix["_w"].values, w_va=es.weight.values
    )
    return only, mixed
