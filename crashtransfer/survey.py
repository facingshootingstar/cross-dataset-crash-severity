"""Design-based replicate weights for CRSS (Rao-Wu bootstrap over PSU_VAR within PSUSTRAT) and paired comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import paths
from .metrics import ece
from .schema import SEED

B_POST = 500


def add_design(df):
    d = pd.read_parquet(paths.interim("crss_posthoc.parquet"), columns=["case_id", "psustrat", "psu_var"])
    out = df.merge(d, on="case_id", how="left")
    assert len(out) == len(df)
    assert out.loc[out.source == "CRSS", "psu_var"].notna().all()
    return out


def replicates(d, B=B_POST, seed=SEED):
    """Yields B weight vectors for one source and split; STATS19 is a census of reported collisions, so rows are resampled."""
    rng = np.random.default_rng(seed)
    n = len(d)
    if d.source.iloc[0] == "STATS19":
        for _ in range(B):
            yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        return
    w = d.weight.values.astype(float)
    key = d.psustrat.astype(int).astype(str) + "_" + d.psu_var.astype(int).astype(str)
    psu_codes, psu_idx = np.unique(key.values, return_inverse=True)
    psu_stratum = pd.Series(psu_codes).str.split("_").str[0].values
    strata = {h: np.where(psu_stratum == h)[0] for h in np.unique(psu_stratum)}
    for _ in range(B):
        mult = np.ones(len(psu_codes))
        for ps in strata.values():
            nh = len(ps)
            if nh < 2:
                continue
            cnt = np.bincount(rng.integers(0, nh, nh - 1), minlength=nh)
            mult[ps] = cnt * nh / (nh - 1)
        yield w * mult[psu_idx]


def design_info(d):
    key = d.psustrat.astype(int).astype(str) + "_" + d.psu_var.astype(int).astype(str)
    s = pd.DataFrame({"h": d.psustrat.values, "k": key.values}).drop_duplicates()
    return dict(
        n_strata=int(s.h.nunique()), n_psu=int(s.k.nunique()), min_psu_per_stratum=int(s.groupby("h").size().min())
    )


def light_metrics(y, p, w):
    m = w > 0
    y, p, w = y[m], np.clip(p[m], 1e-6, 1 - 1e-6), w[m]
    return dict(
        auroc=roc_auc_score(y, p, sample_weight=w),
        auprc=average_precision_score(y, p, sample_weight=w),
        ece=ece(y, p, w),
    )


def point_weight(d):
    return d.weight.values.astype(float) if d.source.iloc[0] == "CRSS" else np.ones(len(d))


def compare(d, preds, pairs, B=B_POST):
    """preds: {name: p}; pairs: [(a, b)] for a minus b. All predictions share the same replicates."""
    y = d.y_ksi.values
    w0 = point_weight(d)
    est = {k: light_metrics(y, p, w0) for k, p in preds.items()}
    reps = {k: [] for k in preds}
    for wr in replicates(d, B):
        for k, p in preds.items():
            reps[k].append(light_metrics(y, p, wr))
    rows = []
    for k in preds:
        r = pd.DataFrame(reps[k])
        for m in r.columns:
            rows.append(
                dict(
                    item=k,
                    metric=m,
                    value=est[k][m],
                    ci_lo=r[m].quantile(0.025),
                    ci_hi=r[m].quantile(0.975),
                    se=r[m].std(),
                )
            )
    for a, b in pairs:
        ra, rb = pd.DataFrame(reps[a]), pd.DataFrame(reps[b])
        for m in ra.columns:
            diff = ra[m] - rb[m]
            rows.append(
                dict(
                    item=f"{a} minus {b}",
                    metric=m,
                    value=est[a][m] - est[b][m],
                    ci_lo=diff.quantile(0.025),
                    ci_hi=diff.quantile(0.975),
                    se=diff.std(),
                )
            )
    return pd.DataFrame(rows)
