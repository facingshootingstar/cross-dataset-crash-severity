"""TreeSHAP of the primary model on 20,000 rows of 2023 per country: rank agreement by Spearman's rho and Kendall's W. Descriptive only."""

import json

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.models import load_model, primary
from crashtransfer.schema import COUNTRIES, FEATS, SEED

B = 50
df, prim = load(baac=True), primary()
A, rngs, weights = {}, {}, {}
for s in COUNTRIES:
    te = split_df(df, s, "test")
    rngs[s] = np.random.default_rng(SEED)
    d = te.iloc[rngs[s].choice(len(te), min(20_000, len(te)), replace=False)].reset_index(drop=True)
    X, cmap = design(d)
    sv = np.asarray(shap.TreeExplainer(load_model(f"{s}_{prim}_ksi.joblib").m).shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    A[s] = pd.DataFrame(
        {f: sv[:, [i for i, c in enumerate(X.columns) if cmap[c] == f]].sum(1) for f in FEATS}
    ).abs()
    weights[s] = d.weight.values

res = []
for s in ("STATS19", "CRSS"):
    mean = A[s].mean()
    wmean = pd.Series(np.average(A[s].values, axis=0, weights=weights[s]), index=FEATS)
    ranks = mean.rank(ascending=False)
    brank = np.array(
        [
            A[s].iloc[rngs[s].choice(len(A[s]), len(A[s]), replace=True)].mean().rank(ascending=False).values
            for _ in range(B)
        ]
    )
    for j, f in enumerate(FEATS):
        res.append(
            dict(
                source=s,
                feature=f,
                mean_abs_shap=mean[f],
                mean_abs_shap_design_weighted=wmean[f],
                rank=ranks[f],
                rank_lo=float(np.quantile(brank[:, j], 0.025)),
                rank_hi=float(np.quantile(brank[:, j], 0.975)),
                n_rows=len(A[s]),
            )
        )
pd.DataFrame(res).to_csv(paths.result("shap_importance.csv"), index=False)

mu, mc = A["STATS19"].mean(), A["CRSS"].mean()
rho, p = spearmanr(mu[FEATS], mc[FEATS])
rng = np.random.default_rng(SEED)
rb = []
for _ in range(B):
    a = A["STATS19"].iloc[rng.choice(len(A["STATS19"]), len(A["STATS19"]))].mean()
    c = A["CRSS"].iloc[rng.choice(len(A["CRSS"]), len(A["CRSS"]))].mean()
    rb.append(spearmanr(a[FEATS], c[FEATS])[0])
json.dump(
    {
        "model": prim,
        "n_features": len(FEATS),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "bootstrap50_ci": [float(np.quantile(rb, 0.025)), float(np.quantile(rb, 0.975))],
        "note": "12 features only: underpowered; descriptive, no causal reading",
        "top3_STATS19": mu.sort_values(ascending=False).index[:3].tolist(),
        "top3_CRSS": mc.sort_values(ascending=False).index[:3].tolist(),
    },
    open(paths.result("shap_rank_correlation.json"), "w"),
    indent=2,
)

res3 = []
for s in COUNTRIES:
    mean, rank = A[s].mean(), A[s].mean().rank(ascending=False)
    res3 += [dict(source=s, feature=f, mean_abs_shap=mean[f], rank=rank[f]) for f in FEATS]
pd.DataFrame(res3).to_csv(paths.result("shap_importance_3way.csv"), index=False)


def kendall_w(means):
    R = np.array([m.rank(ascending=False).values for m in means])
    k, n = R.shape
    return float(12 * ((R.sum(0) - R.sum(0).mean()) ** 2).sum() / (k**2 * (n**3 - n)))


W = kendall_w([A[s].mean()[FEATS] for s in COUNTRIES])
rng = np.random.default_rng(SEED)
wb = [kendall_w([A[s].iloc[rng.choice(len(A[s]), len(A[s]))].mean()[FEATS] for s in COUNTRIES]) for _ in range(B)]
pair = {
    f"{a}~{b}": float(spearmanr(A[a].mean()[FEATS], A[b].mean()[FEATS])[0])
    for i, a in enumerate(COUNTRIES)
    for b in COUNTRIES[i + 1 :]
}
json.dump(
    {
        "kendall_w": W,
        "bootstrap50_ci": [float(np.quantile(wb, 0.025)), float(np.quantile(wb, 0.975))],
        "spearman_pairwise": pair,
        "note": "12 features, 3 countries: underpowered, descriptive",
    },
    open(paths.result("shap_kendall_w.json"), "w"),
    indent=2,
)
