"""Tiers T0-T3 and the prior correction on test 2023, repeated over five model seeds for all six directed pairs.

Besides AUPRC, ECE and AUROC, each arm is read in two operational terms: the expected KSI count (sum of predicted
probabilities) against the observed count, overall and in urban/rural x light cells, and the share of KSI crashes
that falls in the top 5, 10 or 20% of predicted risk. Paired bootstrap intervals against T1 use seed 42.
Output: seed_ladder.csv, seed_ladder_summary.csv, seed_ladder_paired.csv
"""

import argparse

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot, delta_metrics
from crashtransfer.models import Model, norm_w, primary, prior_shift, recalibrate
from crashtransfer.schema import COUNTRIES, PAIRS_FRANCE, PAIRS_UK_US, SEED, target_weight
from crashtransfer.transfer import importance_weights

SEEDS = [42, 1, 2, 3, 4]
BUDGETS = [0.05, 0.10, 0.20]
MIN_CELL_KSI = 100


def capture(y, p, w, k):
    """Share of (weighted) KSI crashes among the top k of (weighted) crashes by p. Rows tied at the cut count pro rata."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    d = pd.DataFrame(dict(p=p, w=w, k=w * y)).groupby("p", sort=True)[["w", "k"]].sum().iloc[::-1]
    cw, ck = d.w.cumsum().values, d.k.cumsum().values
    cut = k * w.sum()
    i = int(np.searchsorted(cw, cut))
    if i >= len(d):
        return 1.0
    before_w, before_k = (cw[i - 1], ck[i - 1]) if i else (0.0, 0.0)
    part = (cut - before_w) / d.w.values[i]
    return float((before_k + part * d.k.values[i]) / ck[-1])


def count_errors(d, p, w):
    w = np.ones(len(d)) if w is None else np.asarray(w, float)
    y = d.y_ksi.values
    out = dict(count_ratio=float((w * p).sum() / (w * y).sum()))
    cell = d.urban_rural.fillna("missing").astype(str) + "/" + d.lighting_dark.fillna(-1).astype(int).astype(str)
    g = pd.DataFrame(dict(c=cell, e=w * p, o=w * y, n=y)).groupby("c").sum()
    g = g[g.n >= MIN_CELL_KSI]
    out["cell_max_abs_rel_err"] = float((g.e / g.o - 1).abs().max())
    out["n_cells"] = int(len(g))
    return out


def readings(d, p, w):
    r = binary_metrics(d.y_ksi.values, p, w)
    r.update(count_errors(d, p, w))
    for k in BUDGETS:
        r[f"capture_top{int(k * 100)}"] = capture(d.y_ksi.values, p, w, k)
    return r


def arms(prim, seed, S, X, M, src, tgt):
    s_tr, s_va = S[src]["train"], S[src]["val"]
    t_tr, t_va = S[tgt]["train"], S[tgt]["val"]
    p1, pva = M[src].p_ksi(X[tgt]["test"]), M[src].p_ksi(X[tgt]["val"])
    pi_s = float(np.average(s_tr.y_ksi, weights=s_tr.weight))
    pi_train = float(np.average(t_tr.y_ksi, weights=t_tr.weight))
    pi_2022 = float(np.average(t_va.y_ksi, weights=t_va.weight))
    iw, _ = importance_weights(s_tr, t_tr, X[src]["train"])
    T3 = Model(prim, "ksi", seed=seed).fit(
        X[src]["train"], s_tr, X[src]["val"], s_va, w_tr=norm_w(s_tr.weight.values) * iw, w_va=s_va.weight.values
    )
    return {
        "T0": M[tgt].p_ksi(X[tgt]["test"]),
        "T1": p1,
        "T2": recalibrate("isotonic", pva, t_va.y_ksi.values, target_weight(t_va))(p1),
        "prior_train_share": prior_shift(p1, pi_s, pi_train),
        "prior_2022_share": prior_shift(p1, pi_s, pi_2022),
        "T3": T3.p_ksi(X[tgt]["test"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="defaults to the primary model chosen by within_domain")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    prim = a.model or primary()
    df = load(baac=True)
    S = {c: {k: split_df(df, c, k) for k in ("train", "val", "test")} for c in COUNTRIES}
    X = {c: {k: design(v)[0] for k, v in S[c].items()} for c in COUNTRIES}
    rows, paired = [], []
    for seed in a.seeds:
        M = {
            c: Model(prim, "ksi", seed=seed).fit(
                X[c]["train"],
                S[c]["train"],
                X[c]["val"],
                S[c]["val"],
                w_tr=S[c]["train"].weight.values,
                w_va=S[c]["val"].weight.values,
            )
            for c in COUNTRIES
        }
        for src, tgt in PAIRS_UK_US + PAIRS_FRANCE:
            te = S[tgt]["test"]
            w = target_weight(te)
            P = arms(prim, seed, S, X, M, src, tgt)
            for arm, p in P.items():
                rows.append(dict(seed=seed, direction=f"{src}->{tgt}", arm=arm, **readings(te, p, w)))
            if seed == SEED:
                mode = "weighted" if w is not None else "uniform"
                y = te.y_ksi.values
                for arm in ("T0", "T2", "prior_train_share", "prior_2022_share", "T3"):
                    pp = np.c_[P["T1"], P[arm]]
                    est = delta_metrics(y, pp, w)
                    ci = boot(delta_metrics, y, pp, te.weight.values, mode)
                    paired += [
                        dict(
                            direction=f"{src}->{tgt}",
                            arm=arm,
                            vs="T1",
                            metric=k,
                            diff=v,
                            ci_lo=ci[k][0],
                            ci_hi=ci[k][1],
                        )
                        for k, v in est.items()
                    ]
            print(seed, src, tgt, flush=True)
        pd.DataFrame(rows).to_csv(paths.result("seed_ladder.csv"), index=False)
    r = pd.DataFrame(rows)
    keys = [c for c in r.columns if c not in ("seed", "direction", "arm", "n_cells")]
    s = r.groupby(["direction", "arm"])[keys].agg(["mean", "std", "min", "max"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    s.reset_index().to_csv(paths.result("seed_ladder_summary.csv"), index=False)
    pd.DataFrame(paired).to_csv(paths.result("seed_ladder_paired.csv"), index=False)


if __name__ == "__main__":
    main()
