"""Amendment A2 (docs/protocol/PROTOCOL_VIC.md): Victoria as a fourth jurisdiction, six new directed pairs.

Within Victoria for the five learners; the transfer ladder over five model seeds with readings and paired intervals
(as seed_ladder); the learning curve and prior correction (as learning_curve); the locked year 2024, written once;
and the outcome of predictions P11-P16 under the rules locked in the amendment.
Output: victoria_within.csv, victoria_ladder*.csv, posthoc_learning_curve*_vic.csv, posthoc_prior_shift_vic.csv,
locked_2024_vic.csv, victoria_predictions.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot, delta_metrics
from crashtransfer.models import Model, primary, recalibrate, save_model
from crashtransfer.schema import COUNTRIES, PAIRS_VIC, SEED, target_weight
from tools import learning_curve
from tools.seed_ladder import SEEDS, arms, readings

ALL = COUNTRIES + ["VIC"]
LEARNERS = ["majority", "logreg", "rf", "xgboost", "lightgbm"]


def fit(name, seed, S, X, c):
    tr, va = S[c]["train"], S[c]["val"]
    return Model(name, "ksi", seed=seed).fit(
        X[c]["train"], tr, X[c]["val"], va, w_tr=tr.weight.values, w_va=va.weight.values
    )


def within(S, X):
    rows = []
    te = S["VIC"]["test"]
    for name in LEARNERS:
        p = fit(name, SEED, S, X, "VIC").p_ksi(X["VIC"]["test"])
        est, ci = (
            binary_metrics(te.y_ksi.values, p, None),
            boot(binary_metrics, te.y_ksi.values, p, None, "uniform"),
        )
        rows += [dict(model=name, metric=k, value=v, ci_lo=ci[k][0], ci_hi=ci[k][1]) for k, v in est.items()]
    pd.DataFrame(rows).to_csv(paths.result("victoria_within.csv"), index=False)
    return pd.DataFrame(rows)


def ladder(prim, S, X):
    rows, paired = [], []
    for seed in SEEDS:
        M = {c: fit(prim, seed, S, X, c) for c in ALL}
        if seed == SEED:
            for c in ALL:
                save_model(M[c], f"{c}_{prim}_ksi.joblib")
        for src, tgt in PAIRS_VIC:
            te = S[tgt]["test"]
            w = target_weight(te)
            P = arms(prim, seed, S, X, M, src, tgt)
            rows += [dict(seed=seed, direction=f"{src}->{tgt}", arm=a, **readings(te, p, w)) for a, p in P.items()]
            if seed == SEED:
                y, mode = te.y_ksi.values, "weighted" if w is not None else "uniform"
                for arm in ("T0", "T2", "prior_train_share", "prior_2022_share", "T3"):
                    pp = np.c_[P["T1"], P[arm]]
                    est, ci = delta_metrics(y, pp, w), boot(delta_metrics, y, pp, te.weight.values, mode)
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
            print("ladder", seed, src, tgt, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(paths.result("victoria_ladder.csv"), index=False)
    pd.DataFrame(paired).to_csv(paths.result("victoria_ladder_paired.csv"), index=False)
    keys = [c for c in r.columns if c not in ("seed", "direction", "arm", "n_cells")]
    s = r.groupby(["direction", "arm"])[keys].agg(["mean", "std"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    s.reset_index().to_csv(paths.result("victoria_ladder_summary.csv"), index=False)
    return s.reset_index(), pd.DataFrame(paired)


def locked(prim, df):
    out = paths.result("locked_2024_vic.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: the locked year is scored once")
    D = {c: {k: split_df(df, c, k) for k in ("train", "val", "locked")} for c in ALL}
    X = {c: {k: design(v)[0] for k, v in D[c].items()} for c in ALL}
    M = {
        c: Model(prim, "ksi").fit(
            X[c]["train"],
            D[c]["train"],
            X[c]["val"],
            D[c]["val"],
            w_tr=D[c]["train"].weight.values,
            w_va=D[c]["val"].weight.values,
        )
        for c in ALL
    }
    rows = []
    for src, tgt in PAIRS_VIC:
        lk, va = D[tgt]["locked"], D[tgt]["val"]
        w = target_weight(lk)
        p1 = M[src].p_ksi(X[tgt]["locked"])
        g = recalibrate("isotonic", M[src].p_ksi(X[tgt]["val"]), va.y_ksi.values, target_weight(va))
        for arm, p in (("T0", M[tgt].p_ksi(X[tgt]["locked"])), ("T1", p1), ("T2", g(p1))):
            rows.append(dict(direction=f"{src}->{tgt}", arm=arm, **binary_metrics(lk.y_ksi.values, p, w)))
    pd.DataFrame(rows).to_csv(out, index=False)


def predictions(wi, s, paired):
    m = s.set_index(["direction", "arm"])
    dirs = [f"{a}->{b}" for a, b in PAIRS_VIC]
    pr = paired[(paired.arm == "T0") & (paired.metric == "d_auprc")].set_index("direction")
    lc = pd.read_csv(paths.result("posthoc_learning_curve_vic.csv"))
    g = lc.groupby(["direction", "n", "arm"]).auroc.mean().unstack()
    t1 = lc.groupby("direction").auroc_T1.first()
    reach = {d: bool((g.loc[d].loc[g.loc[d].index <= 2500, "target_only"] >= t1[d]).any()) for d in dirs}
    ft = {
        d: bool(
            (
                g.loc[d].loc[g.loc[d].index <= 1000, "finetune"]
                > g.loc[d].loc[g.loc[d].index <= 1000, "target_only"]
            ).all()
        )
        for d in dirs
    }
    w = wi.set_index(["model", "metric"])
    res = {
        "P11": sum(pr.loc[d, "ci_lo"] > 0 for d in dirs),
        "P12": sum(m.loc[(d, "T2"), "ece_mean"] < m.loc[(d, "T1"), "ece_mean"] for d in dirs),
        "P13": sum(reach.values()),
        "P14": sum(ft.values()),
        "P15": sum(m.loc[(d, "prior_train_share"), "ece_mean"] < m.loc[(d, "T1"), "ece_mean"] for d in dirs),
    }
    need = {"P12": 5}  # P12 fails if ECE is not lower on 2 or more pairs; the others fail on 3 or more
    out = {k: dict(pairs_meeting=int(v), of=6, supported=bool(v >= need.get(k, 4))) for k, v in res.items()}
    xg, lr = w.loc[("xgboost", "auroc")], w.loc[("logreg", "auroc")]
    out["P16"] = dict(
        xgboost=[xg.value, xg.ci_lo, xg.ci_hi],
        logreg=[lr.value, lr.ci_lo, lr.ci_hi],
        supported=bool(xg.ci_lo > lr.ci_hi),
    )
    json.dump(out, open(paths.result("victoria_predictions.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="defaults to the primary model chosen by within_domain")
    prim = ap.parse_args().model or primary()
    df = load(baac=True, vic=True)
    S = {c: {k: split_df(df, c, k) for k in ("train", "val", "test")} for c in ALL}
    X = {c: {k: design(v)[0] for k, v in S[c].items()} for c in ALL}
    wi = within(S, X)
    s, paired = ladder(prim, S, X)
    learning_curve.run(df, prim, PAIRS_VIC, "_vic")
    locked(prim, df)
    predictions(wi, s, paired)


if __name__ == "__main__":
    main()
