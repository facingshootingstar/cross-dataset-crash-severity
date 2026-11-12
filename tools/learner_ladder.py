"""T0, T1 and T2 on test 2023 for every learner family, six directed pairs, to check that the transfer loss does not
depend on the learner. Readings as in seed_ladder (AUPRC, ECE, AUROC, expected/observed KSI count, KSI share in the
top 5/10/20% of risk). Logistic regression is deterministic and runs once; the other learners run over three seeds.
Output: learner_ladder.csv, learner_ladder_summary.csv
"""

import argparse

import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.models import Model, recalibrate
from crashtransfer.schema import COUNTRIES, PAIRS_FRANCE, PAIRS_UK_US, target_weight
from tools.seed_ladder import readings

LEARNERS = {
    "logreg": [42],
    "rf": [42, 1, 2],
    "lightgbm": [42, 1, 2],
    "catboost": [42, 1, 2],
    "xgboost": [42, 1, 2],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", nargs="+", default=list(LEARNERS))
    a = ap.parse_args()
    df = load(baac=True)
    S = {c: {k: split_df(df, c, k) for k in ("train", "val", "test")} for c in COUNTRIES}
    X = {c: {k: design(v)[0] for k, v in S[c].items()} for c in COUNTRIES}
    rows = []
    for name in a.learners:
        for seed in LEARNERS[name]:
            M = {
                c: Model(name, "ksi", seed=seed).fit(
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
                te, va = S[tgt]["test"], S[tgt]["val"]
                w = target_weight(te)
                p1 = M[src].p_ksi(X[tgt]["test"])
                g = recalibrate("isotonic", M[src].p_ksi(X[tgt]["val"]), va.y_ksi.values, target_weight(va))
                arms = {"T0": M[tgt].p_ksi(X[tgt]["test"]), "T1": p1, "T2": g(p1)}
                for arm, p in arms.items():
                    rows.append(
                        dict(learner=name, seed=seed, direction=f"{src}->{tgt}", arm=arm, **readings(te, p, w))
                    )
            print(name, seed, flush=True)
            pd.DataFrame(rows).to_csv(paths.result("learner_ladder.csv"), index=False)
    r = pd.DataFrame(rows)
    keys = ["auprc", "auroc", "ece", "count_ratio", "capture_top10"]
    s = r.groupby(["learner", "direction", "arm"])[keys].agg(["mean", "std"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    s = s.reset_index()
    wide = s.pivot_table(index=["learner", "direction"], columns="arm", values=["auprc_mean", "auroc_mean"])
    loss = pd.DataFrame(
        {
            "auprc_loss": wide[("auprc_mean", "T0")] - wide[("auprc_mean", "T1")],
            "auroc_loss": wide[("auroc_mean", "T0")] - wide[("auroc_mean", "T1")],
        }
    ).reset_index()
    s.merge(loss, on=["learner", "direction"]).to_csv(paths.result("learner_ladder_summary.csv"), index=False)
    print(loss.groupby("learner")[["auprc_loss", "auroc_loss"]].agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
