"""Label-permutation control. Training labels are permuted; early stopping and the depth choice use real 2022 labels, as in the protocol.
--repeats adds five permutations per country and a variant in which no real label is used anywhere (15% holdout of the permuted rows)."""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.models import Model
from crashtransfer.schema import SEED

STOP = 0.60


def permuted(tr, seed):
    d = tr.copy()
    d["y_ksi"] = tr["y_ksi"].values[np.random.default_rng(seed).permutation(len(tr))]
    return d


def protocol_control(df):
    out = {"rule": "stop if AUROC(KSI, test 2023) > 0.60", "results": {}}
    for src in ("STATS19", "CRSS"):
        tr, va, te = (split_df(df, src, s) for s in ("train", "val", "test"))
        Xtr, Xva, Xte = design(tr)[0], design(va)[0], design(te)[0]
        m = Model("xgboost", "ksi").fit(
            Xtr, permuted(tr, SEED), Xva, va, w_tr=tr.weight.values, w_va=va.weight.values
        )
        p = m.p_ksi(Xte)
        r = {
            "auroc_unweighted": float(roc_auc_score(te.y_ksi, p)),
            "auroc_weighted": float(roc_auc_score(te.y_ksi, p, sample_weight=te.weight)),
            "n_trees": int(m.m.best_iteration + 1),
        }
        real = Model("xgboost", "ksi").fit(Xtr, tr, Xva, va, w_tr=tr.weight.values, w_va=va.weight.values)
        r["auroc_real_labels_unweighted"] = float(roc_auc_score(te.y_ksi, real.p_ksi(Xte)))
        out["results"][src] = r
    worst = max(max(r["auroc_unweighted"], r["auroc_weighted"]) for r in out["results"].values())
    out["passed"] = bool(worst <= STOP)
    json.dump(out, open(paths.result("negative_control.json"), "w"), indent=2)
    tr, va, te = (split_df(df, "BAAC", s) for s in ("train", "val", "test"))
    m = Model("xgboost", "ksi").fit(design(tr)[0], permuted(tr, SEED), design(va)[0], va)
    a = float(roc_auc_score(te.y_ksi, m.p_ksi(design(te)[0])))
    json.dump(
        {"rule": "stop if AUROC(KSI, BAAC test 2023) > 0.60", "auroc_permuted_labels": a, "passed": a <= STOP},
        open(paths.result("negative_control_baac.json"), "w"),
        indent=2,
    )
    if not out["passed"] or a > STOP:
        raise SystemExit("permutation control above 0.60: stop and look for leakage")


def repeats(df):
    rows = []
    for src in ("BAAC", "STATS19", "CRSS"):
        tr, va, te = (split_df(df, src, s) for s in ("train", "val", "test"))
        Xtr, Xva, Xte = design(tr)[0], design(va)[0], design(te)[0]
        for seed in (42, 1, 2, 3, 4):
            d = permuted(tr, seed)
            for name, args, kw in (
                ("real_2022", (Xva, va), dict(w_va=va.weight.values)),
                ("permuted_holdout", (None, None), {}),
            ):
                m = Model("xgboost", "ksi", seed=seed).fit(Xtr, d, *args, w_tr=tr.weight.values, **kw)
                p = m.p_ksi(Xte)
                rows.append(
                    dict(
                        source=src,
                        seed=seed,
                        selection=name,
                        auroc_unweighted=roc_auc_score(te.y_ksi, p),
                        auroc_weighted=roc_auc_score(te.y_ksi, p, sample_weight=te.weight),
                        n_trees=int(m.m.best_iteration + 1),
                        depth=m.hp["max_depth"],
                        n_test=len(te),
                    )
                )
    pd.DataFrame(rows).to_csv(paths.result("e1_negative_control.csv"), index=False)


df = load(baac=True)
protocol_control(df)
if "--repeats" in sys.argv:
    repeats(df)
