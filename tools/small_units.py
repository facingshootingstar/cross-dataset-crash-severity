"""Analysis note B2 (docs/protocol/ANALYSIS_SMALL_UNITS.md): police force areas and French departements as small new
jurisdictions. For each unit: foreign zero-shot, foreign + Platt and foreign fine-tuned on the unit's training-year
collisions, a local-only model and the national model, all scored on the unit's 2023 collisions.
Needs results of prepare_data with region keys (interim/region_keys.parquet).
Output: small_units.csv, small_units_summary.csv
"""

import argparse

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics
from crashtransfer.models import Model, norm_w, primary, recalibrate
from crashtransfer.schema import SEED
from tools.learning_curve import finetune

FOREIGN = {"STATS19": ["CRSS", "BAAC"], "BAAC": ["STATS19", "CRSS"]}
BANDS = [(0, 1000, "N < 1,000"), (1000, 2500, "1,000 <= N <= 2,500"), (2500, 10**9, "N > 2,500")]


def fit(prim, c, S, X):
    tr, va = S[c]["train"], S[c]["val"]
    return Model(prim, "ksi").fit(X[c]["train"], tr, X[c]["val"], va, w_tr=tr.weight.values, w_va=va.weight.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="defaults to the primary model chosen by within_domain")
    prim = ap.parse_args().model or primary()
    df = load(baac=True)
    keys = pd.read_parquet(paths.interim("region_keys.parquet"))
    df = df.merge(keys, on=["source", "case_id"], how="left")
    countries = ["STATS19", "CRSS", "BAAC"]
    S = {c: {k: split_df(df, c, k) for k in ("train", "val", "test")} for c in countries}
    X = {c: {k: design(v)[0] for k, v in S[c].items()} for c in countries}
    M = {c: fit(prim, c, S, X) for c in countries}
    rows = []
    for tgt, srcs in FOREIGN.items():
        tr_all, te_all = S[tgt]["train"], S[tgt]["test"]
        for u in sorted(tr_all.region.dropna().unique()):
            tr, te = tr_all[tr_all.region == u], te_all[te_all.region == u]
            if len(tr) < 100 or len(te) < 300 or tr.y_ksi.nunique() < 2 or te.y_ksi.nunique() < 2:
                continue
            Xtr, Xte = X[tgt]["train"].loc[tr.index], X[tgt]["test"].loc[te.index]
            y, yn, wn = te.y_ksi.values, tr.y_ksi.values, norm_w(tr.weight.values)
            arms = {"national": M[tgt].p_ksi(Xte)}
            local = Model(prim, "ksi", seed=SEED).fit(Xtr.reset_index(drop=True), tr.reset_index(drop=True))
            arms["local_only"] = local.p_ksi(Xte)
            for s in srcs:
                p1 = M[s].p_ksi(Xte)
                arms[f"{s}_zero_shot"] = p1
                arms[f"{s}_platt"] = recalibrate("platt", M[s].p_ksi(Xtr), yn, wn)(p1)
                ft = finetune(M[s], Xtr.reset_index(drop=True), yn, wn, SEED)
                arms[f"{s}_finetuned"] = ft.predict_proba(Xte)[:, 1]
            for arm, p in arms.items():
                rows.append(
                    dict(country=tgt, unit=u, n_train=len(tr), n_test=len(te), arm=arm, **binary_metrics(y, p))
                )
            print(tgt, u, len(tr), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(paths.result("small_units.csv"), index=False)
    if r.empty:
        pd.DataFrame().to_csv(paths.result("small_units_summary.csv"), index=False)
        print("no unit has 100 training-year and 300 test-year collisions")
        return
    out = []
    for tgt, srcs in FOREIGN.items():
        w = r[r.country == tgt].pivot_table(index=["unit", "n_train"], columns="arm", values=["auroc", "ece"])
        for s in srcs:
            for lo, hi, band in BANDS:
                b = w[(w.index.get_level_values("n_train") >= lo) & (w.index.get_level_values("n_train") < hi)]
                if b.empty:
                    continue
                out.append(
                    dict(
                        country=tgt,
                        source=s,
                        band=band,
                        units=len(b),
                        local_beats_zero_shot=float(
                            np.mean(b[("auroc", "local_only")] >= b[("auroc", f"{s}_zero_shot")])
                        ),
                        finetuned_beats_local=float(
                            np.mean(b[("auroc", f"{s}_finetuned")] > b[("auroc", "local_only")])
                        ),
                        platt_lowers_ece=float(np.mean(b[("ece", f"{s}_platt")] < b[("ece", f"{s}_zero_shot")])),
                    )
                )
    pd.DataFrame(out).to_csv(paths.result("small_units_summary.csv"), index=False)
    print(pd.DataFrame(out).round(2).to_string())


if __name__ == "__main__":
    main()
