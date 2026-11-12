"""Pooled sources (leave one country out) and the sensitivity refit of STATS19 and CRSS on the French training years."""

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot
from crashtransfer.models import Model, em_prior, norm_w, primary, prior_shift, recalibrate, save_model
from crashtransfer.schema import COUNTRIES, eval_settings, primary_weighting, target_weight

df, prim = load(baac=True), primary()
S = {s: {k: split_df(df, s, k) for k in ("train", "val", "test")} for s in COUNTRIES}
X = {s: {k: design(v)[0] for k, v in S[s].items()} for s in COUNTRIES}


def pool(sources, split):
    parts = [
        S[s][split].assign(_w=norm_w(S[s][split].weight.values) / len(S[s][split])) for s in sources
    ]  # equal total weight per source
    return pd.concat(parts, ignore_index=True), pd.concat([X[s][split] for s in sources], ignore_index=True)


rows = []
for h in COUNTRIES:
    srcs = [s for s in COUNTRIES if s != h]
    ptr, Xptr = pool(srcs, "train")
    pva, Xpva = pool(srcs, "val")
    L = Model(prim, "ksi").fit(Xptr, ptr, Xpva, pva, w_tr=ptr._w.values, w_va=pva._w.values)
    save_model(L, f"LOCO_{h}_{prim}_ksi.joblib")
    va, te = S[h]["val"], S[h]["test"]
    p1, pv = L.p_ksi(X[h]["test"]), L.p_ksi(X[h]["val"])
    pi_s = float(np.average(ptr.y_ksi, weights=ptr._w))
    pi_em = em_prior(pv, pi_s, va.weight.values)
    for tier, var, p in (
        ("T1", "pooled zero-shot", p1),
        (
            "T2",
            "pooled + isotonic on target 2022",
            recalibrate("isotonic", pv, va.y_ksi.values, target_weight(va))(p1),
        ),
        ("T2b", "pooled + EM prior correction", prior_shift(p1, pi_s, pi_em)),
    ):
        y = te.y_ksi.values
        for wn, w, mode in eval_settings(h, te):
            est, ci = binary_metrics(y, p, w), boot(binary_metrics, y, p, te.weight.values, mode)
            rows += [
                dict(
                    held_out=h,
                    sources="+".join(srcs),
                    tier=tier,
                    variant=var,
                    weighting=wn,
                    metric=k,
                    value=v,
                    ci_lo=ci[k][0],
                    ci_hi=ci[k][1],
                )
                for k, v in est.items()
            ]
pd.DataFrame(rows).to_csv(paths.result("loco.csv"), index=False)

years = {"train": range(2019, 2022), "val": [2022], "test": [2023]}
out = []
for s in ("STATS19", "CRSS"):
    a, v = split_df(df, s, "train", years), split_df(df, s, "val", years)
    m = Model(prim, "ksi").fit(design(a)[0], a, design(v)[0], v, w_tr=a.weight.values, w_va=v.weight.values)
    for t in COUNTRIES:
        d = S[t]["test"]
        out.append(
            dict(
                source=s,
                train_years="2019-2021",
                target=t,
                weighting=primary_weighting(t),
                **binary_metrics(d.y_ksi.values, m.p_ksi(X[t]["test"]), target_weight(d)),
            )
        )
pd.DataFrame(out).to_csv(paths.result("sensitivity_train2019_2021.csv"), index=False)
