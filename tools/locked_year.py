"""Scores the locked year 2024 with frozen models. Each result file can be written only once; 2024 is used for nothing else."""

import datetime
import json
import os

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot
from crashtransfer.models import em_prior, load_model, norm_w, primary, prior_shift, recalibrate
from crashtransfer.schema import COUNTRIES, FRACTIONS, PAIRS_FRANCE, PAIRS_UK_US, eval_settings, target_weight


def recorder(results):
    def record(cond, tgt, p, d):
        y = d.y_ksi.values
        for wn, w, mode in eval_settings(tgt, d):
            est, ci = binary_metrics(y, p, w), boot(binary_metrics, y, p, d.weight.values, mode)
            results.append(
                dict(
                    condition=cond,
                    target=tgt,
                    weighting=wn,
                    n=int(len(d)),
                    **{k: dict(value=float(v), ci=[ci[k][0], ci[k][1]]) for k, v in est.items()},
                )
            )

    return record


def original(df, prim, out):
    res = {
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "primary_model": prim,
        "results": [],
    }
    record = recorder(res["results"])
    for src, tgt in PAIRS_UK_US:
        lk, va = split_df(df, tgt, "locked"), split_df(df, tgt, "val")
        Xlk, Xva = design(lk)[0], design(va)[0]
        record(f"T0 {tgt} within-domain ({prim})", tgt, load_model(f"{tgt}_{prim}_ksi.joblib").p_ksi(Xlk), lk)
        for b in ("logreg", "majority"):
            record(f"T0 {tgt} within-domain ({b})", tgt, load_model(f"{tgt}_{b}_ksi.joblib").p_ksi(Xlk), lk)
        S = load_model(f"{src}_{prim}_ksi.joblib")
        p1 = S.p_ksi(Xlk)
        record(f"T1 {src}->{tgt} zero-shot", tgt, p1, lk)
        record(
            f"T2 {src}->{tgt} isotonic on target 2022",
            tgt,
            recalibrate("isotonic", S.p_ksi(Xva), va.y_ksi.values, target_weight(va))(p1),
            lk,
        )
    json.dump(res, open(out, "w"), indent=2)


def france(df, prim, out):
    res = {
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "primary_model": prim,
        "results": [],
    }
    record = recorder(res["results"])
    D = {s: {k: split_df(df, s, k) for k in ("train", "val", "locked")} for s in COUNTRIES}
    X = {s: {k: design(v)[0] for k, v in D[s].items()} for s in COUNTRIES}
    for b in (prim, "logreg", "majority"):
        record(
            f"T0 BAAC within-domain ({b})",
            "BAAC",
            load_model(f"BAAC_{b}_ksi.joblib").p_ksi(X["BAAC"]["locked"]),
            D["BAAC"]["locked"],
        )
    for src, tgt in PAIRS_FRANCE:
        d, lk, va, Xlk = f"{src}->{tgt}", D[tgt]["locked"], D[tgt]["val"], X[tgt]["locked"]
        S = load_model(f"{src}_{prim}_ksi.joblib")
        record(f"T0 {tgt} within-domain ({prim})", tgt, load_model(f"{tgt}_{prim}_ksi.joblib").p_ksi(Xlk), lk)
        p1, pv = S.p_ksi(Xlk), S.p_ksi(X[tgt]["val"])
        record(f"T1 {d} zero-shot", tgt, p1, lk)
        record(
            f"T2 {d} isotonic on target 2022",
            tgt,
            recalibrate("isotonic", pv, va.y_ksi.values, target_weight(va))(p1),
            lk,
        )
        s_tr = D[src]["train"]
        pi_s = float(np.average(s_tr.y_ksi, weights=s_tr.weight))
        record(
            f"T2b {d} EM prior correction", tgt, prior_shift(p1, pi_s, em_prior(pv, pi_s, va.weight.values)), lk
        )
        record(f"T3 {d} importance weights", tgt, load_model(f"T3_{src}-{tgt}.joblib").p_ksi(Xlk), lk)
        for frac in FRACTIONS:
            record(
                f"T4 {d} target-only {frac}", tgt, load_model(f"T4only_{src}-{tgt}_{frac}.joblib").p_ksi(Xlk), lk
            )
            record(f"T4 {d} mix {frac}", tgt, load_model(f"T4mix_{src}-{tgt}_{frac}.joblib").p_ksi(Xlk), lk)
    for h in COUNTRIES:
        srcs = [s for s in COUNTRIES if s != h]
        lk, va = D[h]["locked"], D[h]["val"]
        L = load_model(f"LOCO_{h}_{prim}_ksi.joblib")
        ptr = pd.concat(
            [D[s]["train"].assign(_w=norm_w(D[s]["train"].weight.values) / len(D[s]["train"])) for s in srcs],
            ignore_index=True,
        )
        pi_s = float(np.average(ptr.y_ksi, weights=ptr._w))
        p1, pv = L.p_ksi(X[h]["locked"]), L.p_ksi(X[h]["val"])
        name = "+".join(srcs)
        record(f"LOCO T1 {name}->{h}", h, p1, lk)
        record(
            f"LOCO T2 {name}->{h} isotonic",
            h,
            recalibrate("isotonic", pv, va.y_ksi.values, target_weight(va))(p1),
            lk,
        )
        record(
            f"LOCO T2b {name}->{h} EM prior", h, prior_shift(p1, pi_s, em_prior(pv, pi_s, va.weight.values)), lk
        )
    json.dump(res, open(out, "w"), indent=2)


df, prim = load(baac=True), primary()
for name, fn in (("locked_2024.json", original), ("locked_2024_baac.json", france)):
    out = paths.result(name)
    if os.path.exists(out):
        raise SystemExit(f"{name} exists: the locked year has already been scored once")
    fn(df, prim, out)
