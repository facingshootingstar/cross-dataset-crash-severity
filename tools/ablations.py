"""Retrains the primary model without one feature group per country, and audits the STATS19 injury-based reporting change."""

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot, delta_metrics
from crashtransfer.models import Model, load_model, primary
from crashtransfer.schema import eval_settings

DROPS = [("drop_vru", ["vru_involved"]), ("drop_speed_limit", ["speed_limit_mph"]), ("drop_hour", ["hour"])]
METRICS = ["auprc", "auroc", "ece"]


def ablations(df, prim, sources, out):
    rows = []
    for src in sources:
        tr, va, te = (split_df(df, src, s) for s in ("train", "val", "test"))
        y, pf = te.y_ksi.values, load_model(f"{src}_{prim}_ksi.joblib").p_ksi(design(te)[0])
        for name, drop in DROPS:
            m = Model(prim, "ksi").fit(
                design(tr, drop=drop)[0],
                tr,
                design(va, drop=drop)[0],
                va,
                w_tr=tr.weight.values,
                w_va=va.weight.values,
            )
            pa = m.p_ksi(design(te, drop=drop)[0])
            for wn, w, mode in eval_settings(src, te):
                ef, ea = binary_metrics(y, pf, w), binary_metrics(y, pa, w)
                ci = boot(delta_metrics, y, np.c_[pf, pa], te.weight.values, mode)
                rows += [
                    dict(
                        source=src,
                        weighting=wn,
                        ablation=name,
                        metric=k,
                        full=ef[k],
                        ablated=ea[k],
                        delta=ea[k] - ef[k],
                        delta_ci_lo=ci[f"d_{k}"][0],
                        delta_ci_hi=ci[f"d_{k}"][1],
                    )
                    for k in METRICS
                ]
    pd.DataFrame(rows).to_csv(paths.result(out), index=False)


def reporting_change(df, prim):
    tr, va, te = (split_df(df, "STATS19", s) for s in ("train", "val", "test"))
    y, pf = te.y_ksi.values, load_model(f"STATS19_{prim}_ksi.joblib").p_ksi(design(te)[0])
    r = []
    for lab, mask in (
        ("all", np.ones(len(te), bool)),
        ("injury_based=1", te.collision_injury_based.values == 1),
        ("injury_based=0", te.collision_injury_based.values == 0),
    ):
        est, ci = binary_metrics(y[mask], pf[mask]), boot(binary_metrics, y[mask], pf[mask], None, "uniform")
        r += [
            dict(
                analysis="base model scored on test subset",
                subset=lab,
                n=int(mask.sum()),
                metric=k,
                value=v,
                ci_lo=ci[k][0],
                ci_hi=ci[k][1],
            )
            for k, v in est.items()
        ]
    extra = ["collision_injury_based"]
    pi = (
        Model(prim, "ksi")
        .fit(design(tr, extra=extra)[0], tr, design(va, extra=extra)[0], va)
        .p_ksi(design(te, extra=extra)[0])
    )
    est, ci, base = (
        binary_metrics(y, pi),
        boot(delta_metrics, y, np.c_[pf, pi], None, "uniform"),
        binary_metrics(y, pf),
    )
    r += [
        dict(
            analysis="base vs model with injury_based indicator (audit feature, STATS19 only)",
            subset="all",
            n=len(te),
            metric=k,
            value=est[k],
            ci_lo=ci[f"d_{k}"][0],
            ci_hi=ci[f"d_{k}"][1],
            base_value=base[k],
            delta=est[k] - base[k],
        )
        for k in METRICS
    ]
    keep = [d[d.collision_injury_based.values == 1].reset_index(drop=True) for d in (tr, va, te)]
    p1 = Model(prim, "ksi").fit(design(keep[0])[0], keep[0], design(keep[1])[0], keep[1]).p_ksi(design(keep[2])[0])
    y1, sub = keep[2].y_ksi.values, pf[te.collision_injury_based.values == 1]
    est, ci, base = (
        binary_metrics(y1, p1),
        boot(delta_metrics, y1, np.c_[sub, p1], None, "uniform"),
        binary_metrics(y1, sub),
    )
    r += [
        dict(
            analysis="train+test restricted to injury_based=1 vs base model on same rows",
            subset="injury_based=1",
            n=len(sub),
            metric=k,
            value=est[k],
            ci_lo=ci[f"d_{k}"][0],
            ci_hi=ci[f"d_{k}"][1],
            base_value=base[k],
            delta=est[k] - base[k],
        )
        for k in METRICS
    ]
    for name, d in (("train", tr), ("val", va), ("test", te)):
        r.append(
            dict(
                analysis="descriptive: injury_based share of train/val/test",
                subset=name,
                n=len(d),
                metric="share_injury_based",
                value=float(d.collision_injury_based.mean()),
            )
        )
    pd.DataFrame(r).to_csv(paths.result("ibrs_sensitivity.csv"), index=False)


df = load(baac=True)
prim = primary()
ablations(df, prim, ["STATS19", "CRSS"], "ablations.csv")
ablations(df, prim, ["BAAC"], "ablations_baac.csv")
reporting_change(df, prim)
