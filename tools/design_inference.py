"""Design-based intervals for CRSS and paired differences on test 2023, and the recall and false-positive audit by driver group."""

import json

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.models import load_model, primary, recalibrate
from crashtransfer.schema import AGE_BANDS, PAIRS_UK_US, target_weight
from crashtransfer.survey import add_design, compare, design_info, point_weight, replicates

prim = primary()
df = add_design(load())


def M(src, name):
    return load_model(f"{src}_{name}_ksi.joblib")


k0 = f"T0 within ({prim})"
out, meta = [], {}
for src, tgt in PAIRS_UK_US:
    te, va = split_df(df, tgt, "test"), split_df(df, tgt, "val")
    Xte, Xva = design(te)[0], design(va)[0]
    S = M(src, prim)
    p1 = S.p_ksi(Xte)
    f = recalibrate("isotonic", S.p_ksi(Xva), va.y_ksi.values, target_weight(va))
    preds = {
        k0: M(tgt, prim).p_ksi(Xte),
        "T0 within (logreg)": M(tgt, "logreg").p_ksi(Xte),
        "T1 zero-shot": p1,
        "T2 isotonic": f(p1),
    }
    r = compare(te, preds, [(k0, "T0 within (logreg)"), (k0, "T1 zero-shot"), ("T2 isotonic", "T1 zero-shot")])
    r.insert(0, "target", tgt)
    r.insert(1, "interval", "design-based (Rao-Wu, PSU within stratum)" if tgt == "CRSS" else "row bootstrap")
    out.append(r)
    if tgt == "CRSS":
        meta["CRSS_test2023"] = design_info(te)
res = pd.concat(out, ignore_index=True)
res.to_csv(paths.result("posthoc_inference.csv"), index=False)

wd = pd.read_csv(paths.result("within_domain_metrics.csv"))
old = wd[
    (wd.source == "CRSS")
    & (wd.weighting == "weighted")
    & (wd.task == "ksi")
    & (wd.split == "test2023")
    & (wd.model == prim)
]
new = res[(res.target == "CRSS") & (res.item == k0)]


def width(d, m):
    return d[d.metric == m].ci_hi.iloc[0] - d[d.metric == m].ci_lo.iloc[0]


meta["width_ratio_design_vs_locked_run"] = {
    m: float(width(new, m) / width(old, m)) for m in ("auroc", "auprc", "ece")
}
meta["rq1_posthoc"] = {}
for tgt in ("STATS19", "CRSS"):
    auroc = res[(res.target == tgt) & (res.metric == "auroc")].set_index("item")
    a, b, d = auroc.loc[k0], auroc.loc["T0 within (logreg)"], auroc.loc[k0 + " minus T0 within (logreg)"]
    meta["rq1_posthoc"][tgt] = dict(
        overlap=bool(a.ci_lo <= b.ci_hi),
        rule_verdict="Rejected" if a.ci_lo <= b.ci_hi else "Not rejected",
        paired_diff=float(d.value),
        paired_ci=[float(d.ci_lo), float(d.ci_hi)],
    )

fair = []
for src in ("STATS19", "CRSS"):
    va, te = split_df(df, src, "val"), split_df(df, src, "test")
    m = M(src, prim)
    pv, pt = m.p_ksi(design(va)[0]), m.p_ksi(design(te)[0])
    wv = point_weight(va)
    prev = np.average(va.y_ksi.values, weights=wv)
    o = np.argsort(-pv)  # threshold: the flagged weighted share of 2022 equals the 2022 KSI prevalence
    thr = float(pv[o][np.searchsorted(np.cumsum(wv[o]) / wv.sum(), prev)])
    flag, y = (pt >= thr).astype(float), te.y_ksi.values

    def stats(wr, groups):
        d = {}
        for g, mk in groups.items():
            ww = wr * mk
            pos, neg = ww * y, ww * (1 - y)
            d[g] = dict(
                recall=(pos * flag).sum() / pos.sum(),
                fpr=(neg * flag).sum() / neg.sum(),
                prevalence=pos.sum() / ww.sum(),
            )
        return d

    for attr, levels in (("driver_sex", ["M", "F"]), ("driver_age_band", AGE_BANDS)):
        groups = {g: (te[attr] == g).values.astype(float) for g in levels if (te[attr] == g).sum() >= 200}
        est, reps = stats(point_weight(te), groups), [stats(wr, groups) for wr in replicates(te)]
        n_all = int(sum(groups[g].sum() for g in groups))

        def row(g, n, k, val, v):
            return dict(
                source=src,
                attribute=attr,
                group=g,
                n=n,
                metric=k,
                value=val,
                ci_lo=np.quantile(v, 0.025),
                ci_hi=np.quantile(v, 0.975),
                threshold=thr,
            )

        for g in groups:
            for k in ("recall", "fpr", "prevalence"):
                fair.append(row(g, int(groups[g].sum()), k, est[g][k], [r[g][k] for r in reps]))
        if attr == "driver_sex":
            for k in ("recall", "fpr"):
                fair.append(
                    row("F minus M", n_all, k, est["F"][k] - est["M"][k], [r["F"][k] - r["M"][k] for r in reps])
                )
        else:

            def spread(r):
                return max(r[g]["recall"] for g in groups) - min(r[g]["recall"] for g in groups)

            fair.append(row("max minus min", n_all, "recall", spread(est), [spread(r) for r in reps]))
    fair.append(
        dict(
            source=src,
            attribute="driver_sex",
            group="missing",
            n=int(te.driver_sex.isna().sum()),
            metric="n_missing",
            value=float(te.driver_sex.isna().mean()),
            ci_lo=np.nan,
            ci_hi=np.nan,
            threshold=thr,
        )
    )
pd.DataFrame(fair).to_csv(paths.result("posthoc_fairness.csv"), index=False)
json.dump(meta, open(paths.result("posthoc_design.json"), "w"), indent=2)
