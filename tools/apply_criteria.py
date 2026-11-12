"""Applies the pre-specified criteria of the protocol amendment to the result files. RQ6 to RQ10 are the protocol's labels for these criteria."""

import json

import pandas as pd

from crashtransfer import paths
from crashtransfer.schema import COUNTRIES, primary_weighting


def rd(f):
    return pd.read_csv(paths.result(f))


tm, lo = rd("transfer_matrix_3way.csv"), rd("loco.csv")
kw = json.load(open(paths.result("shap_kendall_w.json")))


def pick(direction, tier, metric):
    target = tm[tm.direction == direction].target.iloc[0]
    return tm[
        (tm.direction == direction)
        & (tm.tier == tier)
        & (tm.metric == metric)
        & (tm.weighting == primary_weighting(target))
    ].iloc[0]


v, rq6 = {}, {}
for h in COUNTRIES:
    p = lo[
        (lo.held_out == h) & (lo.tier == "T1") & (lo.metric == "auroc") & (lo.weighting == primary_weighting(h))
    ].iloc[0]
    best = max((pick(f"{s}->{h}", "T1", "auroc") for s in COUNTRIES if s != h), key=lambda r: r.value)
    rq6[h] = dict(
        pooled=[p.value, p.ci_lo, p.ci_hi],
        best_single=best.direction,
        best=[best.value, best.ci_lo, best.ci_hi],
        overlap=bool(p.ci_lo <= best.ci_hi and best.ci_lo <= p.ci_hi),
        pooled_better=bool(p.value > best.value),
    )
v["RQ6"] = dict(
    per_country=rq6,
    n_overlap=sum(r["overlap"] for r in rq6.values()),
    rule="rejected if on >= 2 of 3 held-out countries the pooled CI overlaps the best single source (or pooled is lower)",
    rejected=sum(not r["pooled_better"] or r["overlap"] for r in rq6.values()) >= 2,
)
rq7 = {
    d: dict(ece_T1=pick(d, "T1", "ece").value, ece_T2b=pick(d, "T2b", "ece").value) for d in tm.direction.unique()
}
for r in rq7.values():
    r["lower"] = r["ece_T2b"] < r["ece_T1"]
v["RQ7"] = dict(
    per_pair=rq7, n_lower=sum(r["lower"] for r in rq7.values()), rejected=sum(r["lower"] for r in rq7.values()) < 4
)
v["RQ8"] = dict(kendall_w=kw["kendall_w"], ci=kw["bootstrap50_ci"], rejected=bool(kw["bootstrap50_ci"][0] <= 0))
json.dump(v, open(paths.result("rq_verdicts_baac.json"), "w"), indent=2, default=lambda o: o.item())

s = pd.concat(
    [rd(f"posthoc_learning_curve_summary{t}.csv").assign(registered=t == "_baac") for t in ("", "_baac")]
)
p = pd.concat([rd(f"posthoc_prior_shift{t}.csv").assign(registered=t == "_baac") for t in ("", "_baac")])
rq9 = {}
for d, g in s.groupby("direction"):
    nstar = g[g.within_001_of_T0].groupby("arm").n.min().to_dict()
    only = nstar.get("target_only")
    arms = {a: n for a, n in nstar.items() if a in ("platt", "finetune", "mix")}
    rq9[d] = dict(
        pre_registered=bool(g.registered.iloc[0]),
        nstar=nstar,
        transfer_beats_target_only=any(only is None or n < only for n in arms.values()),
    )
rq10 = {}
for d, g in p.groupby("direction"):
    e = g.set_index("arm").ece
    rq10[d] = dict(
        pre_registered=bool(g.registered.iloc[0]),
        ece_T1=e["T1"],
        ece_official=e["prior_official"],
        ece_minus20=e["prior_minus20"],
        ece_plus20=e["prior_plus20"],
        ece_em=e["prior_em"],
        official_lower=bool(e["prior_official"] < e["T1"]),
    )
n9, n10 = (
    sum(r["transfer_beats_target_only"] for r in rq9.values()),
    sum(r["official_lower"] for r in rq10.values()),
)
json.dump(
    {
        "RQ9": dict(
            per_pair=rq9,
            n_pairs_transfer_reaches_T0_minus_001_before_target_only=n9,
            rule="rejected if on 4 of 6 pairs no transfer arm reaches T0-0.01 at a smaller N than target-only",
            rejected=6 - n9 >= 4,
        ),
        "RQ10": dict(
            per_pair=rq10,
            n_pairs_official_prior_lowers_ece=n10,
            rule="rejected if ECE after correction with the official KSI share is not lower than T1 on 4 of 6 pairs",
            rejected=6 - n10 >= 4,
        ),
    },
    open(paths.result("rq9_rq10_verdicts.json"), "w"),
    indent=2,
)
