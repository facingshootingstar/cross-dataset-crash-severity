"""Transfer ladder on test 2023: T0 target model, T1 zero-shot, T2 isotonic on target 2022, T2b EM prior, T3 importance weights, T4 few-shot."""

import json

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot
from crashtransfer.models import Model, em_prior, load_model, norm_w, primary, prior_shift, recalibrate, save_model
from crashtransfer.schema import (
    COUNTRIES,
    FRACTIONS,
    PAIRS_FRANCE,
    PAIRS_UK_US,
    SEED,
    eval_settings,
    primary_weighting,
    target_weight,
)
from crashtransfer.transfer import few_shot_models, importance_weights

SPLIT_NAMES = ("train", "val", "test")


class Ladder:
    def __init__(self, df, prim):
        self.prim, self.rows, self.shift, self.em = prim, [], {}, {}
        self.S = {s: {k: split_df(df, s, k) for k in SPLIT_NAMES} for s in COUNTRIES}
        self.X = {s: {k: design(v)[0] for k, v in self.S[s].items()} for s in COUNTRIES}

    def score(self, direction, tier, variant, frac, tgt, p, d):
        y = d.y_ksi.values
        for wn, w, mode in eval_settings(tgt, d):
            est, ci = binary_metrics(y, p, w), boot(binary_metrics, y, p, d.weight.values, mode)
            self.rows += [
                dict(
                    direction=direction,
                    target=tgt,
                    tier=tier,
                    variant=variant,
                    fraction=frac,
                    weighting=wn,
                    metric=k,
                    value=v,
                    ci_lo=ci[k][0],
                    ci_hi=ci[k][1],
                )
                for k, v in est.items()
            ]

    def run(self, src, tgt, with_t2b, with_t3_t4, keep_models):
        d = f"{src}->{tgt}"
        S, T, Xt = (
            load_model(f"{src}_{self.prim}_ksi.joblib"),
            load_model(f"{tgt}_{self.prim}_ksi.joblib"),
            self.X[tgt],
        )
        s_tr, s_va = self.S[src]["train"], self.S[src]["val"]
        t_tr, t_va, t_te = (self.S[tgt][k] for k in SPLIT_NAMES)
        self.score(d, "T0", "target within-domain", 0, tgt, T.p_ksi(Xt["test"]), t_te)
        p1, pva = S.p_ksi(Xt["test"]), S.p_ksi(Xt["val"])
        self.score(d, "T1", "zero-shot", 0, tgt, p1, t_te)
        self.score(
            d,
            "T2",
            "source + isotonic on target 2022",
            0,
            tgt,
            recalibrate("isotonic", pva, t_va.y_ksi.values, target_weight(t_va))(p1),
            t_te,
        )
        if with_t2b:
            pi_s = float(np.average(s_tr.y_ksi, weights=s_tr.weight))
            pi_em = em_prior(pva, pi_s, t_va.weight.values)
            self.em[d] = dict(
                pi_source=pi_s,
                pi_em=pi_em,
                pi_target_test=float(np.average(t_te.y_ksi, weights=t_te.weight)),
                degenerate=bool(pi_em <= 1.001e-3 or pi_em >= 1 - 1.001e-3),
            )
            label = "EM prior correction, target 2022 features only" + (
                " (post-hoc for this pair)" if (src, tgt) in PAIRS_UK_US else ""
            )
            self.score(d, "T2b", label, 0, tgt, prior_shift(p1, pi_s, pi_em), t_te)
        if not with_t3_t4:
            return
        iw, self.shift[d] = importance_weights(s_tr, t_tr, self.X[src]["train"])
        T3 = Model(self.prim, "ksi").fit(
            self.X[src]["train"],
            s_tr,
            self.X[src]["val"],
            s_va,
            w_tr=norm_w(s_tr.weight.values) * iw,
            w_va=s_va.weight.values,
        )
        if keep_models:
            save_model(T3, f"T3_{src}-{tgt}.joblib")
        self.score(d, "T3", "domain-classifier importance weights", 0, tgt, T3.p_ksi(Xt["test"]), t_te)
        for frac in FRACTIONS:
            only, mix = few_shot_models(self.prim, s_tr, self.X[src]["train"], t_tr, frac, SEED)
            if keep_models:
                save_model(only, f"T4only_{src}-{tgt}_{frac}.joblib")
                save_model(mix, f"T4mix_{src}-{tgt}_{frac}.joblib")
            self.score(d, "T4", "target-only retrain", frac, tgt, only.p_ksi(Xt["test"]), t_te)
            self.score(d, "T4", "source + few-shot mix", frac, tgt, mix.p_ksi(Xt["test"]), t_te)


def gap_closed(tm):
    g = []
    for d in tm.direction.unique():
        t = tm[(tm.direction == d) & (tm.metric == "auprc")]
        t = t[t.weighting == primary_weighting(t.target.iloc[0])]
        t0, t1 = float(t[t.tier == "T0"].value.iloc[0]), float(t[t.tier == "T1"].value.iloc[0])
        for fr in FRACTIONS:
            mix = float(
                t[(t.tier == "T4") & (t.variant == "source + few-shot mix") & (t.fraction == fr)].value.iloc[0]
            )
            only = float(
                t[(t.tier == "T4") & (t.variant == "target-only retrain") & (t.fraction == fr)].value.iloc[0]
            )
            gap = t0 - t1
            g.append(
                dict(
                    direction=d,
                    fraction=fr,
                    auprc_T0=t0,
                    auprc_T1=t1,
                    auprc_mix=mix,
                    auprc_target_only_fewshot=only,
                    gap_T0_minus_T1=gap,
                    gap_closed_by_mix=(mix - t1) / gap if gap > 1e-9 else np.nan,
                )
            )
    pd.DataFrame(g).to_csv(paths.result("fewshot_gap_closed.csv"), index=False)


def main():
    df, prim = load(baac=True), primary()
    a = Ladder(df, prim)
    for src, tgt in PAIRS_UK_US:
        a.run(src, tgt, False, True, False)
    tm = pd.DataFrame(a.rows)
    tm.to_csv(paths.result("transfer_matrix.csv"), index=False)
    json.dump(a.shift, open(paths.result("domain_shift.json"), "w"), indent=2)
    gap_closed(tm)
    b = Ladder(df, prim)
    for src, tgt in PAIRS_UK_US:
        b.run(src, tgt, True, False, False)
    for src, tgt in PAIRS_FRANCE:
        b.run(src, tgt, True, True, True)
    pd.DataFrame(b.rows).to_csv(paths.result("transfer_matrix_3way.csv"), index=False)
    json.dump(
        {"proxy_a_distance": b.shift, "em_prior": b.em},
        open(paths.result("domain_shift_3way.json"), "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
