"""Learning curve (N labelled target collisions, five draws) and correction of the KSI prior without target microdata. Test 2023 only."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics
from crashtransfer.models import Model, em_prior, load_model, norm_w, primary, prior_shift, recalibrate
from crashtransfer.schema import PAIRS_FRANCE, PAIRS_UK_US, target_weight

NS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
SEEDS = [42, 1, 2, 3, 4]


def finetune(S, X, y, w, seed):
    """Continues boosting from the frozen source model on the N rows, with a 15% holdout for early stopping."""
    itr, ies = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
    prm = S.m.get_params()
    prm.update(n_estimators=300, early_stopping_rounds=30, random_state=seed)
    m = xgb.XGBClassifier(**prm)
    m.fit(
        X.iloc[itr],
        y[itr],
        sample_weight=w[itr],
        eval_set=[(X.iloc[ies], y[ies])],
        sample_weight_eval_set=[w[ies]],
        verbose=False,
        xgb_model=S.m.get_booster(),
    )
    return m


def run(df, prim, pairs, tag):
    rows, prow = [], []
    for src, tgt in pairs:
        d = f"{src}->{tgt}"
        s_tr = split_df(df, src, "train")
        t_tr, t_va, t_te = (split_df(df, tgt, s) for s in ("train", "val", "test"))
        Xs_tr, Xt_te, Xt_va = design(s_tr)[0], design(t_te)[0], design(t_va)[0]
        wt, y = target_weight(t_te), t_te.y_ksi.values
        T, S = load_model(f"{tgt}_{prim}_ksi.joblib"), load_model(f"{src}_{prim}_ksi.joblib")
        pS = S.p_ksi(Xt_te)
        ref = {"T0": binary_metrics(y, T.p_ksi(Xt_te), wt), "T1": binary_metrics(y, pS, wt)}
        pi_s, pi_t = (
            float(np.average(s_tr.y_ksi, weights=s_tr.weight)),
            float(np.average(t_tr.y_ksi, weights=t_tr.weight)),
        )
        pi_em = em_prior(S.p_ksi(Xt_va), pi_s, t_va.weight.values)
        for arm, pt in (
            ("T0", None),
            ("T1", None),
            ("prior_official", pi_t),
            ("prior_minus20", pi_t * 0.8),
            ("prior_plus20", pi_t * 1.2),
            ("prior_em", pi_em),
        ):
            m = ref[arm] if pt is None else binary_metrics(y, prior_shift(pS, pi_s, pt), wt)
            prow.append(
                dict(
                    direction=d,
                    arm=arm,
                    pi_source=pi_s,
                    pi_target_used=pt,
                    pi_target_true_test=m["prevalence"],
                    **m,
                )
            )
        for n in [n for n in NS if n < len(t_tr)]:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                samp = t_tr.iloc[rng.choice(len(t_tr), n, replace=False)].reset_index(drop=True)
                Xn = design(samp)[0]
                yn, wn = samp.y_ksi.values, norm_w(samp.weight.values)
                res = {
                    "target_only": Model(prim, "ksi", seed=seed)
                    .fit(Xn, samp, w_tr=samp.weight.values)
                    .p_ksi(Xt_te),
                    "platt": recalibrate("platt", S.p_ksi(Xn), yn, wn)(pS),
                    "finetune": finetune(S, Xn, yn, wn, seed).predict_proba(Xt_te)[:, 1],
                    "prior_N": prior_shift(pS, pi_s, float(np.clip(np.average(yn, weights=wn), 1e-3, 1 - 1e-3))),
                }
                if n >= 1000:  # below this the mix is the source model in practice
                    i_fit, i_es = train_test_split(np.arange(n), test_size=0.15, stratify=yn, random_state=seed)
                    fit, es = samp.iloc[i_fit].reset_index(drop=True), samp.iloc[i_es].reset_index(drop=True)
                    mix = pd.concat(
                        [s_tr.assign(_w=norm_w(s_tr.weight.values)), fit.assign(_w=norm_w(fit.weight.values))],
                        ignore_index=True,
                    )
                    res["mix"] = (
                        Model(prim, "ksi", seed=seed)
                        .fit(
                            pd.concat([Xs_tr, Xn.iloc[i_fit]], ignore_index=True),
                            mix,
                            Xn.iloc[i_es].reset_index(drop=True),
                            es,
                            w_tr=mix["_w"].values,
                            w_va=es.weight.values,
                        )
                        .p_ksi(Xt_te)
                    )
                for arm, p in res.items():
                    rows.append(
                        dict(
                            direction=d,
                            n=n,
                            seed=seed,
                            arm=arm,
                            **binary_metrics(y, p, wt),
                            auroc_T0=ref["T0"]["auroc"],
                            auroc_T1=ref["T1"]["auroc"],
                            auprc_T0=ref["T0"]["auprc"],
                            auprc_T1=ref["T1"]["auprc"],
                            ece_T0=ref["T0"]["ece"],
                            ece_T1=ref["T1"]["ece"],
                        )
                    )
    r = pd.DataFrame(rows)
    r.to_csv(paths.result(f"posthoc_learning_curve{tag}.csv"), index=False)
    pd.DataFrame(prow).to_csv(paths.result(f"posthoc_prior_shift{tag}.csv"), index=False)
    s = (
        r.groupby(["direction", "arm", "n"])
        .agg(
            auroc=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc=("auprc", "mean"),
            ece=("ece", "mean"),
            auroc_T0=("auroc_T0", "first"),
            auroc_T1=("auroc_T1", "first"),
        )
        .reset_index()
    )
    s["within_001_of_T0"] = s.auroc >= s.auroc_T0 - 0.01
    s.to_csv(paths.result(f"posthoc_learning_curve_summary{tag}.csv"), index=False)


def main():
    df, prim = load(baac=True), primary()
    run(df, prim, PAIRS_UK_US, "")
    run(df, prim, PAIRS_FRANCE, "_baac")


if __name__ == "__main__":
    main()
