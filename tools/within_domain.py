"""T0: five learners per country, binary KSI and three classes, test 2023. Platt and isotonic recalibration are fitted on 2022 only."""

import json

import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics, boot, reliability, three_class_metrics
from crashtransfer.models import Model, recalibrate, save_model
from crashtransfer.schema import eval_settings

LEARNERS = ["majority", "logreg", "rf", "xgboost", "lightgbm"]


def country(df, src):
    rows, cal, rel, val = [], [], [], []
    tr, va, te = (split_df(df, src, s) for s in ("train", "val", "test"))
    Xtr, Xva, Xte = design(tr)[0], design(va)[0], design(te)[0]
    for name in LEARNERS:
        for task in ("ksi", "3class"):
            m = Model(name, task).fit(Xtr, tr, Xva, va, w_tr=tr.weight.values, w_va=va.weight.values)
            save_model(m, f"{src}_{name}_{task}.joblib")
            if task == "ksi":
                pv, pt = m.p_ksi(Xva), m.p_ksi(Xte)
                y = te.y_ksi.values
                for wn, w, _ in eval_settings(src, va):
                    vm = binary_metrics(va.y_ksi.values, pv, w)
                    val.append(
                        dict(
                            source=src,
                            model=name,
                            weighting=wn,
                            auprc_val2022=vm["auprc"],
                            auroc_val2022=vm["auroc"],
                        )
                    )
                for wn, w, mode in eval_settings(src, te):
                    est, ci = binary_metrics(y, pt, w), boot(binary_metrics, y, pt, te.weight.values, mode)
                    rows += [
                        dict(
                            source=src,
                            weighting=wn,
                            model=name,
                            task="ksi",
                            split="test2023",
                            metric=k,
                            value=v,
                            ci_lo=ci[k][0],
                            ci_hi=ci[k][1],
                        )
                        for k, v in est.items()
                    ]
                    wv = va.weight.values if wn == "weighted" else None
                    for kind in ("none", "platt", "isotonic"):
                        pc = recalibrate(kind, pv, va.y_ksi.values, wv)(pt)
                        cm, cci = binary_metrics(y, pc, w), boot(binary_metrics, y, pc, te.weight.values, mode)
                        cal += [
                            dict(
                                source=src,
                                weighting=wn,
                                model=name,
                                calibration=kind,
                                metric=k,
                                value=cm[k],
                                ci_lo=cci[k][0],
                                ci_hi=cci[k][1],
                            )
                            for k in ("ece", "brier", "logloss", "auroc", "auprc")
                        ]
                        r = reliability(y, pc, w)
                        for col, v in (("calibration", kind), ("model", name), ("weighting", wn), ("source", src)):
                            r.insert(0, col, v)
                        rel.append(r)
            else:
                P, y3 = m.predict_proba(Xte), te.y_3class.values
                for wn, w, mode in eval_settings(src, te):
                    est, ci = (
                        three_class_metrics(y3, P, w),
                        boot(three_class_metrics, y3, P, te.weight.values, mode),
                    )
                    rows += [
                        dict(
                            source=src,
                            weighting=wn,
                            model=name,
                            task="3class",
                            split="test2023",
                            metric=k,
                            value=v,
                            ci_lo=ci[k][0],
                            ci_hi=ci[k][1],
                        )
                        for k, v in est.items()
                    ]
            print(src, name, task, flush=True)
    return rows, cal, rel, val


def main():
    df = load(baac=True)
    parts = [country(df, s) for s in ("STATS19", "CRSS")]
    pd.DataFrame(parts[0][0] + parts[1][0]).to_csv(paths.result("within_domain_metrics.csv"), index=False)
    pd.DataFrame(parts[0][1] + parts[1][1]).to_csv(paths.result("calibration_metrics.csv"), index=False)
    pd.concat(parts[0][2] + parts[1][2]).to_csv(paths.result("reliability_bins.csv"), index=False)
    vr = pd.DataFrame(parts[0][3] + parts[1][3])
    vr.to_csv(paths.result("validation2022_metrics.csv"), index=False)
    # Primary model: mean 2022 AUPRC over the two original sources (CRSS weighted), boosting learners only.
    keep = vr[
        (vr.model.isin(["xgboost", "lightgbm"]))
        & (((vr.source == "CRSS") & (vr.weighting == "weighted")) | (vr.source == "STATS19"))
    ]
    score = keep.groupby("model").auprc_val2022.mean()
    json.dump(
        {
            "selection": "mean validation-2022 AUPRC (CRSS weighted) over STATS19 and CRSS; XGBoost vs LightGBM",
            "scores": score.to_dict(),
            "primary": score.idxmax(),
        },
        open(paths.result("primary_model.json"), "w"),
        indent=2,
    )
    rows, cal, _, val = country(df, "BAAC")
    pd.DataFrame(rows).to_csv(paths.result("within_domain_baac.csv"), index=False)
    pd.DataFrame(cal).drop(columns="weighting").to_csv(paths.result("calibration_baac.csv"), index=False)
    pd.DataFrame(val)[["model", "auprc_val2022", "auroc_val2022"]].to_csv(
        paths.result("validation2022_baac.csv"), index=False
    )


if __name__ == "__main__":
    main()
