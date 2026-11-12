"""Repeats the T4 few-shot draws over five seeds for the STATS19 and CRSS pair."""

import numpy as np
import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics
from crashtransfer.models import load_model, primary
from crashtransfer.schema import FRACTIONS, PAIRS_UK_US, target_weight
from crashtransfer.transfer import few_shot_models

prim, df = primary(), load()
rows = []
for src, tgt in PAIRS_UK_US:
    s_tr, t_tr, t_te = split_df(df, src, "train"), split_df(df, tgt, "train"), split_df(df, tgt, "test")
    Xs_tr, Xt_te = design(s_tr)[0], design(t_te)[0]
    wt, y = target_weight(t_te), t_te.y_ksi.values
    t0 = binary_metrics(y, load_model(f"{tgt}_{prim}_ksi.joblib").p_ksi(Xt_te), wt)["auprc"]
    t1 = binary_metrics(y, load_model(f"{src}_{prim}_ksi.joblib").p_ksi(Xt_te), wt)["auprc"]
    for frac in FRACTIONS:
        for seed in (42, 1, 2, 3, 4):
            only, mix = few_shot_models(prim, s_tr, Xs_tr, t_tr, frac, seed)
            a, b = binary_metrics(y, only.p_ksi(Xt_te), wt), binary_metrics(y, mix.p_ksi(Xt_te), wt)
            rows.append(
                dict(
                    direction=f"{src}->{tgt}",
                    fraction=frac,
                    seed=seed,
                    auprc_T0=t0,
                    auprc_T1=t1,
                    auprc_mix=b["auprc"],
                    auprc_target_only=a["auprc"],
                    auroc_mix=b["auroc"],
                    auroc_target_only=a["auroc"],
                    gap_closed_by_mix=(b["auprc"] - t1) / (t0 - t1),
                    gap_closed_by_target_only=(a["auprc"] - t1) / (t0 - t1),
                )
            )
r = pd.DataFrame(rows)
r.to_csv(paths.result("posthoc_fewshot_seeds.csv"), index=False)
r.groupby(["direction", "fraction"]).agg(
    mix_mean=("gap_closed_by_mix", "mean"),
    mix_sd=("gap_closed_by_mix", "std"),
    mix_min=("gap_closed_by_mix", "min"),
    mix_max=("gap_closed_by_mix", "max"),
    only_mean=("gap_closed_by_target_only", "mean"),
    only_sd=("gap_closed_by_target_only", "std"),
    mix_beats_only=("auprc_mix", lambda v: int((v.values > r.loc[v.index, "auprc_target_only"].values).sum())),
    n_seeds=("seed", "count"),
).reset_index().to_csv(paths.result("posthoc_fewshot_summary.csv"), index=False)
v = r.groupby(["direction", "seed"]).gap_closed_by_mix.max().reset_index(name="max_gap_closed")
v["rq5_verdict_this_seed"] = np.where(v.max_gap_closed >= 0.5, "Not rejected", "Rejected")
v.to_csv(paths.result("posthoc_fewshot_rq5_by_seed.csv"), index=False)
