"""Builds the harmonised tables from the raw files and writes the descriptive result files."""

import json
import os
import zipfile

import numpy as np
import pandas as pd

from crashtransfer import harmonise, paths
from crashtransfer.schema import FEATS, YEARS

COLS = ["source", "case_id", "year"] + FEATS + ["driver_sex", "weight", "y_3class", "y_ksi"]
RULES = {
    0: "PDO dropped",
    5: "injured, severity unknown dropped",
    6: "died prior to crash dropped",
    8: "no person involved dropped",
}


def collisions_parquet():
    out = paths.interim("stats19_collision_2016_latest_bulk.parquet")
    if os.path.exists(out):
        return
    src = os.path.join(paths.RAW, "stats19", "bulk", harmonise.BULK % "collision")
    parts = [
        ch[ch["collision_year"].between(*harmonise.STATS19_YEARS)]
        for ch in pd.read_csv(src, chunksize=500_000, low_memory=False)
    ]
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)


def extract_crss():
    for y in YEARS:
        d = os.path.join(paths.interim("crss_extracted"), str(y))
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(os.path.join(paths.RAW, "crss", f"CRSS{y}CSV.zip")) as z:
            names = [
                n
                for n in z.namelist()
                if os.path.basename(n).lower() in ("accident.csv", "vehicle.csv", "person.csv")
            ]
            assert len(names) == 3, (y, names)
            for n in names:
                with open(os.path.join(d, os.path.basename(n).lower()), "wb") as f:
                    f.write(z.read(n))


def split_name(y):
    return "train" if 2016 <= y <= 2021 else {2022: "validation", 2023: "test", 2024: "locked"}[y]


def describe(uk, us, excl, held_out):
    sh = pd.concat([uk[COLS + ["collision_injury_based"]], us[COLS + ["maxsev_im"]]], ignore_index=True)
    sh.to_parquet(paths.interim("shared_collisions.parquet"), index=False)
    sh.groupby("source").head(10).drop(columns=["case_id"]).to_csv(
        paths.result("harmonised_sample_head.csv"), index=False
    )
    sh.groupby(["source", "year"]).agg(
        n_rows=("y_ksi", "size"), weight_sum=("weight", "sum")
    ).reset_index().to_csv(paths.result("yearly_counts.csv"), index=False)
    rows = []
    for (src, y), g in sh.groupby(["source", "year"]):
        for kind, w in [("unweighted", None)] + ([("weighted", g["weight"])] if src == "CRSS" else []):
            ww = np.ones(len(g)) if w is None else w.values
            for cls, name in [(1, "Fatal"), (2, "Serious"), (3, "Slight")]:
                rows.append(
                    dict(
                        source=src,
                        year=y,
                        weighting=kind,
                        severity=name,
                        share=float((ww * (g.y_3class == cls)).sum() / ww.sum()),
                    )
                )
            rows.append(
                dict(
                    source=src,
                    year=y,
                    weighting=kind,
                    severity="KSI",
                    share=float((ww * g.y_ksi).sum() / ww.sum()),
                )
            )
    pd.DataFrame(rows).to_csv(paths.result("severity_mix.csv"), index=False)
    excl["pct_of_all_crashes"] = excl["n"] / excl["n_all"] * 100
    excl["rule"] = excl["code"].map(RULES)
    excl[["year", "code", "rule", "n", "n_all", "pct_of_all_crashes"]].sort_values(["year", "code"]).to_csv(
        paths.result("crss_excluded_codes.csv"), index=False
    )
    sh.groupby(["source", "year"])[FEATS + ["driver_sex"]].apply(lambda d: d.isna().mean()).reset_index().to_csv(
        paths.result("missingness.csv"), index=False
    )
    sh["split"] = sh["year"].map(split_name)
    sc = (
        sh.groupby(["source", "split"])
        .agg(n_rows=("y_ksi", "size"), ksi_rate_unweighted=("y_ksi", "mean"), weight_sum=("weight", "sum"))
        .reset_index()
    )
    sc["ksi_rate_weighted"] = [
        float(np.average(g.y_ksi, weights=g.weight)) for _, g in sh.groupby(["source", "split"])
    ]
    sc.to_csv(paths.result("split_counts.csv"), index=False)
    json.dump(
        {
            "role": "held out; outside the protocol window; not fitted, not evaluated",
            "n_collisions": int(len(held_out)),
            "severity_counts": {str(k): int(v) for k, v in held_out.y_3class.value_counts().sort_index().items()},
        },
        open(paths.result("provisional_or_heldout_2025.json"), "w"),
        indent=2,
    )
    prof = {"window": "2016-2024", "stats19": {}, "crss": {}}
    for src, key in [("STATS19", "stats19"), ("CRSS", "crss")]:
        g = sh[sh.source == src]
        prof[key] = {
            "n_rows": int(len(g)),
            "weight_sum_total": float(g.weight.sum()),
            "ksi_rate_unweighted": float(g.y_ksi.mean()),
            "ksi_rate_weighted": float(np.average(g.y_ksi, weights=g.weight)),
            "n_3class": {str(k): int(v) for k, v in g.y_3class.value_counts().sort_index().items()},
        }
    s19 = sh[sh.source == "STATS19"]
    prof["stats19"]["injury_based_share_by_year"] = {
        int(k): round(float(v), 4) for k, v in s19.groupby("year").collision_injury_based.mean().items()
    }
    prof["stats19"]["ksi_rate_by_year"] = {
        int(k): round(float(v), 4) for k, v in s19.groupby("year").y_ksi.mean().items()
    }
    json.dump(prof, open(paths.result("dataset_profile.json"), "w"), indent=2)


def main():
    collisions_parquet()
    extract_crss()
    f = harmonise.stats19()
    f[f.year == 2025].to_parquet(paths.interim("stats19_heldout_2025.parquet"), index=False)
    uk = f[f.year <= 2024]
    uk.to_parquet(paths.interim("stats19_shared.parquet"), index=False)
    parts = [harmonise.crss_year(y) for y in YEARS]
    us = pd.concat([p[0] for p in parts], ignore_index=True)
    us.to_parquet(paths.interim("crss_shared.parquet"), index=False)
    describe(uk, us, pd.concat([p[1] for p in parts]), f[f.year == 2025])
    fr = harmonise.baac()
    fr.to_parquet(paths.interim("baac_shared.parquet"), index=False)
    fr.groupby("year").agg(
        n=("y_ksi", "size"),
        ksi_share=("y_ksi", "mean"),
        fatal_share=("y_3class", lambda s: (s == 1).mean()),
        speed_limit_missing=("speed_limit_mph", lambda s: s.isna().mean()),
    ).reset_index().to_csv(paths.result("baac_label_audit.csv"), index=False)
    pd.concat([harmonise.crss_extra_year(y) for y in YEARS], ignore_index=True).to_parquet(
        paths.interim("crss_posthoc.parquet"), index=False
    )
    harmonise.stats19_extra().to_parquet(paths.interim("stats19_posthoc.parquet"), index=False)
    harmonise.vic().to_parquet(paths.interim("vic_shared.parquet"), index=False)
    harmonise.region_keys().to_parquet(paths.interim("region_keys.parquet"), index=False)


if __name__ == "__main__":
    main()
