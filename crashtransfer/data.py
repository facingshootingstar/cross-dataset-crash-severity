"""Harmonised table: loading, temporal splits and the design matrix."""

import os

import pandas as pd

from . import paths
from .schema import AGE_BANDS, CATS, FORBIDDEN, NUMERIC, SPLITS, SPLITS_BY_SOURCE


def load(baac=False, vic=False):
    """Set CRASH_SMOKE=<rows per source-year> for a quick dry run on a sample."""
    df = pd.read_parquet(paths.interim("shared_collisions.parquet"))
    if baac:
        b = pd.read_parquet(paths.interim("baac_shared.parquet"))
        df = pd.concat([df, b.astype({"month": "int64", "day_of_week": "int64"})], ignore_index=True)
    if vic:
        v = pd.read_parquet(paths.interim("vic_shared.parquet"))
        df = pd.concat([df, v.astype({"month": "int64", "day_of_week": "int64"})], ignore_index=True)
    smoke = os.environ.get("CRASH_SMOKE")
    if smoke:
        df = df.sample(frac=1, random_state=0).groupby(["source", "year"]).head(int(smoke)).reset_index(drop=True)
    return df


def split_df(df, source, split, splits=None):
    years = (splits or SPLITS_BY_SOURCE.get(source, SPLITS))[split]
    return df[(df.source == source) & df.year.isin(list(years))].reset_index(drop=True)


def design(df, drop=(), extra=()):
    """NaN-native design matrix and a map from column to source feature. `extra` admits one named audit column."""
    cols, cmap = {}, {}
    for f in NUMERIC:
        cols[f] = df[f].astype(float).values
        cmap[f] = f
    cols["driver_age_band"] = (
        df["driver_age_band"].map({b: i for i, b in enumerate(AGE_BANDS)}).astype(float).values
    )
    cmap["driver_age_band"] = "driver_age_band"
    for f, levels in CATS.items():
        for lev in levels:
            cols[f"{f}={lev}"] = (df[f] == lev).astype(float).values
            cmap[f"{f}={lev}"] = f
        cols[f"{f}=missing"] = df[f].isna().astype(float).values
        cmap[f"{f}=missing"] = f
    for e in extra:
        cols[e] = df[e].astype(float).values
        cmap[e] = e
    X = pd.DataFrame(cols)
    X = X[[c for c in X.columns if cmap[c] not in drop]]
    checked = [c for c in X.columns if c not in extra]
    assert not any(any(b in c.lower() for b in FORBIDDEN) for c in checked), "forbidden column in X"
    return X, {c: cmap[c] for c in X.columns}
