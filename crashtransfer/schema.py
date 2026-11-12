"""Shared features, temporal splits and country pairs, as fixed in the study protocol."""

import pandas as pd

SEED = 42
YEARS = list(range(2016, 2025))
FEATS = [
    "hour",
    "day_of_week",
    "month",
    "lighting_dark",
    "weather_coarse",
    "speed_limit_mph",
    "junction",
    "n_vehicles",
    "urban_rural",
    "vru_involved",
    "vehicle_type_mode",
    "driver_age_band",
]
NUMERIC = [
    "hour",
    "day_of_week",
    "month",
    "lighting_dark",
    "speed_limit_mph",
    "junction",
    "n_vehicles",
    "vru_involved",
]
CATS = {
    "weather_coarse": ["clear", "rain", "snow", "fog", "other"],
    "urban_rural": ["urban", "rural"],
    "vehicle_type_mode": ["car", "goods", "bus", "motorcycle", "other"],
}
FORBIDDEN = [
    "lat",
    "long",
    "easting",
    "northing",
    "police",
    "psu",
    "index",
    "domain",
    "casualt",
    "injur",
    "severity",
    "maxsev",
    "weight",
    "source",
    "case_id",
    "year",
]
CLOSE = ["hour", "day_of_week", "month", "driver_age_band"]
SHIFTED = [f for f in FEATS if f not in CLOSE]

AGE_BANDS = ["<16", "16-20", "21-25", "26-35", "36-45", "46-55", "56-65", "66-75", ">75"]
VT_PRIORITY = ["motorcycle", "goods", "bus", "car", "other"]  # tie-break for vehicle_type_mode

SPLITS = {"train": range(2016, 2022), "val": [2022], "test": [2023], "locked": [2024]}
# French training starts in 2019: the hospitalised category breaks in 2018 and the speed limit starts in 2019.
SPLITS_BY_SOURCE = {
    "BAAC": {
        "train": range(2019, 2022),
        "val": [2022],
        "test": [2023],
        "locked": [2024],
        "audit": range(2016, 2019),
    }
}

COUNTRIES = ["STATS19", "CRSS", "BAAC"]
PAIRS_UK_US = [("STATS19", "CRSS"), ("CRSS", "STATS19")]
PAIRS_FRANCE = [("BAAC", "STATS19"), ("BAAC", "CRSS"), ("STATS19", "BAAC"), ("CRSS", "BAAC")]
# amendment A2 (docs/protocol/PROTOCOL_VIC.md)
PAIRS_VIC = [
    ("VIC", "STATS19"),
    ("VIC", "CRSS"),
    ("VIC", "BAAC"),
    ("STATS19", "VIC"),
    ("CRSS", "VIC"),
    ("BAAC", "VIC"),
]
FRACTIONS = [0.01, 0.05, 0.10]


def age_band(age):
    a = pd.to_numeric(age, errors="coerce")
    bins = [-1, 15, 20, 25, 35, 45, 55, 65, 75, 200]
    return pd.cut(a, bins=bins, labels=AGE_BANDS).astype(object)


def mode_with_priority(df, key, col):
    """Most frequent value of col per key, ties broken by VT_PRIORITY."""
    d = df[[key, col]].dropna()
    cnt = d.groupby([key, col]).size().rename("n").reset_index()
    cnt["p"] = cnt[col].map({v: i for i, v in enumerate(VT_PRIORITY)})
    cnt = cnt.sort_values([key, "n", "p"], ascending=[True, False, True])
    return cnt.drop_duplicates(key).set_index(key)[col]


def primary_weighting(target):
    return "weighted" if target == "CRSS" else "unweighted"


def eval_settings(target, d):
    """(column label, metric weights, bootstrap mode) for each way a target is scored. CRSS is scored both ways."""
    if target == "CRSS":
        return [("weighted", d.weight.values, "weighted"), ("unweighted", None, "uniform")]
    return [("unweighted", None, "uniform")]


def target_weight(d):
    return d.weight.values if d.source.iloc[0] == "CRSS" else None
