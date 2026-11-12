"""Effect of each crash factor on the KSI log-odds, by country and by which country's model reads the crash.

For every target country the test year is explained twice with TreeSHAP: by the target's own model and by each
foreign model. A factor's effect is the mean SHAP value of its feature on crashes with the factor minus the mean on
crashes in the reference group, so own and foreign effects are measured on the same crashes. Intervals come from
200 bootstrap resamples of the crashes (in proportion to WEIGHT for CRSS).
Output: factor_effects.csv
"""

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.models import Model, primary
from crashtransfer.schema import COUNTRIES, SEED

KMH = 1.609344
# (factor, feature, test on the row, reference group); speed limits are compared in km/h
FACTORS = [
    ("VRU involved", "vru_involved", lambda d: d.vru_involved == 1, lambda d: d.vru_involved == 0),
    ("Speed limit 51-80 km/h", "speed_limit_mph", lambda d: d.kmh.between(50.5, 80.5), lambda d: d.kmh <= 50.5),
    ("Speed limit over 80 km/h", "speed_limit_mph", lambda d: d.kmh > 80.5, lambda d: d.kmh <= 50.5),
    ("Single vehicle", "n_vehicles", lambda d: d.n_vehicles == 1, lambda d: d.n_vehicles > 1),
    ("Dark", "lighting_dark", lambda d: d.lighting_dark == 1, lambda d: d.lighting_dark == 0),
    ("Rural", "urban_rural", lambda d: d.urban_rural == "rural", lambda d: d.urban_rural == "urban"),
    (
        "Weather not clear",
        "weather_coarse",
        lambda d: d.weather_coarse.isin(["rain", "snow", "fog", "other"]),
        lambda d: d.weather_coarse == "clear",
    ),
    (
        "Motorcycle",
        "vehicle_type_mode",
        lambda d: d.vehicle_type_mode == "motorcycle",
        lambda d: d.vehicle_type_mode == "car",
    ),
    (
        "Goods vehicle",
        "vehicle_type_mode",
        lambda d: d.vehicle_type_mode == "goods",
        lambda d: d.vehicle_type_mode == "car",
    ),
    (
        "Driver 16-25",
        "driver_age_band",
        lambda d: d.driver_age_band.isin(["16-20", "21-25"]),
        lambda d: d.driver_age_band.isin(["26-35", "36-45", "46-55", "56-65"]),
    ),
    (
        "Driver over 65",
        "driver_age_band",
        lambda d: d.driver_age_band.isin(["66-75", ">75"]),
        lambda d: d.driver_age_band.isin(["26-35", "36-45", "46-55", "56-65"]),
    ),
    ("Night 20:00-05:59", "hour", lambda d: (d.hour >= 20) | (d.hour < 6), lambda d: d.hour.between(6, 19)),
]


def feature_shap(model, X, cmap):
    """TreeSHAP of the primary model, summed over the columns of each source feature (log-odds)."""
    booster = model.m.get_booster()
    it = (0, model.m.best_iteration + 1)
    c = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=it)[:, :-1]
    cols = pd.Series(list(X.columns)).map(cmap)
    return pd.DataFrame(c, columns=X.columns).T.groupby(cols.values).sum().T


def effect(v, a, b, w):
    wa, wb = w * a, w * b
    if wa.sum() == 0 or wb.sum() == 0:
        return np.nan
    return float((v * wa).sum() / wa.sum() - (v * wb).sum() / wb.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="defaults to the primary model chosen by within_domain")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    prim = a.model or primary()
    if prim != "xgboost":
        raise SystemExit("factor_effects reads TreeSHAP from XGBoost")
    df = load(baac=True)
    S = {c: {k: split_df(df, c, k) for k in ("train", "val", "test")} for c in COUNTRIES}
    M = {}
    for c in COUNTRIES:
        Xtr, _ = design(S[c]["train"])
        Xva, _ = design(S[c]["val"])
        M[c] = Model(prim, "ksi").fit(
            Xtr, S[c]["train"], Xva, S[c]["val"], w_tr=S[c]["train"].weight.values, w_va=S[c]["val"].weight.values
        )
    rng = np.random.default_rng(SEED)
    rows = []
    for tgt in COUNTRIES:
        te = S[tgt]["test"].assign(kmh=lambda d: d.speed_limit_mph * KMH)
        Xte, cmap = design(te)
        w = te.weight.values if tgt == "CRSS" else np.ones(len(te))
        masks = [
            (name, feat, f(te).fillna(False).values, g(te).fillna(False).values) for name, feat, f, g in FACTORS
        ]
        idx = [rng.choice(len(te), len(te), replace=True, p=w / w.sum()) for _ in range(a.boot)]
        ones = np.ones(len(te))
        for src in COUNTRIES:
            sh = feature_shap(M[src], Xte, cmap)
            for name, feat, m1, m0 in masks:
                v = sh[feat].values
                est = effect(v, m1, m0, w)
                bs = [effect(v[i], m1[i], m0[i], ones) for i in idx]
                rows.append(
                    dict(
                        target=tgt,
                        model=src,
                        own=src == tgt,
                        factor=name,
                        effect=est,
                        ci_lo=float(np.nanquantile(bs, 0.025)),
                        ci_hi=float(np.nanquantile(bs, 0.975)),
                        share_with_factor=float(np.average(m1, weights=w)),
                    )
                )
            print(tgt, src, flush=True)
    pd.DataFrame(rows).to_csv(paths.result("factor_effects.csv"), index=False)


if __name__ == "__main__":
    main()
