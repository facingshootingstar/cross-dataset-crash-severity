"""Maps STATS19, CRSS and BAAC onto one row per collision with the shared features and severity labels.

Each country gets its own codebook rules; the mapping is documented in results/crosswalk.csv and results/crosswalk_baac.csv.
"""

import glob
import os

import numpy as np
import pandas as pd

from . import paths
from .schema import YEARS, age_band, mode_with_priority

# STATS19
BULK = "dft-road-casualty-statistics-%s-1979-latest-published-year.csv"
STATS19_YEARS = (2016, 2025)


def stats19_filtered(table, usecols):
    src = os.path.join(paths.RAW, "stats19", "bulk", BULK % table)
    parts = []
    for ch in pd.read_csv(src, usecols=usecols, chunksize=500_000, low_memory=False):
        parts.append(ch[ch["collision_year"].between(*STATS19_YEARS)])
    return pd.concat(parts, ignore_index=True)


def stats19_vehicle_type(v):
    out = pd.Series(np.nan, index=v.index, dtype=object)
    out[v.isin([8, 9])] = "car"
    out[v.isin([19, 20, 21, 98])] = "goods"
    out[v.isin([10, 11])] = "bus"
    out[v.isin([2, 3, 4, 5, 23, 97])] = "motorcycle"
    out[v.isin([17, 18, 22, 33, 90])] = "other"
    return out  # pedal cycles, ridden horses and unknowns stay missing


def stats19():
    c = pd.read_parquet(paths.interim("stats19_collision_2016_latest_bulk.parquet"))
    v = stats19_filtered(
        "vehicle",
        [
            "collision_index",
            "collision_year",
            "vehicle_reference",
            "vehicle_type",
            "sex_of_driver",
            "age_of_driver",
        ],
    )
    k = stats19_filtered("casualty", ["collision_index", "collision_year", "casualty_type"])
    f = pd.DataFrame({"case_id": c["collision_index"].astype(str), "year": c["collision_year"]})
    f["month"] = pd.to_datetime(c["date"], format="%d/%m/%Y", errors="coerce").dt.month.values
    f["day_of_week"] = c["day_of_week"].where(c["day_of_week"].between(1, 7)).values  # 1 = Sunday, as CRSS
    h = pd.to_numeric(c["time"].astype(str).str[:2], errors="coerce")
    f["hour"] = h.where(h.between(0, 23)).values
    lc = c["light_conditions"]
    f["lighting_dark"] = np.select([lc.isin([4, 5, 6, 7]), lc == 1], [1, 0], default=np.nan)
    w = c["weather_conditions"]
    wc = pd.Series(np.nan, index=c.index, dtype=object)
    wc[w.isin([1, 4])] = "clear"
    wc[w.isin([2, 5])] = "rain"
    wc[w.isin([3, 6])] = "snow"
    wc[w == 7] = "fog"
    wc[w == 8] = "other"
    f["weather_coarse"] = wc.values
    jd = c["junction_detail"]
    f["junction"] = np.select(
        [jd == 0, jd.isin([1, 2, 3, 5, 6, 7, 8, 9, 13, 16, 17, 18, 19])], [0, 1], default=np.nan
    )
    f["n_vehicles"] = c["number_of_vehicles"].where(c["number_of_vehicles"] >= 1).values
    f["urban_rural"] = c["urban_or_rural_area"].map({1: "urban", 2: "rural"}).values
    f["speed_limit_mph"] = c["speed_limit"].where(c["speed_limit"].isin([20, 30, 40, 50, 60, 70])).values

    v["_vt"] = stats19_vehicle_type(v["vehicle_type"])
    idx = c["collision_index"]
    f["vehicle_type_mode"] = idx.map(mode_with_priority(v, "collision_index", "_vt")).values
    moto = v.loc[v["_vt"] == "motorcycle", "collision_index"].unique()
    cycle = v.loc[v["vehicle_type"] == 1, "collision_index"].unique()
    ped = k.loc[k["casualty_type"].isin([0, 1]), "collision_index"].unique()
    f["vru_involved"] = (idx.isin(moto) | idx.isin(cycle) | idx.isin(ped)).astype(int).values

    d1 = v[v["vehicle_reference"] == 1].drop_duplicates("collision_index").set_index("collision_index")
    f["driver_age_band"] = idx.map(d1["age_of_driver"].where(d1["age_of_driver"] >= 0)).pipe(age_band).values
    f["driver_sex"] = idx.map(d1["sex_of_driver"].map({1: "M", 2: "F"})).values

    f["y_3class"] = c["collision_severity"].values
    f["y_ksi"] = (c["collision_severity"] <= 2).astype(int).values
    f["weight"] = 1.0
    f["source"] = "STATS19"
    f["collision_injury_based"] = c["collision_injury_based"].values  # audit only, never a feature
    return f


# CRSS
def crss_read(y, table):
    root = paths.interim("crss_extracted")
    files = {
        os.path.basename(p).lower(): p for p in glob.glob(f"{root}/{y}/**/*", recursive=True) if os.path.isfile(p)
    }
    return pd.read_csv(files[table + ".csv"], encoding="latin1", low_memory=False)


def crss_vehicle_type(b):
    b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=b.index, dtype=object)
    out[(b >= 1) & (b <= 20)] = "car"
    out[
        b.isin(
            [
                21,
                22,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                39,
                40,
                41,
                45,
                48,
                49,
                60,
                61,
                62,
                63,
                64,
                66,
                67,
                71,
                72,
                78,
                79,
            ]
        )
    ] = "goods"  # vans, pickups, trucks
    out[b.isin([50, 51, 52, 55, 58, 59])] = "bus"
    out[(b >= 80) & (b <= 89)] = "motorcycle"
    out[b.isin([42, 65, 73, 90, 91, 92, 93, 94, 95, 96, 97])] = "other"
    return out


def crss_weather(w):
    w = pd.to_numeric(w, errors="coerce")
    out = pd.Series(np.nan, index=w.index, dtype=object)
    out[w.isin([1, 10])] = "clear"
    out[w.isin([2, 12])] = "rain"
    out[w.isin([3, 4, 11])] = "snow"
    out[w == 5] = "fog"
    out[w.isin([6, 7, 8])] = "other"
    return out


def crss_year(y):
    """One CRSS year and the count of excluded severity codes. MAXSEV_IM 4, 3, 2/1 map to Fatal, Serious, Slight."""
    a = crss_read(y, "accident")
    keep = a["MAXSEV_IM"].isin([1, 2, 3, 4])
    exc = a.loc[~keep, "MAXSEV_IM"].value_counts().rename_axis("code").reset_index(name="n")
    exc["year"] = y
    exc["n_all"] = len(a)
    a = a[keep].copy()
    v, p = crss_read(y, "vehicle"), crss_read(y, "person")
    v = v[v["CASENUM"].isin(a["CASENUM"])]
    p = p[p["CASENUM"].isin(a["CASENUM"])]

    f = pd.DataFrame({"case_id": a["CASENUM"].astype(str), "year": a["YEAR"]})
    f["hour"] = a["HOUR"].where(a["HOUR"].between(0, 23)).values
    f["day_of_week"] = a["DAY_WEEK"].where(a["DAY_WEEK"].between(1, 7)).values
    f["month"] = a["MONTH"].where(a["MONTH"].between(1, 12)).values
    lc = a["LGT_COND"]
    f["lighting_dark"] = np.select([lc.isin([2, 3, 6]), lc.isin([1, 4, 5])], [1, 0], default=np.nan)
    f["weather_coarse"] = crss_weather(a["WEATHER"]).values
    rj = a["RELJCT2"]
    f["junction"] = np.select([rj.isin([2, 3, 4, 8]), rj == 1], [1, 0], default=np.nan)
    f["n_vehicles"] = a["VE_TOTAL"].where(a["VE_TOTAL"] >= 1).values
    f["urban_rural"] = a["URBANICITY"].map({1: "urban", 2: "rural"}).values
    # The US limit is recorded per vehicle; take the highest valid one (0, 98 and 99 are not limits).
    f["speed_limit_mph"] = (
        a["CASENUM"].map(v[v["VSPD_LIM"].between(5, 95)].groupby("CASENUM")["VSPD_LIM"].max()).values
    )

    v["_vt"] = crss_vehicle_type(v["BODY_TYP"])
    f["vehicle_type_mode"] = a["CASENUM"].map(mode_with_priority(v, "CASENUM", "_vt")).values
    ped = p.loc[p["PER_TYP"].isin([5, 6, 7]), "CASENUM"].unique()
    moto = v.loc[v["_vt"] == "motorcycle", "CASENUM"].unique()
    f["vru_involved"] = (a["CASENUM"].isin(ped) | a["CASENUM"].isin(moto)).astype(int).values

    d = p[(p["PER_TYP"] == 1) & (p["VEH_NO"] == 1) & p["AGE"].between(0, 120)].drop_duplicates("CASENUM")
    f["driver_age_band"] = a["CASENUM"].map(d.set_index("CASENUM")["AGE"]).pipe(age_band).values
    ds = p[(p["PER_TYP"] == 1) & (p["VEH_NO"] == 1) & p["SEX"].isin([1, 2])].drop_duplicates("CASENUM")
    f["driver_sex"] = a["CASENUM"].map(ds.set_index("CASENUM")["SEX"].map({1: "M", 2: "F"})).values

    sev = a["MAXSEV_IM"].map({4: 1, 3: 2, 2: 3, 1: 3})
    f["y_3class"] = sev.values
    f["y_ksi"] = (sev <= 2).astype(int).values
    f["weight"] = a["WEIGHT"].values
    f["source"] = "CRSS"
    f["maxsev_im"] = a["MAXSEV_IM"].values  # audit only, never a feature
    return f, exc


# BAAC
LEGAL_KMH = [20, 30, 50, 70, 80, 90, 110, 130]  # other vma values are coding errors


def baac_read(table, y):
    f = os.path.join(paths.RAW, "baac", f"{table}_{y}.csv")
    with open(f, "rb") as h:
        head = h.read(2000)
    sep = ";" if head.count(b";") > head.count(b",") else ("\t" if head.count(b"\t") > head.count(b",") else ",")
    for enc in ("utf-8", "latin1"):
        try:
            d = pd.read_csv(f, sep=sep, dtype=str, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    d.columns = [c.strip().strip('"').lstrip("\ufeff") for c in d.columns]
    d = d.rename(columns={"Accident_Id": "Num_Acc"})
    for c in d.columns:
        d[c] = d[c].str.strip()
    return d


def baac_num(s):
    return pd.to_numeric(s, errors="coerce").where(lambda x: x >= 0)


def baac_vehicle_type(cv):
    cv = baac_num(cv)
    out = pd.Series(np.nan, index=cv.index, dtype=object)
    out[cv.isin([3, 7])] = "car"
    out[cv.isin([10, 13, 14, 15, 16, 17])] = "goods"
    out[cv.isin([37, 38])] = "bus"
    out[cv.isin([2, 30, 31, 32, 33, 34, 41, 42, 43])] = "motorcycle"
    out[cv.isin([20, 21, 35, 36, 39, 40, 50, 99])] = "other"
    return out


def baac_year(y):
    """Metropolitan France, worst user per collision: killed within 30 days is Fatal, hospitalised over 24 h is Serious."""
    c, l, v, u = (baac_read(t, y) for t in ("caract", "lieux", "vehicules", "usagers"))
    c = c[~c["dep"].fillna("").str.startswith(("97", "98"))].reset_index(drop=True)
    idx = c["Num_Acc"]
    f = pd.DataFrame({"case_id": "FR" + idx, "year": y})
    an = baac_num(c["an"])
    an = an.where(an > 100, an + 2000)
    dt = pd.to_datetime(dict(year=an, month=baac_num(c["mois"]), day=baac_num(c["jour"])), errors="coerce")
    f["month"] = dt.dt.month.values
    f["day_of_week"] = ((dt.dt.dayofweek + 1) % 7 + 1).values  # 1 = Sunday
    hm = c["hrmn"].fillna("")
    h = np.where(
        hm.str.contains(":"),
        pd.to_numeric(hm.str.split(":").str[0], errors="coerce"),
        pd.to_numeric(hm, errors="coerce") // 100,
    )
    f["hour"] = pd.Series(h).where(lambda x: x.between(0, 23)).values
    lum = baac_num(c["lum"])
    f["lighting_dark"] = np.select([lum.isin([3, 4, 5]), lum.isin([1, 2])], [1, 0], default=np.nan)
    atm = baac_num(c["atm"])
    wc = pd.Series(np.nan, index=c.index, dtype=object)
    wc[atm.isin([1, 7, 8])] = "clear"
    wc[atm.isin([2, 3])] = "rain"
    wc[atm == 4] = "snow"
    wc[atm == 5] = "fog"
    wc[atm.isin([6, 9])] = "other"
    f["weather_coarse"] = wc.values
    it = baac_num(c["int"])
    f["junction"] = np.select([it == 1, it.between(2, 9)], [0, 1], default=np.nan)
    f["n_vehicles"] = idx.map(v.groupby("Num_Acc").size()).values
    f["urban_rural"] = baac_num(c["agg"]).map({2: "urban", 1: "rural"}).values
    if "vma" in l.columns:  # absent before 2019
        vma = baac_num(l["vma"]).where(lambda x: x.isin(LEGAL_KMH))
        f["speed_limit_mph"] = idx.map(vma.groupby(l["Num_Acc"]).max() * 0.621371).round(1).values
    else:
        f["speed_limit_mph"] = np.nan

    v["_vt"] = baac_vehicle_type(v["catv"])
    f["vehicle_type_mode"] = idx.map(mode_with_priority(v, "Num_Acc", "_vt")).values
    vru = v.loc[baac_num(v["catv"]).isin([1, 80, 50, 60]) | (v["_vt"] == "motorcycle"), "Num_Acc"].unique()
    ped = u.loc[baac_num(u["catu"]) == 3, "Num_Acc"].unique()
    f["vru_involved"] = (idx.isin(vru) | idx.isin(ped)).astype(int).values

    first = v.sort_values(["Num_Acc", "num_veh"]).drop_duplicates("Num_Acc").set_index("Num_Acc")["num_veh"]
    drv = (
        u[(baac_num(u["catu"]) == 1) & (u["num_veh"] == u["Num_Acc"].map(first))]
        .drop_duplicates("Num_Acc")
        .set_index("Num_Acc")
    )
    f["driver_age_band"] = (
        idx.map((y - baac_num(drv["an_nais"])).where(lambda x: x.between(0, 120))).pipe(age_band).values
    )
    f["driver_sex"] = idx.map(baac_num(drv["sexe"]).map({1: "M", 2: "F"})).values

    sev = baac_num(u["grav"]).map({2: 1, 3: 2, 4: 3}).groupby(u["Num_Acc"]).min()
    f["y_3class"] = idx.map(sev).values
    f = f[f["y_3class"].notna()].copy()
    f["y_3class"] = f["y_3class"].astype(int)
    f["y_ksi"] = (f["y_3class"] <= 2).astype(int)
    f["weight"] = 1.0
    f["source"] = "BAAC"
    return f


def baac():
    parts = []
    for y in YEARS:
        if glob.glob(os.path.join(paths.RAW, "baac", f"usagers_{y}.csv")):
            parts.append(baac_year(y))
    return pd.concat(parts, ignore_index=True)


def region_keys():
    """Police force area (STATS19) and departement (BAAC) of each collision, for analysis note B2. Not a feature."""
    c = stats19_filtered("collision", ["collision_index", "collision_year", "police_force"])
    uk = pd.DataFrame(
        {
            "source": "STATS19",
            "case_id": c["collision_index"].astype(str),
            "region": "PF" + c["police_force"].astype(str),
        }
    )
    parts = []
    for y in YEARS:
        # the departement code changes format before 2019; French modelling starts in 2019
        if y >= 2019 and glob.glob(os.path.join(paths.RAW, "baac", f"caract_{y}.csv")):
            d = baac_read("caract", y)
            dep = "D" + d["dep"].str.upper().str.lstrip("0")
            parts.append(pd.DataFrame({"source": "BAAC", "case_id": "FR" + d["Num_Acc"], "region": dep}))
    return pd.concat([uk] + parts, ignore_index=True).drop_duplicates(["source", "case_id"])


# Victoria (amendment A2, docs/protocol/PROTOCOL_VIC.md)
VIC_AGE = {
    "0-4": "<16",
    "5-12": "<16",
    "13-15": "<16",
    "16-17": "16-20",
    "18-21": "16-20",
    "22-25": "21-25",
    "26-29": "26-35",
    "30-39": "26-35",
    "40-49": "36-45",
    "50-59": "46-55",
    "60-64": "56-65",
    "65-69": "66-75",
    "70+": "66-75",
}
VIC_VT = {
    "car": ["01", "02", "03"],
    "goods": ["04", "05", "06", "07", "60", "61", "62", "63", "71", "72"],
    "bus": ["08", "09"],
    "motorcycle": ["10", "11", "12"],
    "other": ["14", "15", "16", "17", "20", "21", "27"],
}


def vic_read(name, cols):
    return pd.read_csv(os.path.join(paths.RAW, "vic", f"{name}.csv"), usecols=cols, dtype=str)


def vic():
    a = vic_read(
        "Accident",
        [
            "ACCIDENT_NO",
            "ACCIDENT_DATE",
            "ACCIDENT_TIME",
            "LIGHT_CONDITION",
            "NO_OF_VEHICLES",
            "ROAD_GEOMETRY",
            "SEVERITY",
            "SPEED_ZONE",
        ],
    )
    a = a[a["SEVERITY"].isin(["1", "2", "3"])].copy()
    date = pd.to_datetime(a["ACCIDENT_DATE"])
    a = a[date.dt.year.between(2016, 2024)]
    date = pd.to_datetime(a["ACCIDENT_DATE"])
    f = pd.DataFrame(
        {"case_id": a["ACCIDENT_NO"].values, "year": date.dt.year.values, "month": date.dt.month.values}
    )
    f["day_of_week"] = ((date.dt.dayofweek + 1) % 7 + 1).values  # Sunday = 1
    f["hour"] = pd.to_numeric(a["ACCIDENT_TIME"].str[:2], errors="coerce").values
    lc = pd.to_numeric(a["LIGHT_CONDITION"], errors="coerce")
    f["lighting_dark"] = np.select([lc.isin([3, 4, 5, 6]), lc.isin([1, 2])], [1, 0], default=np.nan)
    at = vic_read("Atmospheric_Condition", ["ACCIDENT_NO", "ATMOSPH_COND", "ATMOSPH_COND_SEQ"])
    at = (
        at.sort_values(["ACCIDENT_NO", "ATMOSPH_COND_SEQ"]).drop_duplicates("ACCIDENT_NO").set_index("ACCIDENT_NO")
    )
    wmap = {"1": "clear", "2": "rain", "3": "snow", "4": "fog", "5": "other", "6": "other", "7": "other"}
    f["weather_coarse"] = f["case_id"].map(at["ATMOSPH_COND"].map(wmap)).values
    rg = pd.to_numeric(a["ROAD_GEOMETRY"], errors="coerce")
    f["junction"] = np.select([rg.between(1, 4), rg.isin([5, 6])], [1, 0], default=np.nan)
    f["n_vehicles"] = pd.to_numeric(a["NO_OF_VEHICLES"]).astype(int).values
    node = (
        vic_read("Node", ["ACCIDENT_NO", "DEG_URBAN_NAME"]).drop_duplicates("ACCIDENT_NO").set_index("ACCIDENT_NO")
    )
    urb = node["DEG_URBAN_NAME"].map(
        lambda x: "rural" if x == "RURAL_VICTORIA" else ("urban" if isinstance(x, str) else None)
    )
    f["urban_rural"] = f["case_id"].map(urb).values
    sz = pd.to_numeric(a["SPEED_ZONE"], errors="coerce")
    f["speed_limit_mph"] = (sz.where(sz < 777) / 1.609344).values
    v = vic_read("Vehicle", ["ACCIDENT_NO", "VEHICLE_ID", "VEHICLE_TYPE"])
    v["_vt"] = v["VEHICLE_TYPE"].map({c: t for t, cs in VIC_VT.items() for c in cs})
    f["vehicle_type_mode"] = f["case_id"].map(mode_with_priority(v, "ACCIDENT_NO", "_vt")).values
    p = vic_read("Person", ["ACCIDENT_NO", "VEHICLE_ID", "SEX", "AGE_GROUP", "ROAD_USER_TYPE", "SEATING_POSITION"])
    vru = set(p.loc[p["ROAD_USER_TYPE"].isin(["1", "4", "5", "6", "16"]), "ACCIDENT_NO"])
    vru |= set(v.loc[v["VEHICLE_TYPE"].isin(["10", "11", "12", "13"]), "ACCIDENT_NO"])
    f["vru_involved"] = f["case_id"].isin(vru).astype(int).values
    drv = (
        p[(p["VEHICLE_ID"] == "A") & (p["SEATING_POSITION"] == "D")]
        .drop_duplicates("ACCIDENT_NO")
        .set_index("ACCIDENT_NO")
    )
    f["driver_age_band"] = f["case_id"].map(drv["AGE_GROUP"].map(VIC_AGE)).values
    f["driver_sex"] = f["case_id"].map(drv["SEX"].map({"M": "M", "F": "F"})).values
    f["y_3class"] = pd.to_numeric(a["SEVERITY"]).astype(int).values
    f["y_ksi"] = (f["y_3class"] <= 2).astype(int)
    f["weight"] = 1.0
    f["source"] = "VIC"
    return f


# extra columns for the design-based intervals, crosswalk variants and the native-feature ceiling
CRSS_ACC = [
    "MAN_COLL",
    "HARM_EV",
    "REL_ROAD",
    "TYP_INT",
    "INT_HWY",
    "WRK_ZONE",
    "ALCOHOL",
    "PEDS",
    "PERMVIT",
    "PERNOTMVIT",
    "PVH_INVL",
    "VE_FORMS",
    "SCH_BUS",
    "RELJCT1",
    "RELJCT2",
    "LGT_COND",
    "WEATHER",
]
CRSS_VEH1 = [
    "IMPACT1",
    "P_CRASH1",
    "P_CRASH2",
    "VTRAFWAY",
    "VNUM_LAN",
    "VALIGN",
    "VPROFILE",
    "VSURCOND",
    "VTRAFCON",
    "BODY_TYP",
]
S19_COL = [
    "first_road_class",
    "road_type",
    "junction_detail",
    "junction_control",
    "second_road_class",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "carriageway_hazards",
    "trunk_road_flag",
]
S19_VEH1 = [
    "vehicle_type",
    "vehicle_manoeuvre",
    "first_point_of_impact",
    "junction_location",
    "skidding_and_overturning",
    "hit_object_in_carriageway",
    "vehicle_leaving_carriageway",
    "hit_object_off_carriageway",
    "towing_and_articulation",
]


def crss_extra_year(y):
    a = crss_read(y, "accident")
    a = a[a["MAXSEV_IM"].isin([1, 2, 3, 4])].copy()
    v = crss_read(y, "vehicle")
    v = v[v["CASENUM"].isin(a["CASENUM"])].copy()
    f = pd.DataFrame(
        {
            "case_id": a["CASENUM"].astype(str).values,
            "year": a["YEAR"].values,
            "psustrat": a["PSUSTRAT"].values,
            "psu_var": a["PSU_VAR"].values,
        }
    )
    for c in CRSS_ACC:
        f["n_" + c.lower()] = a[c].values
    v1 = v[v["VEH_NO"] == 1].drop_duplicates("CASENUM").set_index("CASENUM")
    for c in CRSS_VEH1:
        f["n_v1_" + c.lower()] = a["CASENUM"].map(v1[c]).values
    g = v.groupby("CASENUM")
    f["n_any_rollover"] = a["CASENUM"].map(g["ROLLOVER"].apply(lambda s: int(s.isin([1, 2, 3]).any()))).values
    f["n_any_fire"] = a["CASENUM"].map(g["FIRE_EXP"].apply(lambda s: int((s == 1).any()))).values
    f["n_any_hitrun"] = a["CASENUM"].map(g["HIT_RUN"].apply(lambda s: int((s == 1).any()))).values
    f["n_any_speedrel"] = a["CASENUM"].map(g["SPEEDREL"].apply(lambda s: int(s.isin([2, 3, 4, 5]).any()))).values
    f["n_occupants"] = a["CASENUM"].map(g["NUMOCCS"].apply(lambda s: s.where(s < 97).sum())).values
    vt = crss_vehicle_type(v["BODY_TYP"])
    vt[v["BODY_TYP"].isin([30, 31, 32, 33, 34, 39])] = "car"  # variant: pickups as cars
    v["_vt_alt"] = vt
    f["vehicle_type_mode_alt"] = a["CASENUM"].map(mode_with_priority(v, "CASENUM", "_vt_alt")).values
    lc = a["LGT_COND"]
    f["lighting_dark_alt"] = np.select(
        [lc.isin([2, 3, 6, 4, 5]), lc == 1], [1, 0], default=np.nan
    )  # variant: dawn and dusk as dark
    f["source"] = "CRSS"
    return f


def stats19_extra():
    c = pd.read_parquet(paths.interim("stats19_collision_2016_latest_bulk.parquet"))
    c = c[c["collision_year"].between(2016, 2024)]
    f = pd.DataFrame({"case_id": c["collision_index"].astype(str).values, "year": c["collision_year"].values})
    for k in S19_COL:
        f["n_" + k] = pd.to_numeric(c[k], errors="coerce").values
    v = stats19_filtered("vehicle", ["collision_index", "collision_year", "vehicle_reference"] + S19_VEH1)
    v = v[v["collision_year"].between(2016, 2024)]
    v1 = v[v["vehicle_reference"] == 1].drop_duplicates("collision_index").set_index("collision_index")
    for k in S19_VEH1:
        f["n_v1_" + k] = c["collision_index"].map(v1[k]).values
    g = v.groupby("collision_index")
    f["n_any_overturn"] = (
        c["collision_index"]
        .map(g["skidding_and_overturning"].apply(lambda s: int(s.isin([2, 4, 5]).any())))
        .values
    )
    f["n_any_offroad_object"] = (
        c["collision_index"].map(g["hit_object_off_carriageway"].apply(lambda s: int((s > 0).any()))).values
    )
    f["n_any_left_carriageway"] = (
        c["collision_index"].map(g["vehicle_leaving_carriageway"].apply(lambda s: int((s > 0).any()))).values
    )
    f["source"] = "STATS19"
    return f
