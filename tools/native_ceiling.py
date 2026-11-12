"""Within-country ceiling: the primary model refitted on the 12 shared features plus native at-scene fields of the same source."""

import pandas as pd

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.models import Model, load_model, primary
from crashtransfer.survey import add_design, compare

COUNTS = [
    "n_permvit",
    "n_pernotmvit",
    "n_pvh_invl",
    "n_ve_forms",
    "n_peds",
    "n_occupants",
    "n_any_rollover",
    "n_any_fire",
    "n_any_hitrun",
    "n_any_speedrel",
    "n_any_overturn",
    "n_any_offroad_object",
    "n_any_left_carriageway",
]


class NativeModel(Model):
    def _make(self, hp, k):
        m = super()._make(hp, k)
        m.set_params(enable_categorical=True, max_cat_to_onehot=1)
        return m


prim = primary()
assert prim == "xgboost"
df = add_design(load())
out, imp = [], []
for src, f in (("CRSS", "crss_posthoc.parquet"), ("STATS19", "stats19_posthoc.parquet")):
    nat = pd.read_parquet(paths.interim(f))
    ncols = [c for c in nat.columns if c.startswith("n_")]
    nat = nat[["case_id"] + ncols]
    tr, va, te = (
        split_df(df, src, s).merge(nat, on="case_id", how="left", validate="1:1") for s in ("train", "val", "test")
    )
    cats = [c for c in ncols if c not in COUNTS]
    levels = {c: sorted(int(x) for x in pd.to_numeric(tr[c], errors="coerce").dropna().unique()) for c in cats}

    def matrix(d):
        x = design(d)[0]
        for c in ncols:
            v = pd.to_numeric(d[c], errors="coerce")
            if c in cats:
                assert (v.dropna() % 1 == 0).all(), c
            x["nat_" + c[2:]] = (
                pd.Categorical(v.astype("Int64"), categories=levels[c]) if c in cats else v.astype(float).values
            )
        return x

    Xte = matrix(te)
    m = NativeModel(prim, "ksi").fit(matrix(tr), tr, matrix(va), va, w_tr=tr.weight.values, w_va=va.weight.values)
    preds = {
        "native + shared (ceiling)": m.p_ksi(Xte),
        "12 shared features (locked model)": load_model(f"{src}_{prim}_ksi.joblib").p_ksi(design(te)[0]),
    }
    r = compare(te, preds, [("native + shared (ceiling)", "12 shared features (locked model)")])
    r.insert(0, "source", src)
    r.insert(1, "n_native_fields", len(ncols))
    r.insert(2, "hp", str(m.hp))
    out.append(r)
    imp += [
        dict(source=src, feature=k, gain=v) for k, v in m.m.get_booster().get_score(importance_type="gain").items()
    ]
pd.concat(out, ignore_index=True).to_csv(paths.result("posthoc_native_ceiling.csv"), index=False)
i = pd.DataFrame(imp)
i["share"] = i.gain / i.groupby("source").gain.transform("sum")
i.sort_values(["source", "gain"], ascending=[True, False]).to_csv(
    paths.result("posthoc_native_importance.csv"), index=False
)
