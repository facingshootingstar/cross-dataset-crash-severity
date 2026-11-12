"""Where the STATS19 and CRSS shift comes from (PAD by feature) and how the AUROC loss responds to alternative crosswalks."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics
from crashtransfer.models import Model, primary
from crashtransfer.schema import CLOSE, FEATS, PAIRS_UK_US, SEED, SHIFTED, target_weight
from crashtransfer.transfer import domain_classifier, draw

prim, df = primary(), load()
alt = pd.read_parquet(
    paths.interim("crss_posthoc.parquet"), columns=["case_id", "vehicle_type_mode_alt", "lighting_dark_alt"]
)
df = df.merge(alt, on="case_id", how="left")
assert df.loc[df.source == "CRSS", "lighting_dark_alt"].notna().mean() > 0.9
s_tr, c_tr = split_df(df, "STATS19", "train"), split_df(df, "CRSS", "train")
rng = np.random.default_rng(SEED)
n = min(len(s_tr), len(c_tr), 150_000)
a, b = draw(s_tr, n, rng), draw(c_tr, n, rng)


def pad(keep):
    drop = [f for f in FEATS if f not in keep]
    eps = domain_classifier(design(a, drop=drop)[0], design(b, drop=drop)[0], n)[1]
    return eps, 2 * (1 - 2 * eps)


def jsd(f):
    def dist(d, w):
        col = d[f].clip(upper=4) if f == "n_vehicles" else d[f]
        v = col.astype(object).where(d[f].notna(), "missing").astype(str)
        return pd.Series(w, index=v.index).groupby(v.values).sum() / w.sum()

    ps, pc = dist(s_tr, np.ones(len(s_tr))), dist(c_tr, c_tr.weight.values.astype(float))
    idx = ps.index.union(pc.index)
    return float(jensenshannon(ps.reindex(idx, fill_value=0), pc.reindex(idx, fill_value=0), base=2) ** 2)


rows = []
e, p = pad(FEATS)
rows.append(dict(analysis="all 12 features", feature="all", domain_error=e, pad=p, jsd=np.nan))
for f in FEATS:
    e1, p1 = pad([f])
    e2, p2 = pad([g for g in FEATS if g != f])
    rows.append(dict(analysis="single feature", feature=f, domain_error=e1, pad=p1, jsd=jsd(f)))
    rows.append(dict(analysis="all but this feature", feature=f, domain_error=e2, pad=p2, jsd=np.nan))
for label, keep in (("close-only (4 features)", CLOSE), ("shifted-only (8 features)", SHIFTED)):
    e, p = pad(keep)
    rows.append(dict(analysis=label, feature="+".join(keep), domain_error=e, pad=p, jsd=np.nan))
sf = pd.DataFrame(rows)
sf["category"] = sf.feature.map(lambda f: "close" if f in CLOSE else ("shifted" if f in SHIFTED else ""))
sf.to_csv(paths.result("posthoc_shift_features.csv"), index=False)

variants = [
    ("base (retrained)", [], None),
    ("close-only (4 features)", SHIFTED, None),
    ("CRSS pickups -> car", [], "vehicle_type_mode"),
    ("CRSS dawn/dusk -> dark", [], "lighting_dark"),
] + [(f"drop {f}", [f], None) for f in SHIFTED]
cw = []
for name, drop, swap in variants:
    d = df.copy()
    if swap is not None:
        m = d.source == "CRSS"
        d.loc[m, swap] = d.loc[m, swap + "_alt"]
    models, data = {}, {}
    for src in ("STATS19", "CRSS"):
        tr, va, te = (split_df(d, src, s) for s in ("train", "val", "test"))
        models[src] = Model(prim, "ksi").fit(
            design(tr, drop=drop)[0],
            tr,
            design(va, drop=drop)[0],
            va,
            w_tr=tr.weight.values,
            w_va=va.weight.values,
        )
        data[src] = (te, design(te, drop=drop)[0])
    for src, tgt in PAIRS_UK_US:
        te, Xte = data[tgt]
        w, y = target_weight(te), te.y_ksi.values
        t0, t1 = binary_metrics(y, models[tgt].p_ksi(Xte), w), binary_metrics(y, models[src].p_ksi(Xte), w)
        cw.append(
            dict(
                variant=name,
                n_features=len(FEATS) - len(drop),
                direction=f"{src}->{tgt}",
                auroc_T0=t0["auroc"],
                auroc_T1=t1["auroc"],
                auprc_T0=t0["auprc"],
                auprc_T1=t1["auprc"],
                ece_T1=t1["ece"],
                auroc_gap=t0["auroc"] - t1["auroc"],
                auprc_gap=t0["auprc"] - t1["auprc"],
            )
        )
pd.DataFrame(cw).to_csv(paths.result("posthoc_crosswalk_sensitivity.csv"), index=False)
