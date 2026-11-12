"""Feature alignment on test 2023: classical CORAL, Deep CORAL and DANN, each against a source-only model of the same kind, over five seeds."""

import json
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from crashtransfer import adaptation as ad
from crashtransfer import paths
from crashtransfer.data import design, load, split_df
from crashtransfer.metrics import binary_metrics
from crashtransfer.models import Imputer, norm_w, recalibrate
from crashtransfer.schema import PAIRS_FRANCE, PAIRS_UK_US, eval_settings, target_weight
from crashtransfer.transfer import draw

SEEDS, B_BOOT = [42, 1, 2, 3, 4], 200
METRICS = ["auroc", "auprc", "ece", "brier", "logloss"]


def T(a):
    return torch.tensor(a, device=ad.DEV)


def boot_idx(n, w, mode, seed=42):
    rng = np.random.default_rng(seed)
    pr = None if mode == "uniform" else w / w.sum()
    return [rng.choice(n, n, replace=True, p=pr) for _ in range(B_BOOT)]


def run_pair(df, src, tgt, rows, paired, align, t0):
    D = f"{src}->{tgt}"
    s_tr, s_va = split_df(df, src, "train"), split_df(df, src, "val")
    t_tr, t_va, t_te = (split_df(df, tgt, k) for k in ("train", "val", "test"))
    t_un = draw(
        t_tr, min(len(t_tr), 200_000) if tgt == "CRSS" else len(t_tr), np.random.default_rng(42)
    ).reset_index(drop=True)
    imp = Imputer().fit(design(s_tr)[0])
    Z = {
        k: imp.transform(design(d)[0]).values.astype(np.float32)
        for k, d in (
            ("s_tr", s_tr),
            ("s_va", s_va),
            ("t_un", t_un),
            ("t_tr", t_tr),
            ("t_va", t_va),
            ("t_te", t_te),
        )
    }
    W = {
        k: norm_w(d.weight.values).astype(np.float32)
        for k, d in (("s_tr", s_tr), ("s_va", s_va), ("t_tr", t_tr), ("t_va", t_va))
    }
    y_te, wt_va = t_te.y_ksi.values, target_weight(t_va)
    G = {k: T(v) for k, v in Z.items()}
    ys, yva_s = T(s_tr.y_ksi.values.astype(np.float32)), T(s_va.y_ksi.values.astype(np.float32))
    yt, yva_t = T(t_tr.y_ksi.values.astype(np.float32)), T(t_va.y_ksi.values.astype(np.float32))
    sets = eval_settings(tgt, t_te)
    BI = {wn: boot_idx(len(y_te), t_te.weight.values, mode) for wn, _, mode in sets}
    preds, p_val = {}, {}

    def record(method, seed, p_raw, extra=None):
        p_iso = recalibrate("isotonic", p_val[(method, seed)], t_va.y_ksi.values, wt_va)(p_raw)
        preds[(method, seed)] = (p_raw, p_iso)
        for calib, p in (("none", p_raw), ("T2 isotonic", p_iso)):
            for wn, w, _ in sets:
                est, ci = binary_metrics(y_te, p, w), {}
                if seed == 42:
                    bs = pd.DataFrame([binary_metrics(y_te[i], p[i], None) for i in BI[wn]])
                    ci = {k: (bs[k].quantile(0.025), bs[k].quantile(0.975)) for k in METRICS}
                for k in METRICS:
                    rows.append(
                        dict(
                            direction=D,
                            target=tgt,
                            method=method,
                            seed=seed,
                            calibration=calib,
                            weighting=wn,
                            metric=k,
                            value=est[k],
                            ci_lo=ci.get(k, (np.nan, np.nan))[0],
                            ci_hi=ci.get(k, (np.nan, np.nan))[1],
                            **(extra or {}),
                        )
                    )
        print(D, method, seed, f"{(time.time() - t0) / 60:.1f} min", flush=True)

    for seed in SEEDS:
        for mode, name in (("src", "mlp_src"), ("deep_coral", "deep_coral"), ("dann", "dann"), ("tgt", "mlp_tgt")):
            if mode == "tgt":
                net, ep, vl = ad.train_net(
                    mode, G["t_tr"], yt, T(W["t_tr"]), G["t_va"], yva_t, T(W["t_va"]), G["t_un"], seed
                )
            else:
                net, ep, vl = ad.train_net(
                    mode, G["s_tr"], ys, T(W["s_tr"]), G["s_va"], yva_s, T(W["s_va"]), G["t_un"], seed
                )
            p_val[(name, seed)] = ad.predict(net, G["t_va"])
            record(name, seed, ad.predict(net, G["t_te"]), dict(epochs=ep, sel_logloss=vl))
            if seed == 42 and mode != "tgt":
                align.append(
                    dict(
                        direction=D,
                        method=name,
                        space="hidden-128",
                        **ad.pad(ad.features(net, G["s_tr"]), ad.features(net, G["t_un"])),
                    )
                )
            del net
            torch.cuda.empty_cache()
        cm = ad.coral_map(Z["s_tr"].astype(np.float64), Z["t_un"].astype(np.float64))
        m = ad.fit_xgb(
            cm(Z["s_tr"]), s_tr.y_ksi.values, W["s_tr"], cm(Z["s_va"]), s_va.y_ksi.values, W["s_va"], seed
        )
        p_val[("coral_xgb", seed)] = m.predict_proba(Z["t_va"])[:, 1]
        record("coral_xgb", seed, m.predict_proba(Z["t_te"])[:, 1])
        m0 = ad.fit_xgb(Z["s_tr"], s_tr.y_ksi.values, W["s_tr"], Z["s_va"], s_va.y_ksi.values, W["s_va"], seed)
        p_val[("xgb_src", seed)] = m0.predict_proba(Z["t_va"])[:, 1]
        record("xgb_src", seed, m0.predict_proba(Z["t_te"])[:, 1])
        if seed == 42:
            for name, Zs in (("xgb_src", Z["s_tr"]), ("coral_xgb", cm(Z["s_tr"]))):
                align.append(dict(direction=D, method=name, space="input", **ad.pad(Zs, Z["t_un"])))

    for da, ref in (("deep_coral", "mlp_src"), ("dann", "mlp_src"), ("coral_xgb", "xgb_src")):
        for c, calib in enumerate(("none", "T2 isotonic")):
            pa, pr = preds[(da, 42)][c], preds[(ref, 42)][c]
            for wn, w, _ in sets:
                for k in ("auroc", "auprc", "ece"):
                    est = binary_metrics(y_te, pa, w)[k] - binary_metrics(y_te, pr, w)[k]
                    bs = [binary_metrics(y_te[i], pa[i])[k] - binary_metrics(y_te[i], pr[i])[k] for i in BI[wn]]
                    sd = [
                        binary_metrics(y_te, preds[(da, s)][c], w)[k]
                        - binary_metrics(y_te, preds[(ref, s)][c], w)[k]
                        for s in SEEDS
                    ]
                    paired.append(
                        dict(
                            direction=D,
                            method=da,
                            reference=ref,
                            calibration=calib,
                            weighting=wn,
                            metric=k,
                            diff=est,
                            ci_lo=float(np.quantile(bs, 0.025)),
                            ci_hi=float(np.quantile(bs, 0.975)),
                            seed_mean=float(np.mean(sd)),
                            seed_min=float(np.min(sd)),
                            seed_max=float(np.max(sd)),
                            seeds_positive=int(np.sum(np.array(sd) > 0)),
                        )
                    )


df = load(baac=True)
for pairs, tag in ((PAIRS_UK_US, ""), (PAIRS_FRANCE, "_baac")):
    rows, paired, align = [], [], []
    t0 = time.time()
    for src, tgt in pairs:
        run_pair(df, src, tgt, rows, paired, align, t0)
    pd.DataFrame(rows).to_csv(paths.result(f"posthoc_da{tag}_metrics.csv"), index=False)
    pd.DataFrame(paired).to_csv(paths.result(f"posthoc_da{tag}_paired.csv"), index=False)
    pd.DataFrame(align).to_csv(paths.result(f"posthoc_da{tag}_alignment.csv"), index=False)
    json.dump(
        dict(
            net=ad.NET,
            lambda_coral=ad.LAMBDA_CORAL,
            lambda_dann_max=ad.LAMBDA_DANN_MAX,
            seeds=SEEDS,
            boot=B_BOOT,
            device=ad.DEV,
            torch=torch.__version__,
            xgboost=xgb.__version__,
            minutes=(time.time() - t0) / 60,
        ),
        open(paths.result(f"posthoc_da{tag}_config.json"), "w"),
        indent=2,
    )
