"""Five baseline learners, recalibration maps and the prior-shift correction."""

import json

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import paths
from .schema import SEED

HP = {
    "xgboost": [dict(max_depth=4), dict(max_depth=6)],
    "lightgbm": [dict(num_leaves=15), dict(num_leaves=31)],
    "catboost": [dict(depth=4), dict(depth=6)],
}


def norm_w(w):
    w = np.asarray(w, float)
    return w / w.mean()


def class_w(y, w, k):
    tot = np.array([w[y == c].sum() for c in range(k)])
    return w.sum() / (k * np.maximum(tot, 1e-12))


class Imputer:
    """Median imputation, missing indicators, standardisation and cyclic hour; fitted on source training rows only."""

    def fit(self, X):
        self.med = X.median()
        self.na_cols = [c for c in X.columns if X[c].isna().any()]
        Z = self._raw(X)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        return self

    def _raw(self, X):
        Z = X.fillna(self.med)
        for c in self.na_cols:
            Z[c + "_na"] = X[c].isna().astype(float)
        if "hour" in X.columns:
            Z["hour_sin"] = np.sin(2 * np.pi * Z["hour"] / 24)
            Z["hour_cos"] = np.cos(2 * np.pi * Z["hour"] / 24)
        return Z

    def transform(self, X):
        Z = self._raw(X)
        for c in self.mu.index:
            if c not in Z:
                Z[c] = 0.0
        return (Z[self.mu.index] - self.mu) / self.sd


class Model:
    """name: majority, logreg, rf, xgboost, lightgbm or catboost; task: 'ksi' or '3class'.

    Sample weights are normalised to mean 1 inside fit. Logistic regression and random forest are class-balanced for both
    tasks, boosting only for the three-class task. Without a validation set, early stopping uses a stratified holdout.
    """

    def __init__(self, name, task, seed=SEED, es_frac=0.15):
        self.name, self.task, self.seed, self.es_frac = name, task, seed, es_frac

    def _target(self, d):
        return d["y_ksi"].values if self.task == "ksi" else d["y_3class"].values - 1

    def fit(self, Xtr, dtr, Xva=None, dva=None, w_tr=None, w_va=None):
        y = self._target(dtr)
        k = 2 if self.task == "ksi" else 3
        w = np.ones(len(y)) if w_tr is None else norm_w(w_tr)
        self.k = k
        if self.name == "majority":
            self.prior = np.array([w[y == c].sum() for c in range(k)]) / w.sum()
            return self
        cw = class_w(y, w, k)
        sw = w * cw[y] if (self.task == "3class" or self.name in ("logreg", "rf")) else w
        if self.name == "logreg":
            self.imp = Imputer().fit(Xtr)
            self.m = LogisticRegression(max_iter=500, C=1.0).fit(self.imp.transform(Xtr), y, sample_weight=sw)
        elif self.name == "rf":
            self.imp = Imputer().fit(Xtr)
            self.m = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=self.seed
            )
            self.m.fit(self.imp.transform(Xtr), y, sample_weight=sw)
        else:
            if Xva is None:
                itr, iva = train_test_split(
                    np.arange(len(y)), test_size=self.es_frac, stratify=y, random_state=self.seed
                )
                yva = y[iva]
                swva = (w[iva] * cw[yva]) if self.task == "3class" else w[iva]
                Xva = Xtr.iloc[iva]
                Xtr, y, sw = Xtr.iloc[itr], y[itr], sw[itr]
            else:
                yva = self._target(dva)
                wv = np.ones(len(yva)) if w_va is None else norm_w(w_va)
                swva = wv * cw[yva] if self.task == "3class" else wv
            best = None
            for hp in HP[self.name]:
                m = self._make(hp, k)
                if self.name == "xgboost":
                    m.fit(
                        Xtr,
                        y,
                        sample_weight=sw,
                        eval_set=[(Xva, yva)],
                        sample_weight_eval_set=[swva],
                        verbose=False,
                    )
                    score = m.best_score
                elif self.name == "catboost":
                    from catboost import Pool

                    m.fit(Pool(Xtr, y, weight=sw), eval_set=Pool(Xva, yva, weight=swva), verbose=False)
                    score = m.get_best_score()["validation"]["Logloss" if k == 2 else "MultiClass"]
                else:
                    m.fit(
                        Xtr,
                        y,
                        sample_weight=sw,
                        eval_set=[(Xva, yva)],
                        eval_sample_weight=[swva],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    score = list(m.best_score_["valid_0"].values())[0]
                if best is None or score < best[0]:
                    best = (score, m, hp)
            self.m, self.hp = best[1], best[2]
        return self

    def _make(self, hp, k):
        if self.name == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                early_stopping_rounds=50,
                random_state=self.seed,
                n_jobs=12,
                objective="binary:logistic" if k == 2 else "multi:softprob",
                eval_metric="logloss" if k == 2 else "mlogloss",
                **hp,
            )
        if self.name == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                od_type="Iter",
                od_wait=50,
                use_best_model=True,
                random_seed=self.seed,
                thread_count=-1,
                loss_function="Logloss" if k == 2 else "MultiClass",
                **hp,
            )
        return lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=12,
            verbose=-1,
            **hp,
        )

    def predict_proba(self, X):
        if self.name == "majority":
            return np.tile(self.prior, (len(X), 1))
        if self.name in ("logreg", "rf"):
            return self.m.predict_proba(self.imp.transform(X))
        return self.m.predict_proba(X)

    def p_ksi(self, X):
        P = self.predict_proba(X)
        return P[:, 1] if self.task == "ksi" else P[:, :2].sum(1)


def recalibrate(kind, p_val, y_val, w_val=None):
    """Fit on validation rows; returns a map p -> p'. kind: none, platt or isotonic."""
    p_val = np.clip(p_val, 1e-6, 1 - 1e-6)
    if kind == "none":
        return lambda p: np.asarray(p)
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
            p_val, y_val, sample_weight=w_val
        )
        return lambda p: iso.predict(np.asarray(p))
    z = np.log(p_val / (1 - p_val)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=500).fit(z, y_val, sample_weight=w_val)

    def apply(p):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return lr.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]

    return apply


def prior_shift(p, pi_s, pi_t):
    """Bayes prior correction in logit space; monotone, so ranking metrics do not change."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    pi_t = float(np.clip(pi_t, 1e-3, 1 - 1e-3))
    z = np.log(p / (1 - p)) + np.log(pi_t / pi_s) - np.log((1 - pi_t) / (1 - pi_s))
    return 1 / (1 + np.exp(-z))


def em_prior(p, pi_s, w=None, it=500):
    """Saerens et al. (2002), bounded to [1e-3, 1-1e-3]. Reaching the bound is reported as degenerate."""
    w = np.ones(len(p)) if w is None else w
    pi = pi_s
    for _ in range(it):
        new = float(np.clip(np.average(prior_shift(p, pi_s, pi), weights=w), 1e-3, 1 - 1e-3))
        if abs(new - pi) < 1e-7:
            break
        pi = new
    return pi


def primary():
    return json.load(open(paths.result("primary_model.json")))["primary"]


def load_model(name):
    return joblib.load(paths.model(name))


def save_model(m, name):
    joblib.dump(m, paths.model(name))
