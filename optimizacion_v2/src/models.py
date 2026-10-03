"""Entrenamiento LightGBM / CatBoost / XGBoost + Optuna."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .metrics import gini_score
from .validation import PurgedTimeSeriesSplit, TUNING_MONTHS

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42


def scale_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    return neg / pos if pos > 0 else 1.0


def _lgb_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def _cb_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype(str).fillna("MISSING")
    return out


def train_lgbm(
    X_tr,
    y_tr,
    X_va,
    y_va,
    cat_cols,
    params=None,
    num_boost_round=1500,
    early_stopping_rounds=50,
):
    params = dict(params or {})
    params.setdefault("objective", "binary")
    params.setdefault("metric", "auc")
    params.setdefault("verbosity", -1)
    params.setdefault("random_state", SEED)
    params.setdefault("seed", SEED)
    params.setdefault("n_jobs", -1)
    use_spw = not params.pop("scale_pos_weight_off", False)
    if use_spw:
        params["scale_pos_weight"] = scale_pos_weight(y_tr)
    else:
        params.pop("scale_pos_weight", None)
    Xtr = _lgb_frame(X_tr, cat_cols)
    dtr = lgb.Dataset(Xtr, y_tr, categorical_feature=cat_cols, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    Xva = None
    if X_va is not None and y_va is not None:
        Xva = _lgb_frame(X_va, cat_cols)
        dva = lgb.Dataset(Xva, y_va, reference=dtr, categorical_feature=cat_cols, free_raw_data=False)
        valid_sets = [dva]
        if early_stopping_rounds:
            callbacks = [lgb.early_stopping(int(early_stopping_rounds), verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    it = int(model.best_iteration or num_boost_round)
    pred = model.predict(Xva, num_iteration=it) if Xva is not None else None
    return model, pred, it


def predict_lgbm(model, X, cat_cols):
    return model.predict(_lgb_frame(X, cat_cols), num_iteration=model.best_iteration)


def train_xgb(X_tr, y_tr, X_va, y_va, cat_cols, params=None, num_boost_round=1500):
    params = dict(params or {})
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("tree_method", "hist")
    params.setdefault("enable_categorical", True)
    params.setdefault("random_state", SEED)
    params.setdefault("seed", SEED)
    params["scale_pos_weight"] = scale_pos_weight(y_tr)
    Xtr = _lgb_frame(X_tr, cat_cols)
    Xva = _lgb_frame(X_va, cat_cols)
    dtr = xgb.DMatrix(Xtr, label=y_tr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, label=y_va, enable_categorical=True)
    model = xgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "valid")],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        verbose_eval=False,
    )
    best = getattr(model, "best_iteration", None)
    if best is None:
        best = num_boost_round
    else:
        best = int(best) + 1
    pred = model.predict(dva)
    return model, pred, best


def predict_xgb(model, X, cat_cols):
    return model.predict(xgb.DMatrix(_lgb_frame(X, cat_cols), enable_categorical=True))


def train_catboost(X_tr, y_tr, X_va, y_va, cat_cols, params=None, iterations=1500):
    params = dict(params or {})
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "AUC")
    params.setdefault("random_seed", SEED)
    params.setdefault("verbose", 0)
    params.setdefault("early_stopping_rounds", 50)
    params.setdefault("iterations", iterations)
    params["scale_pos_weight"] = scale_pos_weight(y_tr)
    Xtr = _cb_frame(X_tr, cat_cols)
    Xva = _cb_frame(X_va, cat_cols)
    tr_pool = Pool(Xtr, y_tr, cat_features=cat_cols)
    va_pool = Pool(Xva, y_va, cat_features=cat_cols)
    model = CatBoostClassifier(**params)
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    pred = model.predict_proba(va_pool)[:, 1]
    return model, pred, int(model.get_best_iteration() or params["iterations"])


def predict_catboost(model, X, cat_cols):
    pool = Pool(_cb_frame(X, cat_cols), cat_features=cat_cols)
    return model.predict_proba(pool)[:, 1]


def _cv_mean_gini(df, feature_cols, cat_cols, val_months, trainer, params):
    splitter = PurgedTimeSeriesSplit(val_months=val_months)
    scores = []
    for fold in splitter.split(df):
        X_tr = df.loc[fold.train_idx, feature_cols]
        y_tr = df.loc[fold.train_idx, "objetivo"].to_numpy()
        X_va = df.loc[fold.val_idx, feature_cols]
        y_va = df.loc[fold.val_idx, "objetivo"].to_numpy()
        _, pred, _ = trainer(X_tr, y_tr, X_va, y_va, cat_cols, params)
        scores.append(gini_score(y_va, pred))
    return float(np.nanmean(scores))


def tune_lgbm(df, feature_cols, cat_cols, n_trials=30, out_dir: Path | None = None) -> dict:
    trials_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
        }
        use_spw = trial.suggest_categorical("use_spw", [True, False])
        pfit = dict(params)
        if not use_spw:
            pfit["scale_pos_weight_off"] = True
        score = _cv_mean_gini(
            df,
            feature_cols,
            cat_cols,
            TUNING_MONTHS,
            lambda Xtr, ytr, Xva, yva, cats, p, pf=pfit: train_lgbm(
                Xtr, ytr, Xva, yva, cats, params=pf, num_boost_round=500
            ),
            pfit,
        )
        trials_rows.append({"model": "lgbm", "trial": trial.number, "gini": score, **trial.params})
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.pop("use_spw", None)
    if out_dir:
        _dump_params(out_dir / "best_params_lgbm.json", best)
        _append_trials(out_dir / "optuna_trials.csv", trials_rows)
    return best


def tune_catboost(df, feature_cols, cat_cols, n_trials=30, out_dir: Path | None = None) -> dict:
    trials_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 8.0),
            "verbose": 0,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": SEED,
            "early_stopping_rounds": 40,
        }
        score = _cv_mean_gini(
            df,
            feature_cols,
            cat_cols,
            TUNING_MONTHS,
            lambda Xtr, ytr, Xva, yva, cats, p, pf=params: train_catboost(
                Xtr, ytr, Xva, yva, cats, params=pf, iterations=400
            ),
            params,
        )
        trials_rows.append({"model": "catboost", "trial": trial.number, "gini": score, **trial.params})
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    if out_dir:
        _dump_params(out_dir / "best_params_catboost.json", best)
        _append_trials(out_dir / "optuna_trials.csv", trials_rows)
    return best


def tune_xgb(df, feature_cols, cat_cols, n_trials=30, out_dir: Path | None = None) -> dict:
    trials_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        }
        score = _cv_mean_gini(
            df,
            feature_cols,
            cat_cols,
            TUNING_MONTHS,
            lambda Xtr, ytr, Xva, yva, cats, p, pf=params: train_xgb(
                Xtr, ytr, Xva, yva, cats, params=pf, num_boost_round=500
            ),
            params,
        )
        trials_rows.append({"model": "xgb", "trial": trial.number, "gini": score, **trial.params})
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    if out_dir:
        _dump_params(out_dir / "best_params_xgb.json", best)
        _append_trials(out_dir / "optuna_trials.csv", trials_rows)
    return best


def importance_lgbm(model, feature_cols) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain")
    return pd.DataFrame({"feature": feature_cols, "gain": gain}).sort_values("gain", ascending=False)


def importance_catboost(model, feature_cols) -> pd.DataFrame:
    gain = model.get_feature_importance()
    return pd.DataFrame({"feature": feature_cols, "gain": gain}).sort_values("gain", ascending=False)


def _dump_params(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")


def _append_trials(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


MONO_MAP = {"dias_ultima_transaccion": -1, "numero_productos": 1, "banda_ord": 1}
SEEDS_GBM = [42, 777, 2024]


def monotone_params(feats: list[str]) -> dict:
    mono = [MONO_MAP.get(f, 0) for f in feats]
    return {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 16,
        "max_depth": -1,
        "min_child_samples": 400,
        "reg_lambda": 20,
        "reg_alpha": 0,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "monotone_constraints": mono,
        "monotone_constraints_method": "advanced",
        "scale_pos_weight_off": True,
        "n_jobs": -1,
    }


def gbm_feature_list(feats: list[str]) -> list[str]:
    out = [("banda_ord" if f == "banda_riesgo" else f) for f in feats]
    seen = []
    for f in out:
        if f not in seen:
            seen.append(f)
    return [f for f in seen if f != "banda_riesgo"]


def _band_design(d: pd.DataFrame, spl=None):
    dur = d["meses_observados"] if "meses_observados" in d.columns else d["duracion"]
    cont = d[["dias_ultima_transaccion", "ratio_deuda_ingresos"]].to_numpy()
    if spl is None:
        spl = SplineTransformer(n_knots=6, degree=3).fit(cont)
    S = pd.DataFrame(spl.transform(cont), index=d.index).add_prefix("s")
    npd = pd.get_dummies(d["numero_productos"], prefix="np", drop_first=True).astype(float)
    b = d[["activo_movil", "tiene_tarjeta_credito"]].astype(float)
    X = pd.concat(
        [
            npd,
            pd.get_dummies(d["canal_adquisicion"].astype(str), prefix="cn", drop_first=True).astype(float),
            pd.get_dummies(dur.clip(upper=8), prefix="du", drop_first=True).astype(float),
            S,
            b,
            (b["activo_movil"] * b["tiene_tarjeta_credito"]).rename("am_x_tc").to_frame(),
            S.mul(b["activo_movil"], axis=0).add_suffix("_xam"),
            npd.mul(b["activo_movil"], axis=0).add_suffix("_xam"),
            npd.mul(b["tiene_tarjeta_credito"], axis=0).add_suffix("_xtc"),
        ],
        axis=1,
    )
    return X, spl


def predict_band_logistic(trn: pd.DataFrame, obj: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(obj), dtype=float)
    bands = trn["banda_riesgo"].astype(str).unique()
    for banda in bands:
        mt = (trn["banda_riesgo"].astype(str) == banda).to_numpy()
        mo = (obj["banda_riesgo"].astype(str) == banda).to_numpy()
        if mo.sum() == 0 or mt.sum() < 30:
            continue
        Xt, spl = _band_design(trn.loc[mt])
        Xo, _ = _band_design(obj.loc[mo], spl)
        Xo = Xo.reindex(columns=Xt.columns, fill_value=0.0)
        sc = StandardScaler().fit(Xt)
        y = trn.loc[mt, "objetivo"].to_numpy()
        if np.unique(y).size < 2:
            out[mo] = float(y.mean()) if len(y) else 0.5
            continue
        m = LogisticRegression(max_iter=5000, C=0.05)
        m.fit(sc.transform(Xt), y)
        out[mo] = m.predict_proba(sc.transform(Xo))[:, 1]
    return out


def pred_lgbm_seeds(X_tr, y_tr, X_obj, cat_cols, params, num_boost_round, seeds=None) -> np.ndarray:
    seeds = seeds or SEEDS_GBM
    preds = []
    for s in seeds:
        p = dict(params)
        p["seed"] = s
        p["random_state"] = s
        p["bagging_seed"] = s
        p["feature_fraction_seed"] = s
        model, _, _ = train_lgbm(
            X_tr, y_tr, None, None, cat_cols, p, num_boost_round=num_boost_round, early_stopping_rounds=0
        )
        preds.append(predict_lgbm(model, X_obj, cat_cols))
    return np.mean(preds, axis=0)


def select_boost_rounds(df, feats, cats, params, max_rounds=180) -> int:
    splitter = PurgedTimeSeriesSplit(val_months=TUNING_MONTHS)
    curves = []
    for fold in splitter.split(df):
        Xtr = _lgb_frame(df.loc[fold.train_idx, feats], cats)
        Xva = _lgb_frame(df.loc[fold.val_idx, feats], cats)
        ytr = df.loc[fold.train_idx, "objetivo"].to_numpy()
        yva = df.loc[fold.val_idx, "objetivo"].to_numpy()
        p = dict(params)
        p.setdefault("objective", "binary")
        p.setdefault("metric", "auc")
        p.setdefault("verbosity", -1)
        p.pop("scale_pos_weight_off", None)
        p.pop("scale_pos_weight", None)
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cats, free_raw_data=False)
        ev = {}
        lgb.train(
            p,
            dtr,
            num_boost_round=max_rounds,
            valid_sets=[dva],
            valid_names=["v"],
            callbacks=[lgb.record_evaluation(ev), lgb.log_evaluation(0)],
        )
        curves.append(ev["v"]["auc"])
    mean_auc = np.mean(np.asarray(curves), axis=0)
    return max(20, int(np.argmax(mean_auc)) + 1)
