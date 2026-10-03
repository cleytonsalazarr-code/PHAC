from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from .metrics import lgbm_gini_eval, gini_score, auc_score

def get_scale_pos_weight(y_train: np.ndarray) -> float:
    pos = np.sum(y_train)
    neg = len(y_train) - pos
    return neg / pos if pos > 0 else 1.0

# --- LightGBM ---
def train_lgbm(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray, cat_features=None) -> tuple[lgb.Booster, dict]:
    spw = get_scale_pos_weight(y_train)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.015,
        "max_depth": 9,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
        "scale_pos_weight": spw
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_features, free_raw_data=False)
    
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    booster = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[valid_data], valid_names=["valid"], feval=lgbm_gini_eval, callbacks=callbacks)
    
    preds = booster.predict(X_valid)
    return booster, {"gini": gini_score(y_valid, preds), "auc": auc_score(y_valid, preds)}

# --- XGBoost ---
def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray, cat_features=None) -> tuple[xgb.Booster, dict]:
    spw = get_scale_pos_weight(y_train)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.015,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "random_state": 42,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    
    booster = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dvalid, "valid")], early_stopping_rounds=50, verbose_eval=False)
    
    preds = booster.predict(dvalid)
    return booster, {"gini": gini_score(y_valid, preds), "auc": auc_score(y_valid, preds)}

# --- CatBoost ---
def train_catboost(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray, cat_features=None) -> tuple[CatBoostClassifier, dict]:
    spw = get_scale_pos_weight(y_train)
    
    if cat_features is None:
        cat_features = []
        
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.015,
        depth=7,
        eval_metric="AUC",
        random_seed=42,
        scale_pos_weight=spw,
        verbose=0,
        early_stopping_rounds=50
    )
    
    model.fit(train_pool, eval_set=valid_pool)
    preds = model.predict_proba(valid_pool)[:, 1]
    
    return model, {"gini": gini_score(y_valid, preds), "auc": auc_score(y_valid, preds)}
