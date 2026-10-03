import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import optuna

from src.features import build_features_temporal, CAT_COLS, NUM_COLS
from src.validation import PurgedTimeSeriesSplit
from src.metrics import gini_score
from src.leakage_audit import audit_leakage

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    all_data = pd.concat([train, test], ignore_index=True)
    all_features = build_features_temporal(all_data)
    
    df_train_feat = all_features[all_features["mes"] <= 202611].copy().reset_index(drop=True)
    df_test_feat = all_features[all_features["mes"] == 202612].copy().reset_index(drop=True)
    return df_train_feat, df_test_feat

def optimize_lgbm(df, base_cols, cat_features):
    print("Running Optuna tuning for LightGBM on folds 1-3...")
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "verbosity": -1,
            "random_state": 42
        }
        
        cv = PurgedTimeSeriesSplit(val_months=[202606, 202607, 202608])
        ginis = []
        for train_idx, valid_idx in cv.split(df):
            X_train, y_train = df.iloc[train_idx][base_cols].copy(), df.iloc[train_idx]["objetivo"].values
            X_valid, y_valid = df.iloc[valid_idx][base_cols].copy(), df.iloc[valid_idx]["objetivo"].values
            
            for c in cat_features:
                X_train[c] = X_train[c].astype("category")
                X_valid[c] = X_valid[c].astype("category")
            
            pos = np.sum(y_train)
            spw = (len(y_train) - pos) / pos if pos > 0 else 1.0
            params["scale_pos_weight"] = spw
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
            
            model = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X_valid)
            ginis.append(gini_score(y_valid, preds))
        return np.mean(ginis)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def train_models(X_train, y_train, X_valid, y_valid, X_test, cat_features, lgbm_best_params):
    pos = np.sum(y_train)
    spw = (len(y_train) - pos) / pos if pos > 0 else 1.0
    
    # LGBM
    lgbm_params = lgbm_best_params.copy()
    lgbm_params.update({"objective": "binary", "metric": "auc", "random_state": 42, "verbosity": -1, "scale_pos_weight": spw})
    
    # Cast categories correctly for LGBM
    X_train_lgb = X_train.copy()
    X_valid_lgb = X_valid.copy()
    X_test_lgb = X_test.copy()
    for c in cat_features:
        X_train_lgb[c] = X_train_lgb[c].astype("category")
        X_valid_lgb[c] = X_valid_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")
        
    dlgb_train = lgb.Dataset(X_train_lgb, label=y_train)
    dlgb_valid = lgb.Dataset(X_valid_lgb, label=y_valid, reference=dlgb_train)
    lgbm_model = lgb.train(lgbm_params, dlgb_train, valid_sets=[dlgb_valid], num_boost_round=2000, callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgbm = lgbm_model.predict(X_valid_lgb)
    t_lgbm = lgbm_model.predict(X_test_lgb)
    
    # XGB
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc", "learning_rate": lgbm_params.get("learning_rate", 0.02),
        "max_depth": lgbm_params.get("max_depth", 6), "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": spw, "random_state": 42
    }
    dxgb_train = xgb.DMatrix(X_train_lgb, label=y_train, enable_categorical=True)
    dxgb_valid = xgb.DMatrix(X_valid_lgb, label=y_valid, enable_categorical=True)
    dxgb_test = xgb.DMatrix(X_test_lgb, enable_categorical=True)
    xgb_model = xgb.train(xgb_params, dxgb_train, num_boost_round=2000, evals=[(dxgb_valid, "valid")], early_stopping_rounds=50, verbose_eval=False)
    p_xgb = xgb_model.predict(dxgb_valid)
    t_xgb = xgb_model.predict(dxgb_test)
    
    # CatBoost
    X_train_cb, X_valid_cb, X_test_cb = X_train.copy(), X_valid.copy(), X_test.copy()
    for c in cat_features:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_valid_cb[c] = X_valid_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)
        
    train_pool = Pool(X_train_cb, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid_cb, y_valid, cat_features=cat_features)
    test_pool = Pool(X_test_cb, cat_features=cat_features)
    
    cb_model = CatBoostClassifier(iterations=2000, learning_rate=lgbm_params.get("learning_rate", 0.02), depth=lgbm_params.get("max_depth", 6), eval_metric="AUC", random_seed=42, scale_pos_weight=spw, verbose=0, early_stopping_rounds=50)
    cb_model.fit(train_pool, eval_set=valid_pool)
    p_cat = cb_model.predict_proba(valid_pool)[:, 1]
    t_cat = cb_model.predict_proba(test_pool)[:, 1]
    
    return p_lgbm, p_xgb, p_cat, t_lgbm, t_xgb, t_cat

def main():
    print("=========================================================")
    print("STARTING FULL PIPELINE")
    print("=========================================================")
    
    df_train, df_test = load_data()
    base_cols = [c for c in df_train.columns if c not in ["id_cliente", "mes", "objetivo"]]
    cat_features = [c for c in CAT_COLS if c in base_cols]
    
    print("\nRunning Leakage Audit...")
    is_clean = audit_leakage(df_train, base_cols)
    if not is_clean:
        print("LEAKAGE FAILED! Stopping.")
        return
        
    lgbm_best_params = optimize_lgbm(df_train, base_cols, cat_features)
    print(f"Optuna Best Params: {lgbm_best_params}")
    
    print("\nRunning 6-Fold Final Evaluation...")
    cv = PurgedTimeSeriesSplit(val_months=[202606, 202607, 202608, 202609, 202610, 202611])
    
    fold_metrics = []
    t_lgbm_folds, t_xgb_folds, t_cat_folds, t_blend_folds = [], [], [], []
    
    for fold, (train_idx, valid_idx) in enumerate(cv.split(df_train)):
        print(f"\n--- FOLD {fold+1} ---")
        X_train, y_train = df_train.iloc[train_idx][base_cols].copy(), df_train.iloc[train_idx]["objetivo"].values
        X_valid, y_valid = df_train.iloc[valid_idx][base_cols].copy(), df_train.iloc[valid_idx]["objetivo"].values
        X_test = df_test[base_cols].copy()
                
        p_lgbm, p_xgb, p_cat, t_lgbm, t_xgb, t_cat = train_models(X_train, y_train, X_valid, y_valid, X_test, cat_features, lgbm_best_params)
        
        # Rank Average Blend
        # LightGBM receives the bulk of the weight since it was optimized via Optuna
        blend_valid = rankdata(p_lgbm) * 0.85 + rankdata(p_cat) * 0.10 + rankdata(p_xgb) * 0.05
        blend_test = rankdata(t_lgbm) * 0.85 + rankdata(t_cat) * 0.10 + rankdata(t_xgb) * 0.05
        
        g_lgbm = gini_score(y_valid, p_lgbm)
        g_xgb = gini_score(y_valid, p_xgb)
        g_cat = gini_score(y_valid, p_cat)
        g_blend = gini_score(y_valid, blend_valid)
        
        print(f"LGBM: {g_lgbm:.5f} | XGB: {g_xgb:.5f} | Cat: {g_cat:.5f} | Blend: {g_blend:.5f}")
        
        fold_metrics.append({"lgbm": g_lgbm, "xgb": g_xgb, "cat": g_cat, "blend": g_blend})
        
        t_lgbm_folds.append(t_lgbm)
        t_xgb_folds.append(t_xgb)
        t_cat_folds.append(t_cat)
        t_blend_folds.append(blend_test)

    print("\n=========================================================")
    print("FINAL GINI OPTIMIZATION REPORT")
    print("=========================================================")
    
    mean_blend = np.mean([f["blend"] for f in fold_metrics])
    std_blend = np.std([f["blend"] for f in fold_metrics])
    
    print("Benchmark:")
    print("0.258813")
    print("\nMean Fold Gini:")
    print(f"LGBM: {np.mean([f['lgbm'] for f in fold_metrics]):.6f}")
    print(f"XGB:  {np.mean([f['xgb'] for f in fold_metrics]):.6f}")
    print(f"Cat:  {np.mean([f['cat'] for f in fold_metrics]):.6f}")
    print(f"Rank Blend: {mean_blend:.6f}")
    
    print("\nStd (Blend):")
    print(f"{std_blend:.6f}")
    print("\nMinimum Fold (Blend):")
    print(f"{np.min([f['blend'] for f in fold_metrics]):.6f}")
    print("\nMaximum Fold (Blend):")
    print(f"{np.max([f['blend'] for f in fold_metrics]):.6f}")
    print("\nNovember / last temporal fold (Blend):")
    print(f"{fold_metrics[-1]['blend']:.6f}")
    
    print("\nAbsolute Improvement:")
    print(f"{mean_blend - 0.258813:.6f}")
    
    print("\nTarget > 0.265:")
    print("YES" if mean_blend > 0.265 else "NO")
    print("\nTarget >= 0.275:")
    print("YES" if mean_blend >= 0.275 else "NO")
    print("\nLeakage Audit:\nPASSED")
    
    os.makedirs("submissions", exist_ok=True)
    sub = pd.DataFrame({
        "id_cliente": df_test["id_cliente"],
        "prediccion": np.mean(t_blend_folds, axis=0)
    })
    sub.to_csv("submissions/sub_final_ensemble.csv", index=False)
    print("\nFinal Submission:\nsubmissions/sub_final_ensemble.csv")
    print("=========================================================")
    
if __name__ == "__main__":
    main()
