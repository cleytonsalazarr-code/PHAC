"""
optuna_tune.py
--------------
Búsqueda de hiperparámetros para el modelo longitudinal
usando PurgedTimeSeriesSplit (0 leakage).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna

from src.validation import PurgedTimeSeriesSplit
from src.features import encode_base_features, build_longitudinal_features, get_feature_cols_longitudinal, prepare_X_y

def load_and_prepare_data():
    data_dir = Path("data")
    df_train = pd.read_csv(data_dir / "train.csv")
    df_test = pd.read_csv(data_dir / "test.csv")
    
    df_train = encode_base_features(df_train)
    df_test = encode_base_features(df_test)
    
    df_train_long, df_test_long = build_longitudinal_features(df_train, df_test)
    features = get_feature_cols_longitudinal()
    X, y = prepare_X_y(df_train_long, features)
    # We need the full dataframe for PurgedTimeSeriesSplit to extract mes and objetivo
    return X, y, df_train_long

def objective(trial, X, y, df_train_long):
    # Optuna search space
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": 6.0, # Approximate imbalance
        "random_state": 42,
        "verbose": -1,
        "n_estimators": 500
    }
    
    # Validamos sobre septiembre, octubre, noviembre para ser robustos
    cv = PurgedTimeSeriesSplit(val_months=[202609, 202610, 202611])
    
    ginis = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df_train_long)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        
        # Calculate Gini
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_preds)
        gini = 2 * auc - 1
        ginis.append(gini)
    
    # Queremos maximizar el Gini promedio OOT
    return np.mean(ginis)

def run_optuna():
    print("[*] Cargando datos y creando features rodantes (Rolling)...")
    X, y, df_train_long = load_and_prepare_data()
    
    print("[*] Iniciando búsqueda con Optuna (esto tomará varios minutos)...")
    study = optuna.create_study(direction="maximize")
    
    # Pasar los datos al objective function
    func = lambda trial: objective(trial, X, y, df_train_long)
    study.optimize(func, n_trials=15) # 15 pruebas de ejemplo
    
    print("\n======================================")
    print("MEJORES HIPERPARÁMETROS OBTENIDOS:")
    print("======================================")
    print(f"Mejor Gini promedio: {study.best_value:.4f}")
    print("Parámetros:")
    for k, v in study.best_params.items():
        print(f"  '{k}': {v},")

if __name__ == "__main__":
    run_optuna()
