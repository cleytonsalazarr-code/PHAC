import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# 1. Load data
df_train = pd.read_csv("data/train.csv")

# 2. Basic features
CAT_COLS = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
BOOL_COLS = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
NUM_COLS = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos", "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago", "dias_ultima_interaccion"]

df_train["mes_num"] = df_train["mes"] - df_train["mes"].min()
for col in CAT_COLS:
    df_train[f"{col}_enc"] = df_train[col].astype("category").cat.codes.astype(np.int16)
for col in BOOL_COLS:
    df_train[col] = df_train[col].astype(int)

feature_cols = NUM_COLS + [f"{c}_enc" for c in CAT_COLS] + BOOL_COLS + ["mes_num"]
cat_cols = [f"{c}_enc" for c in CAT_COLS]

# 3. Validation
val_months = [202606, 202607, 202608, 202609, 202610, 202611]
ginis = []

for val_month in val_months:
    train_mask = df_train["mes"] < val_month
    val_mask = df_train["mes"] == val_month
    
    X_train = df_train.loc[train_mask, feature_cols]
    y_train = df_train.loc[train_mask, "objetivo"].values.astype(np.int8)
    
    X_val = df_train.loc[val_mask, feature_cols]
    y_val = df_train.loc[val_mask, "objetivo"].values.astype(np.int8)
    
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "scale_pos_weight": spw
    }
    
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_cols)
    
    model = lgb.train(
        params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    gini = 2 * auc - 1
    ginis.append(gini)
    print(f"Val Month: {val_month}, Gini: {gini:.6f}")

print(f"Mean Gini: {np.mean(ginis):.6f}, Std: {np.std(ginis):.6f}")
