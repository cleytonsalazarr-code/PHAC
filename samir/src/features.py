import numpy as np
import pandas as pd

CAT_COLS = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
BOOL_COLS = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
NUM_COLS = [
    "edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", 
    "numero_productos", "saldo_promedio", "dias_ultima_transaccion", 
    "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias", 
    "distancia_sucursal_km", "dia_preferido_pago", "dias_ultima_interaccion"
]

def build_features_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Base transform
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(float)
            
    # Convert categorical to string for CatBoost and category for LGBM/XGB
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("MISSING").astype(str)
            
    # 2. Sort to compute temporal features (Strictly shift(1) per client)
    df = df.sort_values(["id_cliente", "mes"]).reset_index(drop=True)
    
    # 3. Lag features (only past information)
    lag_cols = ["saldo_promedio", "numero_productos", "visitas_web_ultimos_90_dias", "ratio_deuda_ingresos", "dias_ultima_transaccion"]
    
    for c in lag_cols:
        df[f"{c}_lag_1"] = df.groupby("id_cliente")[c].shift(1)
        df[f"{c}_lag_2"] = df.groupby("id_cliente")[c].shift(2)
        
        # Differences
        df[f"{c}_diff_1"] = df[c] - df[f"{c}_lag_1"]
        df[f"{c}_diff_2"] = df[f"{c}_lag_1"] - df[f"{c}_lag_2"]
        df[f"{c}_diff_1_rel"] = df[f"{c}_diff_1"] / (df[f"{c}_lag_1"].replace(0, np.nan).abs() + 1e-5)
    
    # 4. Aggregations (Expanding mean of past)
    # MUST shift(1) to avoid data leakage (don't include current month in past mean)
    df["saldo_promedio_hist_mean"] = df.groupby("id_cliente")["saldo_promedio"].transform(lambda x: x.shift(1).expanding().mean())
    df["saldo_promedio_hist_std"] = df.groupby("id_cliente")["saldo_promedio"].transform(lambda x: x.shift(1).expanding().std())
    df["saldo_promedio_hist_max"] = df.groupby("id_cliente")["saldo_promedio"].transform(lambda x: x.shift(1).expanding().max())
    df["saldo_promedio_hist_min"] = df.groupby("id_cliente")["saldo_promedio"].transform(lambda x: x.shift(1).expanding().min())
    
    # Ratios (Cross-sectional)
    df["saldo_por_producto"] = df["saldo_promedio"] / (df["numero_productos"] + 1)
    df["deuda_total_estimada"] = df["ratio_deuda_ingresos"] * df["ingresos"]
    df["interaccion_frecuente"] = (df["visitas_web_ultimos_90_dias"] > 10).astype(int)
    
    # Ratios over lags
    df["saldo_to_hist_mean"] = df["saldo_promedio"] / (df["saldo_promedio_hist_mean"].replace(0, np.nan) + 1e-5)
    
    df["visitas_hist_mean"] = df.groupby("id_cliente")["visitas_web_ultimos_90_dias"].transform(lambda x: x.shift(1).expanding().mean())
    df["visitas_hist_sum"] = df.groupby("id_cliente")["visitas_web_ultimos_90_dias"].transform(lambda x: x.shift(1).expanding().sum())

    # 5. Month index
    if "mes" in df.columns:
        df["mes_num"] = df["mes"] - 202601
        
    return df
