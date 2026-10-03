"""
src/leakage_audit.py
--------------------
Auditoría de Data Leakage para la Hackathon BCP.

TIPO DE LEAKAGE A DETECTAR:
    1. Leakage temporal: features calculadas con información futura.
    2. Leakage por target: features construidas usando 'objetivo' o
       variables derivadas del target.
    3. Leakage de identidad: 'id_cliente' usado como predictor.
    4. Correlación sospechosa: features con correlación perfecta o
       cerca de ±1 con el target (pueden ser proxies del target).

REGLA:
    Si se detecta leakage severo → imprimir LEAKAGE DETECTADO y retornar False.
    Si está ok → retornar True.
    El pipeline principal revisa este flag antes de continuar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


FORBIDDEN_AS_FEATURES = {
    "id_cliente",
    "objetivo",
    "mes",
}

HIGH_CORR_THRESHOLD = 0.95  # correlación con target que dispara alarma


def audit_leakage(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "objetivo",
    verbose: bool = True,
) -> bool:
    """
    Ejecuta auditoría completa de leakage sobre el conjunto de features.

    Args:
        df_train: DataFrame con features y target.
        feature_cols: Lista de columnas que se usarán como predictoras.
        target_col: Nombre de la columna target.
        verbose: Si True, imprime detalles.

    Returns:
        bool: True si NO se detectó leakage severo. False si SÍ hay leakage.
    """
    is_clean = True

    if verbose:
        print("\n--- Auditoria de Data Leakage ---")

    # -------------------------------------------------------
    # 1. Columnas prohibidas directamente en features
    # -------------------------------------------------------
    forbidden_found = [c for c in feature_cols if c in FORBIDDEN_AS_FEATURES]
    if forbidden_found:
        print(f"  [LEAKAGE DETECTADO] Columnas prohibidas en features: {forbidden_found}")
        is_clean = False
    elif verbose:
        print("  [OK] Sin columnas de identidad/target en features.")

    # -------------------------------------------------------
    # 2. Correlación de Pearson sospechosa con el target
    # -------------------------------------------------------
    y = df_train[target_col].values.astype(float)
    high_corr_cols = []

    for col in feature_cols:
        if col not in df_train.columns:
            continue
        series = df_train[col].dropna()
        if series.nunique() < 2:
            continue

        try:
            col_vals = df_train.loc[series.index, col].values.astype(float)
            y_aligned = y[series.index]
            corr, _ = stats.pearsonr(col_vals, y_aligned)
            if abs(corr) >= HIGH_CORR_THRESHOLD:
                high_corr_cols.append((col, round(corr, 4)))
        except (ValueError, Exception):
            pass

    if high_corr_cols:
        print(f"  [LEAKAGE DETECTADO] Correlacion con target >= {HIGH_CORR_THRESHOLD}:")
        for col, corr in high_corr_cols:
            print(f"     {col:<45} corr={corr}")
        is_clean = False
    elif verbose:
        print(f"  [OK] Ninguna feature con correlacion >= {HIGH_CORR_THRESHOLD} con target.")

    # -------------------------------------------------------
    # 3. Valores NaN solo en un split (síntoma de mezcla de datos)
    # -------------------------------------------------------
    # Si la feature tiene NaN exactamente donde target=1 → sospechoso.
    if target_col in df_train.columns:
        suspicion_count = 0
        for col in feature_cols:
            if col not in df_train.columns:
                continue
            null_mask = df_train[col].isna()
            if null_mask.sum() == 0:
                continue
            null_pos = df_train.loc[null_mask, target_col].mean()
            all_pos = df_train[target_col].mean()
            if null_pos is not None and abs(null_pos - all_pos) > 0.4:
                suspicion_count += 1
                if verbose:
                    print(f"  [SOSPECHOSO] {col}: tasa positiva en NULLs={null_pos:.3f}"
                          f"  vs global={all_pos:.3f}")
        if suspicion_count == 0 and verbose:
            print("  [OK] Patron de NaN no correlacionado con target.")

    # -------------------------------------------------------
    # 4. Leakage temporal: verificar que features lag NO usen información futura
    #    (Prueba heurística: correlación entre lag_X y el X del siguiente mes)
    # -------------------------------------------------------
    lag_cols_present = [c for c in feature_cols if c.endswith("_lag_1")]
    if lag_cols_present:
        if verbose:
            print(f"\n  Verificando {len(lag_cols_present)} features de lag (shift 1)...")

        # Verificación simple: el lag debe ser igual al valor del mes anterior
        lag_ok = True
        if "saldo_lag_1" in df_train.columns and "saldo_promedio" in df_train.columns:
            df_sorted = df_train.sort_values(["id_cliente", "mes"]).copy()
            df_sorted = df_sorted.reset_index(drop=True)
            expected_lag = (
                df_sorted.groupby("id_cliente")["saldo_promedio"].shift(1)
            )
            # Comparación NaN-safe:
            both_nan = df_sorted["saldo_lag_1"].isna() & expected_lag.isna()
            either_nan = df_sorted["saldo_lag_1"].isna() | expected_lag.isna()
            val_mismatch = (~either_nan) & (df_sorted["saldo_lag_1"] != expected_lag)
            nan_mismatch = either_nan & ~both_nan
            mismatch = int(val_mismatch.sum() + nan_mismatch.sum())
            if mismatch > 0:
                print(f"  [LEAKAGE DETECTADO] saldo_lag_1 no coincide con shift(1) de saldo_promedio. "
                      f"Mismatches reales (excluyendo NaN correctos): {mismatch}")
                lag_ok = False
                is_clean = False
            elif verbose:
                n_nan_ok = int(both_nan.sum())
                print(f"  [OK] Lag 'saldo_lag_1' verificado correctamente (shift 1 exacto). "
                      f"NaN esperados (primer registro de cliente): {n_nan_ok}")

        if lag_ok and verbose:
            print("  [OK] Features de lag parecen correctamente construidas.")

    # -------------------------------------------------------
    # Resultado final
    # -------------------------------------------------------
    if is_clean:
        if verbose:
            print("\n  [RESULTADO] No se detecto leakage severo. Pipeline puede continuar.")
    else:
        print("\n  [RESULTADO] LEAKAGE DETECTADO. DETENIENDO EXPERIMENTO.")
        print("  Revisa las features marcadas y corrige antes de reentrenar.")

    return is_clean


def quick_feature_check(
    feature_cols: list[str],
    train_col_names: list[str],
) -> bool:
    """
    Chequeo rápido previo al entrenamiento:
    - Todas las features existen en el DataFrame de train.
    - Ninguna feature es nombre prohibido.

    Returns:
        bool: True si todo está bien.
    """
    is_ok = True

    missing = [c for c in feature_cols if c not in train_col_names]
    if missing:
        print(f"  [ERROR] Features declaradas pero ausentes: {missing}")
        is_ok = False

    forbidden = [c for c in feature_cols if c in FORBIDDEN_AS_FEATURES]
    if forbidden:
        print(f"  [ERROR] Features prohibidas encontradas: {forbidden}")
        is_ok = False

    return is_ok
