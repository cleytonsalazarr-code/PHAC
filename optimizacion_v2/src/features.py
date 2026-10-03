"""Feature engineering modular y anti-leakage.

Regla: feature(t) usa información de < t para lags/rolling/target encoding.
Las covariables del mes t (saldo, visitas, etc.) sí están disponibles en train y test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = [
    "ocupacion",
    "region",
    "canal_adquisicion",
    "banda_riesgo",
    "dispositivo_principal",
]
BOOL_COLS = [
    "tiene_tarjeta_credito",
    "activo_movil",
    "es_nuevo_cliente",
    "tiene_prestamo",
    "tiene_seguro",
]
NUM_COLS = [
    "edad",
    "ingresos",
    "ratio_deuda_ingresos",
    "antiguedad_cuenta_meses",
    "numero_productos",
    "saldo_promedio",
    "dias_ultima_transaccion",
    "antiguedad_direccion_meses",
    "visitas_web_ultimos_90_dias",
    "distancia_sucursal_km",
    "dia_preferido_pago",
    "dias_ultima_interaccion",
]
# En test esta variable es ruido (permutación de dias_ultima_transaccion).
EXCLUDE_FROM_MODEL = {"dias_ultima_interaccion"}
LAG_CANDIDATES = [
    "saldo_promedio",
    "numero_productos",
    "visitas_web_ultimos_90_dias",
    "ratio_deuda_ingresos",
    "dias_ultima_transaccion",
    "dias_ultima_interaccion",
]
ID_COLS = {"id_cliente", "mes", "objetivo"}


def _safe_pct(a, b):
    return (a - b) / (np.abs(b) + 1e-5)


def detect_time_varying(train: pd.DataFrame, cols: list[str], min_change_frac: float = 0.01) -> list[str]:
    """Columnas que realmente cambian dentro del mismo cliente."""
    varying = []
    g = train.groupby("id_cliente", sort=False)
    n_clients = train["id_cliente"].nunique()
    for c in cols:
        if c not in train.columns:
            continue
        nunq = g[c].nunique(dropna=False)
        frac = float((nunq > 1).sum()) / max(n_clients, 1)
        if frac >= min_change_frac:
            varying.append(c)
    return varying


def concat_history(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    te = test.copy()
    if "objetivo" not in te.columns:
        te["objetivo"] = np.nan
    df = pd.concat([train.copy(), te], ignore_index=True)
    df["anio"] = (df["mes"] // 100).astype(np.int16)
    df["mes_calendario"] = (df["mes"] % 100).astype(np.int8)
    df["mes_idx"] = (df["anio"] - 2026) * 12 + df["mes_calendario"]
    df = df.sort_values(["id_cliente", "mes_idx"]).reset_index(drop=True)
    return df


def add_client_history(df: pd.DataFrame, varying_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("id_cliente", sort=False)

    # Recencia / meses observados: solo pasado (cumcount no usa futuro).
    df["meses_observados"] = g.cumcount().astype(np.int16)
    df["recencia_cliente"] = df["meses_observados"]  # alias interpretativo

    for c in varying_cols:
        lag1 = g[c].shift(1)
        lag2 = g[c].shift(2)
        lag3 = g[c].shift(3)
        df[f"{c}_lag1"] = lag1
        df[f"{c}_lag2"] = lag2
        df[f"{c}_lag3"] = lag3
        df[f"{c}_d1"] = df[c] - lag1
        df[f"{c}_d3"] = df[c] - lag3
        df[f"{c}_pct1"] = _safe_pct(df[c], lag1)

        shifted = g[c].shift(1)
        gb_shift = shifted.groupby(df["id_cliente"], sort=False)
        for win, name in ((3, "r3"), (6, "r6")):
            rolled = gb_shift.rolling(win, min_periods=1)
            df[f"{c}_{name}_mean"] = rolled.mean().reset_index(level=0, drop=True)
            df[f"{c}_{name}_std"] = rolled.std().reset_index(level=0, drop=True)
            df[f"{c}_{name}_min"] = rolled.min().reset_index(level=0, drop=True)
            df[f"{c}_{name}_max"] = rolled.max().reset_index(level=0, drop=True)
            df[f"{c}_{name}_median"] = rolled.median().reset_index(level=0, drop=True)

        # tendencia / volatilidad / aceleración sobre pasado
        df[f"{c}_trend"] = lag1 - lag3
        df[f"{c}_accel"] = (df[c] - lag1) - (lag1 - lag2)
        df[f"{c}_vol"] = df[f"{c}_r3_std"]

    return df


def add_static_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(np.int8)
    if "banda_riesgo" in df.columns:
        df["banda_ord"] = df["banda_riesgo"].map({"high": 0, "medium": 1, "low": 2}).astype("float32")
    if "saldo_promedio" in df.columns and "numero_productos" in df.columns:
        df["saldo_por_producto"] = df["saldo_promedio"] / (df["numero_productos"] + 1.0)
    if "ratio_deuda_ingresos" in df.columns and "ingresos" in df.columns:
        df["deuda_estimada"] = df["ratio_deuda_ingresos"] * df["ingresos"]
    if "ingresos" in df.columns:
        df["log_ingresos"] = np.log1p(df["ingresos"].clip(lower=0))
    if "saldo_promedio" in df.columns:
        df["log_saldo"] = np.log1p(df["saldo_promedio"].clip(lower=0))
    if "dias_ultima_transaccion" in df.columns:
        df["tx_reciente_30"] = (df["dias_ultima_transaccion"] <= 30).astype(np.int8)
        df["tx_reciente_7"] = (df["dias_ultima_transaccion"] <= 7).astype(np.int8)
    return df


def _hist_mean_by_group(df: pd.DataFrame, key_cols: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    """Media de value_col por grupo usando SOLO meses anteriores (no t, no futuro)."""
    work = df[["mes_idx", value_col] + key_cols].copy()
    monthly = (
        work.groupby(["mes_idx"] + key_cols, observed=True)[value_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    monthly = monthly.sort_values(["mes_idx"] + key_cols)
    monthly["csum"] = monthly.groupby(key_cols, observed=True)["sum"].cumsum()
    monthly["ccnt"] = monthly.groupby(key_cols, observed=True)["count"].cumsum()
    monthly["psum"] = monthly.groupby(key_cols, observed=True)["csum"].shift(1)
    monthly["pcnt"] = monthly.groupby(key_cols, observed=True)["ccnt"].shift(1)
    monthly[out_col] = monthly["psum"] / monthly["pcnt"]
    mapped = df.merge(
        monthly[["mes_idx"] + key_cols + [out_col]],
        on=["mes_idx"] + key_cols,
        how="left",
    )
    return mapped


def add_historical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency + target encoding histórico y medias relacionales (solo < t)."""
    df = df.copy()
    if "objetivo" in df.columns:
        for cat in CAT_COLS:
            if cat in df.columns:
                df = _hist_mean_by_group(df, [cat], "objetivo", f"te_{cat}")
        # smoothing global past rate
        month_stats = (
            df.groupby("mes_idx")["objetivo"]
            .agg(["sum", "count"])
            .sort_index()
        )
        month_stats["psum"] = month_stats["sum"].cumsum().shift(1)
        month_stats["pcnt"] = month_stats["count"].cumsum().shift(1)
        month_stats["prior"] = month_stats["psum"] / month_stats["pcnt"]
        df = df.merge(month_stats[["prior"]], left_on="mes_idx", right_index=True, how="left")
        for cat in CAT_COLS:
            col = f"te_{cat}"
            if col in df.columns:
                df[col] = df[col].fillna(df["prior"])
                # Laplace-ish: already historical mean; keep prior fill

    for cat in CAT_COLS:
        if cat not in df.columns:
            continue
        cnt = df.groupby(["mes_idx", cat], observed=True).size().rename("cnt").reset_index()
        cnt = cnt.sort_values(["mes_idx", cat])
        cnt["csum"] = cnt.groupby(cat, observed=True)["cnt"].cumsum()
        cnt["freq_hist"] = cnt.groupby(cat, observed=True)["csum"].shift(1)
        df = df.merge(
            cnt[["mes_idx", cat, "freq_hist"]].rename(columns={"freq_hist": f"freq_{cat}"}),
            on=["mes_idx", cat],
            how="left",
        )

    # Relacionales: saldo / visitas vs región, ocupación, canal, banda, dispositivo (pasado)
    for by in CAT_COLS:
        if by not in df.columns:
            continue
        if "saldo_promedio" in df.columns:
            df = _hist_mean_by_group(df, [by], "saldo_promedio", f"saldo_mean_{by}_hist")
            df[f"saldo_vs_{by}"] = df["saldo_promedio"] / (df[f"saldo_mean_{by}_hist"] + 1e-5)
        if "visitas_web_ultimos_90_dias" in df.columns:
            df = _hist_mean_by_group(df, [by], "visitas_web_ultimos_90_dias", f"vis_mean_{by}_hist")
            df[f"vis_vs_{by}"] = df["visitas_web_ultimos_90_dias"] / (df[f"vis_mean_{by}_hist"] + 1e-5)
    return df


def prepare_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    varying = detect_time_varying(train, [c for c in LAG_CANDIDATES if c in train.columns])
    df = concat_history(train, test)
    df = add_static_transforms(df)
    df = add_client_history(df, varying)
    df = add_historical_encodings(df)

    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    train_f = df[df["mes"] <= 202611].copy().reset_index(drop=True)
    test_f = df[df["mes"] == 202612].copy().reset_index(drop=True)
    meta = {"varying_cols": varying, "n_train": len(train_f), "n_test": len(test_f)}
    return train_f, test_f, meta


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(ID_COLS) | EXCLUDE_FROM_MODEL | {"prior"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if df[c].dtype == "object":
            continue
        cols.append(c)
    return cols


def core_feature_columns(df: pd.DataFrame) -> list[str]:
    """Covariables estáticas + duración. No usa lags (21/22 cols están congeladas)."""
    wanted = (
        [c for c in NUM_COLS if c not in EXCLUDE_FROM_MODEL]
        + CAT_COLS
        + BOOL_COLS
        + ["meses_observados", "banda_ord", "anio", "mes_calendario", "mes_idx"]
    )
    return [c for c in wanted if c in df.columns]


def cat_feature_names(cols: list[str]) -> list[str]:
    return [c for c in CAT_COLS if c in cols]
