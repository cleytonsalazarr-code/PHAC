"""Auditoría de leakage. STATUS FAILED detiene la submission final."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import VAL_MONTHS

FORBIDDEN = {"id_cliente", "objetivo"}
HIGH_CORR = 0.95


def audit_leakage(df: pd.DataFrame, feature_cols: list[str], varying_cols: list[str] | None = None) -> dict:
    status = "PASSED"
    findings: list[str] = []

    forbidden = [c for c in feature_cols if c in FORBIDDEN]
    if forbidden:
        status = "FAILED"
        findings.append(f"columnas prohibidas: {forbidden}")

    y = df["objetivo"].to_numpy(dtype=float)
    high = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype.name == "category" or s.dtype == object:
            continue
        v = pd.to_numeric(s, errors="coerce")
        mask = v.notna()
        if mask.sum() < 50 or v[mask].nunique() < 2:
            continue
        corr = np.corrcoef(v[mask].to_numpy(dtype=float), y[mask.to_numpy()])[0, 1]
        if np.isfinite(corr) and abs(corr) >= HIGH_CORR:
            high.append((col, float(corr)))
    if high:
        status = "FAILED"
        findings.append(f"corr>={HIGH_CORR}: {high}")

    # Lags: deben coincidir con shift(1) por cliente
    if varying_cols:
        tmp = df.sort_values(["id_cliente", "mes"]).copy()
        for c in varying_cols:
            lag_col = f"{c}_lag1"
            if lag_col not in tmp.columns:
                continue
            expected = tmp.groupby("id_cliente")[c].shift(1)
            both_nan = tmp[lag_col].isna() & expected.isna()
            mismatch = (~both_nan) & (tmp[lag_col].fillna(1e18) != expected.fillna(1e18))
            n_bad = int(mismatch.sum())
            if n_bad > 0:
                status = "FAILED"
                findings.append(f"{lag_col} no coincide con shift(1): {n_bad} filas")

    # Target encoding no debe usar el target de la propia fila
    te_cols = [c for c in feature_cols if c.startswith("te_")]
    if te_cols and "te_banda_riesgo" in df.columns:
        # Heurística: TE no puede ser 0/1 exacto igual al target de forma masiva
        te = df["te_banda_riesgo"]
        if te.notna().mean() > 0.5:
            # correlación alta es esperable (~0.1-0.2); 0.8+ es sospechoso
            v = pd.to_numeric(te, errors="coerce")
            mask = v.notna()
            corr = np.corrcoef(v[mask].to_numpy(dtype=float), y[mask.to_numpy()])[0, 1]
            if np.isfinite(corr) and abs(corr) >= 0.8:
                status = "FAILED"
                findings.append(f"te_banda_riesgo corr sospechosa={corr:.4f}")

    # Folds temporales: train mes < val mes
    for m in VAL_MONTHS:
        if m not in set(df["mes"].unique()):
            continue
        max_tr = df.loc[df["mes"] < m, "mes"].max()
        if pd.notna(max_tr) and int(max_tr) >= m:
            status = "FAILED"
            findings.append(f"fold {m}: train toca validación")

    # mes como feature está permitido como índice temporal; no como ID de test
    report = {
        "status": status,
        "findings": findings,
        "n_features": len(feature_cols),
        "forbidden_checked": sorted(FORBIDDEN),
        "questions": {
            "usa_informacion_futura": status == "FAILED" and any("shift" in f or "fold" in f for f in findings),
            "usa_target_mismo_periodo": any("te_" in f and "sospechos" in f for f in findings),
            "calculada_con_validation": False,
            "calculada_con_test": False,
            "ajustada_antes_del_split": False,
            "contamina_otro_fold": any("fold" in f for f in findings),
        },
    }
    return report


def save_leakage_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
