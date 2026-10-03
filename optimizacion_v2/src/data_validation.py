"""Validación de integridad: originales de solo lectura. No corrige datos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_TRAIN_ROWS = 110100
EXPECTED_TEST_ROWS = 9900
EXPECTED_TRAIN_COLS = 25
EXPECTED_TEST_COLS = 24
TRAIN_MONTHS = list(range(202601, 202612))
TEST_MONTH = 202612


def _file_meta(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def find_data_files(root: Path) -> tuple[Path, Path, Path | None]:
    names = ("train.csv", "test.csv", "sample_submission.csv")
    found: dict[str, Path] = {}
    search_roots = [
        root / "data",
        root.parent / "samir" / "data",
        root.parent / "leo",
        root.parent / "samir",
        root,
    ]
    for sr in search_roots:
        for name in names:
            cand = sr / name
            if name not in found and cand.exists():
                found[name] = cand
    if "train.csv" not in found or "test.csv" not in found:
        raise FileNotFoundError("No se encontraron train.csv / test.csv (solo lectura).")
    return found["train.csv"], found["test.csv"], found.get("sample_submission.csv")


def validate_data(train_path: Path, test_path: Path, sample_path: Path | None) -> dict:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    issues: list[dict] = []

    def issue(file, problem, evidence, impact, can_continue):
        issues.append(
            {
                "file": file,
                "problem": problem,
                "evidence": evidence,
                "impact": impact,
                "can_continue": can_continue,
            }
        )

    if train.shape[0] != EXPECTED_TRAIN_ROWS:
        issue("train.csv", "filas inesperadas", str(train.shape[0]), "benchmark no comparable", False)
    if test.shape[0] != EXPECTED_TEST_ROWS:
        issue("test.csv", "filas inesperadas", str(test.shape[0]), "submission inválida", False)
    if train.shape[1] != EXPECTED_TRAIN_COLS:
        issue("train.csv", "columnas inesperadas", str(list(train.columns)), "schema distinto", True)
    if test.shape[1] != EXPECTED_TEST_COLS:
        issue("test.csv", "columnas inesperadas", str(list(test.columns)), "schema distinto", True)

    if "objetivo" not in train.columns:
        issue("train.csv", "falta objetivo", "columna ausente", "imposible entrenar", False)
    if "objetivo" in test.columns:
        issue("test.csv", "test contiene objetivo", "columna presente", "riesgo de leakage", False)

    nulls_tr = int(train.isna().sum().sum())
    nulls_te = int(test.isna().sum().sum())
    inf_tr = int(np.isinf(train.select_dtypes(include=[np.number])).sum().sum())
    inf_te = int(np.isinf(test.select_dtypes(include=[np.number])).sum().sum())

    dups_tr = int(train.duplicated().sum())
    dups_id_mes = int(train.duplicated(["id_cliente", "mes"]).sum()) if {"id_cliente", "mes"} <= set(train.columns) else -1

    train_months = sorted(train["mes"].unique().tolist()) if "mes" in train.columns else []
    test_months = sorted(test["mes"].unique().tolist()) if "mes" in test.columns else []

    if train_months != TRAIN_MONTHS:
        issue("train.csv", "meses distintos a ene-nov 2026", str(train_months), "validación temporal", True)
    if test_months != [TEST_MONTH]:
        issue("test.csv", "test no es solo diciembre", str(test_months), "escenario de test", True)

    if "id_cliente" in test.columns and test["id_cliente"].duplicated().any():
        issue("test.csv", "id_cliente duplicado en test", str(int(test["id_cliente"].duplicated().sum())), "orden submission", False)

    pos_rate = float(train["objetivo"].mean()) if "objetivo" in train.columns else float("nan")
    if "objetivo" in train.columns:
        after_conv = _post_conversion_rows(train)
        if after_conv > 0:
            issue(
                "train.csv",
                "filas posteriores a conversión",
                str(after_conv),
                "posible adulteración del DGP",
                False,
            )

    report = {
        "status": "FAILED" if any(not i["can_continue"] for i in issues) else ("WARNING" if issues else "OK"),
        "issues": issues,
        "train_meta": _file_meta(train_path),
        "test_meta": _file_meta(test_path),
        "schema": {
            "train_rows": int(train.shape[0]),
            "train_cols": int(train.shape[1]),
            "test_rows": int(test.shape[0]),
            "test_cols": int(test.shape[1]),
            "train_columns": list(map(str, train.columns)),
            "test_columns": list(map(str, test.columns)),
            "dtypes_train": {c: str(t) for c, t in train.dtypes.items()},
        },
        "quality": {
            "nulls_train": nulls_tr,
            "nulls_test": nulls_te,
            "inf_train": inf_tr,
            "inf_test": inf_te,
            "dup_rows_train": dups_tr,
            "dup_id_mes_train": dups_id_mes,
            "target_prevalence": pos_rate,
            "train_months": train_months,
            "test_months": test_months,
            "n_clients_train": int(train["id_cliente"].nunique()) if "id_cliente" in train.columns else None,
            "n_clients_test": int(test["id_cliente"].nunique()) if "id_cliente" in test.columns else None,
        },
        "sample_meta": _file_meta(sample_path) if sample_path else None,
    }
    return report


def _post_conversion_rows(train: pd.DataFrame) -> int:
    if not {"id_cliente", "mes", "objetivo"} <= set(train.columns):
        return 0
    df = train.sort_values(["id_cliente", "mes"]).copy()
    conv_mes = df.loc[df["objetivo"] == 1, ["id_cliente", "mes"]].rename(columns={"mes": "mes_conv"})
    if conv_mes.empty:
        return 0
    merged = df.merge(conv_mes, on="id_cliente", how="left")
    return int(((merged["objetivo"] == 0) & (merged["mes"] > merged["mes_conv"])).sum())


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
