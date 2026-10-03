"""
src/validation.py
-----------------
Motor de validación temporal estricta para la Hackathon BCP.

REGLA ABSOLUTA:
    NO usar train_test_split(), KFold() ni StratifiedKFold() como validación principal.
    El escenario real es: entrenar con pasado, predecir el futuro inmediato.

Esquema de folds:
    Fold 1: train < 202608  |  val = 202608
    Fold 2: train < 202609  |  val = 202609
    Fold 3: train < 202610  |  val = 202610
    Fold 4: train < 202611  |  val = 202611  ← más cercano a diciembre

El último fold tiene peso interpretativo especial porque replica el escenario de test.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class FoldResult:
    """Resultado de un fold temporal."""
    fold: int
    val_month: int
    n_train: int
    n_val: int
    pos_rate_train: float
    pos_rate_val: float
    train_idx: np.ndarray = field(repr=False)
    val_idx: np.ndarray = field(repr=False)
    auc: float = float("nan")
    gini: float = float("nan")

    def summary(self) -> str:
        return (
            f"Fold {self.fold} | val={self.val_month}"
            f" | train={self.n_train:,}  val={self.n_val:,}"
            f" | pos_train={self.pos_rate_train:.3f}"
            f" | pos_val={self.pos_rate_val:.3f}"
            f" | AUC={self.auc:.5f}  Gini={self.gini:.5f}"
        )


VAL_MONTHS = [202606, 202607, 202608, 202609, 202610, 202611]


class PurgedTimeSeriesSplit:
    """Generador de folds temporales estrictos para evitar data leakage."""
    def __init__(self, month_col: str = "mes", val_months: list[int] = None):
        self.month_col = month_col
        self.val_months = val_months if val_months is not None else VAL_MONTHS

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        available_months = set(df[self.month_col].unique())
        for m in self.val_months:
            if m not in available_months:
                raise ValueError(f"Mes de validación {m} no existe en los datos.")
                
        for fold_num, val_month in enumerate(self.val_months, start=1):
            train_mask = df[self.month_col] < val_month
            val_mask = df[self.month_col] == val_month

            train_idx = df.index[train_mask].values
            val_idx = df.index[val_mask].values

            if len(train_idx) == 0:
                raise ValueError(f"Fold {fold_num}: no hay datos de entrenamiento para val_month={val_month}")
            if len(val_idx) == 0:
                raise ValueError(f"Fold {fold_num}: no hay datos de validación para val_month={val_month}")

            # PROTECCIÓN CONTRA LEAKAGE
            max_mes_train = df.loc[train_idx, self.month_col].max()
            min_mes_val = df.loc[val_idx, self.month_col].min()
            
            if max_mes_train >= min_mes_val:
                raise ValueError(f"LEAKAGE DETECTADO: max train ({max_mes_train}) >= min val ({min_mes_val})")

            pos_train = float(df.loc[train_idx, "objetivo"].mean())
            pos_val = float(df.loc[val_idx, "objetivo"].mean())
            
            print(f"FOLD {fold_num}")
            print(f"Train months: <= {max_mes_train}")
            print(f"Validation month: {val_month}")
            print(f"Train rows: {len(train_idx)}")
            print(f"Validation rows: {len(val_idx)}")
            print(f"Positive rate train: {pos_train:.4f}")
            print(f"Positive rate validation: {pos_val:.4f}")
            print("-" * 30)

            yield train_idx, val_idx

