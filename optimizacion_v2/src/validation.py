"""Validación temporal estricta. 6 folds oficiales: 202606–202611."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VAL_MONTHS = [202606, 202607, 202608, 202609, 202610, 202611]
TUNING_MONTHS = [202606, 202607, 202608]


@dataclass
class Fold:
    fold: int
    val_month: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    n_train: int
    n_val: int
    pos_rate_train: float
    pos_rate_val: float


class PurgedTimeSeriesSplit:
    def __init__(self, month_col: str = "mes", val_months: list[int] | None = None):
        self.month_col = month_col
        self.val_months = list(val_months) if val_months is not None else list(VAL_MONTHS)

    def split(self, df: pd.DataFrame):
        months = set(df[self.month_col].unique())
        for m in self.val_months:
            if m not in months:
                raise ValueError(f"Mes de validación {m} no existe en los datos.")

        for fold_num, val_month in enumerate(self.val_months, start=1):
            train_mask = df[self.month_col] < val_month
            val_mask = df[self.month_col] == val_month
            train_idx = df.index[train_mask].to_numpy()
            val_idx = df.index[val_mask].to_numpy()
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(f"Fold {fold_num} vacío para val_month={val_month}")

            max_train = int(df.loc[train_idx, self.month_col].max())
            min_val = int(df.loc[val_idx, self.month_col].min())
            if max_train >= min_val:
                raise ValueError(
                    f"LEAKAGE: max train mes ({max_train}) >= min val mes ({min_val})"
                )

            y_train = df.loc[train_idx, "objetivo"].to_numpy(dtype=float)
            y_val = df.loc[val_idx, "objetivo"].to_numpy(dtype=float)
            yield Fold(
                fold=fold_num,
                val_month=val_month,
                train_idx=train_idx,
                val_idx=val_idx,
                n_train=len(train_idx),
                n_val=len(val_idx),
                pos_rate_train=float(np.nanmean(y_train)),
                pos_rate_val=float(np.nanmean(y_val)),
            )
