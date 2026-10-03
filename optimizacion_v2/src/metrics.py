"""Gini = 2 * ROC-AUC - 1. Optimiza ranking, no accuracy."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def gini_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(2.0 * roc_auc_score(y_true, y_pred) - 1.0)


def auc_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def gini_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "last": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "last": float(arr[-1]),
    }
