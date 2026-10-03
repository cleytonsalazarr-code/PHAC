"""
src/metrics.py
--------------
Métricas oficiales para la Hackathon BCP.
Métrica principal: Gini = 2 * ROC_AUC - 1

IMPORTANTE:
- NO usar Accuracy, F1, precision ni recall como criterio principal.
- El objetivo es maximizar el ranking global (AUC/Gini).
- Todas las funciones manejan de forma segura casos degenerados
  donde validation solo tiene una clase.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def gini_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Gini normalizado a partir de ROC-AUC.

    Args:
        y_true: Etiquetas binarias reales.
        y_pred: Probabilidades predichas continuas en [0, 1].

    Returns:
        float: Gini = 2 * AUC - 1. Retorna NaN si solo hay una clase.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return float("nan")

    auc = roc_auc_score(y_true, y_pred)
    return float(2.0 * auc - 1.0)


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula ROC-AUC de forma segura.

    Returns:
        float: AUC. Retorna NaN si solo hay una clase.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_pred))


def lgbm_gini_eval(preds: np.ndarray, train_data) -> tuple:
    """
    Función de evaluación personalizada para la API nativa de LightGBM.

    Invocada internamente por lgb.train() en cada iteración de validación.

    Args:
        preds: Probabilidades predichas (post-sigmoide) para el conjunto de validación.
        train_data: lightgbm.Dataset con etiquetas reales.

    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    labels = train_data.get_label()

    # Normalizar si llegaron como márgenes raw
    if preds.ndim > 1:
        preds = preds.ravel()
    if (preds < 0.0).any() or (preds > 1.0).any():
        preds = 1.0 / (1.0 + np.exp(-preds))

    gini = gini_score(labels, preds)

    # Caso degenerado: solo una clase en el fold
    if np.isnan(gini):
        return "gini", 0.0, True

    return "gini", gini, True


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """
    Calcula e imprime AUC y Gini. Retorna un diccionario con los valores.

    Args:
        y_true: Etiquetas reales.
        y_pred: Probabilidades predichas.
        prefix: Prefijo descriptivo para el log.

    Returns:
        dict: {"auc": float, "gini": float}
    """
    auc = auc_score(y_true, y_pred)
    gini = gini_score(y_true, y_pred)

    tag = f"[{prefix}] " if prefix else ""
    print(f"  {tag}AUC: {auc:.5f}  |  Gini: {gini:.5f}")

    return {"auc": auc, "gini": gini}


# Aliases para compatibilidad con evaluate_model.py
compute_gini = gini_score
lgb_gini_eval = lgbm_gini_eval
