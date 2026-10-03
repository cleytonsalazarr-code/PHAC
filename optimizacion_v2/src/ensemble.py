"""Rank averaging, blend ponderado nested y stacking regularizado."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge

from .metrics import gini_score


def ranks(pred: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    return rankdata(p, method="average") / len(p)


def rank_average(preds: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    names = list(preds)
    if weights is None:
        w = {k: 1.0 / len(names) for k in names}
    else:
        w = weights
    s = np.zeros(len(next(iter(preds.values()))), dtype=float)
    tot = 0.0
    for k in names:
        wk = float(w.get(k, 0.0))
        if wk <= 0:
            continue
        s += wk * ranks(preds[k])
        tot += wk
    if tot <= 0:
        raise ValueError("pesos nulos")
    return s / tot


def fit_weights(preds: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    names = list(preds)
    r = np.column_stack([ranks(preds[k]) for k in names])
    y = np.asarray(y)

    def loss(w):
        w = np.clip(w, 0, None)
        if w.sum() <= 1e-12:
            return 1.0
        w = w / w.sum()
        blend = r @ w
        g = gini_score(y, blend)
        return -g if np.isfinite(g) else 1.0

    w0 = np.ones(len(names)) / len(names)
    cons = {"type": "eq", "fun": lambda w: np.clip(w, 0, None).sum() - 1.0}
    res = minimize(loss, w0, method="SLSQP", bounds=[(0.0, 1.0)] * len(names), constraints=cons)
    w = np.clip(res.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else w0
    return {n: float(wi) for n, wi in zip(names, w)}


def nested_weighted_oof(oof_by_fold: list[dict]) -> tuple[np.ndarray, list[dict[str, float]], list[float]]:
    """Cada fold usa pesos ajustados solo en folds anteriores (nested temporal)."""
    names = list(oof_by_fold[0]["preds"])
    all_idx = []
    all_pred = []
    weights_hist = []
    ginis = []
    for i, fold in enumerate(oof_by_fold):
        if i == 0:
            w = {k: 1.0 / len(names) for k in names}
        else:
            # concat previous OOF
            prev_preds = {k: np.concatenate([oof_by_fold[j]["preds"][k] for j in range(i)]) for k in names}
            prev_y = np.concatenate([oof_by_fold[j]["y"] for j in range(i)])
            w = fit_weights(prev_preds, prev_y)
        blend = rank_average(fold["preds"], w)
        weights_hist.append(w)
        ginis.append(gini_score(fold["y"], blend))
        all_idx.append(fold["idx"])
        all_pred.append(blend)
    order = np.concatenate(all_idx)
    pred = np.concatenate(all_pred)
    sorter = np.argsort(order)
    oof = np.empty_like(pred)
    oof[sorter] = pred[sorter]  # placeholder
    full = np.full(int(order.max()) + 1, np.nan)
    full[order] = pred
    return full, weights_hist, ginis


def stack_predict(train_preds: np.ndarray, y_train: np.ndarray, test_preds: np.ndarray, kind: str = "logistic"):
    if kind == "ridge":
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(train_preds, y_train)
        return model.predict(test_preds), model
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000)
    model.fit(train_preds, y_train)
    return model.predict_proba(test_preds)[:, 1], model
