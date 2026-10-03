"""Orquestador Gini BCP: audita, reproduce, tunea, ensambla, audita leakage y entrega."""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_validation import find_data_files, save_report, validate_data
from src.ensemble import fit_weights, rank_average
from src.features import (
    BOOL_COLS,
    CAT_COLS,
    NUM_COLS,
    cat_feature_names,
    core_feature_columns,
    feature_columns,
    prepare_frames,
)
from src.leakage_audit import audit_leakage, save_leakage_report
from src.metrics import gini_score, gini_stats
from src.models import (
    SEED,
    _cb_frame,
    _lgb_frame,
    gbm_feature_list,
    importance_catboost,
    importance_lgbm,
    monotone_params,
    pred_lgbm_seeds,
    predict_band_logistic,
    scale_pos_weight,
    select_boost_rounds,
    train_catboost,
    train_lgbm,
    train_xgb,
    tune_catboost,
    tune_lgbm,
    tune_xgb,
)
from src.validation import PurgedTimeSeriesSplit, VAL_MONTHS

OUT = ROOT / "outputs"
SUB = ROOT / "submissions"
BENCH_GINI = 0.258813
N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "12"))
N_TRIALS_CB = int(os.environ.get("OPTUNA_TRIALS_CB", str(min(N_TRIALS, 8))))
W_GBM_PRIOR = 0.75


def stage(name: str, action: str, result: str, gini=None, change=None, nxt=""):
    g = f"{gini:.6f}" if isinstance(gini, float) and np.isfinite(gini) else "-"
    c = f"{change:+.6f}" if isinstance(change, float) and np.isfinite(change) else "-"
    print(f"\nETAPA {name}\nACCIÓN {action}\nRESULTADO {result}\nGINI {g}\nCAMBIO {c}\nSIGUIENTE {nxt}")


def repo_inventory(base: Path) -> pd.DataFrame:
    skip = {".git", ".venv", "node_modules", "__pycache__", "catboost_info"}
    rows = []
    for p in base.rglob("*"):
        if any(part in skip for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in {".png", ".jpeg", ".jpg", ".gif", ".pkl", ".bin", ".exe"}:
            kind = "binario"
        elif p.suffix.lower() in {".csv", ".parquet"}:
            kind = "datos"
        elif p.suffix.lower() == ".py":
            kind = "script"
        elif p.suffix.lower() in {".md", ".txt"}:
            kind = "docs"
        else:
            kind = "otro"
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        rel = str(p.relative_to(base))
        rows.append({"ruta": rel, "tipo": kind, "tamano": size, "funcion": _guess(rel, kind), "relevancia": _rel(rel)})
        if len(rows) >= 400:
            break
    return pd.DataFrame(rows)


def _guess(rel: str, kind: str) -> str:
    r = rel.lower()
    if "train.csv" in r:
        return "train original"
    if "test.csv" in r:
        return "test original"
    if "feature" in r:
        return "features"
    if "valid" in r:
        return "validacion"
    if "model" in r:
        return "modelos"
    if "metric" in r:
        return "metricas"
    if "leak" in r:
        return "leakage"
    if "ensemble" in r or "blend" in r:
        return "ensemble"
    if "sub" in r:
        return "submission"
    return kind


def _rel(rel: str) -> str:
    r = rel.lower()
    if any(x in r for x in ["samir", "leo", "optimizacion_v2", "instrucciones"]):
        return "alta"
    if "informacion de soluciones" in r:
        return "baja"
    return "media"


def benchmark_lgbm(train: pd.DataFrame) -> dict:
    df = train.copy()
    df["mes_num"] = df["mes"] - df["mes"].min()
    for col in CAT_COLS:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes.astype(np.int16)
    for col in BOOL_COLS:
        df[col] = df[col].astype(int)
    feat = [c for c in NUM_COLS if c in df.columns] + [f"{c}_enc" for c in CAT_COLS] + BOOL_COLS + ["mes_num"]
    cat = [f"{c}_enc" for c in CAT_COLS]
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
    }
    rows = []
    ginis = []
    splitter = PurgedTimeSeriesSplit()
    for fold in splitter.split(df):
        Xtr, ytr = df.loc[fold.train_idx, feat], df.loc[fold.train_idx, "objetivo"].to_numpy()
        Xva, yva = df.loc[fold.val_idx, feat], df.loc[fold.val_idx, "objetivo"].to_numpy()
        _, p, _ = train_lgbm(Xtr, ytr, Xva, yva, cat, params, num_boost_round=3000)
        g = gini_score(yva, p)
        ginis.append(g)
        rows.append({"fold": fold.fold, "val_month": fold.val_month, "model": "lgbm", "gini": g})
        print(f"  bench LGBM fold {fold.fold} mes {fold.val_month}: {g:.6f}")
    stats = gini_stats(ginis)
    cmp = {
        "esperado": BENCH_GINI,
        "obtenido": stats["mean"],
        "diferencia": stats["mean"] - BENCH_GINI,
        "causa_probable": (
            "reproduccion razonable"
            if abs(stats["mean"] - BENCH_GINI) < 0.01
            else "revisar folds/features/semilla/version librerias"
        ),
    }
    return {"rows": rows, "stats": stats, "compare": cmp, "ginis": ginis}


def oof_boosters(df, feats, cats, params_map: dict):
    splitter = PurgedTimeSeriesSplit()
    fold_pack = []
    recs = []
    for fold in splitter.split(df):
        Xtr, ytr = df.loc[fold.train_idx, feats], df.loc[fold.train_idx, "objetivo"].to_numpy()
        Xva, yva = df.loc[fold.val_idx, feats], df.loc[fold.val_idx, "objetivo"].to_numpy()
        preds = {}
        _, preds["lgbm"], _ = train_lgbm(Xtr, ytr, Xva, yva, cats, params_map["lgbm"], 1200)
        _, preds["catboost"], _ = train_catboost(Xtr, ytr, Xva, yva, cats, params_map["catboost"], 800)
        _, preds["xgb"], _ = train_xgb(Xtr, ytr, Xva, yva, cats, params_map["xgb"], 1200)
        combo = {
            "lgbm_cat": rank_average({"lgbm": preds["lgbm"], "catboost": preds["catboost"]}),
            "lgbm_xgb": rank_average({"lgbm": preds["lgbm"], "xgb": preds["xgb"]}),
            "cat_xgb": rank_average({"catboost": preds["catboost"], "xgb": preds["xgb"]}),
            "rank3": rank_average(preds),
        }
        for name, p in {**preds, **combo}.items():
            recs.append(
                {
                    "fold": fold.fold,
                    "val_month": fold.val_month,
                    "model": name,
                    "gini": gini_score(yva, p),
                    "n_train": fold.n_train,
                    "n_val": fold.n_val,
                }
            )
        fold_pack.append({"idx": fold.val_idx, "y": yva, "preds": preds, "fold": fold.fold, "val_month": fold.val_month})
        print(
            f"  fold {fold.fold} {fold.val_month} "
            f"LGBM={gini_score(yva, preds['lgbm']):.5f} "
            f"CB={gini_score(yva, preds['catboost']):.5f} "
            f"XGB={gini_score(yva, preds['xgb']):.5f} "
            f"R3={gini_score(yva, combo['rank3']):.5f}"
        )
    return fold_pack, pd.DataFrame(recs)


def nested_blend_and_stack(fold_pack: list[dict]):
    names = ["lgbm", "catboost", "xgb"]
    w_hist, g_w, g_s_log, g_s_ridge = [], [], [], []
    oof_rows = []
    for i, fold in enumerate(fold_pack):
        if i == 0:
            w = {k: 1.0 / 3.0 for k in names}
            p_w = rank_average(fold["preds"], w)
            p_log = p_w
            p_ridge = p_log
        else:
            prev_p = {k: np.concatenate([fold_pack[j]["preds"][k] for j in range(i)]) for k in names}
            prev_y = np.concatenate([fold_pack[j]["y"] for j in range(i)])
            w = fit_weights(prev_p, prev_y)
            p_w = rank_average(fold["preds"], w)
            Xtr = np.column_stack([prev_p[k] for k in names])
            Xva = np.column_stack([fold["preds"][k] for k in names])
            log = LogisticRegression(C=1.0, penalty="l2", max_iter=2000)
            log.fit(Xtr, prev_y)
            p_log = log.predict_proba(Xva)[:, 1]
            from sklearn.linear_model import Ridge

            ridge = Ridge(alpha=1.0)
            ridge.fit(Xtr, prev_y)
            p_ridge = ridge.predict(Xva)
        w_hist.append(w)
        g_w.append(gini_score(fold["y"], p_w))
        g_s_log.append(gini_score(fold["y"], p_log))
        g_s_ridge.append(gini_score(fold["y"], p_ridge))
        oof_rows.append(
            pd.DataFrame(
                {
                    "row_idx": fold["idx"],
                    "fold": fold["fold"],
                    "val_month": fold["val_month"],
                    "y": fold["y"],
                    "oof_lgbm": fold["preds"]["lgbm"],
                    "oof_catboost": fold["preds"]["catboost"],
                    "oof_xgb": fold["preds"]["xgb"],
                    "oof_rank3": rank_average(fold["preds"]),
                    "oof_weighted": p_w,
                    "oof_stack_log": p_log,
                    "oof_stack_ridge": p_ridge,
                }
            )
        )
    oof = pd.concat(oof_rows, ignore_index=True)
    return oof, w_hist, {"weighted": g_w, "stack_log": g_s_log, "stack_ridge": g_s_ridge}


def oof_mono_band(df, gbm_feats, cats, rounds: int):
    recs = []
    oof_extra = []
    splitter = PurgedTimeSeriesSplit()
    mparams = monotone_params(gbm_feats)
    g_mono, g_band, g_blend = [], [], []
    for fold in splitter.split(df):
        trn = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
        yva = val["objetivo"].to_numpy()
        p_g = pred_lgbm_seeds(
            trn[gbm_feats],
            trn["objetivo"].to_numpy(),
            val[gbm_feats],
            [c for c in cats if c in gbm_feats],
            mparams,
            rounds,
            seeds=[42],
        )
        p_b = predict_band_logistic(trn, val)
        p_m = W_GBM_PRIOR * (pd.Series(p_g).rank(method="average").to_numpy() / len(p_g)) + (
            1 - W_GBM_PRIOR
        ) * (pd.Series(p_b).rank(method="average").to_numpy() / len(p_b))
        gm, gb, gx = gini_score(yva, p_g), gini_score(yva, p_b), gini_score(yva, p_m)
        g_mono.append(gm)
        g_band.append(gb)
        g_blend.append(gx)
        recs.extend(
            [
                {"fold": fold.fold, "val_month": fold.val_month, "model": "lgbm_mono", "gini": gm},
                {"fold": fold.fold, "val_month": fold.val_month, "model": "band_logistic", "gini": gb},
                {"fold": fold.fold, "val_month": fold.val_month, "model": "lgbm_mono_bandlog", "gini": gx},
            ]
        )
        oof_extra.append(
            pd.DataFrame(
                {
                    "row_idx": fold.val_idx,
                    "fold": fold.fold,
                    "oof_lgbm_mono": p_g,
                    "oof_band_logistic": p_b,
                    "oof_lgbm_mono_bandlog": p_m,
                }
            )
        )
        print(f"  mono/band fold {fold.fold} {fold.val_month} MONO={gm:.5f} BAND={gb:.5f} BLEND={gx:.5f}")
    return pd.DataFrame(recs), pd.concat(oof_extra, ignore_index=True), {"lgbm_mono": g_mono, "band_logistic": g_band, "lgbm_mono_bandlog": g_blend}


def validate_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    assert list(sub.columns) == ["id_cliente", "prediccion"], sub.columns.tolist()
    assert len(sub) == len(test), (len(sub), len(test))
    assert sub["id_cliente"].equals(test["id_cliente"].reset_index(drop=True))
    assert sub["prediccion"].notna().all()
    assert np.isfinite(sub["prediccion"]).all()
    assert sub["prediccion"].between(0, 1).all()
    assert "mes" not in sub.columns


def main():
    OUT.mkdir(exist_ok=True)
    SUB.mkdir(exist_ok=True)
    log_path = OUT / "run_log.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg)
        log_f.write(str(msg) + "\n")
        log_f.flush()

    log(f"start {datetime.now().isoformat(timespec='seconds')}")
    log(f"python {sys.version.split()[0]} trials={N_TRIALS} cb_trials={N_TRIALS_CB}")

    inv = repo_inventory(ROOT.parent)
    inv.to_csv(OUT / "repo_inventory.csv", index=False)
    stage("1", "inventario repositorio", f"{len(inv)} archivos indexados", nxt="validación datos")

    train_path, test_path, sample_path = find_data_files(ROOT)
    data_rep = validate_data(train_path, test_path, sample_path)
    save_report(data_rep, OUT / "data_validation.json")
    if data_rep["status"] == "FAILED":
        log("STOP: datos corruptos/inconsistentes. Ver outputs/data_validation.json")
        log(json.dumps(data_rep["issues"], indent=2, default=str))
        log_f.close()
        return
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    stage("2", "validación schema/nulos/hashes", data_rep["status"], nxt="benchmark")

    bench = benchmark_lgbm(train_raw)
    pd.DataFrame(bench["rows"]).to_csv(OUT / "benchmark.csv", index=False)
    cmp = bench["compare"]
    log(
        f"benchmark esperado={cmp['esperado']:.6f} obtenido={cmp['obtenido']:.6f} "
        f"diff={cmp['diferencia']:.6f} causa={cmp['causa_probable']}"
    )
    if abs(cmp["diferencia"]) > 0.03:
        log("STOP: no se reproduce el benchmark sin explicación suficiente.")
        log_f.close()
        return
    stage("3", "reproducir LGBM 6 folds", cmp["causa_probable"], cmp["obtenido"], cmp["diferencia"], "features")

    train_f, test_f, feat_meta = prepare_frames(train_raw, test_raw)
    rich_feats = feature_columns(train_f)
    keep = []
    for c in rich_feats:
        s = train_f[c]
        if getattr(s.dtype, "name", "") == "category":
            keep.append(c)
            continue
        if s.notna().sum() == 0:
            continue
        keep.append(c)
    rich_feats = keep
    core_feats = core_feature_columns(train_f)
    gbm_feats = gbm_feature_list(core_feats)
    cats_core = cat_feature_names(core_feats)
    cats_gbm = [c for c in cats_core if c in gbm_feats]
    log(f"varying={feat_meta['varying_cols']} n_rich={len(rich_feats)} n_core={len(core_feats)} n_gbm={len(gbm_feats)}")
    stage("4", "features temporales/hist/core", f"core={len(core_feats)} rich={len(rich_feats)} varying={feat_meta['varying_cols']}", nxt="leakage + tuning")

    leak0 = audit_leakage(train_f, rich_feats, feat_meta["varying_cols"])
    save_leakage_report(leak0, OUT / "leakage_report.json")
    if leak0["status"] == "FAILED":
        log("STOP leakage: " + "; ".join(leak0["findings"]))
        log_f.close()
        return

    log("Optuna LGBM (core feats, folds de tuning)...")
    best_lgbm = tune_lgbm(train_f, core_feats, cats_core, n_trials=N_TRIALS, out_dir=OUT)
    log("Optuna CatBoost...")
    best_cb = tune_catboost(train_f, core_feats, cats_core, n_trials=N_TRIALS_CB, out_dir=OUT)
    log("Optuna XGB...")
    best_xgb = tune_xgb(train_f, core_feats, cats_core, n_trials=N_TRIALS_CB, out_dir=OUT)
    params_map = {"lgbm": best_lgbm, "catboost": best_cb, "xgb": best_xgb}
    stage("5", f"Optuna LGBM={N_TRIALS} CB/XGB={N_TRIALS_CB}", "params en outputs/best_params_*.json", nxt="OOF 6 folds")

    fold_pack, exp_df = oof_boosters(train_f, core_feats, cats_core, params_map)
    oof, w_hist, nested_g = nested_blend_and_stack(fold_pack)

    extra_rows = []
    for key, gs in nested_g.items():
        for i, g in enumerate(gs, start=1):
            extra_rows.append({"fold": i, "val_month": VAL_MONTHS[i - 1], "model": key, "gini": g})
    exp_df = pd.concat([exp_df, pd.DataFrame(extra_rows)], ignore_index=True)

    mparams = monotone_params(gbm_feats)
    rounds = select_boost_rounds(train_f, gbm_feats, cats_gbm, mparams, max_rounds=180)
    log(f"rondas GBM monótono (curva media folds de tuning)={rounds}")
    rec_mono, oof_mono, _ = oof_mono_band(train_f, gbm_feats, cats_gbm, rounds)
    exp_df = pd.concat([exp_df, rec_mono], ignore_index=True)
    oof = oof.merge(oof_mono, on=["row_idx", "fold"], how="left")
    oof.to_csv(OUT / "oof_master.csv", index=False)
    exp_df.to_csv(OUT / "experiment_results.csv", index=False)

    summary = []
    for model, part in exp_df.groupby("model"):
        st = gini_stats(part.sort_values("fold")["gini"].tolist())
        summary.append({"model": model, **st, "delta_vs_baseline": st["mean"] - BENCH_GINI})
    sum_df = pd.DataFrame(summary).sort_values("mean", ascending=False)
    sum_df.to_csv(OUT / "experiment_summary.csv", index=False)
    best_row = sum_df.iloc[0]
    log("ranking modelos OOF:\n" + sum_df.to_string(index=False))
    stage("6-9", "OOF + rank + nested blend + stacking + mono/band", str(best_row["model"]), float(best_row["mean"]), float(best_row["delta_vs_baseline"]), "selección")

    leak1 = audit_leakage(train_f, rich_feats, feat_meta["varying_cols"])
    save_leakage_report(leak1, OUT / "leakage_report.json")
    if leak1["status"] == "FAILED":
        log("STOP leakage final")
        log_f.close()
        return

    best_name = str(best_row["model"])
    Xall, yall = train_f[core_feats], train_f["objetivo"].to_numpy()
    Xte = test_f[core_feats]
    last = [f for f in PurgedTimeSeriesSplit().split(train_f)][-1]
    Xl, yl = train_f.loc[last.train_idx, core_feats], train_f.loc[last.train_idx, "objetivo"].to_numpy()
    Xv, yv = train_f.loc[last.val_idx, core_feats], train_f.loc[last.val_idx, "objetivo"].to_numpy()
    _, _, it_l = train_lgbm(Xl, yl, Xv, yv, cats_core, best_lgbm, 1200)
    _, _, it_c = train_catboost(Xl, yl, Xv, yv, cats_core, best_cb, 800)
    _, _, it_x = train_xgb(Xl, yl, Xv, yv, cats_core, best_xgb, 1200)

    lgb_params = dict(best_lgbm)
    lgb_params.update(
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": SEED,
            "scale_pos_weight": scale_pos_weight(yall),
        }
    )
    dtr = lgb.Dataset(_lgb_frame(Xall, cats_core), yall, categorical_feature=cats_core, free_raw_data=False)
    final_lgb = lgb.train(lgb_params, dtr, num_boost_round=max(it_l, 80))
    p_lgb = final_lgb.predict(_lgb_frame(Xte, cats_core))
    importance_lgbm(final_lgb, core_feats).to_csv(OUT / "feature_importance_lgbm.csv", index=False)

    cb_params = dict(best_cb)
    cb_params.update(
        {
            "iterations": max(it_c, 80),
            "verbose": 0,
            "random_seed": SEED,
            "scale_pos_weight": scale_pos_weight(yall),
            "loss_function": "Logloss",
        }
    )
    cb_params.pop("early_stopping_rounds", None)
    final_cb = CatBoostClassifier(**cb_params)
    final_cb.fit(Pool(_cb_frame(Xall, cats_core), yall, cat_features=cats_core))
    p_cb = final_cb.predict_proba(Pool(_cb_frame(Xte, cats_core), cat_features=cats_core))[:, 1]
    importance_catboost(final_cb, core_feats).to_csv(OUT / "feature_importance_catboost.csv", index=False)

    xgb_params = dict(best_xgb)
    xgb_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": SEED,
            "scale_pos_weight": scale_pos_weight(yall),
        }
    )
    dx = xgb.DMatrix(_lgb_frame(Xall, cats_core), label=yall, enable_categorical=True)
    dte = xgb.DMatrix(_lgb_frame(Xte, cats_core), enable_categorical=True)
    final_xgb = xgb.train(xgb_params, dx, num_boost_round=max(it_x, 80))
    p_xgb = final_xgb.predict(dte)

    w_final = w_hist[-1]
    test_preds = {"lgbm": p_lgb, "catboost": p_cb, "xgb": p_xgb}
    p_rank = rank_average(test_preds)
    p_w = rank_average(test_preds, w_final)
    Xoof = oof[["oof_lgbm", "oof_catboost", "oof_xgb"]].to_numpy()
    yoof = oof["y"].to_numpy()
    meta = LogisticRegression(C=1.0, penalty="l2", max_iter=2000)
    meta.fit(Xoof, yoof)
    p_stack = meta.predict_proba(np.column_stack([p_lgb, p_cb, p_xgb]))[:, 1]

    p_mono = pred_lgbm_seeds(
        train_f[gbm_feats],
        train_f["objetivo"].to_numpy(),
        test_f[gbm_feats],
        cats_gbm,
        mparams,
        rounds,
        seeds=[42, 777, 2024],
    )
    p_band = predict_band_logistic(train_f, test_f)
    p_blend = W_GBM_PRIOR * (pd.Series(p_mono).rank(method="average").to_numpy() / len(p_mono)) + (
        1 - W_GBM_PRIOR
    ) * (pd.Series(p_band).rank(method="average").to_numpy() / len(p_band))
    # remap a escala de probabilidad del GBM (monótono; Gini invariante)
    orden = pd.Series(p_blend).rank(method="ordinal").astype(int).to_numpy() - 1
    p_blend_prob = np.sort(p_mono)[orden]

    cand = {
        "lgbm": p_lgb,
        "catboost": p_cb,
        "xgb": p_xgb,
        "rank3": p_rank,
        "weighted": p_w,
        "stack_log": p_stack,
        "lgbm_mono": p_mono,
        "band_logistic": p_band,
        "lgbm_mono_bandlog": p_blend_prob,
        "lgbm_cat": rank_average({"lgbm": p_lgb, "catboost": p_cb}),
        "lgbm_xgb": rank_average({"lgbm": p_lgb, "xgb": p_xgb}),
        "cat_xgb": rank_average({"catboost": p_cb, "xgb": p_xgb}),
        "stack_ridge": p_stack,
    }
    pred_key = best_name if best_name in cand else "lgbm_mono_bandlog"
    raw = np.asarray(cand[pred_key], dtype=float)
    raw = np.clip(raw, 0, 1)
    raw = np.nan_to_num(raw, nan=0.5)

    order = test_raw[["id_cliente"]].copy()
    te_map = test_f[["id_cliente"]].copy()
    te_map["prediccion"] = raw
    sub = order.merge(te_map, on="id_cliente", how="left")
    validate_submission(sub, test_raw.reset_index(drop=True))
    sub_path = SUB / "sub_final_ensemble.csv"
    sub.to_csv(sub_path, index=False)

    st = {
        "mean": float(best_row["mean"]),
        "std": float(best_row["std"]),
        "min": float(best_row["min"]),
        "median": float(best_row["median"]),
        "max": float(best_row["max"]),
        "last": float(best_row["last"]),
    }
    gain = st["mean"] - BENCH_GINI
    report = {
        "benchmark_gini": BENCH_GINI,
        "best_model": best_name,
        "best_gini_mean": st["mean"],
        "best_gini_std": st["std"],
        "best_gini_min": st["min"],
        "best_gini_median": st["median"],
        "best_gini_max": st["max"],
        "last_fold_gini": st["last"],
        "absolute_gain": gain,
        "target_0265": bool(st["mean"] > 0.265),
        "target_0270": bool(st["mean"] >= 0.269),
        "target_0275": bool(st["mean"] >= 0.275),
        "leakage_status": leak1["status"],
        "submission_path": "submissions/sub_final_ensemble.csv",
        "benchmark_repro_mean": bench["stats"]["mean"],
        "final_weights": w_final,
        "n_features": len(core_feats),
        "varying_cols": feat_meta["varying_cols"],
        "mono_rounds": rounds,
        "selected_pred_key": pred_key,
    }
    (OUT / "final_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(
        f"""
========================================
BCP GINI OPTIMIZATION
========================================

Benchmark:        {BENCH_GINI:.6f}
Best Model:       {best_name}
Best Gini:        {st['mean']:.6f}
Std:              {st['std']:.6f}
Min Fold:         {st['min']:.6f}
Last Fold:        {st['last']:.6f}
Absolute Gain:    {gain:+.6f}
Target > 0.265:   {"YES" if report["target_0265"] else "NO"}
Target ≈ 0.270:   {"YES" if report["target_0270"] else "NO"}
Target >= 0.275:  {"YES" if report["target_0275"] else "NO"}
Leakage Audit:    {leak1["status"]}

Submission:
{sub_path}

========================================
"""
    )
    log_f.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
