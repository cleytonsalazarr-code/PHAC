"""
run_exp02.py
------------
Ejecuta solo EXP_02 (features longitudinales) con el leakage audit corregido.
Reutiliza EXP_01 ya ejecutado.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import (
    encode_base_features,
    build_longitudinal_features,
    get_feature_cols_longitudinal,
)
from src.leakage_audit import audit_leakage, quick_feature_check
from src.validation import get_temporal_folds, print_fold_summary, print_cv_summary
from src.models import (
    LGBM_BASE_PARAMS,
    train_fold,
    predict_test,
    get_feature_importance,
)
from src.metrics import gini_score, auc_score

OUTPUT_DIR = ROOT_DIR / "outputs"
SUBMISSION_DIR = ROOT_DIR / "submissions"


def _find(name: str) -> Path:
    for candidate in [ROOT_DIR / "data" / name, ROOT_DIR / name]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro: {name}")


def main():
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("  EXP_02 — BASELINE LONGITUDINAL (LAGS + DELTAS + RATIOS)")
    print("=" * 70)

    # -------------------------------------------------------
    # 1. Cargar datos
    # -------------------------------------------------------
    df_train_raw = pd.read_csv(_find("train.csv"))
    df_test_raw = pd.read_csv(_find("test.csv"))
    df_sample = pd.read_csv(_find("sample_submission.csv"))

    print(f"\nTrain: {df_train_raw.shape}  |  Test: {df_test_raw.shape}")

    # -------------------------------------------------------
    # 2. Construir features longitudinales (anti-leakage: shift(1))
    # -------------------------------------------------------
    print("\nConstruyendo features longitudinales...")
    df_train_long, df_test_long = build_longitudinal_features(
        df_train_raw, df_test_raw
    )

    # Encodificar categoricas y booleanas
    df_train_long = encode_base_features(df_train_long)
    df_test_long = encode_base_features(df_test_long)

    # Normalizar mes_num con rango completo train+test
    all_mes = pd.concat([df_train_long["mes"], df_test_long["mes"]])
    min_mes = int(all_mes.min())
    df_train_long["mes_num"] = df_train_long["mes"] - min_mes
    df_test_long["mes_num"] = df_test_long["mes"] - min_mes

    # -------------------------------------------------------
    # 3. Verificar features disponibles
    # -------------------------------------------------------
    feat_longitudinal = get_feature_cols_longitudinal()
    feat_valid = [c for c in feat_longitudinal if c in df_train_long.columns]
    feat_missing = [c for c in feat_longitudinal if c not in df_train_long.columns]

    print(f"\nFeatures declaradas: {len(feat_longitudinal)}")
    print(f"Features presentes:  {len(feat_valid)}")
    if feat_missing:
        print(f"Features ausentes:   {feat_missing}")

    # -------------------------------------------------------
    # 4. Auditoria de leakage (con comparacion NaN-safe)
    # -------------------------------------------------------
    print("\nAuditoria de leakage...")
    is_clean = audit_leakage(df_train_long, feat_valid, verbose=True)

    if not is_clean:
        print("\n[DETENIDO] EXP_02 cancelado por leakage. Revisar src/features.py")
        return

    # -------------------------------------------------------
    # 5. Inspeccionar lags manualmente (muestra de 5 clientes)
    # -------------------------------------------------------
    print("\nInspeccion manual de lags (muestra de 3 clientes con historia):")
    clientes_multi = (
        df_train_long.groupby("id_cliente").filter(lambda g: len(g) >= 2)
        ["id_cliente"].unique()[:3]
    )
    for cid in clientes_multi:
        subset = (
            df_train_long[df_train_long["id_cliente"] == cid]
            [["id_cliente", "mes", "saldo_promedio", "prev_saldo", "delta_saldo"]]
            .sort_values("mes")
        )
        print(subset.to_string(index=False))
        print()

    # -------------------------------------------------------
    # 6. Cross-validation temporal (4 folds: ago, sep, oct, nov)
    # -------------------------------------------------------
    folds = get_temporal_folds(df_train_long, month_col="mes")
    print_fold_summary(folds)

    boosters = []

    for fold in folds:
        X_train = df_train_long.loc[fold.train_idx, feat_valid]
        y_train = df_train_long.loc[fold.train_idx, "objetivo"].values.astype(np.int8)
        X_val = df_train_long.loc[fold.val_idx, feat_valid]
        y_val = df_train_long.loc[fold.val_idx, "objetivo"].values.astype(np.int8)

        params = LGBM_BASE_PARAMS.copy()

        booster, val_preds, metrics = train_fold(
            X_train, y_train, X_val, y_val,
            params=params,
            fold_num=fold.fold,
            verbose=True,
        )
        fold.gini = metrics["gini"]
        fold.auc = metrics["auc"]
        boosters.append(booster)

    # -------------------------------------------------------
    # 7. Resumen CV
    # -------------------------------------------------------
    agg = print_cv_summary("EXP_02_longitudinal", folds)

    # -------------------------------------------------------
    # 8. Feature importance
    # -------------------------------------------------------
    imp_df = get_feature_importance(
        boosters,
        feature_names=feat_valid,
        importance_type="gain",
        top_n=25,
    )
    print("\n  Top 25 features por importancia (gain):")
    print(imp_df[["feature", "importance"]].to_string(index=False))

    imp_path = OUTPUT_DIR / "feature_importance_EXP_02_longitudinal.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"\n  Feature importance guardada: {imp_path}")

    # -------------------------------------------------------
    # 9. Prediccion sobre test y submission
    # -------------------------------------------------------
    feat_test_valid = [c for c in feat_valid if c in df_test_long.columns]
    X_test = df_test_long[feat_test_valid]
    test_preds = predict_test(boosters, X_test, average=True)

    submission = pd.DataFrame({
        "id_cliente": df_test_long["id_cliente"].values,
        "probabilidad_conversion": test_preds,
    })
    sub_path = SUBMISSION_DIR / "sub_v2_longitudinal.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\n  Submission EXP_02 guardada: {sub_path}")
    print(f"  Filas: {len(submission)}")

    # -------------------------------------------------------
    # 10. Comparacion EXP_01 vs EXP_02
    # -------------------------------------------------------
    log_path = OUTPUT_DIR / "experiment_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame()

    exp02_row = {
        "experiment": "EXP_02_longitudinal",
        "status": "OK",
        "gini_mean": round(agg["gini_mean"], 5),
        "gini_std": round(agg["gini_std"], 5),
        "gini_last_fold": round(agg["gini_last_fold"], 5),
        "auc_mean": round(agg["auc_mean"], 5),
        "auc_last_fold": round(agg["auc_last_fold"], 5),
        "n_features": len(feat_test_valid),
        "n_folds": agg["n_folds"],
    }

    # Actualizar o agregar fila EXP_02
    if not log_df.empty and "experiment" in log_df.columns:
        log_df = log_df[log_df["experiment"] != "EXP_02_longitudinal"]
        log_df = pd.concat([log_df, pd.DataFrame([exp02_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([exp02_row])

    log_df.to_csv(log_path, index=False)

    print("\n" + "=" * 70)
    print("  TABLA COMPARATIVA FINAL")
    print("=" * 70)
    display_cols = ["experiment", "n_features", "gini_mean", "gini_std",
                    "gini_last_fold", "auc_mean", "status"]
    available_cols = [c for c in display_cols if c in log_df.columns]
    print(log_df[available_cols].to_string(index=False))

    # Ganador
    ok_rows = log_df[log_df["status"] == "OK"]
    if not ok_rows.empty:
        best = ok_rows.loc[ok_rows["gini_last_fold"].idxmax()]
        print(f"\n  Mejor experimento (Gini ultimo fold): {best['experiment']}")
        print(f"    Gini Promedio:    {best['gini_mean']:.5f} +/- {best['gini_std']:.5f}")
        print(f"    Gini Ultimo Fold: {best['gini_last_fold']:.5f}")

        improvement = None
        if len(ok_rows) >= 2:
            ginis = ok_rows.set_index("experiment")["gini_last_fold"]
            if "EXP_01_baseline_estatico" in ginis and "EXP_02_longitudinal" in ginis:
                improvement = ginis["EXP_02_longitudinal"] - ginis["EXP_01_baseline_estatico"]
                print(f"\n    Mejora EXP_02 vs EXP_01 (Gini ultimo fold): {improvement:+.5f}")
                if improvement > 0:
                    print("    -> Lags y deltas aportan valor. Usar sub_v2_longitudinal.csv")
                else:
                    print("    -> Baseline estatico es igual o mejor en el ultimo fold.")

    print("\n  Pipeline EXP_02 completado.")


if __name__ == "__main__":
    main()
