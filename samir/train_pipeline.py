"""
train_pipeline.py
-----------------
Pipeline principal — Hackathon BCP.

ORDEN DE EJECUCION:
    1. Auditoria de datos (empirica, no assumptions)
    2. Configuracion de folds temporales
    3. EXP_01 — Baseline estatico (features sin historia)
    4. EXP_02 — Baseline longitudinal (lags, deltas, ratios)
    5. Comparacion de experimentos (tabla)
    6. Generacion de submission para el mejor experimento

REGLAS:
    - NO usar train_test_split / KFold / StratifiedKFold como validacion principal.
    - Toda feature historica usa informacion disponible antes de la observacion actual.
    - Si se detecta leakage: DETENER experimento y reportar.
    - No asumir valores: todo resultado proviene de la ejecucion real.

OUTPUTS:
    outputs/experiment_log.csv              — tabla comparativa de experimentos
    submissions/sub_v1_baseline.csv         — EXP_01 submission
    submissions/sub_v2_longitudinal.csv     — EXP_02 submission
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------
# Importar modulos del proyecto
# ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_audit import run_full_audit
from src.validation import PurgedTimeSeriesSplit
from src.features import (
    CAT_COLS,
    encode_base_features,
    build_longitudinal_features,
    get_feature_cols_baseline,
    get_feature_cols_longitudinal,
    prepare_X_y,
)
from src.leakage_audit import audit_leakage, quick_feature_check
from src.models import (
    train_lgbm,
    get_lgbm_params,
    predict_test,
    get_feature_importance,
)
from src.metrics import gini_score, auc_score

OUTPUT_DIR = ROOT_DIR / "outputs"
SUBMISSION_DIR = ROOT_DIR / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file(name: str) -> Path:
    """Busca archivo en data/ o en raíz."""
    for candidate in [ROOT_DIR / "data" / name, ROOT_DIR / name]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro el archivo: {name}")


def prepare_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Prepara features base para entrenamiento o evaluacion."""
    df_train_enc = encode_base_features(df_train)
    df_test_enc = encode_base_features(df_test)
    feature_cols = [c for c in get_feature_cols_baseline() if c in df_train_enc.columns]
    cat_cols = [f"{c}_enc" for c in CAT_COLS if f"{c}_enc" in df_train_enc.columns]
    return df_train_enc, df_test_enc, feature_cols, cat_cols


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, test y sample_submission."""
    df_train = pd.read_csv(find_data_file("train.csv"))
    df_test = pd.read_csv(find_data_file("test.csv"))
    df_sample = pd.read_csv(find_data_file("sample_submission.csv"))
    return df_train, df_test, df_sample


def _section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_experiment(
    experiment_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_sample: pd.DataFrame,
    feature_cols: list[str],
    submission_path: Path,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta un experimento completo con validacion temporal de 4 folds.
    """
    _section(f"EXPERIMENTO: {experiment_name}")

    # A. Chequeo rapido de features
    all_train_cols = df_train.columns.tolist()
    if not quick_feature_check(feature_cols, all_train_cols):
        print(f"  [ERROR] Experimento {experiment_name} cancelado por features invalidas.")
        return {"experiment": experiment_name, "status": "ERROR_FEATURES"}

    # B. Auditoria de leakage
    if verbose:
        print("\nAuditoria de leakage...")
    is_clean = audit_leakage(df_train, feature_cols, verbose=verbose)

    if not is_clean:
        print(f"\n  [DETENIDO] Experimento {experiment_name} cancelado por leakage.")
        return {"experiment": experiment_name, "status": "LEAKAGE_DETECTED"}

    # C. Configurar folds temporales con PurgedTimeSeriesSplit
    cat_cols = [f"{c}_enc" for c in CAT_COLS if f"{c}_enc" in df_train.columns]
    splitter = PurgedTimeSeriesSplit()
    
    boosters = []
    fold_ginis = []
    fold_aucs = []
    fold_num = 0

    for train_idx, val_idx in splitter.split(df_train):
        fold_num += 1
        X_train = df_train.loc[train_idx, feature_cols]
        y_train = df_train.loc[train_idx, "objetivo"].values.astype(np.int8)
        X_val = df_train.loc[val_idx, feature_cols]
        y_val = df_train.loc[val_idx, "objetivo"].values.astype(np.int8)

        booster, metrics = train_lgbm(
            X_train, y_train, X_val, y_val,
            categorical_features=cat_cols,
        )

        if verbose:
            print(f"  Fold {fold_num} | Mejor iter: {metrics['best_iteration']} | "
                  f"AUC={metrics['auc']:.5f} | Gini={metrics['gini']:.5f}")

        boosters.append(booster)
        fold_ginis.append(metrics["gini"])
        fold_aucs.append(metrics["auc"])

    # E. Resumen de CV
    mean_gini = float(np.mean(fold_ginis))
    std_gini = float(np.std(fold_ginis))
    last_gini = fold_ginis[-1]
    mean_auc = float(np.mean(fold_aucs))
    last_auc = fold_aucs[-1]

    print(f"\n  Resumen CV — {experiment_name}:")
    print(f"    Gini Promedio:       {mean_gini:.5f}")
    print(f"    Gini Desv. Est.:     {std_gini:.5f}")
    print(f"    Gini Último Fold:    {last_gini:.5f}  (nov 2026 ← escenario real)")
    print(f"    AUC  Promedio:       {mean_auc:.5f}")

    # F. Feature importance
    print("\n  Top 20 features por importancia (gain, promedio de folds):")
    imp_df = get_feature_importance(
        boosters,
        feature_names=[c for c in feature_cols if c in df_train.columns],
        importance_type="gain",
        top_n=20,
    )
    print(imp_df[["feature", "importance"]].to_string(index=False))

    imp_path = OUTPUT_DIR / f"feature_importance_{experiment_name}.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"\n  Feature importance guardada en: {imp_path}")

    # G. Prediccion sobre test
    test_feature_cols = [c for c in feature_cols if c in df_test.columns]
    X_test = df_test[test_feature_cols]
    test_preds = predict_test(boosters, X_test, average=True)

    # Construccion de submission
    submission = pd.DataFrame({
        "id_cliente": df_test["id_cliente"].values,
        "prediccion": test_preds,
    })

    submission.to_csv(submission_path, index=False)
    print(f"\n  Submission guardada: {submission_path}")
    print(f"  Shape submission: {submission.shape}")

    result = {
        "experiment": experiment_name,
        "status": "OK",
        "gini_mean": round(mean_gini, 5),
        "gini_std": round(std_gini, 5),
        "gini_last_fold": round(last_gini, 5),
        "auc_mean": round(mean_auc, 5),
        "auc_last_fold": round(last_auc, 5),
        "n_features": len(test_feature_cols),
        "n_folds": fold_num,
    }

    return result


def main() -> None:
    """Ejecuta el pipeline completo: auditoria + experimentos + comparacion."""
    warnings.filterwarnings("ignore")

    _section("PIPELINE — HACKATHON BCP — INICIO")

    # PASO 1: Cargar datos
    _section("PASO 1 — CARGA DE DATOS")
    df_train_raw, df_test_raw, df_sample = _load_data()
    print(f"  Train shape: {df_train_raw.shape}")
    print(f"  Test  shape: {df_test_raw.shape}")

    # PASO 2: Auditoria de datos
    _section("PASO 2 — AUDITORIA DE DATOS")
    run_full_audit()

    # PASO 3: EXP_01 — Baseline estatico
    df_train_base = encode_base_features(df_train_raw)
    df_test_base = encode_base_features(df_test_raw)

    feat_baseline = get_feature_cols_baseline()
    feat_baseline_valid = [c for c in feat_baseline if c in df_train_base.columns]

    exp01_result = run_experiment(
        experiment_name="EXP_01_baseline_estatico",
        df_train=df_train_base,
        df_test=df_test_base,
        df_sample=df_sample,
        feature_cols=feat_baseline_valid,
        submission_path=SUBMISSION_DIR / "sub_v1_baseline.csv",
    )

    # PASO 4: EXP_02 — Baseline longitudinal
    _section("PASO 3 — CONSTRUCCION DE FEATURES LONGITUDINALES")
    print("  Construyendo lags, deltas y ratios (shift(1) por cliente)...")

    df_train_long, df_test_long = build_longitudinal_features(
        df_train_raw, df_test_raw
    )
    df_train_long = encode_base_features(df_train_long)
    df_test_long = encode_base_features(df_test_long)

    # Recalcular mes_num con el rango completo
    all_mes = pd.concat([df_train_long["mes"], df_test_long["mes"]])
    min_mes = all_mes.min()
    df_train_long["mes_num"] = df_train_long["mes"] - min_mes
    df_test_long["mes_num"] = df_test_long["mes"] - min_mes

    feat_longitudinal = get_feature_cols_longitudinal()
    feat_longitudinal_valid = [c for c in feat_longitudinal if c in df_train_long.columns]

    print(f"  Features baseline:     {len(feat_baseline_valid)}")
    print(f"  Features longitudinal: {len(feat_longitudinal_valid)}")
    print(f"  Features nuevas:       {len(feat_longitudinal_valid) - len(feat_baseline_valid)}")

    exp02_result = run_experiment(
        experiment_name="EXP_02_longitudinal",
        df_train=df_train_long,
        df_test=df_test_long,
        df_sample=df_sample,
        feature_cols=feat_longitudinal_valid,
        submission_path=SUBMISSION_DIR / "sub_v2_longitudinal.csv",
    )

    # PASO 5: Comparacion de experimentos
    _section("PASO 5 — COMPARACION DE EXPERIMENTOS")

    results = [exp01_result, exp02_result]
    results_df = pd.DataFrame(results)

    print("\n  Tabla comparativa:")
    print(results_df[[
        "experiment", "n_features",
        "gini_mean", "gini_std", "gini_last_fold",
        "auc_mean", "status"
    ]].to_string(index=False))

    valid_results = [r for r in results if r.get("status") == "OK"]
    if valid_results:
        best = max(valid_results, key=lambda r: r.get("gini_last_fold", -1))
        print(f"\n  Mejor experimento (Gini ultimo fold): {best['experiment']}")
        print(f"    Gini Promedio:      {best['gini_mean']:.5f}")
        print(f"    Gini Ultimo Fold:   {best['gini_last_fold']:.5f}")

    log_path = OUTPUT_DIR / "experiment_log.csv"
    results_df.to_csv(log_path, index=False)
    print(f"\n  Log de experimentos guardado: {log_path}")

    _section("PIPELINE COMPLETADO")
    print("  Archivos generados:")
    print(f"    submissions/sub_v1_baseline.csv     — EXP_01")
    print(f"    submissions/sub_v2_longitudinal.csv — EXP_02")
    print(f"    outputs/experiment_log.csv          — tabla comparativa")
    print(f"    outputs/data_audit_report.csv       — auditoria de datos")
    print(f"    outputs/feature_importance_*.csv    — importancia por experimento")


if __name__ == "__main__":
    main()
