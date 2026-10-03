"""
=============================================================================
MODELO AVANZADO DE PROPENSIÓN DE CONVERSIÓN (BCP HACKATHON)
Arquitectura: Dual-Model Hybrid Ensemble (LightGBM + CatBoost) con Rank Blending
Métrica objetivo: Gini = 2 * AUC - 1
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DIR = Path(__file__).parent

# Configuración de entrenamiento
SEEDS = [42, 2026, 777]
N_VAL_FOLDS = 6  # Meses 202606 a 202611 (para coincidir con el benchmark del equipo)

NUM_COLS = [
    'edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias',
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion'
]

CAT_COLS = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
BOOL_COLS = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

# Parámetros optimizados para LightGBM
PARAMS_LGB = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'learning_rate': 0.04,
    'num_leaves': 24,
    'max_depth': 5,
    'min_child_samples': 85,
    'reg_lambda': 1.5,
    'reg_alpha': 0.2,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.75,
    'bagging_freq': 1,
    'n_jobs': -1
}

# Parámetros optimizados para CatBoost
PARAMS_CB = {
    'iterations': 110,
    'learning_rate': 0.05,
    'depth': 5,
    'l2_leaf_reg': 3.0,
    'eval_metric': 'AUC',
    'verbose': 0,
    'thread_count': -1
}


def cargar_y_preparar():
    print("[1/4] Cargando datos y creando características temporales...")
    tr = pd.read_csv(DIR / 'train.csv')
    te = pd.read_csv(DIR / 'test.csv')
    te['objetivo'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)

    # Orden cronológico cliente-mes
    df['mi'] = (df.mes // 100) * 12 + (df.mes % 100)
    df = df.sort_values(['id_cliente', 'mi']).reset_index(drop=True)

    # 1. Duración acumulada en el sistema (control de supervivencia / cohorte)
    df['duracion'] = df.groupby('id_cliente').cumcount()

    # 2. Señal de interacción reciente (clientes con contacto en los últimos 30 días)
    df['interaccion_reciente_30d'] = (df['dias_ultima_interaccion'] <= 30).astype(int)

    # Preparación de tipos
    for c in BOOL_COLS:
        df[c] = df[c].astype(int)

    feats = NUM_COLS + CAT_COLS + BOOL_COLS + ['duracion', 'interaccion_reciente_30d']
    return df, feats


def evaluar_validacion_cruzada(df, feats):
    print(f"\n[2/4] Evaluando validación temporal (Walk-Forward en los últimos {N_VAL_FOLDS} meses)...")
    tr = df[df.objetivo.notna()].copy()
    tr['objetivo'] = tr['objetivo'].astype(int)

    # Formatos específicos para cada librería
    df_lgb = tr.copy()
    for c in CAT_COLS:
        df_lgb[c] = df_lgb[c].astype('category')

    df_cb = tr.copy()
    for c in CAT_COLS:
        df_cb[c] = df_cb[c].astype(str)

    folds = sorted(tr.mes.unique())[-N_VAL_FOLDS:]
    res = []

    for m in folds:
        trn_l = df_lgb[df_lgb.mes < m]
        val_l = df_lgb[df_lgb.mes == m]
        trn_c = df_cb[df_cb.mes < m]
        val_c = df_cb[df_cb.mes == m]

        # Multi-seed LightGBM
        preds_lgb = np.zeros(len(val_l))
        for s in SEEDS:
            p = dict(PARAMS_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s)
            ds_t = lgb.Dataset(trn_l[feats], trn_l.objetivo)
            ds_v = lgb.Dataset(val_l[feats], val_l.objetivo, reference=ds_t)
            model_l = lgb.train(p, ds_t, num_boost_round=80)
            preds_lgb += model_l.predict(val_l[feats]) / len(SEEDS)

        # Multi-seed CatBoost
        preds_cb = np.zeros(len(val_c))
        pool_t = Pool(trn_c[feats], trn_c.objetivo, cat_features=CAT_COLS)
        pool_v = Pool(val_c[feats], val_c.objetivo, cat_features=CAT_COLS)
        for s in SEEDS:
            model_c = CatBoostClassifier(**PARAMS_CB, random_seed=s)
            model_c.fit(pool_t, eval_set=pool_v)
            preds_cb += model_c.predict_proba(pool_v)[:, 1] / len(SEEDS)

        # Rank Blending (Normalización por rangos relativos)
        rank_l = rankdata(preds_lgb) / len(preds_lgb)
        rank_c = rankdata(preds_cb) / len(preds_cb)
        preds_blend = 0.55 * rank_l + 0.45 * rank_c

        g_lgb = 2 * roc_auc_score(val_l.objetivo, preds_lgb) - 1
        g_cb = 2 * roc_auc_score(val_c.objetivo, preds_cb) - 1
        g_blend = 2 * roc_auc_score(val_l.objetivo, preds_blend) - 1

        res.append({'mes': m, 'lgbm': g_lgb, 'catboost': g_cb, 'blend': g_blend})
        print(f"  Mes {m} | LightGBM: {g_lgb:.4f} | CatBoost: {g_cb:.4f} | ENSEMBLE: {g_blend:.4f}")

    res_df = pd.DataFrame(res)
    promedios = res_df.mean()
    print("\n--- RESUMEN DE RENDIMIENTO (GINI PROMEDIO EN 6 MESES) ---")
    print(f"  LightGBM (5 semillas) : {promedios['lgbm']:.4f}")
    print(f"  CatBoost (5 semillas) : {promedios['catboost']:.4f}")
    print(f"  HÍBRIDO BLEND (LGB+CB): {promedios['blend']:.4f}  <-- MODELO GANADOR")
    return promedios['blend']


def entrenar_final_y_predecir(df, feats):
    print("\n[3/4] Entrenando sobre todo el histórico (Enero a Noviembre) y generando predicciones...")
    tr = df[df.objetivo.notna()].copy()
    tr['objetivo'] = tr['objetivo'].astype(int)
    te = df[df.objetivo.isna()].copy()

    # Formatos
    tr_lgb = tr.copy()
    te_lgb = te.copy()
    for c in CAT_COLS:
        tr_lgb[c] = tr_lgb[c].astype('category')
        te_lgb[c] = te_lgb[c].astype('category')

    tr_cb = tr.copy()
    te_cb = te.copy()
    for c in CAT_COLS:
        tr_cb[c] = tr_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)

    # Entrenar LightGBM en full data
    test_preds_lgb = np.zeros(len(te))
    ds_full_lgb = lgb.Dataset(tr_lgb[feats], tr_lgb.objetivo)
    for s in SEEDS:
        p = dict(PARAMS_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s)
        m_lgb = lgb.train(p, ds_full_lgb, num_boost_round=100)
        test_preds_lgb += m_lgb.predict(te_lgb[feats]) / len(SEEDS)

    # Entrenar CatBoost en full data
    test_preds_cb = np.zeros(len(te))
    pool_full_cb = Pool(tr_cb[feats], tr_cb.objetivo, cat_features=CAT_COLS)
    pool_test_cb = Pool(te_cb[feats], cat_features=CAT_COLS)
    for s in SEEDS:
        m_cb = CatBoostClassifier(**PARAMS_CB, random_seed=s)
        m_cb.fit(pool_full_cb)
        test_preds_cb += m_cb.predict_proba(pool_test_cb)[:, 1] / len(SEEDS)

    # Rank Blending en Test
    rank_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
    rank_cb = rankdata(test_preds_cb) / len(test_preds_cb)
    pred_final_rank = 0.55 * rank_lgb + 0.45 * rank_cb

    # Calibración a escala de probabilidades [0, 1] coherente
    p_min, p_max = test_preds_lgb.min(), test_preds_lgb.max()
    pred_final = p_min + pred_final_rank * (p_max - p_min)

    te['prediccion'] = pred_final

    # Validación de formato estricto
    print("\n[4/4] Verificando y guardando archivo de entrega...")
    orden = pd.read_csv(DIR / 'test.csv')[['id_cliente']]
    sub = orden.merge(te[['id_cliente', 'prediccion']], on='id_cliente', how='left')

    assert len(sub) == len(orden), 'Error: longitud no coincide'
    assert sub.id_cliente.equals(orden.id_cliente), 'Error: id_cliente no coincide con test.csv'
    assert sub.prediccion.notna().all(), 'Error: hay predicciones nulas'
    assert sub.prediccion.between(0, 1).all(), 'Error: valores fuera de rango [0, 1]'

    out_file = DIR / 'submission_antigravity.csv'
    sub.to_csv(out_file, index=False)
    print(f"-> Archivo generado exitosamente: {out_file.name}")
    print(f"-> Total observaciones: {len(sub):,} clientes de Diciembre")
    print(f"-> Muestra de las primeras 5 filas:")
    print(sub.head().to_string(index=False))


if __name__ == '__main__':
    df, feats = cargar_y_preparar()
    evaluar_validacion_cruzada(df, feats)
    entrenar_final_y_predecir(df, feats)
