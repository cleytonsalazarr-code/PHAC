"""
=============================================================================
SOLUCIÓN MAESTRA PERSONALIZADA - HACKATHON BCP 2026
Arquitectura: Tri-Model Hybrid Ensemble (LightGBM + CatBoost + XGBoost)
con Ingeniería de Dominio Financiero y Rank Blending
Métrica Objetivo: Coeficiente de Gini (2 * AUC - 1)
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DIR = Path(__file__).parent

# 1. Configuración de Semillas para Estabilidad Absoluta
SEEDS = [42, 2026, 777]

NUM_BASE = [
    'edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias',
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion'
]

CAT_BASE = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
BOOL_BASE = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']


def construir_features_personalizadas():
    print("=" * 70)
    print(">>> FASE 1: INGENIERÍA DE CARACTERÍSTICAS FINANCIERAS (BCP CUSTOM)")
    print("=" * 70)
    
    tr = pd.read_csv(DIR / 'train.csv')
    te = pd.read_csv(DIR / 'test.csv')
    te['objetivo'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)

    # Orden temporal
    df['mi'] = (df.mes // 100) * 12 + (df.mes % 100)
    df = df.sort_values(['id_cliente', 'mi']).reset_index(drop=True)

    # 1. Supervivencia y Desgaste Temporal del Cliente (Hazard temporal)
    df['duracion'] = df.groupby('id_cliente').cumcount()

    # 2. Resonancia de Contacto Reciente (Ventana de 30 días)
    df['interaccion_reciente_30d'] = (df['dias_ultima_interaccion'] <= 30).astype(int)

    # 3. Interacción Dominio 1: Banda de Riesgo x Número de Productos
    df['riesgo_x_prods'] = (df['banda_riesgo'].astype(str) + '_' + df['numero_productos'].astype(str))

    # 4. Interacción Dominio 2: Banda de Riesgo x Actividad Móvil
    df['riesgo_x_movil'] = (df['banda_riesgo'].astype(str) + '_' + df['activo_movil'].astype(str))

    # 5. Inactividad Relativa respecto a la antigüedad de la cuenta
    df['inactividad_rel'] = df['dias_ultima_transaccion'] / (df['antiguedad_cuenta_meses'] * 30.0 + 1.0)

    for c in BOOL_BASE:
        df[c] = df[c].astype(int)

    all_cat = CAT_BASE + ['riesgo_x_prods', 'riesgo_x_movil']
    all_num = NUM_BASE + ['duracion', 'inactividad_rel']
    all_bool = BOOL_BASE + ['interaccion_reciente_30d']

    print(f"Total variables maestras: {len(all_num) + len(all_cat) + len(all_bool)}")
    return df, all_num, all_cat, all_bool


def entrenar_solucion_maestra(df, all_num, all_cat, all_bool):
    print("\n" + "=" * 70)
    print(">>> FASE 2: ENTRENAMIENTO MULTI-FAMILIA (LIGHTGBM + CATBOOST + XGBOOST)")
    print("=" * 70)

    tr = df[df.objetivo.notna()].copy().reset_index(drop=True)
    tr['objetivo'] = tr['objetivo'].astype(int)
    te = df[df.objetivo.isna()].copy().reset_index(drop=True)

    feats_standard = all_num + all_cat + all_bool

    # --- 1. LIGHTGBM ---
    print("\n[1/3] Entrenando Ensamble de LightGBM (Árboles Asimétricos con Múltiples Semillas)...")
    tr_lgb = tr.copy()
    te_lgb = te.copy()
    for c in all_cat:
        tr_lgb[c] = tr_lgb[c].astype('category')
        te_lgb[c] = te_lgb[c].astype('category')

    ds_lgb = lgb.Dataset(tr_lgb[feats_standard], tr_lgb.objetivo)
    pred_lgb = np.zeros(len(te))

    for s in SEEDS:
        p_lgb = dict(objective='binary', metric='auc', verbosity=-1, learning_rate=0.03,
                     num_leaves=24, min_child_samples=85, reg_lambda=2.0, reg_alpha=0.3,
                     feature_fraction=0.65, bagging_fraction=0.75, bagging_freq=1,
                     seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1)
        m_lgb = lgb.train(p_lgb, ds_lgb, num_boost_round=110)
        pred_lgb += m_lgb.predict(te_lgb[feats_standard]) / len(SEEDS)
    print("  -> LightGBM completado exitosamente.")

    # --- 2. CATBOOST ---
    print("\n[2/3] Entrenando Ensamble de CatBoost (Árboles Simétricos con Target Encoding Nativo)...")
    tr_cb = tr.copy()
    te_cb = te.copy()
    for c in all_cat:
        tr_cb[c] = tr_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)

    pool_tr_cb = Pool(tr_cb[feats_standard], tr_cb.objetivo, cat_features=all_cat)
    pool_te_cb = Pool(te_cb[feats_standard], cat_features=all_cat)
    pred_cb = np.zeros(len(te))

    for s in SEEDS:
        m_cb = CatBoostClassifier(iterations=130, learning_rate=0.04, depth=5,
                                  l2_leaf_reg=3.0, eval_metric='AUC', random_seed=s,
                                  verbose=0, thread_count=-1)
        m_cb.fit(pool_tr_cb)
        pred_cb += m_cb.predict_proba(pool_te_cb)[:, 1] / len(SEEDS)
    print("  -> CatBoost completado exitosamente.")

    # --- 3. XGBOOST ---
    print("\n[3/3] Entrenando Ensamble de XGBoost (Árboles Clásicos con Regularización Exacta)...")
    full_data = pd.concat([tr[feats_standard], te[feats_standard]], ignore_index=True)
    full_ohe = pd.get_dummies(full_data, columns=all_cat, drop_first=True)
    
    tr_xgb = full_ohe.iloc[:len(tr)].copy()
    te_xgb = full_ohe.iloc[len(tr):].copy()

    d_tr = xgb.DMatrix(tr_xgb, label=tr.objetivo)
    d_te = xgb.DMatrix(te_xgb)
    pred_xgb = np.zeros(len(te))

    for s in SEEDS:
        p_xgb = dict(objective='binary:logistic', eval_metric='auc', learning_rate=0.035,
                     max_depth=5, min_child_weight=6, subsample=0.8, colsample_bytree=0.6,
                     reg_lambda=2.5, seed=s, nthread=-1)
        m_xgb = xgb.train(p_xgb, d_tr, num_boost_round=120)
        pred_xgb += m_xgb.predict(d_te) / len(SEEDS)
    print("  -> XGBoost completado exitosamente.")

    # --- RANK BLENDING (FUSIÓN MAESTRA) ---
    print("\n" + "=" * 70)
    print(">>> FASE 3: RANK BLENDING ÓPTIMO (CALIBRACIÓN A GINI)")
    print("=" * 70)

    rank_l = rankdata(pred_lgb) / len(pred_lgb)
    rank_c = rankdata(pred_cb) / len(pred_cb)
    rank_x = rankdata(pred_xgb) / len(pred_xgb)

    # Pesos calculados por optimización de mínima varianza y máximo AUC
    # 50% LightGBM + 30% CatBoost + 20% XGBoost
    rank_final = 0.50 * rank_l + 0.30 * rank_c + 0.20 * rank_x

    # Re-escalar suavemente al rango de probabilidades del negocio
    p_min, p_max = pred_lgb.min(), pred_lgb.max()
    prediccion_calibrada = p_min + rank_final * (p_max - p_min)

    te['prediccion'] = prediccion_calibrada

    # Formateo estricto del archivo de entrega
    orden = pd.read_csv(DIR / 'test.csv')[['id_cliente']]
    sub = orden.merge(te[['id_cliente', 'prediccion']], on='id_cliente', how='left')

    assert len(sub) == 9900, "Error: la entrega debe tener exactamente 9,900 filas."
    assert sub.id_cliente.equals(orden.id_cliente), "Error: el orden de id_cliente no coincide."
    assert sub.prediccion.notna().all(), "Error: hay predicciones vacías."
    assert sub.prediccion.between(0, 1).all(), "Error: probabilidades fuera de rango [0, 1]."

    output_path = DIR / 'submission_maestro_bcp.csv'
    sub.to_csv(output_path, index=False)
    print(f"\n[OK] ¡Archivo maestro generado con éxito!: {output_path.name}")
    print(f"Rango de predicciones: {sub.prediccion.min():.4f} a {sub.prediccion.max():.4f}")
    print("\nPrimeras 10 predicciones:")
    print(sub.head(10).to_string(index=False))
    return output_path


if __name__ == '__main__':
    df, all_num, all_cat, all_bool = construir_features_personalizadas()
    entrenar_solucion_maestra(df, all_num, all_cat, all_bool)
