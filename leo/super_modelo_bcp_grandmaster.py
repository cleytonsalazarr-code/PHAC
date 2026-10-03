"""
=============================================================================
SUPER MODELO BCP GRANDMASTER (20-MODEL HYBRID ENSEMBLE)
Arquitectura: 
  - 10 Modelos LightGBM (10 semillas con subsampling aleatorio)
  - 5 Modelos CatBoost (5 semillas con árboles simétricos nativos)
  - 5 Modelos XGBoost (5 semillas con muestreo por columnas y L2 estricto)
Total: 20 Modelos Diversos en Ensamble de Rangos Ponderados
Métrica Objetivo: Coeficiente de Gini Máximo
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
from scipy.stats import rankdata

DIR = Path(__file__).parent

# 20 Semillas científicamente distribuidas
SEEDS_LGB = [42, 2026, 777, 31337, 8191, 101, 555, 999, 1337, 2024]
SEEDS_CB = [42, 2026, 777, 101, 555]
SEEDS_XGB = [42, 2026, 777, 101, 555]

NUM_BASE = [
    'edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias',
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion'
]

CAT_BASE = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
BOOL_BASE = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']


def preparar_super_dataset():
    print("=" * 75)
    print(">>> FASE 1: INGENIERÍA DE ATRIBUTOS AVANZADA (SUPER MODELO GRANDMASTER)")
    print("=" * 75)
    
    tr = pd.read_csv(DIR / 'train.csv')
    te = pd.read_csv(DIR / 'test.csv')
    te['objetivo'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)

    # Orden temporal estricto cliente-mes
    df['mi'] = (df.mes // 100) * 12 + (df.mes % 100)
    df = df.sort_values(['id_cliente', 'mi']).reset_index(drop=True)

    # 1. Supervivencia de Cohorte
    df['duracion'] = df.groupby('id_cliente').cumcount()

    # 2. Resonancia de Interacción Reciente
    df['interaccion_reciente_30d'] = (df['dias_ultima_interaccion'] <= 30).astype(int)

    # 3. Interacción Dominio 1: Riesgo x Número de Productos
    df['riesgo_x_prods'] = (df['banda_riesgo'].astype(str) + '_' + df['numero_productos'].astype(str))

    # 4. Interacción Dominio 2: Riesgo x Actividad Móvil
    df['riesgo_x_movil'] = (df['banda_riesgo'].astype(str) + '_' + df['activo_movil'].astype(str))

    # 5. Inactividad Relativa Ponderada
    df['inactividad_rel'] = df['dias_ultima_transaccion'] / (df['antiguedad_cuenta_meses'] * 30.0 + 1.0)

    # 6. Ratio de Estabilidad Domiciliaria vs Cuenta
    df['estabilidad_dir'] = df['antiguedad_direccion_meses'] / (df['antiguedad_cuenta_meses'] + 1.0)

    for c in BOOL_BASE:
        df[c] = df[c].astype(int)

    all_cat = CAT_BASE + ['riesgo_x_prods', 'riesgo_x_movil']
    all_num = NUM_BASE + ['duracion', 'inactividad_rel', 'estabilidad_dir']
    all_bool = BOOL_BASE + ['interaccion_reciente_30d']

    print(f"Dataset consolidado con éxito: {len(df):,} filas | {len(all_num) + len(all_cat) + len(all_bool)} variables")
    return df, all_num, all_cat, all_bool


def entrenar_super_ensamble(df, all_num, all_cat, all_bool):
    print("\n" + "=" * 75)
    print(">>> FASE 2: ENTRENANDO EL SUPER ENSAMBLE DE 20 MODELOS DIVERSOS")
    print("=" * 75)

    tr = df[df.objetivo.notna()].copy().reset_index(drop=True)
    tr['objetivo'] = tr['objetivo'].astype(int)
    te = df[df.objetivo.isna()].copy().reset_index(drop=True)

    feats_all = all_num + all_cat + all_bool

    # -------------------------------------------------------------
    # 1. BATERÍA LIGHTGBM (10 Semillas con submuestreo estocástico)
    # -------------------------------------------------------------
    print(f"\n[1/3] Entrenando Batería de 10 Modelos LightGBM (110k filas c/u)...")
    tr_lgb = tr.copy()
    te_lgb = te.copy()
    for c in all_cat:
        tr_lgb[c] = tr_lgb[c].astype('category')
        te_lgb[c] = te_lgb[c].astype('category')

    ds_lgb = lgb.Dataset(tr_lgb[feats_all], tr_lgb.objetivo)
    pred_lgb = np.zeros(len(te))

    for i, s in enumerate(SEEDS_LGB):
        p_lgb = dict(
            objective='binary',
            metric='auc',
            verbosity=-1,
            learning_rate=0.028,
            num_leaves=24,
            max_depth=5,
            min_child_samples=85,
            reg_lambda=2.2,
            reg_alpha=0.35,
            feature_fraction=0.62,
            bagging_fraction=0.72,
            bagging_freq=1,
            seed=s,
            bagging_seed=s,
            feature_fraction_seed=s,
            n_jobs=-1
        )
        m_lgb = lgb.train(p_lgb, ds_lgb, num_boost_round=125)
        pred_lgb += m_lgb.predict(te_lgb[feats_all]) / len(SEEDS_LGB)
        print(f"   [OK] LightGBM #{i+1:02d} completado (seed={s})")

    # -------------------------------------------------------------
    # 2. BATERÍA CATBOOST (5 Semillas con Oblivious Trees)
    # -------------------------------------------------------------
    print(f"\n[2/3] Entrenando Batería de 5 Modelos CatBoost...")
    tr_cb = tr.copy()
    te_cb = te.copy()
    for c in all_cat:
        tr_cb[c] = tr_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)

    pool_tr_cb = Pool(tr_cb[feats_all], tr_cb.objetivo, cat_features=all_cat)
    pool_te_cb = Pool(te_cb[feats_all], cat_features=all_cat)
    pred_cb = np.zeros(len(te))

    for i, s in enumerate(SEEDS_CB):
        m_cb = CatBoostClassifier(
            iterations=140,
            learning_rate=0.038,
            depth=5,
            l2_leaf_reg=3.2,
            eval_metric='AUC',
            random_seed=s,
            verbose=0,
            thread_count=-1
        )
        m_cb.fit(pool_tr_cb)
        pred_cb += m_cb.predict_proba(pool_te_cb)[:, 1] / len(SEEDS_CB)
        print(f"   [OK] CatBoost #{i+1:02d} completado (seed={s})")

    # -------------------------------------------------------------
    # 3. BATERÍA XGBOOST (5 Semillas con ColSample y L2 Estricto)
    # -------------------------------------------------------------
    print(f"\n[3/3] Entrenando Batería de 5 Modelos XGBoost...")
    full_data = pd.concat([tr[feats_all], te[feats_all]], ignore_index=True)
    full_ohe = pd.get_dummies(full_data, columns=all_cat, drop_first=True)
    
    tr_xgb = full_ohe.iloc[:len(tr)].copy()
    te_xgb = full_ohe.iloc[len(tr):].copy()

    d_tr = xgb.DMatrix(tr_xgb, label=tr.objetivo)
    d_te = xgb.DMatrix(te_xgb)
    pred_xgb = np.zeros(len(te))

    for i, s in enumerate(SEEDS_XGB):
        p_xgb = dict(
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=0.032,
            max_depth=5,
            min_child_weight=6,
            subsample=0.78,
            colsample_bytree=0.60,
            reg_lambda=2.8,
            seed=s,
            nthread=-1
        )
        m_xgb = xgb.train(p_xgb, d_tr, num_boost_round=135)
        pred_xgb += m_xgb.predict(d_te) / len(SEEDS_XGB)
        print(f"   [OK] XGBoost  #{i+1:02d} completado (seed={s})")

    # -------------------------------------------------------------
    # 4. FUSIÓN MAESTRA POR RANK BLENDING PONDERADO
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(">>> FASE 3: FUSIÓN DE 20 MODELOS VÍA RANK BLENDING ÓPTIMO")
    print("=" * 75)

    rank_l = rankdata(pred_lgb) / len(pred_lgb)
    rank_c = rankdata(pred_cb) / len(pred_cb)
    rank_x = rankdata(pred_xgb) / len(pred_xgb)

    # Ponderación maestra óptima para máxima estabilidad
    rank_final = 0.50 * rank_l + 0.30 * rank_c + 0.20 * rank_x

    # Re-escalar suavemente al rango natural de propensión bancaria
    p_min, p_max = pred_lgb.min(), pred_lgb.max()
    prediccion_final = p_min + rank_final * (p_max - p_min)

    te['prediccion'] = prediccion_final

    # Control de calidad y validación de reglas
    orden = pd.read_csv(DIR / 'test.csv')[['id_cliente']]
    sub = orden.merge(te[['id_cliente', 'prediccion']], on='id_cliente', how='left')

    assert len(sub) == 9900, "Error: debe contener exactamente 9,900 observaciones."
    assert sub.id_cliente.equals(orden.id_cliente), "Error: el orden de id_cliente no coincide."
    assert sub.prediccion.notna().all(), "Error: existen valores nulos."
    assert sub.prediccion.between(0, 1).all(), "Error: existen probabilidades fuera de [0, 1]."

    output_path = DIR / 'submission_super_poderoso.csv'
    sub.to_csv(output_path, index=False)
    print(f"\n[ÉXITO TOTAL] ¡Super Archivo generado!: {output_path.name}")
    print(f"Total clientes predichos: {len(sub):,}")
    print(f"Rango de predicciones: {sub.prediccion.min():.4f} a {sub.prediccion.max():.4f}")
    print("\nMuestra de las primeras 10 predicciones definitivas:")
    print(sub.head(10).to_string(index=False))
    return output_path


if __name__ == '__main__':
    df, all_num, all_cat, all_bool = preparar_super_dataset()
    entrenar_super_ensamble(df, all_num, all_cat, all_bool)
