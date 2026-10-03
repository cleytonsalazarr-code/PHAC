"""
=============================================================================
AUDITORÍA Y SOLUCIÓN DEFINITIVA - HACKATHON BCP 2026
Protocolo Estricto de Validación Walk-Forward y Comparación Pareada
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DIR = Path(__file__).parent
SEEDS = [42, 777, 2024]
N_VAL_FOLDS = 7
MAX_ROUNDS = 300
W_GBM = 0.75

NUM = ['edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses', 'numero_productos',
       'saldo_promedio', 'dias_ultima_transaccion', 'antiguedad_direccion_meses',
       'visitas_web_ultimos_90_dias', 'distancia_sucursal_km', 'dia_preferido_pago',
       'dias_ultima_interaccion']
CAT = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
BOOL = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']
FEATS = NUM + CAT + BOOL + ['duracion']

PARAMS = dict(objective='binary', metric='auc', verbosity=-1, learning_rate=0.05, num_leaves=16,
              max_depth=-1, min_child_samples=400, reg_lambda=20, reg_alpha=0,
              feature_fraction=1.0, bagging_fraction=0.7, bagging_freq=1)


def build():
    tr = pd.read_csv(DIR / 'train.csv')
    te = pd.read_csv(DIR / 'test.csv')
    assert tr.notna().all().all() and te.notna().all().all(), 'aparecieron nulos'
    te['objetivo'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df['mi'] = (df.mes // 100) * 12 + (df.mes % 100)
    df = df.sort_values(['id_cliente', 'mi']).reset_index(drop=True)
    df['duracion'] = df.groupby('id_cliente').cumcount()
    for c in CAT:
        df[c] = df[c].astype('category')
    for c in BOOL:
        df[c] = df[c].astype(int)
    return df


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def rk(v):
    return rankdata(v) / len(v)


def mezcla(p_gbm, p_par):
    return W_GBM * rk(p_gbm) + (1 - W_GBM) * rk(p_par)


def diseno(d, spl=None, n_knots=6):
    cont = d[['dias_ultima_transaccion', 'dias_ultima_interaccion', 'ratio_deuda_ingresos']].values
    if spl is None:
        spl = SplineTransformer(n_knots=n_knots, degree=3).fit(cont)
    S = pd.DataFrame(spl.transform(cont), index=d.index).add_prefix('s')
    npd = pd.get_dummies(d.numero_productos, prefix='np', drop_first=True).astype(float)
    b = d[['activo_movil', 'tiene_tarjeta_credito']].astype(float)
    X = pd.concat([
        npd,
        pd.get_dummies(d.canal_adquisicion, prefix='cn', drop_first=True).astype(float),
        pd.get_dummies(d.duracion.clip(upper=8), prefix='du', drop_first=True).astype(float),
        S, b,
        (b.activo_movil * b.tiene_tarjeta_credito).rename('am_x_tc').to_frame(),
        S.mul(b.activo_movil, axis=0).add_suffix('_xam'),
        npd.mul(b.activo_movil, axis=0).add_suffix('_xam'),
        npd.mul(b.tiene_tarjeta_credito, axis=0).add_suffix('_xtc'),
    ], axis=1)
    return X, spl


def pred_param(trn, obj, n_knots=6, c_val=0.05):
    out = np.zeros(len(obj))
    for banda in trn.banda_riesgo.cat.categories:
        mt, mo = (trn.banda_riesgo == banda).values, (obj.banda_riesgo == banda).values
        if mo.sum() == 0:
            continue
        Xt, spl = diseno(trn[mt], n_knots=n_knots)
        Xo, _ = diseno(obj[mo], spl, n_knots=n_knots)
        Xo = Xo.reindex(columns=Xt.columns, fill_value=0.0)
        sc = StandardScaler().fit(Xt)
        m = LogisticRegression(max_iter=5000, C=c_val).fit(sc.transform(Xt), trn[mt].objetivo)
        out[mo] = m.predict_proba(sc.transform(Xo))[:, 1]
    return out


def curva_gbm(trn, val, params, rounds=MAX_ROUNDS):
    ds_t = lgb.Dataset(trn[FEATS], trn.objetivo, free_raw_data=False)
    ds_v = lgb.Dataset(val[FEATS], val.objetivo, reference=ds_t, free_raw_data=False)
    cs = []
    for s in SEEDS:
        ev = {}
        lgb.train(dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1),
                  ds_t, rounds, valid_sets=[ds_v], valid_names=['v'],
                  callbacks=[lgb.record_evaluation(ev)])
        cs.append(ev['v']['auc'])
    return np.mean(cs, axis=0)


def pred_gbm(trn, obj, params, rounds):
    ds_t = lgb.Dataset(trn[FEATS], trn.objetivo, free_raw_data=False)
    ps, m = [], None
    for s in SEEDS:
        m = lgb.train(dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1),
                      ds_t, rounds)
        ps.append(m.predict(obj[FEATS]))
    return np.mean(ps, axis=0), m


def ejecutar_auditoria():
    print("=" * 80)
    print("AUDITORÍA DE HIPÓTESIS SEGÚN PROTOCOLO C (7 FOLDS WALK-FORWARD)")
    print("=" * 80)
    df = build()
    tr = df[df.objetivo.notna()].copy(); tr['objetivo'] = tr.objetivo.astype(int)
    te = df[df.objetivo.isna()].copy()
    folds = sorted(tr.mi.unique())[-N_VAL_FOLDS:]
    meses_nombres = [int(df.mes[df.mi == f].iloc[0]) for f in folds]

    # --- 1. MODELO DE REFERENCIA ---
    print("\n[1/4] Evaluando Modelo de Referencia (solucion.py)...")
    curvas_ref, pares_ref, n_trn = [], [], []
    for vm in folds:
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        curvas_ref.append(curva_gbm(trn, val, PARAMS))
        pares_ref.append((val.objetivo.values, pred_param(trn, val, n_knots=6, c_val=0.05)))
        n_trn.append(len(trn))

    gcurva_ref = 2 * np.mean(curvas_ref, axis=0) - 1
    best_ref = int(np.argmax(gcurva_ref)) + 1

    ref_blend = []
    for (y, pp), vm in zip(pares_ref, folds):
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        pg, _ = pred_gbm(trn, val, PARAMS, best_ref)
        ref_blend.append(gini(y, mezcla(pg, pp)))
    ref_blend = np.array(ref_blend)

    # --- 2. CANDIDATO: RESTRICCIONES MONÓTONAS + PARÁMETROS AJUSTADOS (n_knots=5, C=0.02) ---
    print("\n[2/4] Evaluando Candidato (Restricciones Monótonas + Splines n_knots=5, C=0.02)...")
    mono_params = dict(PARAMS)
    mono_params['monotone_constraints'] = [
        1 if f == 'numero_productos' else (-1 if f == 'dias_ultima_transaccion' else 0)
        for f in FEATS
    ]

    curvas_cand, pares_cand = [], []
    for vm in folds:
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        curvas_cand.append(curva_gbm(trn, val, mono_params))
        pares_cand.append((val.objetivo.values, pred_param(trn, val, n_knots=5, c_val=0.02)))

    gcurva_cand = 2 * np.mean(curvas_cand, axis=0) - 1
    best_cand = int(np.argmax(gcurva_cand)) + 1

    cand_blend = []
    for (y, pp), vm in zip(pares_cand, folds):
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        pg, _ = pred_gbm(trn, val, mono_params, best_cand)
        cand_blend.append(gini(y, mezcla(pg, pp)))
    cand_blend = np.array(cand_blend)

    # --- 3. TABLA COMPARATIVA Y TESTS PAREADOS ---
    diffs = cand_blend - ref_blend
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    t_stat = diffs.mean() / se
    wins = (diffs > 0).sum()

    print("\n" + "=" * 80)
    print(f"{'Mes':<8} | {'Referencia':<12} | {'Candidato':<12} | {'Diferencia Pareada':<18}")
    print("-" * 80)
    for m, r, c, d in zip(meses_nombres, ref_blend, cand_blend, diffs):
        print(f"{m:<8} | {r:<12.4f} | {c:<12.4f} | {d:+12.4f}")
    print("-" * 80)
    print(f"{'MEDIA':<8} | {ref_blend.mean():<12.4f} | {cand_blend.mean():<12.4f} | {diffs.mean():+12.5f} +- {se:.5f}")
    print(f"t-statistic: {t_stat:.2f} (Umbral de aceptación: t > 2.00)")
    print(f"Folds ganados: {wins} de {len(diffs)}")

    # --- 4. VERIFICACIÓN EN FOLDS FRESCOS ---
    print("\n[3/4] Verificación en folds frescos (202603, 202604, 202605)...")
    all_folds = sorted(tr.mi.unique())
    fresh_folds = all_folds[2:5]
    fresh_ref, fresh_cand = [], []
    for vm in fresh_folds:
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        y = val.objetivo.values
        pg_r, _ = pred_gbm(trn, val, PARAMS, best_ref)
        pp_r = pred_param(trn, val, n_knots=6, c_val=0.05)
        fresh_ref.append(gini(y, mezcla(pg_r, pp_r)))

        pg_c, _ = pred_gbm(trn, val, mono_params, best_cand)
        pp_c = pred_param(trn, val, n_knots=5, c_val=0.02)
        fresh_cand.append(gini(y, mezcla(pg_c, pp_c)))

    fresh_diff = np.mean(fresh_cand) - np.mean(fresh_ref)
    print(f"  Folds frescos: Ref = {np.mean(fresh_ref):.4f} | Cand = {np.mean(fresh_cand):.4f} | Diff = {fresh_diff:+.5f}")

    # --- 5. DECISIÓN RIGUROSA SEGÚN CRITERIO C ---
    print("\n" + "=" * 80)
    print("VEREDICTO ESTADÍSTICO SEGÚN CRITERIO C:")
    print("=" * 80)
    cumple_a = t_stat > 2.0
    cumple_b = fresh_diff > 0.0

    print(f"  Criterio (a) Dif. pareada > 2 SE  : {'CUMPLE' if cumple_a else 'NO CUMPLE'} (t={t_stat:.2f} < 2.00)")
    print(f"  Criterio (b) Replica en frescos   : {'CUMPLE' if cumple_b else 'NO CUMPLE'} (diff={fresh_diff:+.5f})")

    if cumple_a and cumple_b:
        print("\n-> VEREDICTO: Se acepta el candidato como nueva entrega.")
        modelo_elegido = 'candidato'
        final_params = mono_params
        final_best = best_cand
        final_knots, final_c = 5, 0.02
        expected_gini = cand_blend.mean()
    else:
        print("\n-> VEREDICTO: NINGUNA IDEA SUPERA ESTADÍSTICAMENTE A LA REFERENCIA.")
        print("-> LA ENTREGA CORRECTA, VERIFICADA Y OFICIAL ES EL MODELO DE REFERENCIA.")
        modelo_elegido = 'referencia'
        final_params = PARAMS
        final_best = best_ref
        final_knots, final_c = 6, 0.05
        expected_gini = ref_blend.mean()

    # --- 6. ENTRENAMIENTO FINAL Y SUBMISSION OFICIAL ---
    print("\n[4/4] Entrenando modelo final con todo train y generando submission.csv...")
    scale = len(tr) / np.mean(n_trn)
    rounds = int(final_best * scale)
    print(f"  Modelo: {modelo_elegido} | Rondas re-escaladas: {rounds} ({final_best} x {scale:.2f})")

    pg, m = pred_gbm(tr, te, final_params, rounds)
    pp = pred_param(tr, te, n_knots=final_knots, c_val=final_c)

    orden = rankdata(mezcla(pg, pp), method='ordinal').astype(int) - 1
    te['prediccion'] = np.sort(pg)[orden]

    # Validaciones obligatorias de formato
    orden_test = pd.read_csv(DIR / 'test.csv')[['id_cliente']]
    sub = orden_test.merge(te[['id_cliente', 'prediccion']], on='id_cliente', how='left')

    assert len(sub) == len(orden_test), 'Error: longitud no coincide'
    assert sub.id_cliente.equals(orden_test.id_cliente), 'Error: orden roto'
    assert sub.prediccion.notna().all(), 'Error: valores nulos'
    assert sub.prediccion.between(0, 1).all(), 'Error: fuera de rango [0, 1]'
    assert sub.prediccion.nunique() > 0.9 * len(sub), 'Error: predicciones poco variadas'

    sub_path = DIR / 'submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f"\n[OK] submission.csv generado correctamente ({len(sub):,} filas) | Gini esperado ~{expected_gini:.4f}")
    print(sub.head(10).to_string(index=False))


if __name__ == '__main__':
    ejecutar_auditoria()
