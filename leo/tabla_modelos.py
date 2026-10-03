"""Tabla comparativa de modelos, mismo protocolo para todos.

Gini = 2*AUC - 1 en 6 folds walk-forward (entrenar con los meses ANTERIORES a cada mes y medir en ese
mes): fold_1 = 202606 ... fold_6 = 202611, igual que la captura de Jhojan.  Rondas fijas (sin early
stopping contra el fold que se reporta).  Los modelos de la captura de Jhojan se re-ejecutan aqui con
parametros propios (no son sus numeros): lgbm, catboost, xgb, logreg y sus ensambles por rangos.
Escribe resultados_modelos.csv (modelo, fold_1..fold_6, media, std con ddof=0 como en su tabla).
"""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from scipy.stats import rankdata
import solucion as S

DIR = Path(__file__).parent
N_FOLDS = 6
SEEDS = [42, 777, 2024]
INTER = 'dias_ultima_interaccion'
rk = lambda v: rankdata(v) / len(v)
gini = S.gini

df = S.build()
tr = df[df.objetivo.notna()].copy(); tr['objetivo'] = tr.objetivo.astype(int)
FOLDS = sorted(tr.mi.unique())[-N_FOLDS:]
CAT, BOOL = S.CAT, S.BOOL
F_BASE = S.NUM + [INTER] + CAT + BOOL + ['duracion']                       # las 22 originales + duracion
LGB_BASE = {k: v for k, v in S.PARAMS.items() if not k.startswith('monotone')}


# ---------------------------------------------------------------- modelos estandar (los de la captura)
def p_lgbm(trn, val):
    ds = lgb.Dataset(trn[F_BASE], trn.objetivo, free_raw_data=False)
    return np.mean([lgb.train(dict(LGB_BASE, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1),
                              ds, 80).predict(val[F_BASE]) for s in SEEDS], axis=0)


def p_xgb(trn, val):
    out = 0
    for s in SEEDS:
        m = xgb.XGBClassifier(n_estimators=90, learning_rate=0.05, max_depth=4, min_child_weight=20,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5, enable_categorical=True,
                              tree_method='hist', random_state=s, n_jobs=-1, verbosity=0)
        out = out + m.fit(trn[F_BASE], trn.objetivo).predict_proba(val[F_BASE])[:, 1] / len(SEEDS)
    return out


def p_cat(trn, val):
    X, Xv = trn[F_BASE].copy(), val[F_BASE].copy()
    for c in CAT:
        X[c] = X[c].astype(str); Xv[c] = Xv[c].astype(str)
    out = 0
    for s in SEEDS[:2]:
        m = CatBoostClassifier(iterations=236, learning_rate=0.05, depth=6, l2_leaf_reg=10, random_seed=s,
                               verbose=0, cat_features=CAT, thread_count=-1, allow_writing_files=False)
        out = out + m.fit(X, trn.objetivo).predict_proba(Xv)[:, 1] / 2
    return out


def p_logreg(trn, val):
    prep = ColumnTransformer([
        ('n', QuantileTransformer(output_distribution='normal', n_quantiles=200), S.NUM + [INTER, 'duracion']),
        ('c', OneHotEncoder(handle_unknown='ignore'), CAT + ['numero_productos']),
        ('b', 'passthrough', BOOL)])
    return make_pipeline(prep, LogisticRegression(max_iter=3000, C=0.1)).fit(
        trn[F_BASE], trn.objetivo).predict_proba(val[F_BASE])[:, 1]


# ---------------------------------------------------------------- "modelo maestro" de Antigravity (replica)
NUM_B = ['edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses', 'numero_productos',
         'saldo_promedio', 'dias_ultima_transaccion', 'antiguedad_direccion_meses',
         'visitas_web_ultimos_90_dias', 'distancia_sucursal_km', 'dia_preferido_pago', INTER]
df['interaccion_reciente_30d'] = (df[INTER] <= 30).astype(int)
df['riesgo_x_prods'] = df.banda_riesgo.astype(str) + '_' + df.numero_productos.astype(str)
df['riesgo_x_movil'] = df.banda_riesgo.astype(str) + '_' + df.activo_movil.astype(str)
df['inactividad_rel'] = df.dias_ultima_transaccion / (df.antiguedad_cuenta_meses * 30.0 + 1.0)
CAT_M = CAT + ['riesgo_x_prods', 'riesgo_x_movil']
F_M = NUM_B + ['duracion', 'inactividad_rel'] + CAT_M + BOOL + ['interaccion_reciente_30d']
tr = df[df.objetivo.notna()].copy(); tr['objetivo'] = tr.objetivo.astype(int)
SEEDS_M = [42, 2026, 777]


def p_maestro(trn, val):
    tl, vl = trn.copy(), val.copy()
    for c in CAT_M:
        tl[c] = tl[c].astype('category'); vl[c] = vl[c].astype('category')
    ds = lgb.Dataset(tl[F_M], tl.objetivo)
    p_l = np.zeros(len(val))
    for s in SEEDS_M:
        p = dict(objective='binary', metric='auc', verbosity=-1, learning_rate=0.03, num_leaves=24,
                 min_child_samples=85, reg_lambda=2.0, reg_alpha=0.3, feature_fraction=0.65,
                 bagging_fraction=0.75, bagging_freq=1, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1)
        p_l += lgb.train(p, ds, 110).predict(vl[F_M]) / len(SEEDS_M)
    tc, vc = trn.copy(), val.copy()
    for c in CAT_M:
        tc[c] = tc[c].astype(str); vc[c] = vc[c].astype(str)
    pt, pv = Pool(tc[F_M], tc.objetivo, cat_features=CAT_M), Pool(vc[F_M], cat_features=CAT_M)
    p_c = np.zeros(len(val))
    for s in SEEDS_M:
        m = CatBoostClassifier(iterations=130, learning_rate=0.04, depth=5, l2_leaf_reg=3.0, eval_metric='AUC',
                               random_seed=s, verbose=0, thread_count=-1, allow_writing_files=False)
        p_c += m.fit(pt).predict_proba(pv)[:, 1] / len(SEEDS_M)
    full = pd.concat([trn[F_M], val[F_M]], ignore_index=True)
    ohe = pd.get_dummies(full, columns=CAT_M, drop_first=True)
    d_tr, d_va = xgb.DMatrix(ohe.iloc[:len(trn)], label=trn.objetivo.values), xgb.DMatrix(ohe.iloc[len(trn):])
    p_x = np.zeros(len(val))
    for s in SEEDS_M:
        p = dict(objective='binary:logistic', eval_metric='auc', learning_rate=0.035, max_depth=5,
                 min_child_weight=6, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.5, seed=s, nthread=-1)
        p_x += xgb.train(p, d_tr, num_boost_round=120).predict(d_va) / len(SEEDS_M)
    return 0.50 * rk(p_l) + 0.30 * rk(p_c) + 0.20 * rk(p_x)


# ---------------------------------------------------------------- nuestro modelo anterior (con dias_ultima_interaccion)
def _diseno(d, cont_cols, spl=None):
    cont = d[cont_cols].values
    if spl is None:
        spl = SplineTransformer(n_knots=6, degree=3).fit(cont)
    Sp = pd.DataFrame(spl.transform(cont), index=d.index).add_prefix('s')
    npd = pd.get_dummies(d.numero_productos, prefix='np', drop_first=True).astype(float)
    b = d[['activo_movil', 'tiene_tarjeta_credito']].astype(float)
    return pd.concat([npd, pd.get_dummies(d.canal_adquisicion, prefix='cn', drop_first=True).astype(float),
                      pd.get_dummies(d.duracion.clip(upper=8), prefix='du', drop_first=True).astype(float), Sp, b,
                      (b.activo_movil * b.tiene_tarjeta_credito).rename('am_x_tc').to_frame(),
                      Sp.mul(b.activo_movil, axis=0).add_suffix('_xam'),
                      npd.mul(b.activo_movil, axis=0).add_suffix('_xam'),
                      npd.mul(b.tiene_tarjeta_credito, axis=0).add_suffix('_xtc')], axis=1), spl


def _param(trn, val, cont_cols):
    out = np.zeros(len(val))
    for b in trn.banda_riesgo.cat.categories:
        mt, mo = (trn.banda_riesgo == b).values, (val.banda_riesgo == b).values
        if not mo.sum():
            continue
        Xt, spl = _diseno(trn[mt], cont_cols); Xo, _ = _diseno(val[mo], cont_cols, spl)
        Xo = Xo.reindex(columns=Xt.columns, fill_value=0.0); sc = StandardScaler().fit(Xt)
        out[mo] = LogisticRegression(max_iter=5000, C=0.05).fit(sc.transform(Xt), trn[mt].objetivo).predict_proba(
            sc.transform(Xo))[:, 1]
    return out


F_V1 = S.FEATS_GBM + [INTER]
MONO_V1 = {'dias_ultima_transaccion': -1, INTER: -1, 'numero_productos': 1, 'banda_ord': 1}


def p_anterior(trn, val):
    ds = lgb.Dataset(trn[F_V1], trn.objetivo, free_raw_data=False)
    pg = np.mean([lgb.train(dict(LGB_BASE, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1,
                                 monotone_constraints=[MONO_V1.get(f, 0) for f in F_V1],
                                 monotone_constraints_method='advanced'), ds, 71).predict(val[F_V1])
                  for s in SEEDS], axis=0)
    pp = _param(trn, val, ['dias_ultima_transaccion', INTER, 'ratio_deuda_ingresos'])
    return S.mezcla(pg, pp)


def p_final(trn, val):
    pg, _ = S.pred_gbm(trn, val, 87)                  # exactamente el pipeline de solucion.py
    return S.mezcla(pg, S.pred_param(trn, val))


MODELOS = {
    'lgbm': p_lgbm, 'catboost': p_cat, 'xgb': p_xgb, 'logreg': p_logreg,
    'antigravity_maestro_bcp': p_maestro, 'nuestro_anterior_(0.2618)': p_anterior,
    'NUESTRO_FINAL': p_final,
}

if __name__ == '__main__':
    P = {k: [] for k in MODELOS}; Y = []
    for vm in FOLDS:
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        Y.append(val.objetivo.values)
        for k, fn in MODELOS.items():
            P[k].append(np.asarray(fn(trn, val)))
        print('fold', int(df.mes[df.mi == vm].iloc[0]), 'listo', flush=True)

    ens = {  # ensambles por rangos, como los de la captura
        'ensemble_LGBM+CB': ['lgbm', 'catboost'],
        'ensemble_LGBM+CB+XGB': ['lgbm', 'catboost', 'xgb'],
        'ensemble_LGBM+CB+LR': ['lgbm', 'catboost', 'logreg'],
        'ensemble_LGBM+CB+XGB+LR': ['lgbm', 'catboost', 'xgb', 'logreg'],
    }
    for k, comp in ens.items():
        P[k] = [np.mean([rk(P[c][i]) for c in comp], axis=0) for i in range(len(FOLDS))]

    orden = ['lgbm', 'catboost', 'xgb', 'logreg', *ens, 'antigravity_maestro_bcp',
             'nuestro_anterior_(0.2618)', 'NUESTRO_FINAL']
    filas = []
    for k in orden:
        g = [gini(Y[i], P[k][i]) for i in range(len(FOLDS))]
        filas.append({'modelo': k, **{f'fold_{i+1}': g[i] for i in range(len(g))},
                      'media': float(np.mean(g)), 'std': float(np.std(g))})
    out = pd.DataFrame(filas)
    out.to_csv(DIR / 'resultados_modelos.csv', index=False)
    print(out.round(6).to_string(index=False))
