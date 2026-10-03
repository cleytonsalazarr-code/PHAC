"""Propension de conversion -- solucion final.  Metrica: Gini = 2*AUC - 1.

MODELO: 75% LightGBM CON RESTRICCIONES MONOTONAS + 25% logistica parametrica ajustada POR BANDA
DE RIESGO, mezclados sobre
rangos (la metrica es de orden, asi que el rango es la escala natural y evita problemas de
calibracion entre familias).

================================================================================================
ESTRUCTURA DE LOS DATOS (medido, no supuesto)
================================================================================================
 1. 21 de los 22 predictores estan CONGELADOS por cliente entre meses.  La unica que varia es
    dias_ultima_interaccion, y es una copia ruidosa de dias_ultima_transaccion (identica en 55%
    de las filas, valor aleatorio en el resto).  => todo feature de delta / tendencia / momentum
    por cliente es identicamente cero.  Esa via no existe aca.

 2. La caida del hazard con la duracion (16.0% -> 12.6%) es PURA COMPOSICION: condicional al
    score del modelo, la duracion no es significativa (p=0.20 lineal; LR chi2=8.79 con 8 gl con
    dummies).  No hay heterogeneidad no observada persistente => sobrevivir k meses sin convertir
    no informa nada.  El DGP es Bernoulli fresco cada mes sobre atributos estaticos.

 3. Por lo tanto es un hazard de tiempo discreto con covariables estaticas.  Las 110.100 filas
    cliente-mes se usan TAL CUAL: cada una es un ensayo independiente a la misma x, lo que
    maximiza la informacion sobre p(x).  Es la verosimilitud correcta: no se deduplica por
    cliente ni se ponderan filas.

 4. banda_riesgo funciona como COMPUERTA: participa en 5 de las 11 interacciones significativas
    (Bonferroni sobre 253 pares), incluidas las dos mayores -- x numero_productos (chi2=421) y
    x dias_ultima_transaccion (chi2=329).  En banda "low" el efecto de numero_productos recorre
    1.05 en logit; en "medium" y "high" es casi plano (0.18 y 0.29).  Por eso el modelo
    parametrico se ajusta SEPARADO POR BANDA: captura todas esas interacciones por construccion.

 5. Descartadas con razon: id_cliente (secuencial por cohorte de entrada => colineal con duracion
    dentro de un mes, y en test sale del rango de train), y mes (tasa base plana 14.1%-15.7%, y
    diciembre no existe en train => extrapolacion pura).

================================================================================================
PROBADO Y DESCARTADO POR MEDICION PAREADA (no por gusto)
================================================================================================
   ratios explicitos (saldo/ingresos, capacidad de pago, ...)   -0.0013 +- 0.0009
   ensamble LGBM+XGBoost                                        -0.0003 +- 0.0006
   agregar un GAM de splines global al blend                    -0.0036 +- 0.0013
   CatBoost (solo)                                       0.2558 vs 0.2583 del LGBM
   promediar 5 semillas                                         -0.0000 +- 0.0015
   podar a las 9 features significativas                 0.2548 vs 0.2583  <- EMPEORA
   dar las interacciones como factores cruzados al GBM    0.2540 vs 0.2583  <- EMPEORA
   el parametrico SOLO                                   0.2517 vs 0.2592  <- peor solo
   busqueda de 45 configs de hiperparametros                    +0.0006 +- 0.0020 (no sig.)

Lo contraintuitivo: las 14 variables sin efecto principal significativo NO son ruido para el GBM
(podarlas empeora).  Los tests tienen poca potencia para efectos debiles; el GBM los agrega.  Y
darle las interacciones masticadas tampoco ayuda: ya las encuentra con sus cortes.

LO UNICO QUE SOBREVIVIO: mezclar el parametrico al 25%.  +0.0013 +- 0.0008, positivo en los 9
folds, y REPLICO en 202603 y 202604, meses no usados para elegirlo (+0.0014 +- 0.0004 sobre 3 meses,
pero 202605 si estaba entre los folds de seleccion, asi que la replica limpia es de 2 meses).  La curva de
peso es suave y unimodal con optimo en 0.7-0.8, que es lo que hace creible el efecto.

================================================================================================
VARIABLE CORRUPTA: dias_ultima_interaccion SE EXCLUYE (arreglo de robustez, no de puntaje)
================================================================================================
Hallazgo (validacion adversarial agrupada por cliente + inspeccion directa): dias_ultima_interaccion
se va corrompiendo con el mes calendario.  Fraccion de filas donde es igual a dias_ultima_transaccion:
ene 100%, feb 91%, mar 82%, ... nov 9.4%, DIC 0.26% (26 de 9.900).  En test es una PERMUTACION exacta
de dias_ultima_transaccion (mismos valores, repartidos al azar): ruido puro.
   - En train, filas corruptas: corr(inter, trans) = 0.0025; AUC de inter vs objetivo = 0.5052
     (nada), mientras trans en esas mismas filas tiene 0.5442.  Filas iguales: AUC 0.5482 = el de trans.
   - Toda su "senal" en train es ser una copia de trans; informacion propia: ninguna.
   - Sin ella el adversarial cae de 0.715 a 0.527 (diciembre deja de distinguirse).
Dano medido en validacion "tipo diciembre" (inter permutada en la fila de validacion, como el test):
modelo con la variable 0.2579 vs 0.2624 normal (-0.0045 +- 0.0026).  Se concentra en los primeros
meses (mar +0.024, abr +0.008, may +0.007 a favor del modelo sin ella), cuando el entrenamiento aun
tiene pocas filas corruptas; de junio en adelante la diferencia es ~0 porque el modelo aprende a
ignorarla.  Con 11 meses de entrenamiento el dano esperado en diciembre es chico, pero quitarla cuesta
-0.0001 +- 0.0009 en validacion normal (empate) y elimina el riesgo.  Se quita de GBM y logistica.

Probado y NO adoptado en esa ronda (9 folds, 5 semillas, pareado contra la referencia):
   interaction_constraints (su agrupacion / guiada por el barrido de pares)  -0.0017 / -0.0021 (t=-3.6)
   CatBoost monotono como 3er modelo (rho con LGB 0.949, no 0.82-0.85)      -0.0001 a -0.0006
   5a monotona ratio_deuda: +0.0006 (t=2.27) con la variable corrupta, +0.0003 +- 0.0004 sin ella
   ponderacion temporal: pico aislado en lambda=0.10 (+0.0010), negativo desde 0.20; +0.0004 sin inter
   (nota: ratio_deuda_ingresos NUNCA vale 0: minimo 0.02, asi que un indicador sin_deuda seria constante)

================================================================================================
RESTRICCIONES MONOTONAS: LA UNICA MEJORA ADICIONAL QUE SOBREVIVIO (+0.0026)
================================================================================================
Se fuerza al GBM a que dias_ultima_transaccion (decreciente), numero_productos y banda_riesgo-
ordinal (crecientes) tengan efecto monotono (la medicion de +0.0026 incluia tambien
dias_ultima_interaccion, luego excluida: el modelo sin ella da 0.2622 vs 0.2624, empate).  Las direcciones
salen de los deciles medidos antes de ver ningun resultado de validacion.  Es regularizacion
con base a priori: el GBM sin restriccion ajusta escalones espurios en variables cuyo efecto
verdadero es suave.  Medido en 9 folds walk-forward, mezcla .75/.25, 5 semillas:
   mezcla sin restricciones                0.2600
   monotona en 3 variables                 +0.0016 +- 0.0004, gana 8/9, con 50-130 rondas
   monotona en 4 (+ banda ordinal)         +0.0026 +- 0.0008, gana 7-8/9   <-- el usado
   banda ordinal SIN restriccion           +0.0001  (la ganancia es la restriccion, no la codificacion)
   dosis-respuesta: 1 restriccion +0.0007, 2 -> +0.0015, 3 -> +0.0018
   agregar activo_movil / tarjeta          sin efecto adicional (se dejan fuera por parsimonia)
Caveat honesto: se eligio entre ~6 variantes mirando los mismos 9 meses, asi que la ganancia
esperada fuera de muestra es algo menor que +0.0026; la estabilidad entre rondas y la dosis-
respuesta indican que es real.  Meseta medida: 60-130 rondas, peso del GBM 0.6-0.8.
Descartadas en la misma ronda: EBM (-0.0004 a -0.0013), DART, tunear C/nudos de la logistica.

================================================================================================
LA RECETA DE LOS GANADORES DE KAGGLE TABULAR, INVESTIGADA Y MEDIDA
================================================================================================
Fuentes: NVIDIA "Kaggle Grandmasters Playbook", soluciones 1er puesto de Chris Deotte (Playground
S5E6, S5E12), writeups de Playground S6E9.  La receta: muchos modelos de familias distintas +
predicciones fuera de fold + hill climbing / stacking + feature engineering masivo (target y
count encoding de columnas y pares).  Resultado aca, evaluado leave-one-fold-out en 9 meses:

   NUESTRO (75% LGBM + 25% logistica por banda)   0.2605
   hill climbing sobre 8 familias                 0.2593   -0.0012 +- 0.0005
   promedio simple de las 8                       0.2590   -0.0015 +- 0.0012
   stacking logistico / NNLS                      0.2588   -0.0017
   XGBoost / CatBoost / red neuronal solos        0.2586 / 0.2577 / 0.2330

Por que no funciona aca: los tres boosting ordenan casi igual (rho 0.93-0.95 entre si); la red y
la logistica simple son distintas (rho ~0.80) pero demasiado debiles.  El unico modelo distinto Y
decente es la logistica por banda (rho 0.81, gini 0.252) -- que ya esta en la mezcla, y que el
hill climbing elige en las 9 rotaciones.  Lo que agrega encima sobreajusta los pesos.

Target/count encoding de numericas: en datos sinteticos suele haber senal por valor exacto.  Aca
NO: el valor exacto de ingresos/saldo identifica al cliente (AUC 0.535 en todos, 0.500 en clientes
nuevos) y lo unico que "aprende" es la duracion.  Peor: como el numero de filas de un cliente
depende de cuando convierte, el count encoding sobre todo el dataset FILTRA EL FUTURO.
XGBoost con objetivo de ranking por pares: 0.08 -- un grupo unico gigante muestrea mal los pares.

================================================================================================
CUANDO PARAR
================================================================================================
Si el DGP es Bernoulli por mes, un ORACULO que conozca p(x) exacto tiene Gini finito: estimado
por simulacion en ~0.2694 +- 0.0066, y estamos en ~0.260 => ~96% del maximo.  Ademas el ruido de
muestreo del propio test (9.900 filas) da sd 0.0143 al Gini final: 10x la resolucion de los tests
pareados de arriba y 11x la mejora del blend.  El puntaje lo decide el azar de diciembre.

VALIDACION: walk-forward por mes (entrenar en meses < m, validar en m), que replica el escenario
real (train ene-nov -> predecir dic).  Un KFold aleatorio inflaria el Gini porque las filas de un
mismo cliente son casi identicas y caerian a ambos lados del split.  Las rondas salen del MAXIMO
DE LA CURVA PROMEDIADA entre folds, no de promediar los argmax de cada fold (eso es sesgado) ni
de early stopping contra el fold que se reporta (eso es optimista).
"""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DIR = Path(__file__).parent
SEEDS = [42, 777, 2024]     # ganancia medida nula, pero cubre contra una semilla desafortunada
N_VAL_FOLDS = 7
MAX_ROUNDS = 300            # el optimo esta ~65; pasadas 150 rondas degrada en serio
W_GBM = 0.75                # meseta medida 0.7-0.8; el medio, para no sobreajustar el peso

NUM = ['edad','ingresos','ratio_deuda_ingresos','antiguedad_cuenta_meses','numero_productos',
       'saldo_promedio','dias_ultima_transaccion','antiguedad_direccion_meses',
       'visitas_web_ultimos_90_dias','distancia_sucursal_km','dia_preferido_pago']
# dias_ultima_interaccion queda FUERA a proposito: en test es una permutacion al azar de
# dias_ultima_transaccion (ruido puro).  Ver "VARIABLE CORRUPTA" en el encabezado.
CAT = ['ocupacion','region','canal_adquisicion','banda_riesgo','dispositivo_principal']
BOOL = ['tiene_tarjeta_credito','activo_movil','es_nuevo_cliente','tiene_prestamo','tiene_seguro']
FEATS = NUM + CAT + BOOL + ['duracion']
# Para el GBM, banda_riesgo entra como ORDINAL (high=0 < medium=1 < low=2; tasas 8.7% < 14.0% < 18.6%)
# para poder imponerle monotonia.  La logistica por banda sigue usando la columna categorica.
FEATS_GBM = [('banda_ord' if f == 'banda_riesgo' else f) for f in FEATS]
MONO = {'dias_ultima_transaccion': -1, 'numero_productos': 1, 'banda_ord': 1}
MONO_VEC = [MONO.get(f, 0) for f in FEATS_GBM]

PARAMS = dict(objective='binary', metric='auc', verbosity=-1, learning_rate=0.05, num_leaves=16,
              max_depth=-1, min_child_samples=400, reg_lambda=20, reg_alpha=0,
              feature_fraction=1.0, bagging_fraction=0.7, bagging_freq=1,
              monotone_constraints=MONO_VEC, monotone_constraints_method='advanced')


def build():
    tr = pd.read_csv(DIR / 'train.csv')
    te = pd.read_csv(DIR / 'test.csv')
    assert tr.notna().all().all() and te.notna().all().all(), 'aparecieron nulos'
    te['objetivo'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df['mi'] = (df.mes // 100) * 12 + (df.mes % 100)
    df = df.sort_values(['id_cliente', 'mi']).reset_index(drop=True)
    # cumcount solo mira hacia atras: usar filas futuras del mismo cliente seria fuga de
    # supervivencia (saber que seguia "vivo" despues del mes que se esta puntuando).
    df['duracion'] = df.groupby('id_cliente').cumcount()
    df['banda_ord'] = df.banda_riesgo.map({'high': 0, 'medium': 1, 'low': 2}).astype(int)
    for c in CAT:
        df[c] = df[c].astype('category')     # categoricas nativas: sin one-hot ni target-encoding
    for c in BOOL:
        df[c] = df[c].astype(int)
    assert np.isfinite(df[[f for f in FEATS if f not in CAT]].to_numpy()).all(), 'inf/nan'
    return df


# ------------------------------------------------------------------ componente 1: LightGBM
def curva_gbm(trn, val, rounds=MAX_ROUNDS):
    """Curva de AUC de validacion por iteracion, promediada sobre semillas."""
    ds_t = lgb.Dataset(trn[FEATS_GBM], trn.objetivo, free_raw_data=False)
    ds_v = lgb.Dataset(val[FEATS_GBM], val.objetivo, reference=ds_t, free_raw_data=False)
    cs = []
    for s in SEEDS:
        ev = {}
        lgb.train(dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1),
                  ds_t, rounds, valid_sets=[ds_v], valid_names=['v'],
                  callbacks=[lgb.record_evaluation(ev)])
        cs.append(ev['v']['auc'])
    return np.mean(cs, axis=0)


def pred_gbm(trn, obj, rounds):
    ds_t = lgb.Dataset(trn[FEATS_GBM], trn.objetivo, free_raw_data=False)
    ps, m = [], None
    for s in SEEDS:
        m = lgb.train(dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, n_jobs=-1),
                      ds_t, rounds)
        ps.append(m.predict(obj[FEATS_GBM]))
    return np.mean(ps, axis=0), m


# ------------------------------------------------- componente 2: logistica por banda de riesgo
def diseno(d, spl=None):
    """Matriz de diseno DENTRO de una banda.  Al ajustar por banda, las 5 interacciones con
    banda_riesgo quedan capturadas sin gastar un solo parametro en estimarlas."""
    cont = d[['dias_ultima_transaccion', 'ratio_deuda_ingresos']].values
    if spl is None:
        spl = SplineTransformer(n_knots=6, degree=3).fit(cont)
    S = pd.DataFrame(spl.transform(cont), index=d.index).add_prefix('s')
    npd = pd.get_dummies(d.numero_productos, prefix='np', drop_first=True).astype(float)
    b = d[['activo_movil', 'tiene_tarjeta_credito']].astype(float)
    X = pd.concat([
        npd,
        pd.get_dummies(d.canal_adquisicion, prefix='cn', drop_first=True).astype(float),
        pd.get_dummies(d.duracion.clip(upper=8), prefix='du', drop_first=True).astype(float),
        S, b,
        (b.activo_movil * b.tiene_tarjeta_credito).rename('am_x_tc').to_frame(),
        S.mul(b.activo_movil, axis=0).add_suffix('_xam'),      # ratio_deuda x activo_movil
        npd.mul(b.activo_movil, axis=0).add_suffix('_xam'),
        npd.mul(b.tiene_tarjeta_credito, axis=0).add_suffix('_xtc'),
    ], axis=1)
    return X, spl


def pred_param(trn, obj):
    out = np.zeros(len(obj))
    for banda in trn.banda_riesgo.cat.categories:
        mt, mo = (trn.banda_riesgo == banda).values, (obj.banda_riesgo == banda).values
        if mo.sum() == 0:
            continue
        Xt, spl = diseno(trn[mt])
        Xo, _ = diseno(obj[mo], spl)
        Xo = Xo.reindex(columns=Xt.columns, fill_value=0.0)
        sc = StandardScaler().fit(Xt)
        m = LogisticRegression(max_iter=5000, C=0.05).fit(sc.transform(Xt), trn[mt].objetivo)
        out[mo] = m.predict_proba(sc.transform(Xo))[:, 1]
    return out


def rk(v):
    return rankdata(v) / len(v)


def mezcla(p_gbm, p_par):
    return W_GBM * rk(p_gbm) + (1 - W_GBM) * rk(p_par)


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


# ------------------------------------------------------------------ CV + entrega
def main():
    df = build()
    tr = df[df.objetivo.notna()].copy(); tr['objetivo'] = tr.objetivo.astype(int)
    te = df[df.objetivo.isna()].copy()
    folds = sorted(tr.mi.unique())[-N_VAL_FOLDS:]
    print(f'train {len(tr):,} | test {len(te):,} | {len(FEATS)} features | mezcla '
          f'{W_GBM:.0%} GBM + {1-W_GBM:.0%} parametrico')
    print(f'walk-forward sobre {[int(df.mes[df.mi==f].iloc[0]) for f in folds]}\n')

    curvas, pares, n_trn = [], [], []
    for vm in folds:
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        curvas.append(curva_gbm(trn, val))
        pares.append((val.objetivo.values, pred_param(trn, val)))
        n_trn.append(len(trn))
        print(f'  {int(df.mes[df.mi==vm].iloc[0])}  trn={len(trn):>6,}  val={len(val):>5,}  '
              f'gini_param={gini(*pares[-1]):.4f}', flush=True)

    gcurva = 2 * np.mean(curvas, axis=0) - 1
    best = int(np.argmax(gcurva)) + 1
    plano = np.where(gcurva >= gcurva.max() - 0.001)[0] + 1
    print(f'\nGBM solo        : {gcurva.max():.4f}  (optimo {best} rondas, '
          f'zona plana {plano.min()}-{plano.max()})')
    print(f'parametrico solo: {np.mean([gini(y,p) for y,p in pares]):.4f}')

    # Gini de la mezcla, con las rondas ya fijadas arriba
    gm = []
    for (y, pp), vm in zip(pares, folds):
        trn, val = tr[tr.mi < vm], tr[tr.mi == vm]
        pg, _ = pred_gbm(trn, val, best)
        gm.append(gini(y, mezcla(pg, pp)))
    cv_gini = float(np.mean(gm))
    print(f'MEZCLA          : {cv_gini:.4f}  (+{cv_gini-gcurva.max():.4f} sobre GBM solo)')
    print(f'  por mes: {np.round(gm,4)}  sd {np.std(gm, ddof=1):.4f}')

    # Rondas SIN escalar por tamano de datos: el optimo por fold no muestra tendencia con las filas
    # de entrenamiento (23-131 rondas con 60k-100k filas) y la meseta medida es 60-130.
    rounds = best
    print(f'\nmodelo final: {len(tr):,} filas, {rounds} rondas')
    pg, m = pred_gbm(tr, te, rounds)
    pp = pred_param(tr, te)
    # La mezcla vive en escala de rangos.  La remapeo a la distribucion de probabilidad del GBM:
    # es monotona, asi que el AUC/Gini no cambia, pero la entrega queda en escala de probabilidad
    # como pide el enunciado en vez de en rangos.
    orden = rankdata(mezcla(pg, pp), method='ordinal').astype(int) - 1
    te['prediccion'] = np.sort(pg)[orden]

    imp = pd.Series(m.feature_importance('gain'), index=FEATS_GBM).sort_values(ascending=False)
    print('\n=== aporte de cada feature al GBM (% de gain) ===')
    print((imp / imp.sum() * 100).round(2).head(12).to_string())

    # Formato exigido: una fila por fila de test.csv, mismo orden, sin la columna mes.
    orden = pd.read_csv(DIR / 'test.csv')[['id_cliente']]
    sub = orden.merge(te[['id_cliente', 'prediccion']], on='id_cliente', how='left')
    assert len(sub) == len(orden), 'el merge duplico filas'
    assert sub.id_cliente.equals(orden.id_cliente), 'se rompio el orden de test.csv'
    assert sub.prediccion.notna().all(), 'faltan predicciones'
    assert sub.prediccion.between(0, 1).all(), 'predicciones fuera de [0,1]'
    assert sub.prediccion.nunique() > 0.9 * len(sub), 'predicciones poco variadas'
    sub.to_csv(DIR / 'submission.csv', index=False)
    print(f'\nsubmission.csv -> {len(sub):,} filas | Gini esperado ~{cv_gini:.4f}')
    print(sub.head().to_string(index=False))
    return cv_gini


def demo():
    """Check minimo: si la CV filtrara informacion o el modelo colapsara, esto falla."""
    g = main()
    assert 0.15 < g < 0.45, f'Gini {g:.4f} fuera del rango plausible -> fuga o bug'
    print('\nOK: Gini en rango plausible, entrega valida.')


if __name__ == '__main__':
    demo()
