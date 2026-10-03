# Propensión de conversión bancaria — Documentación completa del proyecto

> Documento de referencia: qué es el problema, qué se probó (48 pruebas), cómo y por qué se
> hizo cada una, con todos los números, y cuál es el mejor modelo que tenemos.
> Escrito para alguien que parte de cero. Fecha de trabajo: 1 y 2 de octubre de 2026.

---

## Índice

0. [Resumen ejecutivo](#0-resumen-ejecutivo)
1. [Qué es este proyecto (desde cero)](#1-qué-es-este-proyecto-desde-cero)
2. [Los archivos de la carpeta](#2-los-archivos-de-la-carpeta)
3. [Cómo se mide: Gini, AUC y la validación](#3-cómo-se-mide-gini-auc-y-la-validación)
4. [Cómo leer los resultados (pareado, error estándar)](#4-cómo-leer-los-resultados)
5. [Tabla maestra: las 48 pruebas de un vistazo](#5-tabla-maestra-las-48-pruebas-de-un-vistazo)
6. [Detalle de cada prueba](#6-detalle-de-cada-prueba)
   - [Parte A — Exploración de los datos (E01–E11)](#parte-a--exploración-de-los-datos)
   - [Parte B — Primeros modelos y ajustes (E12–E20)](#parte-b--primeros-modelos-y-ajustes)
   - [Parte C — Ingeniería inversa del problema y modelo estructurado (E21–E24)](#parte-c--ingeniería-inversa-y-modelo-estructurado)
   - [Parte D — Comparación con otros modelos y auditoría (E25–E26)](#parte-d--comparación-con-otros-modelos)
   - [Parte E — Investigación en internet y ensambles tipo Kaggle (E27–E32)](#parte-e--investigación-y-ensambles-tipo-kaggle)
   - [Parte F — Últimas ideas y restricciones monótonas (E33–E40)](#parte-f--últimas-ideas-y-restricciones-monótonas)
   - [Parte G — La variable corrupta y las ideas de la investigación de Antigravity (E41–E47)](#parte-g--la-variable-corrupta-y-las-ideas-de-la-investigación-de-antigravity)
   - [Parte H — Comparación final de 11 modelos y frente a Jhojan (E48)](#parte-h--comparación-final-de-11-modelos)
7. [El mejor modelo que tenemos](#7-el-mejor-modelo-que-tenemos)
8. [Qué esperar del resultado real (techo y ruido)](#8-qué-esperar-del-resultado-real)
9. [Errores míos, correcciones y limitaciones honestas](#9-errores-míos-correcciones-y-limitaciones-honestas)
10. [Lo que hicieron otros (Jhojan, Antigravity)](#10-lo-que-hicieron-otros)
11. [Cómo reproducir todo](#11-cómo-reproducir-todo)
12. [Ideas no probadas](#12-ideas-no-probadas)
13. [Glosario](#13-glosario)
14. [Fuentes consultadas](#14-fuentes-consultadas)

---

## 0. Resumen ejecutivo

**El problema.** Un banco quiere saber, para cada cliente, qué tan probable es que "convierta"
(contrate un producto por primera vez) en diciembre de 2026. Se entrega una probabilidad por
cliente (9.900 clientes). La nota es el **Gini** (más alto es mejor).

**El resultado.** Mejor modelo medido de forma honesta: **Gini ≈ 0.2617** (validación walk-forward
de 7 meses; era 0.2618 antes de excluir la variable corrupta de las secciones E41–E43, un empate). Punto de partida del primer modelo: 0.2551 (estimador conservador) / 0.2599 (estimador
optimista). Un modelo "ingenuo" (regresión logística simple): 0.2323.

**El modelo final** (`solucion.py` → `submission.csv`):

> 75 % LightGBM con restricciones monótonas (3 variables)  +  25 % regresión logística ajustada
> por separado dentro de cada banda de riesgo, mezclados por rangos. Sin la variable
> `dias_ultima_interaccion`, que en diciembre es ruido puro.

**Los cuatro hallazgos que definen el problema:**

1. **21 de 22 variables están congeladas por cliente**: nunca cambian entre meses. Solo
   `dias_ultima_interaccion` varía, y es una copia de `dias_ultima_transaccion` que se corrompe mes a
   mes (ver hallazgo 4). Por eso no
   existen "tendencias" ni "cambios" del cliente que explotar (mi plan inicial era justamente ese, y
   estaba equivocado).
2. **El resultado de cada mes es casi una moneda al aire** con una probabilidad que depende de
   atributos fijos. Incluso un modelo perfecto que conociera esa probabilidad exacta solo sacaría
   Gini ≈ 0.27 (techo estimado). Estamos en ≈ 97 % de ese techo.
3. **`banda_riesgo` funciona como una compuerta**: interactúa con casi todo. En banda "low" el
   número de productos importa muchísimo; en "medium" y "high" casi nada.
4. **`dias_ultima_interaccion` se corrompe con el mes calendario**: es igual a
   `dias_ultima_transaccion` en el 100 % de las filas de enero, 91 % en febrero, ... 9 % en
   noviembre y **0.26 % en diciembre**. En el test es una permutación al azar de
   `dias_ultima_transaccion`: ruido puro. Se excluye del modelo (E41–E43).

**Lo que NO funcionó** (≈ 30 ideas, todas medidas): ratios de variables, ensambles de
LightGBM+XGBoost+CatBoost, hill climbing, stacking, redes neuronales, poda de variables,
interacciones explícitas, EBM, DART, target/count encoding, ranking por pares, tuning de
hiperparámetros, promediar más semillas, `interaction_constraints`, CatBoost monótono como tercer
modelo, ponderación temporal de las observaciones y una 5.ª restricción monótona.

**Lo único que mejoró de verdad:** (a) mezclar con la logística por banda (+0.0013 a +0.0016) y
(b) forzar monotonía en el LightGBM (+0.0026 con sesgo de selección; esperar ≈ +0.0015 a +0.002 fuera
de muestra). Además (c) se excluyó una variable que en diciembre es ruido (robustez, sin cambio de
puntaje).

**Comparación final (E48).** Con el mismo protocolo para todos (6 meses, 202606–202611), nuestro modelo
queda **primero entre 11 modelos con Gini 0.2595**: LightGBM, XGBoost y CatBoost estándar dan
0.2535–0.2560, sus ensambles 0.2536–0.2563, el modelo maestro de Antigravity 0.2556 y la logística
simple 0.2302. Frente a un LightGBM como el de Jhojan corrido con el mismo protocolo: **+0.0038 ± 0.0013,
ganando los 6 meses**. Frente a los números de su captura (0.2588): +0.0007, prácticamente un empate.
La tabla está en `tabla_comparacion_modelos.png`.

**Qué esperar:** el puntaje en el leaderboard de diciembre varía **±0.014 solo por azar de
muestreo**, unas diez veces más que cualquiera de las mejoras. El ganador real entre modelos
buenos lo decide el azar.

---

## 1. Qué es este proyecto (desde cero)

### 1.1 El negocio

Un banco vende productos (tarjetas, préstamos, seguros). Cada mes, algunos clientes contratan un
producto por **primera vez**: a eso se le llama **conversión**. Al banco le conviene saber *antes*
quién tiene más probabilidad de convertir para:

- mandar ofertas solo a quienes tienen más chance (no gastar marketing en todos),
- priorizar al call center,
- decidir a quién mostrarle qué en la app.

A ese número (de 0 a 1) se le llama **propensión de conversión**.

### 1.2 La competencia

Es una competencia estilo Kaggle: te dan datos históricos **con la respuesta** (`train.csv`), datos
nuevos **sin la respuesta** (`test.csv`), y debes entregar una predicción para cada cliente del
test. Ellos comparan con la respuesta real (que no te muestran) y calculan tu nota.

- **Train:** enero a noviembre de 2026, 110.100 filas.
- **Test:** diciembre de 2026, 9.900 filas.
- **Objetivo:** `objetivo` = 1 si ese cliente convirtió ese mes, 0 si no.

### 1.3 Qué representa cada fila

**Un cliente observado en un mes.** El mismo cliente aparece en varios meses (una fila por mes en
que fue observado). Regla verificada en los datos (0 excepciones sobre 16.567 conversiones): **si
un cliente convierte, desaparece del dataset para siempre**. Por eso hay 24.628 clientes únicos
repartidos en 110.100 filas.

### 1.4 Las columnas

| Tipo | Columnas |
|---|---|
| Identificadores | `id_cliente`, `mes` (formato AAAAMM, p. ej. 202601) |
| Numéricas | `edad`, `ingresos`, `ratio_deuda_ingresos`, `antiguedad_cuenta_meses`, `numero_productos`, `saldo_promedio`, `dias_ultima_transaccion`, `antiguedad_direccion_meses`, `visitas_web_ultimos_90_dias`, `distancia_sucursal_km`, `dia_preferido_pago`, `dias_ultima_interaccion` |
| Categóricas | `ocupacion` (5), `region` (5), `canal_adquisicion` (5), `banda_riesgo` (3: low/medium/high), `dispositivo_principal` (4) |
| Booleanas | `tiene_tarjeta_credito`, `activo_movil`, `es_nuevo_cliente`, `tiene_prestamo`, `tiene_seguro` |
| Respuesta | `objetivo` (solo en train) |
| A entregar | `prediccion` (probabilidad entre 0 y 1) |

No hay valores faltantes en ningún archivo.

### 1.5 Qué hay que entregar

Un CSV con columnas exactas `id_cliente,prediccion`, **una fila por cada fila de `test.csv`, en el
mismo orden**, sin la columna `mes`.

---

## 2. Los archivos de la carpeta

| Archivo | Qué es | De quién |
|---|---|---|
| `DATASET_DESCRIPTION.md`, `metaData.csv` | Enunciado y diccionario de datos | Organizadores |
| `train.csv`, `test.csv`, `sample_submission.csv` | Datos y formato de entrega | Organizadores |
| **`solucion.py`** | **Pipeline final: valida, entrena y genera la entrega** | Este trabajo |
| **`submission.csv`** | **Entrega final (Gini esperado ≈ 0.2618)** | Este trabajo |
| `DOCUMENTACION_PROYECTO.md` | Este documento | Este trabajo |
| `modelo_maestro_bcp.py`, `submission_maestro_bcp.csv` | Modelo "maestro" de Antigravity (sin validación interna) | Antigravity |
| `modelo_avanzado.py`, `solucion_auditada.py`, `super_modelo_bcp_grandmaster.py` | Otros scripts de Antigravity | Antigravity |
| `submission_antigravity.csv`, `submission_super_poderoso.csv` | Otras entregas de Antigravity | Antigravity |
| `WhatsApp Image 2026-10-02 at 12.51.44 AM.jpeg` | Captura con resultados de Jhojan | Jhojan |
| **`tabla_comparacion_modelos.png`** | **Imagen tipo terminal con la tabla de 11 modelos (E48)** | Este trabajo |
| `tabla_modelos.py` | Recalcula la tabla comparativa (6 folds, mismo protocolo para todos) | Este trabajo |
| `resultados_modelos.csv` | Números de la tabla comparativa | Este trabajo |

Los ~20 scripts de experimentos (`ablacion.py`, `tuning.py`, `ens.py`, `robust_mono.py`, etc.) están
en el directorio temporal de la sesión (ver [sección 11](#11-cómo-reproducir-todo)).

---

## 3. Cómo se mide: Gini, AUC y la validación

### 3.1 AUC y Gini

- **AUC**: toma todos los pares posibles (un cliente que convirtió, uno que no) y mide en qué
  fracción de pares el modelo le puso mayor número al que sí convirtió. 0.5 = moneda al aire,
  1.0 = perfecto.
- **Gini = 2 × AUC − 1.** 0 = azar, 1 = perfecto. **Más alto es mejor** (no confundir con el Gini de
  desigualdad económica, donde menos es mejor).
- Importante: es una métrica de **orden**. Solo importa que los que van a convertir queden por
  encima de los que no; los números no tienen que estar "calibrados". Por eso cualquier
  transformación creciente de las predicciones (por ejemplo, convertirlas a rangos) **no cambia el
  Gini**.

Referencia: Gini 0.26 ⇔ AUC 0.63. Es decir, el modelo ordena bien un par convertido/no convertido el
63 % de las veces.

### 3.2 Validación walk-forward (la parte más importante)

La entrega se evalúa prediciendo **un mes futuro nunca visto** (diciembre) con un modelo entrenado
en los meses anteriores. Así que la validación debe hacer exactamente eso:

> Para cada mes *m*: entrenar con todos los meses **anteriores** a *m* y medir el Gini en *m*.

Un `KFold` aleatorio **sobrestimaría** el Gini: las filas del mismo cliente en meses distintos son
casi idénticas (21 de 22 variables congeladas) y caerían a ambos lados de la división, así que el
modelo "reconocería" al cliente en lugar de generalizar.

Se usaron distintos conjuntos de meses de validación según el experimento:

| Nombre | Meses | Usado en |
|---|---|---|
| 4 folds | 202608–202611 | E12 |
| 5 folds | 202607–202611 | E07, E13, E14, E15, E16, E17, E18, E19 |
| 3 meses | 202603–202605 | E23 |
| 7 folds | 202605–202611 | E21, E22, E25 y la CV de `solucion.py` |
| 9 folds | 202603–202611 | E24 y E27–E40 |

### 3.3 Cómo se eligen las rondas de boosting

Un modelo de boosting construye árboles en secuencia; hay que decidir cuántos. Tres métodos:

1. **Early stopping contra el fold que se reporta** → *optimista* (elige el punto de corte mirando
   los mismos datos con que se puntúa). Inflaba el Gini en ≈ 0.005.
2. **Promediar el mejor número de rondas de cada fold** → sesgado y ruidoso.
3. **Promediar las curvas de AUC de todos los folds y tomar el máximo** → el método usado en el
   modelo final.

---

## 4. Cómo leer los resultados

- **Diferencia pareada.** Cada modelo se evalúa en los *mismos* folds, y se compara fold por fold
  ("el modelo B menos el modelo A en el mes 5, en el mes 6, ..."). Es mucho más sensible que
  comparar promedios: el ruido de cada mes afecta a ambos modelos por igual y se cancela.
- **± (error estándar, EE).** Incertidumbre de esa diferencia promedio. Regla usada: una mejora se
  acepta solo si es **mayor que 2 EE** y **se repite en meses no usados para elegirla**.
- **"Gana k/n"**: en cuántos de los *n* folds el modelo nuevo superó al de referencia.
- **Escala de qué es grande aquí:** el Gini de un mes cualquiera varía ±0.012–0.017 entre meses;
  una diferencia pareada de ±0.0005 ya es resoluble, pero ±0.002 es lo máximo que se espera ganar.

---

## 5. Tabla maestra: las 48 pruebas de un vistazo

Resultado: ✅ adoptado · ❌ descartado · ℹ️ diagnóstico (no es un modelo)

| # | Prueba | Resultado clave | Veredicto |
|---|---|---|---|
| E01 | Estructura del panel (regla de absorción, huecos) | 0 violaciones, 0 huecos | ℹ️ |
| E02 | Tasas de conversión por variable y deriva train/test | `banda_riesgo`: 8.7 % / 14.0 % / 18.6 %; deriva < 3 % | ℹ️ |
| E03 | Tasa de conversión según meses observados (hazard) | cae de 16.0 % a 12.6 % | ℹ️ |
| E04 | ¿Cambian las variables dentro de un mismo cliente? | 21 de 22 congeladas | ℹ️ ❌ plan de panel |
| E05 | Qué es `dias_ultima_interaccion` | copia ruidosa de `dias_ultima_transaccion` | ℹ️ |
| E06 | Qué es `id_cliente` | secuencial por cohorte de entrada | ❌ no usar |
| E07 | ¿La duración aporta, dado el modelo? | p = 0.20; χ² = 8.79 (8 gl) | ❌ composición pura |
| E08 | Target/count encoding de numéricas (¿señal por valor exacto?) | AUC 0.535 con valor exacto, 0.500 en clientes nuevos | ❌ fuga |
| E09 | Forma funcional por deciles + piso de ruido | rangos de logit vs piso 0.082 | ℹ️ |
| E10 | Barrido de 253 interacciones de a pares | 11 significativas tras Bonferroni | ℹ️ |
| E11 | Efectos principales de las 23 variables | 9 significativas, 14 no | ℹ️ |
| E12 | Ablación: 5 conjuntos de variables × 4 modelos | todo dentro del ruido (sd 0.013) | ℹ️ |
| E13 | Búsqueda aleatoria de 45 hiperparámetros | mejor +0.0033 ± 0.0015 (sesgo de selección) | ⚠️ |
| E14 | Grupos de variables, comparación pareada (7 conjuntos) | ratios −0.0013 ± 0.0009 | ❌ ratios |
| E15 | Blend LightGBM + XGBoost + GAM de splines | LGB+XGB −0.0003; +GAM −0.0036 ± 0.0013 | ❌ |
| E16 | Poda de variables (top-4 a top-18) | top-4 −0.0161 ± 0.0033; top-8 −0.0049 | ❌ |
| E17 | Curva de rondas (promedio vs argmax) | óptimo 59 rondas; meseta 44–133 | ✅ método |
| E18 | Sobreajuste por exceso de rondas | 0.2591 → 0.1744 a 5000 rondas | ℹ️ |
| E19 | Techo teórico (modelo oráculo) | 0.2694 ± 0.0066 | ℹ️ |
| E20 | Ruido de muestreo del test de diciembre | sd = 0.0143 | ℹ️ |
| E21 | Modelos estructurados (9sig, cruces, paramétrico, CatBoost) | paramétrico solo 0.2517; todos peores que GBM | ❌ salvo como mezcla |
| E22 | Confirmación pareada del tuning y del blend | tuning +0.0006 ± 0.0020; blend +0.0010 a +0.0011 | ⚠️ |
| E23 | Validación del blend en folds no usados | +0.0009 a +0.0016, 3/3 positivos | ✅ |
| E24 | Barrido del peso del blend (9 folds) | óptimo 0.7–0.8; elegido 0.75 | ✅ |
| E25 | Duelo vs "modelo maestro BCP" (mismo protocolo) | nuestro 0.2603 vs 0.2593: +0.0010 ± 0.0013 | ℹ️ |
| E26 | Análisis de la captura de Jhojan | predicciones = rank/9900 (inocuo para Gini) | ℹ️ |
| E27 | 8 familias de modelos, 9 folds | LGB 0.2593, XGB 0.2586, CatBoost 0.2577, ... | ℹ️ |
| E28 | Hill climbing (leave-one-fold-out) | 0.2593 vs 0.2605: −0.0012 ± 0.0005 | ❌ |
| E29 | Stacking logístico y NNLS | −0.0017 | ❌ |
| E30 | Promedio simple de las 8 familias | −0.0015 ± 0.0012 | ❌ |
| E31 | XGBoost con ranking por pares | Gini 0.08 | ❌ |
| E32 | Red neuronal (MLP) | 0.2330 | ❌ |
| E33 | EBM (Explainable Boosting Machine) | −0.0004 a −0.0013 | ❌ |
| E34 | DART | 0.2579 vs 0.2603 | ❌ |
| E35 | Tuning de la logística por banda (C, nudos) | −0.0001 | ❌ |
| E36 | Restricciones monótonas, 1.ª medición | +0.0010 ± 0.0007 | ⚠️ |
| E37 | Robustez monótonas (5 semillas, 5 rondas, 3 variantes) | +0.0016 a +0.0025, EE 0.0004–0.0007, 8–9/9 | ✅ |
| E38 | Monótonas extendidas (flags, banda ordinal) | 4 restricciones: +0.0026 ± 0.0008 | ✅ |
| E39 | Reajuste de rondas y peso con monótonas | meseta 60–130 rondas × peso 0.6–0.8 | ✅ |
| E40 | Promediar 5 semillas vs 1 | −0.0000 ± 0.0015 | ❌ (se deja por seguridad) |
| E41 | Validación adversarial agrupada por cliente (diciembre vs resto) | AUC 0.715 → 0.527 al quitar `dias_ultima_interaccion` | ℹ️ hallazgo |
| E42 | Autopsia de `dias_ultima_interaccion` | permutación exacta en test; AUC 0.505 en filas corruptas | ℹ️ ✅ excluirla |
| E43 | Simulación "tipo diciembre" (inter permutada en validación) | actual −0.0045 ± 0.0026; sin inter empata (−0.0001 ± 0.0009) | ✅ robustez |
| E44 | 5.ª restricción monótona (`ratio_deuda_ingresos`) | +0.0006 (t = 2.27) con inter; +0.0003 ± 0.0004 sin ella | ❌ no robusta |
| E45 | `interaction_constraints` de LightGBM (2 agrupaciones) | −0.0017 ± 0.0009 y −0.0021 ± 0.0006 (t = −3.6) | ❌ empeora |
| E46 | CatBoost monótono como tercer modelo | −0.0001 a −0.0006; ρ con LightGBM = 0.949 | ❌ |
| E47 | Ponderación temporal de observaciones (λ de 0.02 a 0.50) | pico aislado en 0.10 (+0.0010); negativo desde 0.20 | ❌ no adoptada |
| E48 | Tabla comparativa de 11 modelos, mismo protocolo (6 folds) | nuestro modelo 1.º con 0.2595; vs LightGBM estándar +0.0038 ± 0.0013 (6/6) | ℹ️ comparación |

---

## 6. Detalle de cada prueba

### Parte A — Exploración de los datos

#### E01 — Estructura del panel

**Qué.** Verificar la regla "quien convierte desaparece" y si los clientes tienen meses faltantes.
**Cómo.** Recorrer cada cliente ordenado por mes.
**Resultado.**
- 16.567 conversiones = 16.567 clientes únicos que convierten; **0 violaciones** de la regla.
- Ningún cliente tiene huecos (los meses observados son consecutivos): 0 de 24.628.
- Apariciones por cliente: 1→4.518, 2→4.517, 3→3.104, 4→2.729, 5→1.896, 6→1.763, 7→1.471, 8→825,
  9→944, 10→602, 11→2.259.
- Clientes nuevos que entran al panel por mes: ene 10.400 (cohorte inicial), feb 921, mar 2.043, abr
  827, may 1.973, jun 1.834, jul 950, ago 1.760, sep 1.299, oct 1.893, nov 728.
- De los 9.900 clientes de test: 8.061 ya estaban en train y 1.839 son nuevos (ids 24.629–26.467).

**Por qué importa.** El dataset no es "filas independientes": es un proceso donde el evento saca al
cliente del panel. Esto dio la pista inicial de que había "historia" que explotar (que luego se
refutó en E04).

#### E02 — Tasas por variable y deriva

**Resultado.**
- Tasa de conversión global 15.05 %; por mes entre 14.07 % y 15.74 % (sin tendencia).
- Correlaciones con `objetivo`: `numero_productos` 0.076, `dias_ultima_transaccion` −0.058,
  `dias_ultima_interaccion` −0.036, `ratio_deuda_ingresos` −0.012; el resto |r| < 0.012.
- `banda_riesgo`: high 8.70 %, medium 14.01 %, low 18.61 %. `activo_movil`: False 13.58 % vs True
  15.82 %. `tiene_tarjeta_credito`: False 13.94 % vs True 15.43 %. `es_nuevo_cliente`,
  `tiene_prestamo`, `tiene_seguro` casi sin efecto.
- **Deriva train → test:** las medias de las 12 variables numéricas difieren menos de 3 %
  (máximo: `numero_productos` −2.68 %, `dias_ultima_transaccion` +2.14 %). Test es una continuación
  limpia, no un dominio distinto.

**Conclusión.** Señal individual muy débil; ninguna variable sola predice bien.

#### E03 — Hazard por meses observados

**Qué.** Tasa de conversión según cuántos meses previos lleva observado el cliente (`n_prev`).
**Resultado.**

| n_prev | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tasa | 15.97 % | 16.09 % | 15.12 % | 14.73 % | 15.18 % | 14.06 % | 13.77 % | 12.89 % | 12.59 % | 14.02 % | 13.10 % |

Parece que "cuanto más tiempo sin convertir, menos probable". **Se demostró en E07 que es un efecto
de composición**, no una señal explotable.

#### E04 — ¿Cambian las variables dentro de un cliente?

**Qué.** Para cada variable, qué fracción de clientes con ≥ 2 filas tiene más de un valor distinto.
**Resultado.** **21 de 22 predictores: 0.0000** (nunca cambian). Solo `dias_ultima_interaccion`
cambia (en 80.3 % de los clientes).

**Consecuencia fuerte.** Mi plan inicial era crear features de panel (cambio de saldo vs mes
anterior, tendencias, "momentum"). **Todas habrían sido idénticamente cero.** Plan descartado
*antes* de escribir una línea del modelo.

#### E05 — Qué es `dias_ultima_interaccion`

**Resultado.** Igual a `dias_ultima_transaccion` en **54.9 %** de las filas; correlación 0.553. Para
un mismo cliente, `dias_ultima_transaccion` es constante y `dias_ultima_interaccion` toma ese mismo
valor la mayoría de los meses, y otro valor aleatorio los demás. Es una copia con ruido inyectado,
no una señal temporal real.

#### E06 — Qué es `id_cliente`

**Resultado.** Se asigna secuencialmente por cohorte de entrada (cohorte de enero: ids 1–10.400;
febrero: 10.401–11.321; etc.). Dentro de un mes, es colineal con la duración. Los ids de test nuevos
(24.629–26.467) salen del rango de train. Correlación con `objetivo`: 0.021. **No se usa.**

#### E07 — ¿La duración aporta algo, dado el modelo? (`diag_duracion.py`)

**Hipótesis.** Si hay clientes "intrínsecamente más difíciles" no observados, haber sobrevivido k
meses sin convertir debería ser información extra.
**Cómo.** Entrenar un LightGBM *solo con variables estáticas*, obtener su predicción fuera de fold,
y probar con una regresión logística si `duracion` aporta *además* de esa predicción.
**Resultado.**
- Gini solo-estáticas (5 folds): 0.2572.
- Hazard real vs predicho por duración: las diferencias son pequeñas y sin patrón (de +0.0129 a
  −0.0043).
- Regresión `y ~ logit(score) + duracion`: coeficiente de duración 0.0056, **p = 0.20**.
- Con duración en dummies: **χ² = 8.79 con 8 grados de libertad** (no significativo).

**Conclusión.** La caída del 16 % al 12.6 % es **composición**: los clientes con mayor probabilidad
convierten primero y salen; los que quedan son, en promedio, los de menor probabilidad, que el
modelo ya identifica por sus atributos. Cada mes es "una moneda nueva". No hay nada que extraer de
la duración (aunque se mantiene como variable porque no daña).

#### E08 — Target/count encoding de numéricas (¿señal por valor exacto?)

**Contexto.** Los ganadores de competencias con datos sintéticos a veces encuentran señal oculta
tratando cada valor numérico exacto como una categoría y codificándola por su tasa de conversión
(target encoding).
**Cómo.** Se calculó el encoding con meses ≤ 202608 y se midió el AUC en meses 202609–202611 (29.800
filas), separando las 7.447 filas de **clientes nuevos** (que el encoding nunca vio).

| Variable | Valores únicos | AUC valor exacto (todos) | AUC valor exacto (clientes nuevos) |
|---|---|---|---|
| ingresos | 24.232 | 0.5352 | 0.5007 |
| ratio_deuda_ingresos | 24.582 | 0.5350 | 0.5001 |
| saldo_promedio | 22.985 | 0.5353 | 0.4998 |
| distancia_sucursal_km | 24.588 | 0.5350 | 0.5007 |
| numero_productos | 5 | 0.5499 | 0.5694 |
| dias_ultima_transaccion | 364 | 0.5388 | 0.5289 |

**Conclusión.** En las variables continuas (ingresos, saldo, ratio, distancia) el valor exacto
**identifica al cliente**: parece predecir (0.535) solo porque el cliente ya fue visto y el encoding
"recuerda" su duración; en clientes nuevos es 0.500 (azar puro). Es fuga, no señal. Las variables de
pocos valores (`numero_productos`, `dias_*`) sí tienen señal real, pero ya está capturada.

**Peligro adicional:** el *count encoding* (cuántas filas tiene un cliente) **filtra el futuro**,
porque el número de filas depende de cuándo convierte. Se descartó por diseño.

#### E09 — Forma funcional por deciles + piso de ruido (`forma_dgp.py`)

**Cómo.** Para cada variable numérica, tasa de conversión por decil en escala logit; se calculó cuánto
varía "por puro ruido" con ese tamaño de muestra (error estándar del logit por decil: 0.0267; rango
esperado por azar: media 0.082, percentil 95: 0.119).
**Resultado (rango de logit entre deciles):**

| Variable | Rango | Lectura |
|---|---|---|
| numero_productos (3 niveles: 1, 2, 3+) | 0.487 | señal real |
| dias_ultima_transaccion | 0.470 | señal real, monótona decreciente |
| dias_ultima_interaccion | 0.309 | señal real, monótona decreciente |
| duracion | 0.229 | efecto de composición (E07) |
| ratio_deuda_ingresos | 0.177 | señal solo en el decil más alto |
| edad 0.117, saldo 0.114, antiguedad_cuenta 0.134 | | ≈ ruido |
| ingresos 0.080, distancia 0.083, visitas web 0.075, antiguedad_direccion 0.067, dia_pago 0.066 | | ruido |

**Hallazgo clave — la compuerta.** Logit de conversión por `banda_riesgo` × `numero_productos`:

| productos | high | low | medium |
|---|---|---|---|
| 1 | −2.372 | −1.834 | −1.850 |
| 2 | −2.413 | −1.473 | −1.852 |
| 3 | −2.307 | −0.895 | −1.748 |
| 4 | −2.204 | −0.858 | −1.673 |
| 5 | −2.125 | −0.785 | −1.724 |

En banda **low**, pasar de 1 a 5 productos mueve el logit en **1.05**; en **medium** 0.18 y en
**high** 0.29. **El efecto de `numero_productos` solo existe en la banda low.** Esto es una
interacción fuerte y explica por qué la regresión logística simple se quedaba en 0.2323.

#### E10 — Barrido de 253 interacciones (`scan_interacciones.py`)

**Cómo.** Para cada par de las 23 variables (continuas en 5 intervalos), se compara una logística
con solo efectos principales contra una con la interacción completa (test de razón de verosimilitud).
Corrección de Bonferroni por 253 tests.
**Resultado: 11 interacciones significativas.**

| Par | χ² | gl |
|---|---|---|
| numero_productos × banda_riesgo | 421.3 | 8 |
| dias_ultima_transaccion × banda_riesgo | 328.6 | 8 |
| dias_ultima_interaccion × banda_riesgo | 95.28 | 8 |
| numero_productos × duracion | 56.43 | 12 |
| canal_adquisicion × banda_riesgo | 47.87 | 8 |
| banda_riesgo × activo_movil | 43.5 | 2 |
| banda_riesgo × tiene_tarjeta_credito | 42.1 | 2 |
| numero_productos × tiene_tarjeta_credito | 35.14 | 4 |
| numero_productos × activo_movil | 32.67 | 4 |
| tiene_tarjeta_credito × activo_movil | 31.44 | 1 |
| ratio_deuda_ingresos × activo_movil | 30.84 | 4 |

**5 de las 11 involucran `banda_riesgo`**, incluidas las dos mayores.

#### E11 — Efectos principales

**Resultado (χ² frente al modelo nulo).** Significativas tras Bonferroni: `banda_riesgo` 1351,
`numero_productos` 660, `dias_ultima_transaccion` 344, `dias_ultima_interaccion` 140,
`activo_movil` 99, `duracion` 84, `tiene_tarjeta_credito` 37, `ratio_deuda_ingresos` 21,
`canal_adquisicion` 20. **Las otras 14 no son significativas** como efecto individual (edad,
ingresos, saldo, región, distancia, antigüedad de dirección, día de pago, seguro, nuevo cliente,
préstamo, dispositivo, ocupación, visitas web, antigüedad de cuenta).

**Matiz posterior (E16, E21):** que no sean significativas solas **no** significa que sobren: podarlas
empeora el modelo.

---

### Parte B — Primeros modelos y ajustes

#### E12 — Ablación: conjuntos de variables × modelos (`ablacion.py`)

**Qué.** 5 conjuntos de variables × 4 modelos, 4 folds (202608–202611), early stopping.
**Modelos:** `lgb_shallow` (4 hojas, profundidad 2), `lgb_medio` (16 hojas, profundidad 4),
`lgb_profundo` (64 hojas), regresión logística.
**Conjuntos:** A estáticas; B + duración; C + derivados de `dias_ultima_interaccion`; D + `id_cliente`;
E sin `dias_ultima_interaccion`.

| Conjunto | shallow | medio | profundo | logística |
|---|---|---|---|---|
| A estáticas | 0.2516 | 0.2542 | 0.2524 | 0.2310 |
| B + duración | 0.2504 | 0.2550 | 0.2509 | 0.2306 |
| C + derivados inter. | 0.2519 | **0.2572** | 0.2505 | 0.2302 |
| D + id_cliente | 0.2511 | 0.2554 | 0.2541 | 0.2311 |
| E sin inter. | 0.2517 | 0.2560 | 0.2520 | 0.2306 |

**Lecturas.** (1) LightGBM ≈ 0.255 vs logística 0.231: hay **no linealidad real** (+0.025).
(2) Entre conjuntos de variables las diferencias (0.254–0.257) caen **dentro del ruido** (desvío
entre folds 0.007–0.017). Perseguirlas sería autoengaño.

#### E13 — Búsqueda aleatoria de 45 hiperparámetros (`tuning.py`)

**Cómo.** 45 configuraciones aleatorias (tasa de aprendizaje, hojas, profundidad, hoja mínima,
regularización L1/L2, submuestreo) + la base; 5 folds, 2 semillas por config, comparación pareada
contra la base.
**Resultado.** Base 0.2595. Mejor config: **0.2628 (+0.0033 ± 0.0015)**. Las 8 mejores comparten
patrón: tasa 0.05, `min_child_samples` alto (400–800), `feature_fraction` 1.0.
**Advertencia (maldición del ganador).** El mejor de 45 configuraciones siempre sale favorecido por
puro azar. En E22 se re-evaluó de forma independiente: **+0.0006 ± 0.0020 (no significativo)**.
Aun así, esa región de parámetros (tasa 0.05, hoja mínima 400, λ = 20) se adoptó como base.

#### E14 — Grupos de variables, comparación pareada (`test_features.py`)

5 folds, 2 semillas, LightGBM. Diferencias pareadas contra la base (23 variables):

| Conjunto | Gini | Dif. pareada |
|---|---|---|
| base (22 + duración + derivados inter.) | 0.2595 | — |
| + 14 ratios (saldo/ingresos, deuda absoluta, capacidad de pago...) | 0.2582 | **−0.0013 ± 0.0009** |
| sin duración | 0.2587 | −0.0008 ± 0.0003 |
| sin derivados de inter. | 0.2600 | +0.0005 ± 0.0013 |
| sin `dias_ultima_interaccion` | 0.2591 | −0.0004 ± 0.0012 |
| ratios, sin inter. | 0.2589 | −0.0006 ± 0.0022 |
| solo ratios + categóricas | 0.2569 | −0.0026 ± 0.0024 |

**Hipótesis refutada:** "los árboles aproximan mal los cocientes, darles ratios ayuda". En estos
datos **empeora levemente**. Los derivados de `dias_ultima_interaccion` se quitan por parsimonia.
`duracion` se mantiene (aporta poco, pero de forma consistente).

#### E15 — Blend de familias: LGB + XGB + GAM (`test_features.py`)

Mezcla por rangos sobre las 5 folds.

| Modelo | Gini | Dif. vs LGB solo |
|---|---|---|
| LightGBM | 0.2600 | — |
| LGB + XGB | 0.2597 | −0.0003 ± 0.0006 |
| 0.5 LGB + 0.4 XGB + 0.1 GAM | 0.2594 | −0.0005 ± 0.0006 |
| XGBoost solo | 0.2589 | −0.0011 ± 0.0014 |
| 0.6 LGB + 0.2 XGB + 0.2 GAM | 0.2587 | −0.0013 ± 0.0008 |
| LGB + XGB + GAM (partes iguales) | 0.2564 | **−0.0036 ± 0.0013** |
| GAM de splines solo | 0.2328 | −0.0272 ± 0.0036 |

**Conclusión.** Ensamblar no ayuda; añadir el GAM **empeora de forma significativa**. LightGBM solo
es lo mejor entre estas.

#### E16 — Poda de variables (`test_poda.py`)

Ordenadas por importancia (gain), 5 folds, 3 semillas.

| Variables usadas | Gini | Dif. vs 23 |
|---|---|---|
| top-4 | 0.2441 | −0.0161 ± 0.0033 |
| top-6 | 0.2518 | −0.0084 ± 0.0023 |
| top-8 | 0.2553 | −0.0049 ± 0.0017 |
| top-10 | 0.2598 | −0.0004 ± 0.0023 |
| top-12 | 0.2594 | −0.0009 ± 0.0018 |
| top-15 | 0.2603 | +0.0001 ± 0.0019 |
| top-18 | 0.2595 | −0.0007 ± 0.0018 |
| **todas (23)** | **0.2602** | — |

**Conclusión.** Por debajo de 10 variables empeora de verdad; de 10 a 23 es plano. Las variables
"sin efecto significativo" (E11) **sí cuentan en conjunto**: los tests individuales tienen poca
potencia para efectos débiles, y el GBM los agrega.

#### E17 — Curva de rondas (`curva_rondas.py`)

**Qué.** ¿Cuántas rondas para el modelo final? Argmax individual por fold: 58, 131, 44, 56, 71
(promedio 72, muy disperso). Promediando las **curvas** y tomando el máximo: **59 rondas**, Gini
0.2557.
**Planitud.** Rondas dentro de 0.001 del óptimo: 44 a 133 → la elección exacta importa poco.
Pasadas 150 rondas degrada en serio (−0.0047 a 200, −0.0216 a 600).
**Corrección.** El Gini que reportaba el primer script (0.2599) usaba early stopping contra el
mismo fold; el estimador honesto era 0.2551–0.2557.

#### E18 — Sobreajuste por exceso de rondas (`test_sobreajuste.py`)

Surgió al ver que la captura de Jhojan tenía predicciones llegando a 1.000 (ver E26; resultó ser
una conversión a rangos y no sobreajuste). Se midió igualmente:

| Rondas | Gini CV | Predicción máxima en test |
|---|---|---|
| 80 | 0.2571 | 0.39 |
| 150 | 0.2549 | 0.44 |
| 300 | 0.2457 | 0.46 |
| 600 | 0.2343 | 0.52 |
| 1200 | 0.2194 | 0.59 |
| 2500 | 0.1953 | 0.70 |
| 5000 | 0.1744 | 0.79 |

Óptimo: 50 rondas (0.2591). Entrenar de más destruye el Gini; por eso el modelo final usa un número
de rondas elegido por validación, no uno "grande".

#### E19 — Techo teórico (`techo_bayes.py`)

**Idea.** Si cada mes es una moneda con probabilidad p(x), incluso un **oráculo** que conozca p(x)
exactamente tiene Gini finito. Se estima simulando resultados Bernoulli con nuestras predicciones
como p(x).
**Resultado.**
- Gini real medido (predicciones fuera de fold, 5 meses): 0.2592.
- Pendiente de calibración medida: 1.2624 (el modelo está "sobre-encogido").
- Techo **conservador** (p̂ crudo): 0.2165 ± 0.0069 (ya superado por nuestro modelo, lo que prueba
  que subestima).
- Techo **recalibrado**: **0.2694 ± 0.0066** → capturamos ≈ 96 %.
**Limitación.** Es una **estimación circular**: usa nuestro propio modelo. Si existiera estructura
que ningún modelo detectó, el techo real sería más alto. Se considera improbable (8 familias de
modelos, 3 equipos y el barrido de 253 interacciones convergen en ≈ 0.26).

#### E20 — Ruido de muestreo del test

Simulando los resultados Bernoulli del test (9.900 filas, ≈ 1.470 convertidos esperados) con
nuestro propio modelo: **desviación estándar del Gini ≈ 0.0143**, rango p5–p95 de unos ±0.023. (De
esa simulación solo se usa la dispersión; su media no es comparable con el techo.)
Es ≈ 10 veces la resolución de los tests pareados y mayor que todo el margen hasta el techo.

---

### Parte C — Ingeniería inversa y modelo estructurado

#### E21 — Modelos estructurados (`modelo_estructurado.py`)

**Hipótesis.** Si se conoce la estructura (E09–E11), un modelo paramétrico bien especificado tiene
menos parámetros efectivos y debería tener menos varianza que un GBM.
**7 folds.** Variantes:

| Modelo | Gini |
|---|---|
| **B** LightGBM, 23 variables, parámetros del tuning | **0.2583** |
| A LightGBM, parámetros previos | 0.2566 |
| F CatBoost (236 rondas) | 0.2558 |
| C LightGBM solo con las 9 variables significativas | 0.2548 |
| D LightGBM + 8 interacciones como factores cruzados | 0.2540 |
| E 9 variables + cruces | 0.2506 |
| **G Logística separada por banda de riesgo** | **0.2517** |

**El modelo paramétrico G** ajusta una regresión logística *por separado dentro de cada banda*
(así captura por construcción las interacciones con la banda). Dentro de cada banda usa: dummies de
`numero_productos`, `canal_adquisicion` y `duracion`; splines (6 nudos, grado 3) sobre
`dias_ultima_transaccion`, `dias_ultima_interaccion` y `ratio_deuda_ingresos`; `activo_movil`,
`tiene_tarjeta_credito` y su producto; e interacciones splines × `activo_movil` y productos ×
flags. Estandarizado y regularizado (C = 0.05).
**Lecturas (contra mi hipótesis).**
1. Quitar las 14 variables "de ruido" **empeora** (0.2548).
2. Dar las interacciones masticadas al GBM **empeora** (0.2540): ya las encuentra con sus cortes.
3. El modelo paramétrico **solo pierde** (0.2517): el GBM extrae más de lo que se puede especificar
   a mano.
**Pero** ese modelo débil tiene un sesgo muy distinto al del GBM, lo que motivó E22.

#### E22 — Confirmación pareada y blend (`confirmacion.py`)

7 folds, 3 semillas. Diferencias pareadas:

| Modelo | Gini | Dif. vs A (actual) | Dif. vs B (tuned) |
|---|---|---|---|
| A (LGB parámetros previos) | 0.2587 | — | −0.0006 ± 0.0020 |
| B (LGB tuned) | 0.2592 | +0.0006 ± 0.0020 | — |
| G paramétrico | 0.2517 | −0.0070 ± 0.0048 | −0.0075 ± 0.0034 |
| blend 0.9 B + 0.1 G | 0.2598 | +0.0011 ± 0.0022 | +0.0006 ± 0.0003 |
| blend 0.8 B + 0.2 G | 0.2602 | +0.0015 ± 0.0024 | +0.0010 ± 0.0006 |
| blend 0.7 B + 0.3 G | 0.2603 | +0.0016 ± 0.0027 | +0.0011 ± 0.0010 |

**Lectura.** El tuning **no** sobrevive (+0.0006 ± 0.0020), pero aparece algo: mezclar un 10–30 %
del modelo paramétrico gana +0.0006 a +0.0011 de forma consistente.
**Riesgo.** Tras ~12 ideas probadas, que una dé t ≈ 1.7 es lo que produce el azar. Se validó en E23.

#### E23 — Validación del blend en meses no usados (`validacion_limpia.py`)

Se evaluó en los meses 202603, 202604 y 202605 con entrenamiento solo en meses anteriores.

| Mes | blend 0.9 | blend 0.8 | blend 0.7 |
|---|---|---|---|
| 202603 | +0.0007 | +0.0011 | +0.0012 |
| 202604 | +0.0012 | +0.0021 | +0.0027 |
| 202605 | +0.0006 | +0.0009 | +0.0010 |
| **Media ± EE** | +0.0009 ± 0.0002 | +0.0014 ± 0.0004 | +0.0016 ± 0.0006 |

Positivo en los 3 meses y en los 3 pesos. **Salvedad que debo señalar:** de esos 3 meses, **202605
ya formaba parte de los 7 folds usados para elegir** el blend en E22; solo **202603 y 202604 eran
realmente nuevos**. (Ambos son positivos, pero la réplica es de 2 meses, no de 3 como se dijo
inicialmente.)

#### E24 — Barrido del peso del blend (`peso_blend.py`)

9 folds, mezcla de rangos GBM (tuned) + paramétrico.

| Peso del GBM | Gini | Dif. vs GBM solo | Folds ganados |
|---|---|---|---|
| 1.0 | 0.2592 | — | — |
| 0.9 | 0.2599 | +0.0007 ± 0.0002 | 7/9 |
| 0.8 | 0.2603 | +0.0011 ± 0.0005 | 6/9 |
| **0.7** | **0.2605** | +0.0013 ± 0.0008 | 6/9 |
| 0.6 | 0.2604 | +0.0012 ± 0.0011 | 6/9 |
| 0.5 | 0.2598 | +0.0006 ± 0.0014 | 6/9 |
| 0.4 | 0.2589 | −0.0003 ± 0.0017 | 4/9 |
| 0.3 | 0.2577 | −0.0015 ± 0.0019 | 4/9 |

Curva **suave y de un solo pico** (óptimo entre 0.7 y 0.8): el ruido no produce curvas así, lo que
apoya que el efecto es real. Se eligió **0.75**, el centro de la meseta, para no sobreajustar el
peso exacto.

---

### Parte D — Comparación con otros modelos

#### E25 — Duelo contra el "modelo maestro BCP" de Antigravity (`duelo.py`)

**Hallazgo previo (lectura de código).** `modelo_maestro_bcp.py`, que generó
`submission_maestro_bcp.csv`, **importa `roc_auc_score` pero nunca lo llama**: no tiene folds, ni
validación ni cálculo de la métrica. Entrena con rondas fijas (110/130/120) y pesos fijos
(0.50/0.30/0.20) escritos a mano. El "0.2615" que se reportó no sale de ese script. Por su parte,
`modelo_avanzado.py` sí valida, pero evalúa **otro modelo** (LGB + CatBoost 0.55/0.45) distinto del
que genera la entrega, y usa `eval_set` en CatBoost (que por defecto elige la mejor iteración
mirando el fold evaluado).
**Cómo.** Se replicó su pipeline exacto (variables, 3 modelos, rondas, pesos) y se evaluó con nuestro
protocolo, en los mismos 7 folds.

| Modelo | Gini |
|---|---|
| **Nuestro (para entonces: 0.75 LGB + 0.25 logística por banda)** | **0.2603** |
| Nuestro GBM solo | 0.2593 |
| Su modelo maestro (LGB+CatBoost+XGB) | 0.2593 |
| — su LightGBM solo | 0.2591 |
| — su XGBoost solo | 0.2581 |
| — su CatBoost solo | 0.2556 |

Diferencia pareada: **+0.0010 ± 0.0013** (gana 4 de 7 folds): **empate estadístico**. Su ensamble de
tres librerías saca +0.0002 sobre su propio LightGBM: ensamblar no ayuda aquí. Correlación de rangos
entre sus entregas y la nuestra: 0.94–0.96.

#### E26 — La captura de Jhojan

Resultados mostrados (6 folds): LGBM 0.2588, CatBoost 0.2552, XGB 0.2523, regresión logística 0.2323;
ensambles LGBM+CB 0.2587, LGBM+CB+XGB 0.2576, LGBM+CB+LR 0.2558, los 4 juntos 0.2564; "ensemble solo
LGBM" 0.2588.
Coinciden con lo nuestro (E12–E15, E21): logística ≈ 0.232, ensambles **no** superan a LightGBM
solo, CatBoost no le gana.
**Predicciones que llegan a 1.000.** Inicialmente lo interpreté como sobreajuste. **Error mío,
retirado:** los valores (1.000000, 0.999899, 0.999798, ...) coinciden exactamente con `rank/9900`: la
predicción es una **conversión a rangos**, que no altera el Gini.

---

### Parte E — Investigación y ensambles tipo Kaggle

Se buscó en internet cómo ganan este tipo de competencias (fuentes en la sección 14). Receta
encontrada: (1) validación que imite al test; (2) muchos modelos de **familias distintas**;
(3) **hill climbing** (partir del mejor modelo y añadir el que más mejore, con reemplazo);
(4) **stacking** (un segundo modelo que aprende a combinar); (5) ingeniería de variables masiva
(target y count encoding de columnas y pares); (6) pseudo-etiquetado y reentrenar con todos los
datos.

#### E27 — Ocho familias de modelos, predicciones fuera de fold (`oof_gen.py`)

9 folds, rondas fijas, 3 semillas.

| Modelo | Gini medio |
|---|---|
| LightGBM profundo | 0.2593 |
| XGBoost | 0.2586 |
| CatBoost | 0.2577 |
| LightGBM casi aditivo (profundidad 2) | 0.2544 |
| Logística por banda | 0.2521 |
| LightGBM por banda | 0.2510 |
| Red neuronal (MLP) | 0.2330 |
| Logística simple | 0.2323 |

**Correlaciones entre modelos (rangos).** Los tres boosting entre sí: 0.93–0.96 (casi no se
diferencian). La logística por banda vs boosting: 0.81. Red neuronal vs boosting: ≈ 0.80. La
logística por banda es el único modelo a la vez **distinto y decente**.

#### E28 — Hill climbing (leave-one-fold-out) (`ens.py`)

Para no sobreajustar los pesos, cada método se ajusta con 8 meses y se mide en el noveno, rotando.
**Resultado:** 0.2593 vs **0.2605** de nuestro blend: **−0.0012 ± 0.0005**. Los pesos elegidos son
inestables entre rotaciones (XGB entra y sale), pero la logística por banda aparece en las 9.

#### E29 — Stacking (logística y NNLS)

Leave-one-fold-out. Logístico: 0.2588 (−0.0017 ± 0.0007), NNLS: 0.2588 (−0.0017 ± 0.0009).

#### E30 — Promedio simple de las 8 familias

0.2590 (−0.0015 ± 0.0012). Incluir modelos débiles resta.

#### E31 — XGBoost con ranking por pares (`rank:pairwise`)

Gini **0.08** (con `min_child_weight` 0.01: 0.0829; 0.1: 0.0784; 1.0: 0.0635). Un único grupo
gigante muestrea mal los pares. Descartado.

#### E32 — Red neuronal (MLP)

**Primer intento:** Gini 0.1095. **Causa:** el `early_stopping` de scikit-learn monitorea
*accuracy*; con 15 % de positivos esa métrica queda plana y corta el entrenamiento a las pocas
épocas. **Corrección:** 20 épocas fijas, `alpha = 0.03` (con 40 épocas sobreajusta: 0.1892). Resultado
final: 0.2330. Aun corregida, es demasiado débil para aportar.

**Resumen de los ensambles de la receta Kaggle (E28–E30).**

| Método | Gini (LOFO) | Dif. vs nuestro blend (0.2605) |
|---|---|---|
| Hill climbing | 0.2593 | −0.0012 ± 0.0005 |
| Promedio de 8 | 0.2590 | −0.0015 ± 0.0012 |
| Stacking | 0.2588 | −0.0017 |

**Por qué no funcionan aquí.** Los boosting ordenan casi igual (ρ ≈ 0.95), así que combinarlos no suma
información; los modelos realmente distintos (red, logística simple) rinden demasiado poco. El único
complemento útil ya estaba en el blend.

**Dos técnicas de la receta que no se implementaron como modelo:**
- *Ingeniería masiva de variables (target/count encoding)*: E08 demostró que el valor exacto de las
  numéricas identifica al cliente (fuga) y que el count encoding filtra el futuro.
- *Pseudo-etiquetado*: no se probó (ver sección 12). Reentrenar con todos los datos sí se hace en el
  modelo final (110.100 filas).

---

### Parte F — Últimas ideas y restricciones monótonas

Las cuatro ideas siguientes salieron del script `ideas.py` (9 folds, 3 semillas, comparadas contra la
referencia de entonces, 0.2605).

#### E33 — EBM (Explainable Boosting Machine)

Modelo aditivo con curvas por variable e interacciones de a pares (paquete `interpret`).
Gini solo: **global 0.2510, por banda 0.2524**. Dentro de la mezcla frente a la referencia:
0.75 LGB + 0.25 EBM-banda **−0.0010 ± 0.0008** (4/9); 0.75 LGB + 0.25 EBM global **−0.0013 ± 0.0010**
(3/9); 0.60 monótona + 0.20 paramétrico + 0.20 EBM-banda +0.0005 ± 0.0004 (6/9, pero ese +0.0005 viene
de la parte monótona, no del EBM). **Descartado.**

#### E34 — DART

Boosting con *dropout* de árboles. Gini solo **0.2579**, frente a 0.2603 del LightGBM monótono. Mezcla
0.50 monótona + 0.25 paramétrico + 0.25 DART: +0.0006 ± 0.0005 (4/9), sin ganancia sobre usar solo la
monótona. Antigravity obtuvo lo mismo (0.2580). **Descartado.**

#### E35 — Ajuste fino de la logística por banda

Variando la regularización (C = 0.02 y 0.2) y los nudos de los splines (4 y 8). Gini solo: C = 0.02
0.2529, 4 nudos 0.2529, C = 0.2 0.2509, 8 nudos 0.2504 (la base en 9 folds: 0.2521). Dentro de la
mezcla: 0.75 LGB + 0.25 (C = 0.02) **−0.0001 ± 0.0001** (2/9); (4 nudos) −0.0001 ± 0.0002 (4/9).
**Descartado:** la logística original ya estaba bien configurada.

#### E36 — Restricciones monótonas, primera medición

Se forzó al LightGBM a que `dias_ultima_transaccion` y `dias_ultima_interaccion` solo puedan *bajar* la
probabilidad y `numero_productos` solo *subirla* (3 variables; direcciones leídas de los deciles de
E09).
- Gini solo: **0.2603** vs 0.2593 del LightGBM sin restricciones: **+0.0010 ± 0.0007**. Diferencia por
  fold: +0.0004, −0.0003, +0.0024, +0.0037, +0.0042, +0.0002, −0.0019, −0.0009, +0.0016.
- Dentro de la mezcla: 0.75 monótona + 0.25 paramétrico **+0.0008 ± 0.0006** (gana 6/9); 0.70/0.30
  +0.0008 ± 0.0005 (5/9).

Lectura: la única candidata de las cuatro, pero con t ≈ 1.3, **por debajo del umbral de 2 EE**. En esa
corrida se usaron 3 semillas. Antes de adoptarla se comprobó su robustez (E37).

#### E37 — Robustez de las monótonas (`robust_mono.py`)

**Qué.** Antes de adoptarlas, comprobar si el efecto es real o frágil: 9 folds, **5 semillas**,
rondas 50/65/80/100/130 y tres niveles de restricción. (Dirección de las restricciones: más días sin
transacción/interacción → nunca sube; más productos → nunca baja.)
**Mezcla 0.75/0.25, Gini medio y diferencia pareada vs sin restricciones:**

| Restricciones | 50 | 65 | 80 | 100 | 130 |
|---|---|---|---|---|---|
| ninguna | 0.2595 | 0.2600 | 0.2599 | 0.2599 | 0.2589 |
| 3 variables (trans., inter., productos) | 0.2614 (+0.0019) | 0.2617 (+0.0016) | 0.2617 (+0.0018) | 0.2618 (+0.0020) | 0.2614 (+0.0025) |
| 2 variables (trans., productos) | 0.2611 (+0.0016) | 0.2615 (+0.0015) | 0.2614 (+0.0015) | 0.2613 (+0.0014) | 0.2608 (+0.0019) |
| 1 variable (solo trans.) | 0.2600 (+0.0005) | 0.2608 (+0.0008) | 0.2604 (+0.0005) | 0.2605 (+0.0007) | 0.2596 (+0.0007) |

**Error estándar** de la diferencia con 3 variables: 0.0005, 0.0004, 0.0004, 0.0005, 0.0007 (≈ 4 EE),
con **8/9 folds** ganados (9/9 a 130 rondas).
**Lecturas.** (1) Estable con cualquier número de rondas entre 50 y 130. (2) **Dosis-respuesta**: 1
restricción +0.0007, 2 → +0.0015, 3 → +0.0018: difícil de explicar por azar. (3) En LightGBM solo
(sin mezcla) pasa de 0.2587 a 0.2608.
**Por qué ayuda.** El GBM sin restricción ajusta escalones espurios en variables cuyo efecto
verdadero es suave y monótono; la restricción es una regularización con base a priori (las
direcciones salen de los deciles de E09).

#### E38 — Monótonas extendidas (`mono_ext.py`)

9 folds, 5 semillas. Variantes (mezcla 0.75/0.25), diferencia pareada vs sin restricciones:

| Variante | 65 rondas | 100 rondas |
|---|---|---|
| A: 3 restricciones | +0.0016 ± 0.0004 (8/9) | +0.0020 ± 0.0005 (8/9) |
| B: 3 + `activo_movil` + `tarjeta` | +0.0017 ± 0.0008 (7/9) | +0.0023 ± 0.0008 (7/9) |
| **D: 3 + `banda_riesgo` ordinal** | **+0.0026 ± 0.0008 (7/9)** | **+0.0026 ± 0.0008 (8/9)** |
| C: todo (3 + flags + banda ordinal) | +0.0024 ± 0.0009 (7/9) | +0.0026 ± 0.0009 (8/9) |
| E: banda ordinal **sin** restricción | +0.0001 ± 0.0001 (5/9) | +0.0003 ± 0.0002 (6/9) |

**Lecturas.** (1) Codificar `banda_riesgo` como ordinal (high = 0, medium = 1, low = 2; sus tasas
8.7 % < 14.0 % < 18.6 % son monótonas) **y** restringirla da el mejor resultado (D). (2) **La
variante E prueba que la ganancia viene de la restricción, no de la recodificación.** (3) Las flags no
añaden nada sobre D; se omiten por parsimonia. **Se adopta D.**

#### E39 — Reajuste de rondas y peso con monótonas (`retune_mono.py`)

9 folds, 5 semillas. Gini de la mezcla según rondas (filas) y peso del LightGBM (columnas):

| Rondas | 1.0 | 0.9 | 0.8 | 0.75 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|---|
| 40 | 0.2579 | 0.2590 | 0.2597 | 0.2599 | 0.2600 | 0.2600 | 0.2597 |
| 60 | 0.2612 | 0.2618 | 0.2622 | 0.2623 | 0.2622 | 0.2619 | 0.2611 |
| 80 | 0.2613 | 0.2620 | 0.2623 | 0.2624 | 0.2624 | 0.2621 | 0.2613 |
| **100** | 0.2612 | 0.2620 | 0.2624 | **0.2625** | **0.2625** | 0.2622 | 0.2615 |
| 130 | 0.2602 | 0.2612 | 0.2618 | 0.2620 | 0.2621 | 0.2619 | 0.2613 |
| 170 | 0.2587 | 0.2600 | 0.2608 | 0.2611 | 0.2613 | 0.2614 | 0.2609 |
| 220 | 0.2567 | 0.2583 | 0.2595 | 0.2599 | 0.2603 | 0.2606 | 0.2604 |
| 300 | 0.2527 | 0.2550 | 0.2567 | 0.2574 | 0.2580 | 0.2588 | 0.2590 |

Máximo: 100 rondas con peso 0.7 (0.2625). Meseta (dentro de 0.0005): 60–130 rondas y peso 0.6–0.8.
El peso 0.75 ya usado queda en el centro. En `solucion.py` las rondas salen de la curva promediada de
7 folds (71 rondas).
**Cambio adicional.** Ya no se escalan las rondas por el tamaño de datos: el óptimo por fold no
muestra tendencia con las filas de entrenamiento (23 a 131 rondas con 60k–100k filas).

#### E40 — ¿Sirve promediar semillas?

Promediar 5 semillas vs 1: **−0.0000 ± 0.0015**. No aporta; se mantienen 3 semillas solo como
protección contra una semilla desafortunada.

---

### Parte G — La variable corrupta y las ideas de la investigación de Antigravity

Antigravity entregó un informe de investigación (papers recientes de ML tabular y soluciones ganadoras de
Kaggle) con cinco propuestas. Se probaron todas con el protocolo del proyecto (9 folds, 5 semillas,
80 rondas, diferencia pareada contra la referencia de entonces: LightGBM monótono de 4 variables +
logística por banda, mezcla 0.75/0.25, Gini 0.2624). De esa tanda salió un hallazgo mayor que las cinco
ideas.

#### E41 — Validación adversarial (`adv.py`)

**Idea.** Entrenar un clasificador para distinguir las filas de diciembre del resto; si lo logra, hay
deriva de covariables. **Precaución que la propuesta omitía:** el 81 % de los clientes de diciembre ya
aparece en train con variables idénticas, así que un clasificador ingenuo "memoriza clientes" y da un AUC
falso. Se usó `GroupKFold` por `id_cliente`.

| Variables | AUC adversarial |
|---|---|
| todas, incluida `duracion` | 0.7816 |
| sin `duracion` (esa diferencia es composición por supervivencia: los de diciembre llevan más meses) | 0.7150 |
| sin `duracion` ni `dias_ultima_interaccion` | **0.5274** |
| diciembre vs noviembre, sin `duracion` | 0.5215 |

Variables que más delatan a diciembre (gain): `dias_ultima_transaccion` 54.0 %,
`dias_ultima_interaccion` 26.1 %, el resto ≤ 2.2 %. Medias diciembre vs train: `numero_productos`
−2.68 %, `dias_ultima_transaccion` +2.14 %, `banda_ord` −5.60 %, `activo_movil` −1.41 %,
`tiene_tarjeta_credito` −0.22 %, `ratio_deuda_ingresos` +0.91 % (todo composición por supervivencia).
**Casi toda la deriva está en `dias_ultima_interaccion`.**

#### E42 — Autopsia de `dias_ultima_interaccion`

Hasta aquí se la había descrito como "copia ruidosa de `dias_ultima_transaccion`" (E05). Faltaba lo
esencial: **el ruido crece con el mes calendario.**

| Mes | ene | feb | mar | abr | may | jun | jul | ago | sep | oct | nov | **dic (test)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % de filas con inter = trans | 100 | 90.9 | 81.9 | 72.8 | 63.7 | 54.7 | 45.7 | 36.5 | 27.6 | 18.4 | 9.4 | **0.26** |

(En el test solo 26 de 9.900 filas coinciden, el azar esperado de una permutación.) Además:
- En el test, `dias_ultima_interaccion` es una **permutación exacta** de `dias_ultima_transaccion`
  (los valores ordenados son idénticos). Esto explica un dato que ya estaba en la primera tabla de
  deriva y que no interpreté: las dos medias eran idénticas hasta el sexto decimal (194.307273).
- En train, filas **corruptas** (49.627): correlación con la transacción 0.0025; **AUC de inter contra la
  conversión 0.5052** (nada), mientras `dias_ultima_transaccion` en esas mismas filas tiene 0.5442. Filas
  **iguales** (60.473): AUC 0.5482, el mismo que el de la transacción.
- Los valores corruptos van de 1 a 364 (media 191.7).
- Fracción de filas iguales según meses previos del cliente: 71.4 % (0 meses), 65.2 %, 60.4 %, 54.3 %,
  49.3 %, 42.6 %, 36.6 %, 32.7 %, 24.4 %, 17.4 %, 10.8 % (10 meses).

**Conclusión.** Toda su "señal" en train viene de ser una copia de `dias_ultima_transaccion`; no tiene
información propia. En diciembre es ruido puro. Por eso E12 y E14 no encontraron pérdida al quitarla.

#### E43 — Validación "tipo diciembre" (`dec_like.py`)

**Cómo.** En cada mes de validación se reemplaza `dias_ultima_interaccion` por una permutación de
`dias_ultima_transaccion` (como en el test) y se compara el modelo actual (con la variable, entrenado con
sus valores reales) contra un modelo sin ella. 9 folds, mezcla 0.75/0.25.

| Modelo | Gini | Dif. vs actual en validación normal (0.2624) |
|---|---|---|
| Actual con la variable permutada (como diciembre) | 0.2579 | −0.0045 ± 0.0026 (t = −1.76), gana 2/9 |
| **Sin `dias_ultima_interaccion`** | **0.2622** | −0.0001 ± 0.0009 (t = −0.16), gana 4/9 |

Sin vs actual en condiciones de diciembre: **+0.0044 ± 0.0027** (t = 1.64, 6/9). Por mes: 202603 **+0.0238**,
202604 +0.0080, 202605 +0.0073, 202606 −0.0003, 202607 +0.0001, 202608 −0.0001, 202609 +0.0002,
202610 −0.0004, 202611 +0.0008.

**Lectura.** El daño se concentra en los primeros meses, cuando el entrenamiento aún tiene pocas filas
corruptas y el modelo se apoya en la variable. De junio en adelante la diferencia es ≈ 0: el modelo
aprendió a ignorarla. El modelo final se entrena con 11 meses, así que el daño esperado en diciembre es
**pequeño**, pero quitarla cuesta −0.0001 ± 0.0009 (empate) y elimina el riesgo. **Es un arreglo de
robustez, no una mejora de puntaje.** Se aplicó en `solucion.py` al LightGBM y a la logística.

La validación walk-forward estándar no revelaba esto porque sus meses de validación tienen la variable
parcialmente limpia (82 % de coincidencia en marzo, 9 % en noviembre), a diferencia de diciembre (0.26 %).

#### E44 — 5.ª restricción monótona (`ratio_deuda_ingresos` decreciente)

La propuesta incluía un indicador `sin_deuda = (ratio == 0)`. **No existe ningún cliente con ratio = 0**
(mínimo 0.02, 168 filas topadas ahí; máximo 0.95), así que ese indicador sería una columna constante. La
relación real es plana (~15.3 %) hasta el ventil 17 y cae solo en los dos más altos (13.6 % y 13.2 %).
- Con el modelo con la variable corrupta (`ideas2.py`): solo +0.0008 ± 0.0003, mezcla
  **+0.0006 ± 0.0003** (t = 2.27, 7/9). (Antigravity midió 0.2631 en 9 folds, consistente con mi 0.2630.)
- Sobre el modelo sin la variable corrupta (`final_check.py`): 50/80/110 rondas dan −0.0002 ± 0.0003,
  **+0.0003 ± 0.0004** (t = 0.84) y +0.0003 ± 0.0005. Muestra dividida: +0.0002 ± 0.0006 (3/5) y
  +0.0005 ± 0.0006 (1/4).

**No adoptada:** el efecto no es robusto y no supera 2 EE tras quitar la variable corrupta.

#### E45 — `interaction_constraints` de LightGBM

Restringe qué variables pueden aparecer juntas en una misma rama. Dos agrupaciones (el resto de
variables, como grupos de un solo elemento):
- La propuesta: grupo "compuerta" {banda, productos, transacción, interacción, `activo_movil`, tarjeta} y
  grupo "solvencia" {ingresos, saldo, ratio}.
- Guiada por mi barrido de pares (E10): solo las 9 variables que aparecen en alguna interacción
  significativa pueden interactuar entre sí.

| Variante | Gini solo | Mezcla | Dif. mezcla | Gana |
|---|---|---|---|---|
| Propuesta | 0.2589 (−0.0024 ± 0.0013) | 0.2607 | **−0.0017 ± 0.0009** | 3/9 |
| Guiada por el barrido | 0.2591 (−0.0022 ± 0.0008) | 0.2602 | **−0.0021 ± 0.0006** (t = −3.59) | **0/9** |
| Guiada + 5.ª monótona | 0.2598 (−0.0015 ± 0.0010) | 0.2608 | −0.0016 ± 0.0008 | 4/9 |

**Empeora de forma significativa.** El GBM aprovecha interacciones que el barrido de pares no detecta
(órdenes superiores o entre variables que no pasaron el filtro individual). Mismo patrón que E16 y E21:
restringir lo que "debería ser ruido" daña.

#### E46 — CatBoost monótono como tercer modelo

CatBoost (árboles simétricos) con las mismas restricciones monótonas, 2 semillas, 9 folds, sin la variable
corrupta. Gini solo **0.2558**. **Correlación de rangos con LightGBM: 0.949** (la propuesta asumía
0.82–0.85, que es lo que justificaba que aportara diversidad). Dentro de la mezcla, frente a la
referencia sin la variable corrupta:

| Mezcla | Dif. | t | Gana |
|---|---|---|---|
| 0.55 LGB + 0.25 Cat + 0.20 logística (la propuesta) | −0.0006 ± 0.0003 | −1.77 | 2/9 |
| 0.60 + 0.20 + 0.20 | −0.0004 ± 0.0003 | −1.47 | 3/9 |
| 0.65 + 0.15 + 0.20 | −0.0003 ± 0.0003 | −1.06 | 3/9 |
| 0.70 + 0.10 + 0.20 | −0.0001 ± 0.0002 | −0.68 | 4/9 |

**Descartado:** es el mismo resultado de E27–E30; los árboles ordenan casi igual.

#### E47 — Ponderación temporal de las observaciones (`ideas2.py`, `ideas3.py`)

Peso por fila `w = (1 + λ)^(mes − mes_validado)` (los meses recientes pesan más), normalizado a media 1.
9 folds, mezcla, vs la referencia (0.2624):

| λ | Dif. | t | Gana |
|---|---|---|---|
| 0.02 | +0.0002 ± 0.0003 | 0.60 | 7/9 |
| 0.05 | +0.0006 ± 0.0003 | 1.84 | 7/9 |
| **0.10** | **+0.0010 ± 0.0003** | **3.10** | **8/9** |
| 0.15 | +0.0005 ± 0.0006 | 0.81 | 5/9 |
| 0.20 | −0.0004 ± 0.0005 | −0.78 | 4/9 |
| 0.30 | −0.0006 ± 0.0006 | −0.89 | 5/9 |
| 0.50 | −0.0003 ± 0.0007 | −0.36 | 4/9 |

**Error mío de interpretación en vivo:** tras ver 0.02, 0.05 y 0.10 creciendo, anticipé una dosis-respuesta
y extendí la grilla; **no se sostuvo**. Es un pico aislado en 0.10 entre vecinos nulos o negativos, el
patrón del azar tras probar varios valores. Más pruebas:
- Con 0.20: 50 rondas −0.0020 ± 0.0012; 80 −0.0004; 110 −0.0005. Combinada con la 5.ª monótona: +0.0000
  (0.20) y +0.0002 (0.30). Con la logística también ponderada: −0.0009 y −0.0012 (t = −1.72).
- Muestra dividida: eligiendo λ en 202603–07 sale 0.10 y rinde +0.0011 ± 0.0004 (4/4) en 202608–11;
  eligiendo en 202608–11 sale 0.10 y rinde +0.0009 ± 0.0005 (4/5) en 202603–07. (No es una prueba limpia:
  0.10 ya se había visto con todos los folds.)
- En condiciones de diciembre (E43): con la variable corrupta +0.0007 ± 0.0003 (t = 2.59, 7/9); **sin ella
  +0.0004 ± 0.0002 (t = 1.91, 8/9)**; con λ = 0.30 negativo en ambos (−0.0012 y −0.0010).

**Mecanismo probable:** ponderar los meses recientes le da más peso a las filas donde
`dias_ultima_interaccion` ya está corrupta, así que el modelo aprende a ignorarla. Al quitar esa variable
la ganancia se reduce a +0.0004 y deja de superar 2 EE. **No adoptada.**

---

### Parte H — Comparación final de 11 modelos

#### E48 — Tabla comparativa en el formato de la captura de Jhojan (`tabla_modelos.py`)

**Qué.** Reproducir la tabla de la captura de Jhojan (6 folds, columnas `fold_1`..`fold_6`, `media`,
`std`), pero con todos los modelos medidos con el **mismo protocolo** e incluyendo los nuestros.
**Cómo.** 6 folds walk-forward: `fold_1` = 202606 ... `fold_6` = 202611 (entrenar con los meses
anteriores, medir en ese mes). Rondas fijas, sin early stopping contra el fold. `std` con ddof = 0, como
en su tabla. Las filas lgbm, catboost, xgb y logreg se corrieron con parámetros propios (no son los
números de Jhojan); sus ensambles son promedios de rangos. El modelo de Antigravity es una réplica exacta
de `modelo_maestro_bcp.py`.

| modelo | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | fold_6 | media | std |
|---|---|---|---|---|---|---|---|---|
| lgbm | 0.2542 | 0.2658 | 0.2622 | 0.2455 | 0.2670 | 0.2396 | 0.2557 | 0.0103 |
| catboost | 0.2473 | 0.2620 | 0.2654 | 0.2423 | 0.2660 | 0.2379 | 0.2535 | 0.0114 |
| xgb | 0.2525 | 0.2677 | 0.2599 | 0.2483 | 0.2694 | 0.2378 | 0.2560 | 0.0110 |
| logreg | 0.2173 | 0.2422 | 0.2457 | 0.2297 | 0.2397 | 0.2065 | 0.2302 | 0.0142 |
| ensemble_LGBM+CB | 0.2522 | 0.2656 | 0.2653 | 0.2444 | 0.2676 | 0.2396 | 0.2558 | 0.0111 |
| ensemble_LGBM+CB+XGB | 0.2530 | 0.2669 | 0.2639 | 0.2461 | 0.2687 | 0.2394 | 0.2563 | 0.0110 |
| ensemble_LGBM+CB+LR | 0.2466 | 0.2649 | 0.2661 | 0.2447 | 0.2643 | 0.2348 | 0.2536 | 0.0121 |
| ensemble_LGBM+CB+XGB+LR | 0.2490 | 0.2667 | 0.2653 | 0.2465 | 0.2664 | 0.2364 | 0.2551 | 0.0117 |
| antigravity_maestro_bcp | 0.2534 | 0.2690 | 0.2632 | 0.2455 | 0.2659 | 0.2367 | 0.2556 | 0.0116 |
| nuestro_anterior (con la variable corrupta) | 0.2600 | 0.2710 | 0.2668 | 0.2473 | 0.2668 | 0.2421 | 0.2590 | 0.0107 |
| **NUESTRO_FINAL** | **0.2614** | 0.2680 | **0.2705** | 0.2472 | 0.2676 | **0.2425** | **0.2595** | 0.0108 |

(Seis decimales en `resultados_modelos.csv`.) Lecturas:
- **Nuestro modelo es el primero.** Las dos versiones nuestras (0.2595 y 0.2590) empatan entre sí y le
  sacan unos 0.003–0.004 al resto.
- **Igual que en la captura de Jhojan, ningún ensamble supera claramente a un solo boosting**
  (LGBM+CB+XGB 0.2563 vs XGB 0.2560), y meter la logística simple en el ensamble lo empeora.
- La media aquí (0.2595) es menor que la de 7 folds (0.2617) porque no incluye 202605, un mes alto.

**Diferencias pareadas de nuestro modelo (fold a fold):**
- vs lgbm estándar: +0.0072, +0.0022, +0.0082, +0.0017, +0.0006, +0.0028 → **+0.0038 ± 0.0013**
  (t ≈ 2.9), **gana 6/6**.
- vs el mejor ensamble (LGBM+CB+XGB): **+0.0032**, gana 5/6.

**Frente a los números de la captura de Jhojan** (suponiendo que sus 6 folds son los mismos meses):

| Fold | Jhojan (lgbm) | Nuestro | Diferencia |
|---|---|---|---|
| 1 | 0.2543 | 0.2614 | +0.0072 |
| 2 | 0.2730 | 0.2680 | −0.0051 |
| 3 | 0.2743 | 0.2705 | −0.0038 |
| 4 | 0.2506 | 0.2472 | −0.0035 |
| 5 | 0.2671 | 0.2676 | +0.0004 |
| 6 | 0.2335 | 0.2425 | +0.0089 |
| **Media** | **0.2588** | **0.2595** | **+0.0007** (gana 3/6) |

En Gini **más alto es mejor**, así que nuestro modelo queda por encima, pero frente a sus números es
casi un empate. Esa comparación no es limpia en ninguna dirección: si él usó early stopping contra el
fold reportado, sus números están inflados (en nuestro caso eso inflaba ≈ 0.005); también pudo usar
mejores parámetros que los de mi réplica. Para cerrarlo haría falta su script, corrido con este mismo
protocolo. Además, su modelo casi seguro usa `dias_ultima_interaccion`, que en diciembre es ruido.

**La imagen.** `tabla_comparacion_modelos.png` no es una captura de una ventana: se ejecutaron de verdad
en la carpeta dos comandos (imprimir `resultados_modelos.csv` en markdown y las 10 predicciones más
altas de `submission.csv`) y su salida real se dibujó con fuente Consolas sobre fondo de terminal,
imitando el formato de la captura de Jhojan.

---

## 7. El mejor modelo que tenemos

Archivo: `solucion.py` → genera `submission.csv`.

### 7.1 Arquitectura

```
                   ┌──────────────────────────────┐
 21 variables  ───▶│ LightGBM CON restricciones   │──┐
 + duración        │ monótonas (peso 0.75)        │  │   mezcla por RANGOS
 + banda ordinal   └──────────────────────────────┘  ├──▶ 0.75·rank(A) + 0.25·rank(B)
                   ┌──────────────────────────────┐  │         │
 mismas variables ▶│ Logística separada por banda │──┘         ▼
                   │ de riesgo (peso 0.25)        │     remapeo monótono a la
                   └──────────────────────────────┘     escala de probabilidad
                                                              │
                                                              ▼
                                                    submission.csv (9.900 filas)
```

### 7.2 Variables

- 21 originales (se **excluye `dias_ultima_interaccion`**, ver E41–E43) más `duracion` (meses previos
  observados del cliente; se calcula **solo hacia atrás**, nunca con filas futuras). Categóricas como
  `category` nativa de LightGBM, booleanas como 0/1.
- Para el LightGBM, `banda_riesgo` entra como **ordinal** `banda_ord` (high 0, medium 1, low 2).
- **No** se usan `mes` (tasa plana; diciembre no existe en train) ni `id_cliente` (fuga/colineal).

### 7.3 Componente 1 — LightGBM monótono

| Parámetro | Valor |
|---|---|
| tasa de aprendizaje | 0.05 |
| hojas | 16 |
| profundidad máxima | sin límite |
| `min_child_samples` | 400 |
| `reg_lambda` / `reg_alpha` | 20 / 0 |
| `feature_fraction` | 1.0 |
| `bagging_fraction` / frecuencia | 0.7 / 1 |
| semillas | 42, 777, 2024 (promedio) |
| rondas | 87 (máximo de la curva promediada de 7 folds; meseta 58–139) |
| restricciones monótonas | `dias_ultima_transaccion` −1, `numero_productos` +1, `banda_ord` +1 (método `advanced`) |

### 7.4 Componente 2 — Logística por banda

Un modelo por cada banda (low/medium/high) con splines de 6 nudos (grado 3) en `dias_ultima_transaccion` y
`ratio_deuda_ingresos`; dummies de `numero_productos`, `canal_adquisicion` y `duracion` (recortada a 8);
`activo_movil`, `tiene_tarjeta_credito` y su producto; interacciones splines × `activo_movil` y productos ×
flags; estandarización y `LogisticRegression(C=0.05)`.

### 7.5 Mezcla y formato

`0.75·rank(LGB) + 0.25·rank(logística)`; luego se remapea al reparto de probabilidades del LGB (monótono,
no cambia el Gini) para que la entrega quede en escala de probabilidad (0.026 a 0.435, media 0.142).

### 7.6 Resultados medidos (7 folds, 202605–202611)

| | Gini |
|---|---|
| LightGBM monótono solo | 0.2605 (óptimo 87 rondas; meseta 58–139) |
| Logística por banda sola | 0.2526 |
| **Mezcla (modelo final)** | **0.2617** |

Por mes: 202605 0.2747 · 202606 0.2614 · 202607 0.2680 · 202608 0.2705 · 202609 0.2472 · 202610 0.2676 ·
202611 0.2425 (desvío entre meses 0.012).

La versión anterior (con `dias_ultima_interaccion` y 4 restricciones monótonas) daba 0.2618 en el mismo
protocolo; la nueva ordena a los clientes en un 99.0 % igual (correlación de rangos 0.9902). Se prefirió la
nueva porque, en diciembre, esa variable es ruido.

### 7.7 Qué usa el modelo (importancia por *gain*, LightGBM)

`banda_ord` 33.2 % · `numero_productos` 21.8 % · `dias_ultima_transaccion` 16.0 % · `activo_movil` 7.0 % ·
`tiene_tarjeta_credito` 5.9 % · `ratio_deuda_ingresos` 2.7 % · `antiguedad_cuenta_meses` 2.0 % ·
`ingresos` 1.8 % · `canal_adquisicion` 1.6 % · `edad` 1.3 % · `saldo_promedio` 1.2 % ·
`distancia_sucursal_km` 1.1 %. Cinco variables concentran ≈ 84 % de la señal.

### 7.8 Evolución del Gini (mismo protocolo salvo indicado)

| Etapa | Gini | Observación |
|---|---|---|
| Regresión logística simple | 0.2310–0.2323 | referencia |
| LightGBM, 23 variables (early stopping) | 0.2599 | optimista |
| LightGBM, estimador honesto (curva promediada) | 0.2551 | 5 folds |
| Referencia a 7 folds (comparable) | 0.2587–0.2593 | |
| + logística por banda (0.75/0.25) | 0.2603–0.2605 | E22–E24 |
| + restricciones monótonas (4 variables) | 0.2618 | E37–E39 |
| **Sin la variable corrupta (3 restricciones)** | **0.2617** | E41–E43: empate, más robusto |

### 7.9 Historial de versiones de `solucion.py`

| Versión | Qué cambió | Por qué | Gini |
|---|---|---|---|
| v1 | LightGBM con las 23 variables, early stopping por fold, 5 semillas | primer modelo correcto con validación walk-forward | 0.2599 (optimista, 5 folds) |
| v2 | Rondas por el máximo de la curva de AUC promediada (59 → 80 en el final) | el early stopping contra el fold reportado inflaba ≈ 0.005 | 0.2551 (honesto, 5 folds) |
| v3 | Parámetros de la región buena del tuning; mezcla 0.75 LightGBM + 0.25 logística por banda; remapeo a escala de probabilidad; 7 folds | única mejora de ~12 ideas que replicó fuera de muestra (E22–E24) | 0.2603 (7 folds) |
| v4 | Restricciones monótonas en 4 variables con `banda_riesgo` ordinal; rondas sin escalar (71) | +0.0026 ± 0.0008 en 9 folds, dosis-respuesta, estable entre rondas (E37–E39) | 0.2618 (7 folds) |
| **v5 (actual)** | **Sin `dias_ultima_interaccion`** (3 restricciones monótonas), 87 rondas | esa variable es una permutación al azar en diciembre (E41–E43) | **0.2617** (7 folds) · **0.2595** (6 folds) |

Copias de respaldo de v4 (`solucion_antes_sin_inter.py`, `submission_antes_sin_inter_0.2618.csv`) y de la
entrega de v3 (`submission_anterior_0.2603.csv`) quedaron en la carpeta temporal del sistema (`%TEMP%`).

---

## 8. Qué esperar del resultado real

- **Techo teórico estimado: ≈ 0.269 ± 0.007** (E19; estimación circular, ver limitaciones).
  Estamos en ≈ 97 %. **0.27 como resultado esperado no es alcanzable**: es el techo de un modelo que
  conociera la probabilidad exacta de cada cliente.
- **Ruido de muestreo del test: ±0.0143** (E20). El Gini de diciembre de nuestro mismo modelo puede
  caer en 0.24–0.28 según qué clientes convirtieron. La probabilidad de que un modelo con esperanza
  0.26 saque ≥ 0.28 es ≈ 8 %, y le pasa igual a cualquier modelo bueno.
- Por mes, el mismo modelo dio entre 0.242 y 0.278 en validación (ver §7.6).
- **Para subir de verdad hace falta más información, no más modelos** (por ejemplo historial real de
  transacciones, campañas o fechas de contratación), si la competencia lo permite.
- Los modelos de Jhojan, Antigravity y el nuestro ordenan a los clientes casi igual (ρ ≈ 0.94–0.97);
  quién quede arriba lo decide el azar de diciembre.

---

## 9. Errores míos, correcciones y limitaciones honestas

### 9.1 Errores cometidos y corregidos

1. **Plan inicial equivocado.** Propuse features de historial por cliente (deltas, tendencias). Los
   datos mostraron que 21 de 22 variables nunca cambian (E04). Descartado antes de implementarlo.
2. **Gini inicial inflado.** El primer script daba 0.2599 usando early stopping contra el fold
   evaluado; el estimador honesto era 0.2551 (E17).
3. **Hipótesis falsas que propuse y se refutaron con medición:** que los ratios ayudarían (E14), que
   el ensamble de familias ayudaría (E15), que el modelo paramétrico superaría al GBM (E21).
4. **Interpretación equivocada de la captura de Jhojan.** Leí las predicciones de 1.000 como
   sobreajuste; eran rangos (E26). Retirado.
5. **Error de código en el barrido de interacciones.** El primer diseño de matriz era colineal
   (matriz singular); se corrigió usando el factor cruzado saturado (E10).
6. **Afirmación imprecisa sobre "folds frescos".** Dije que los 3 meses de E23 no se habían usado
   para elegir el blend; **202605 sí estaba entre los 7 folds de selección**. Solo 202603 y 202604
   eran nuevos.
7. **Simulación del ruido del test.** Esa simulación devolvió una media (0.2988) que no es comparable
   con el techo de 0.269; solo se usó su desvío (0.0143).
8. **Una pista que no interpreté durante mucho tiempo.** En la primera tabla de deriva train/test, las
   medias de `dias_ultima_transaccion` y `dias_ultima_interaccion` en el test eran idénticas hasta el sexto
   decimal (194.307273). Era la huella de una permutación y apuntaba a que la segunda variable era ruido en
   diciembre. Lo descubrí recién con la validación adversarial (E41–E42). Mi descripción en E05 ("copia
   ruidosa") era incompleta, y mi estimación de +0.0026 para las 4 restricciones monótonas incluía una
   variable que luego resultó ser ruido en el test.
9. **Extrapolación falsa en E47.** Vi tres valores de λ con ganancia creciente, anuncié una dosis-respuesta
   y extendí la grilla; no se sostuvo (el pico era aislado).
10. **Archivos ajenos.** Al limpiar la carpeta borré `catboost_info/` (registros automáticos de
   CatBoost de los scripts de Antigravity); es regenerable, pero no era mía.

### 9.2 Limitaciones que siguen vigentes

- **Sesgo de selección.** Las restricciones monótonas se eligieron entre ~6 variantes mirando los
  mismos 9 meses; la mejora esperada fuera de muestra es algo menor que +0.0026 (esperar ≈ +0.0015 a
  +0.002). La estabilidad entre rondas y la dosis-respuesta sugieren que es real, pero no hay un mes
  completamente virgen para confirmarlo.
- **Las direcciones monótonas se leyeron de los deciles de todo train** (incluye los meses de
  validación). Es una filtración leve y de bajo impacto (las direcciones son obvias y estables).
- **El techo de 0.269 es una estimación circular** (depende de nuestro propio modelo).
- **Maldición del ganador** en el tuning de hiperparámetros (E13): el +0.0033 no se confirmó
  (+0.0006 ± 0.0020).
- **No está listo para producción:** no hay monitoreo de deriva, reentrenamiento programado,
  versionado de modelo ni pruebas unitarias reales (solo aserciones de formato y de rango).
- **La discusión de negocio no se hizo:** cuánto valen 0.002 de Gini en soles, o si conviene
  optimizar Gini frente a otra métrica de negocio.

---

## 10. Lo que hicieron otros

### 10.1 Jhojan (captura del 2026-10-02)
Validación de 6 folds: LightGBM 0.2588, CatBoost 0.2552, XGB 0.2523, logística 0.2323; ningún
ensamble supera a LightGBM solo (0.2587 el mejor). Entrega en forma de rangos. Coincide con nuestros
hallazgos de que ensamblar no ayuda.

**Comparación (E48).** En Gini, el nuestro queda por encima: 0.2595 vs 0.2588 de su captura (+0.0007,
gana 3 de 6 meses, prácticamente empate; protocolos posiblemente distintos). Frente a un LightGBM como
el suyo corrido con el mismo protocolo: +0.0038 ± 0.0013, ganando los 6 meses. No se tuvo acceso a su
código.

### 10.2 Antigravity
- **Modelo maestro BCP:** sin validación interna (importa `roc_auc_score` y no lo usa). Medido bajo
  nuestro protocolo: **0.2593**, empate estadístico con nuestra versión de entonces (E25).
- **`modelo_avanzado.py`:** valida un modelo distinto del que entrega y usa `eval_set` en CatBoost.
- **Tabla comparativa** "Claude 0.2551 / Jhojan 0.2586 / Maestro 0.2615" mezclaba protocolos
  distintos; no es comparable.
- **Su investigación (Porto Seguro):** RankGauss (−0.0001 ± 0.0007 en la logística; para árboles no
  hace nada porque son invariantes a transformaciones monótonas), monótonas con solo 2 restricciones
  (+0.0005 ± 0.0008; menos potencia que E37), DART (descartado, coincide con E34). Su veredicto de
  que 0.2603 era "la mejor versión posible" no se sostiene: las monótonas de 4 variables mejoraron
  (0.2618).
- **Archivos nuevos que no se auditaron:** `solucion_auditada.py` (describe la versión anterior),
  `super_modelo_bcp_grandmaster.py`, `submission_super_poderoso.csv`.

### 10.3 Informe de investigación de Antigravity (papers y soluciones ganadoras)

Cinco propuestas, todas probadas en E41–E47:

| Propuesta | Resultado | Veredicto |
|---|---|---|
| 1. 5.ª monótona + indicador `sin_deuda` | el indicador es constante (no hay ratio = 0); la monótona da t = 2.27 con la variable corrupta y +0.0003 ± 0.0004 sin ella | no adoptada |
| 2. `interaction_constraints` | −0.0017 a −0.0021 (t hasta −3.6) | empeora |
| 3. CatBoost monótono, 3.er modelo (suponía ρ = 0.82–0.85) | ρ real = 0.949; −0.0001 a −0.0006 | empeora |
| 4. Validación adversarial | AUC 0.715 → 0.527 al quitar una variable | **hallazgo clave** (E41–E43) |
| 5. Ponderación temporal | pico aislado en λ = 0.10; negativo desde 0.20 | no adoptada |

Observaciones sobre el informe:
- La advertencia de que "diciembre tiene estacionalidad de aguinaldos" no aplica: las variables están
  congeladas por cliente, así que la deriva solo puede venir de composición por supervivencia o de la
  variable corrupta, que es justo lo que se encontró.
- Las referencias bibliográficas del informe no se verificaron una por una. Una imprecisión conocida: RealMLP
  (2024) es de Holzmüller, Grinsztajn y Steinwart, no de Gorishniy et al. (que firman TabR).
- La validación adversarial del informe, tal como se describe, habría dado un AUC engañoso por la
  duplicación de clientes entre meses; hubo que agruparla por cliente.

---

## 11. Cómo reproducir todo

### 11.1 Entrega final

```bash
cd "C:\Users\Leonardo\Downloads\drive-download-20261001T084508Z-1-001"
python solucion.py
```

Imprime la CV walk-forward (7 folds), el Gini esperado (≈ 0.2618), la importancia de variables y
genera `submission.csv`. Requisitos: Python 3.11, `pandas`, `numpy`, `scikit-learn`, `scipy`,
`lightgbm`. Incluye comprobaciones de formato (9.900 filas, mismo orden que `test.csv`, valores en
[0,1], sin nulos) y una aserción de que el Gini cae en un rango plausible.

### 11.2 Tabla comparativa e imagen

```bash
python tabla_modelos.py
```

Recalcula los 11 modelos en 6 folds (unos minutos, CatBoost es lo más lento) y escribe
`resultados_modelos.csv`. La imagen `tabla_comparacion_modelos.png` se generó con
`render_terminal.py` (carpeta temporal), que ejecuta los comandos reales y dibuja su salida.

### 11.3 Scripts de experimentos

Están en el directorio temporal de la sesión de trabajo:

`C:\Users\Leonardo\AppData\Local\Temp\claude\C--Users-Leonardo\be4cc4e0-daf3-4dd3-8447-a27d88ae69fb\scratchpad\`

| Script | Pruebas |
|---|---|
| `ablacion.py` | E12 |
| `tuning.py` | E13 |
| `diag_duracion.py` | E07 |
| `test_features.py` | E14, E15 |
| `test_poda.py` | E16 |
| `curva_rondas.py` | E17 |
| `test_sobreajuste.py` | E18 |
| `techo_bayes.py` | E19, E20 |
| `forma_dgp.py` | E09 |
| `scan_interacciones.py` | E10, E11 |
| `modelo_estructurado.py` | E21 |
| `confirmacion.py` | E22 |
| `validacion_limpia.py` | E23 |
| `peso_blend.py` | E24 |
| `duelo.py` | E25 |
| `oof_gen.py`, `ens.py` | E27–E33 |
| `ideas.py` | E36 |
| `robust_mono.py` | E37 |
| `mono_ext.py` | E38 |
| `retune_mono.py` | E39 |
| `ideas2.py`, `ideas3.py` | E44, E45, E47 |
| `adv.py` | E41 |
| `dec_like.py` | E43 |
| `final_check.py` | E44, E46 |
| `render_terminal.py` | imagen de E48 |

> Ese directorio es temporal y puede borrarse. Si se quieren conservar, hay que copiarlos a esta
> carpeta.

---

## 12. Ideas no probadas

Ninguna con expectativa de más de +0.001 a +0.002 (por debajo del ruido del test):

- **Pseudo-etiquetado** con los 9.900 clientes de diciembre.
- **Imputar `dias_ultima_interaccion` como "faltante" en los meses donde está corrupta**, en vez de excluirla.
  Probable ganancia nula: en las filas limpias es idéntica a `dias_ultima_transaccion`.
- **Un LightGBM por banda con restricciones monótonas** (probado por banda sin monótonas: 0.2510).
- **Más datos externos** (si la competencia lo permite): es la única vía con potencial real.

---

## 13. Glosario

| Término | Significado |
|---|---|
| **Conversión** | Que el cliente contrate un producto por primera vez en ese mes |
| **Propensión** | Probabilidad estimada de conversión |
| **AUC** | Fracción de pares (convertido, no convertido) bien ordenados por el modelo |
| **Gini** | 2·AUC − 1; 0 = azar, 1 = perfecto; más alto es mejor |
| **Train / test** | Datos con respuesta (para entrenar) / sin respuesta (para predecir) |
| **Walk-forward** | Entrenar con meses anteriores y validar en el mes siguiente, repetido |
| **Fold** | Una repetición de la validación (aquí, un mes) |
| **LOFO** | *Leave-one-fold-out*: ajustar con k−1 folds y medir en el restante, rotando |
| **GBM / boosting** | Muchos árboles pequeños en secuencia, cada uno corrige errores del anterior |
| **LightGBM / XGBoost / CatBoost** | Tres implementaciones populares de GBM |
| **Ronda** | Un árbol añadido al modelo de boosting |
| **Early stopping** | Parar el entrenamiento cuando la validación deja de mejorar |
| **Regresión logística** | Modelo lineal para probabilidades; aquí con splines (curvas suaves) |
| **Spline** | Función suave por tramos para modelar relaciones no lineales |
| **GAM / EBM** | Modelos aditivos con curvas suaves por variable (EBM añade pares) |
| **Rank blending** | Mezclar modelos promediando sus *rangos* en lugar de sus probabilidades |
| **Restricción monótona** | Obligar al modelo a que una variable solo pueda subir (o bajar) la predicción |
| **Diferencia pareada** | Comparar dos modelos fold por fold y promediar las diferencias |
| **Error estándar (EE)** | Incertidumbre del promedio; diferencia > 2 EE ≈ real |
| **Bonferroni** | Corrección al probar muchas hipótesis a la vez |
| **χ² (chi-cuadrado)** | Estadístico de los tests de razón de verosimilitud |
| **Hazard** | Probabilidad de convertir en un mes dado que aún no ha convertido |
| **Fuga (leakage)** | Que el modelo vea, sin querer, información que no tendría en la realidad |
| **Hill climbing** | Ensamble que añade, paso a paso, el modelo que más mejora la validación |
| **Stacking** | Un segundo modelo que aprende a combinar las predicciones del primero |
| **Oráculo / techo** | Modelo imaginario que conoce la probabilidad verdadera; da el máximo alcanzable |
| **Validación adversarial** | Entrenar un clasificador para distinguir train de test; si lo logra, hay deriva de covariables |
| **Permutación** | Reordenar al azar los valores de una columna entre filas: conserva los valores pero rompe su relación con cada cliente |
| **Maldición del ganador** | Al elegir lo mejor de muchas pruebas, el ganador sale favorecido por azar |

---

## 14. Fuentes consultadas

- [The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data (NVIDIA)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Grandmaster Pro Tip: Winning First Place with Stacking Using cuML (NVIDIA)](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
- [Grandmaster Pro Tip: Winning First Place with Feature Engineering Using cuDF pandas (NVIDIA)](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)
- [1st Place – Fast GPU Experimentation with RAPIDS (Playground S5E6)](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi)
- [1st Place Solution – Hill Climbing + Ridge Ensemble (Playground S5E12)](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)
- [kaggle-s6e9-ev-purchase (numéricas como categóricas, target encoding)](https://github.com/CATastr0phe/kaggle-s6e9-ev-purchase)
- [3rd Place – Target Encoding and 3 Levels (Playground S5E4)](https://www.kaggle.com/c/playground-series-s5e4/discussion/575862)

> Nota: algunas páginas de Kaggle no devolvieron su contenido al consultarlas; de ellas solo se
> usaron los títulos y los resúmenes que sí estuvieron disponibles.
