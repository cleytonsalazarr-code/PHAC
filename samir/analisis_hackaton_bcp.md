# 🏦 Análisis de Problemática — Hackathon BCP

## 🎯 ¿De qué trata la problemática?

El desafío es un problema de **clasificación binaria** en el ámbito **bancario/financiero**. Específicamente:

> **Predecir la probabilidad de que un cliente del banco realice su primera conversión en un mes determinado.**

Una "conversión" significa que el cliente toma una acción de valor para el banco (como adquirir un nuevo producto, activarse, o realizar una transacción significativa). Una vez que un cliente convierte (`objetivo = 1`), **ya no vuelve a aparecer en meses posteriores**, lo que lo hace un evento de "primer uso" o "primera activación".

---

## 📦 Estructura del problema

| Elemento | Detalle |
|---|---|
| **Tipo** | Clasificación binaria con salida probabilística |
| **Target** | `objetivo` — 1 si convirtió ese mes, 0 si no |
| **Métrica** | **Coeficiente de Gini** = `2 × AUC - 1` |
| **Train** | 110,100 observaciones (Enero–Noviembre 2026) |
| **Test** | 9,900 observaciones (Diciembre 2026) |
| **Features** | 24 columnas: numéricas, categóricas y booleanas |

---

## 🔍 Variables disponibles (25 columnas)

### 🔢 Numéricas (12)
| Variable | Descripción |
|---|---|
| `edad` | Edad del cliente |
| `ingresos` | Ingresos anuales |
| `ratio_deuda_ingresos` | Deuda / Ingresos |
| `antiguedad_cuenta_meses` | Antigüedad de cuenta |
| `numero_productos` | Productos bancarios activos |
| `saldo_promedio` | Saldo promedio |
| `dias_ultima_transaccion` | Recencia transaccional |
| `antiguedad_direccion_meses` | Tiempo en la dirección actual |
| `visitas_web_ultimos_90_dias` | Interés digital |
| `distancia_sucursal_km` | Distancia a sucursal |
| `dia_preferido_pago` | Día preferido de pago |
| `dias_ultima_interaccion` | Recencia de interacción con el banco |

### 🏷️ Categóricas (5)
| Variable | Descripción |
|---|---|
| `ocupacion` | Categoría laboral |
| `region` | Región geográfica |
| `canal_adquisicion` | Canal de captación |
| `banda_riesgo` | Segmento de riesgo |
| `dispositivo_principal` | Dispositivo digital principal |

### ✅ Booleanas (5)
| Variable | Descripción |
|---|---|
| `tiene_tarjeta_credito` | ¿Tiene tarjeta? |
| `activo_movil` | ¿Activo en app móvil? |
| `es_nuevo_cliente` | ¿Es cliente nuevo? |
| `tiene_prestamo` | ¿Tiene préstamo activo? |
| `tiene_seguro` | ¿Tiene seguro? |

---

## ⚠️ Retos y consideraciones clave

1. **Desbalance de clases**: En problemas de conversión, el porcentaje de conversión suele ser muy bajo (5-15%). Necesitamos manejar el desbalance.
2. **Temporal split**: El train va de enero a noviembre, el test es solo diciembre. Hay que respetar el orden temporal para evitar data leakage.
3. **Cliente irrepetible tras conversión**: Si `objetivo = 1`, el cliente no aparece más — esto implica que el dataset tiene sesgo de supervivencia.
4. **Métrica Gini**: No optimizamos accuracy, sino la capacidad de **ordenar/rankear** clientes. Necesitamos probabilidades bien calibradas.
5. **Sin valores faltantes**: El dataset está limpo, pero igual se debe hacer feature engineering.

---

## 🛠️ ¿Cómo abordarlo?

### Paso 1 — Análisis Exploratorio (EDA)
- Distribución de la variable `objetivo` (check desbalance)
- Correlaciones entre variables numéricas
- Distribución por mes (¿hay drift temporal?)
- Análisis de las categóricas (`banda_riesgo`, `canal_adquisicion`)

### Paso 2 — Feature Engineering
- **Recencia / Frecuencia / Monetización (RFM)**: Usar `dias_ultima_transaccion`, `dias_ultima_interaccion`, `saldo_promedio`
- **Interacción digital**: Combinar `activo_movil` + `visitas_web_ultimos_90_dias`
- **Ratio ingreso/deuda** ya está dado, pero se puede transformar (log, binning)
- **Variables de riesgo ponderadas**: `banda_riesgo` + `ratio_deuda_ingresos`
- **Encoding de categóricas**: Target encoding o WOE (Weight of Evidence) para variables como `banda_riesgo`

### Paso 3 — Manejo del desbalance
- `scale_pos_weight` en XGBoost/LightGBM
- `class_weight='balanced'` en sklearn
- SMOTE (con cuidado — no en series temporales)
- Submuestreo aleatorio de la clase mayoritaria

### Paso 4 — Modelado
Ver sección de modelos recomendados abajo ⬇️

### Paso 5 — Validación temporal
- Usar **TimeSeriesSplit** respetando el orden mensual
- Nunca mezclar meses futuros en el train de validación

### Paso 6 — Calibración de probabilidades
- `CalibratedClassifierCV` (Platt scaling o isotonic)
- Esencial para que las predicciones entre 0 y 1 sean probabilidades reales

---

## 🤖 Mejores modelos recomendados

### 🥇 Opción 1 — LightGBM (LGBM)
**El más recomendado para esta competencia.**

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    scale_pos_weight=10,  # ajustar según desbalance
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)
```

✅ **Por qué**: Muy rápido, maneja bien variables categóricas nativamente, excelente en rankings/AUC, robusto con desbalance.

---

### 🥈 Opción 2 — XGBoost
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)
```

✅ **Por qué**: Muy estable, ampliamente probado en competencias bancarias. Gran capacidad de tuning.

---

### 🥉 Opción 3 — CatBoost
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    cat_features=['ocupacion','region','canal_adquisicion','banda_riesgo','dispositivo_principal'],
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=100
)
```

✅ **Por qué**: Manejo **nativo de variables categóricas** sin encoding manual. Ideal dado que el dataset tiene 5 variables categóricas.

---

### 🔄 Opción 4 — Ensemble / Stacking (para maximizar Gini)
Combinar los 3 modelos anteriores:
```python
from sklearn.ensemble import StackingClassifier, VotingClassifier

# Promedio simple de probabilidades (blending)
pred_final = (pred_lgbm * 0.4 + pred_xgb * 0.35 + pred_catboost * 0.25)
```

✅ **Por qué**: En competencias, el ensemble suele ganar 1-3 puntos de Gini adicionales.

---

### 📊 Tabla comparativa de modelos

| Modelo | Velocidad | Categóricas | AUC típico | Complejidad |
|---|---|---|---|---|
| **LightGBM** | ⚡⚡⚡ | Buena | 0.80–0.87 | Baja |
| **XGBoost** | ⚡⚡ | Regular | 0.79–0.86 | Media |
| **CatBoost** | ⚡⚡ | Excelente | 0.81–0.87 | Baja |
| **Random Forest** | ⚡ | Regular | 0.75–0.82 | Baja |
| **Logistic Reg.** | ⚡⚡⚡ | Requiere enc. | 0.70–0.78 | Muy baja |
| **Ensemble** | ⚡ | — | 0.83–0.89 | Alta |

---

## 📋 Pipeline recomendado

```
1. EDA → 2. Limpieza/FE → 3. Encoding → 4. Split Temporal
         ↓
5. Train LGBM/XGBoost/CatBoost con CV temporal
         ↓
6. Calibración de probabilidades
         ↓
7. Ensemble de modelos
         ↓
8. Generar submission: id_cliente, prediccion (entre 0 y 1)
```

---

## 🏆 Estrategia para maximizar el Gini

1. **Usa LightGBM como modelo base** — es el más rápido para iterar
2. **Feature Engineering agresivo** — las variables de comportamiento (recencia, visitas web) suelen ser las más poderosas
3. **Valida con split temporal** — nunca mezcles meses
4. **Ensemble al final** — combina 2-3 modelos con distintas semillas
5. **No optimices accuracy** — optimiza directamente AUC/Gini
