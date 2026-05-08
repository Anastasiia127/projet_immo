# Carpeta de notebooks

Aquí cada miembro del equipo coloca su script o notebook de entrenamiento.

## Estructura recomendada

```
notebooks/
├── 01_EDA.py                     # Análisis exploratorio (Spyder)
├── 02_linear_regression.py       # Regresión Lineal / Ridge
├── 03_random_forest.py           # Random Forest
└── 04_mlp.py                     # Red Neuronal (MLP)
```
> Si usáis Jupyter, los archivos serán `.ipynb` en lugar de `.py` — funciona igual.

---

## Plantilla completa — copiar y pegar en vuestro script

```python
import sys
sys.path.insert(0, '..')  # Para importar desde src/

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import load_and_prepare, get_features_and_target
from src.model_loader import get_metrics
from src.anomaly_detection import clasificar_zonas, resumen_anomalias, exportar_resultados

# ── 1. Cargar y preparar datos ─────────────────────────────────────────────
df = load_and_prepare('../data/dataset_corregido.xlsx')
X, y = get_features_and_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Entrenar modelo ─────────────────────────────────────────────────────
# (sustituir por vuestro modelo: LinearRegression, RandomForest, MLP...)
from sklearn.ensemble import RandomForestRegressor

modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)

# ── 3. Predecir y calcular métricas ───────────────────────────────────────
predictions = modelo.predict(X_test)

metrics = get_metrics("Random Forest", y_test, predictions)
print("RMSE:", metrics["RMSE"])
print("MAE: ", metrics["MAE"])
print("R²:  ", metrics["R2"])

# ── 4. Guardar métricas en CSV para el dashboard ───────────────────────────
# ⚠️ IMPORTANTE: esto permite que el dashboard muestre vuestros resultados
# reales sin necesidad de subir el modelo .pkl a GitHub
pd.DataFrame([{
    "RMSE": metrics["RMSE"],
    "MAE":  metrics["MAE"],
    "R2":   metrics["R2"],
}]).to_csv("../outputs/metrics_rf.csv", index=False)
print("Métricas guardadas en outputs/metrics_rf.csv")

# Nombres de archivo por modelo:
# Regresión Lineal → metrics_linear.csv
# Random Forest    → metrics_rf.csv
# MLP              → metrics_mlp.csv

# ── 5. Guardar predicciones en CSV para el dashboard ──────────────────────
# ⚠️ IMPORTANTE: esto permite ver el gráfico Predicción vs Realidad real
pd.DataFrame({
    "y_real": y_test.values,
    "y_pred": predictions,
}).to_csv("../outputs/predictions_rf.csv", index=False)
print("Predicciones guardadas en outputs/predictions_rf.csv")

# Nombres de archivo por modelo:
# Regresión Lineal → predictions_linear.csv
# Random Forest    → predictions_rf.csv
# MLP              → predictions_mlp.csv

# ── 6. Detectar anomalías por zona ─────────────────────────────────────────
df_test = df.iloc[X_test.index].copy()
df_test["price_pred"] = predictions

resultado = clasificar_zonas(df_test, col_zona="city", col_real="price", col_pred="price_pred")
resumen_anomalias(resultado)

# ── 7. Exportar resultados de anomalías ────────────────────────────────────
exportar_resultados(resultado, path="../outputs/predicciones.csv")

# ── 8. Guardar modelo (opcional — solo local, no va a GitHub) ──────────────
joblib.dump(modelo, '../models/random_forest.pkl')
print("Modelo guardado en models/random_forest.pkl")
```

---

## Cómo ver tus resultados en el dashboard (flujo de trabajo)

1. Entrena tu modelo y ejecuta el script completo
2. Se generan automáticamente los CSV en `outputs/`
3. Haz `git add outputs/ && git commit -m "Add RF results" && git push`
4. El resto del equipo hace `git pull` y ve tus resultados reales en el dashboard

> Los archivos `.pkl` son pesados y **no van a GitHub** (están en `.gitignore`).
> Los CSV de métricas y predicciones **sí van a GitHub** — son ligeros y suficientes para el dashboard.

---

## Nombres de archivo esperados por el dashboard

| Modelo | Métricas | Predicciones |
|---|---|---|
| Regresión Lineal / Ridge | `outputs/metrics_linear.csv` | `outputs/predictions_linear.csv` |
| Random Forest | `outputs/metrics_rf.csv` | `outputs/predictions_rf.csv` |
| MLP | `outputs/metrics_mlp.csv` | `outputs/predictions_mlp.csv` |

---

## Nota para usuarios de Spyder

En Spyder, asegúrate de que el **directorio de trabajo** está en la carpeta `notebooks/`.
Se puede cambiar arriba a la derecha donde pone la ruta de la carpeta actual.
