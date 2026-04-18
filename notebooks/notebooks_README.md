# Carpeta de notebooks

Aquí cada miembro del equipo coloca su notebook de entrenamiento.

## Estructura recomendada

```
notebooks/
├── 01_EDA.ipynb                  # Análisis exploratorio
├── 02_linear_regression.ipynb    # Regresión Lineal / Ridge
├── 03_random_forest.ipynb        # Random Forest
└── 04_mlp.ipynb                  # Red Neuronal (MLP)
```

## Cómo usar preprocessing desde el notebook

```python
import sys
sys.path.insert(0, '..')  # Para importar desde src/

from src.preprocessing import load_and_prepare, get_features_and_target
from sklearn.model_selection import train_test_split

# Cargar y preparar datos
df = load_and_prepare('../data/dataset_corregido.xlsx')

# Obtener X e y
X, y = get_features_and_target(df)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Cómo calcular métricas

```python
from src.model_loader import get_metrics

metrics = get_metrics("Random Forest", y_test, predictions)
print(metrics)
# {'RMSE': 72000, 'MAE': 48000, 'R2': 0.78}
```

## Cómo guardar el modelo para el dashboard

```python
import joblib

# Al final del notebook, guardar el modelo entrenado
joblib.dump(modelo, '../models/random_forest.pkl')

# El dashboard lo detecta automáticamente
```

## Nombres de archivo esperados

| Modelo | Archivo |
|---|---|
| Regresión Lineal / Ridge | `../models/linear_regression.pkl` |
| Random Forest | `../models/random_forest.pkl` |
| MLP | `../models/mlp.pkl` |
