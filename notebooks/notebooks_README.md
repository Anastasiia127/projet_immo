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

## Cómo usar preprocessing desde el script

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

# Al final del script, guardar el modelo entrenado
joblib.dump(modelo, '../models/random_forest.pkl')

# El dashboard lo detecta automáticamente
```

## Nombres de archivo esperados

| Modelo | Archivo |
|---|---|
| Regresión Lineal / Ridge | `../models/linear_regression.pkl` |
| Random Forest | `../models/random_forest.pkl` |
| MLP | `../models/mlp.pkl` |

## Nota para usuarios de Spyder

En Spyder, asegúrate de que el **directorio de trabajo** está en la carpeta `notebooks/`.
Se puede cambiar arriba a la derecha donde pone la ruta de la carpeta actual.
