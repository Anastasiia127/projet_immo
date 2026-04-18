# Carpeta de outputs

Aquí se guardan los resultados generados por los modelos.

## Archivo: `predicciones_template.csv`

Plantilla del formato esperado para las predicciones. 
Cada modelo debe rellenar sus columnas correspondientes.

### Columnas

| Columna | Descripción |
|---|---|
| `id_annonce` | ID único de la vivienda |
| `city` | Ciudad |
| `size_m2` | Superficie en m² |
| `nb_rooms` | Número de habitaciones |
| `property_type` | Tipo de propiedad |
| `price_real` | Precio real del dataset (€) |
| `price_pred_linear` | Predicción Regresión Lineal (€) |
| `price_pred_random_forest` | Predicción Random Forest (€) |
| `price_pred_mlp` | Predicción MLP (€) |
| `diferencial_linear` | price_real - price_pred_linear |
| `diferencial_rf` | price_real - price_pred_random_forest |
| `diferencial_mlp` | price_real - price_pred_mlp |
| `clasificacion` | Tendencial / Atípica (IC 95%) |

## Cómo generar el CSV desde el notebook

```python
import pandas as pd

resultados = pd.DataFrame({
    'id_annonce': df_test['id_annonce'],
    'city': df_test['city'],
    'size_m2': df_test['size'],
    'nb_rooms': df_test['nb_rooms'],
    'property_type': df_test['property_type'],
    'price_real': y_test,
    'price_pred_random_forest': predictions,
    'diferencial_rf': y_test.values - predictions,
})

resultados.to_csv('../outputs/predicciones.csv', index=False)
```
