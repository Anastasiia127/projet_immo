# OuiPredict — Predicción de precios de vivienda en Francia

**Proyecto I · IA · UPV 2026**  
David Esteban Platero · Vicente Tralci Sindoni · Laura Muñoz · Lidia Martínez · Anastasiia Nogina

---

## Demo

[ouipredict.streamlit.app](https://ouipredict.streamlit.app/)

---

## Qué es

App interactiva para predecir el precio de viviendas en Francia a partir de sus características, con detección de zonas con precios anómalos. Dataset de 37.368 propiedades del Institut Louis Bachelier.

| Modelo | R² | MAE |
|--------|-----|-----|
| XGBoost | 0.8108 | 71.080 € |
| Random Forest | 0.7540 | 79.268 € |
| MLP | 0.6898 | 92.775 € |
| Ridge | 0.4893 | 118.762 € |

---

## Estructura del repositorio

```
projet_immo/
│
├── app/
│   └── dashboard.py              # Aplicación Streamlit principal
│
├── src/
│   ├── preprocessing.py          # Pipeline de carga y limpieza de datos
│   ├── model_loader.py           # Carga de modelos, métricas y predicciones
│   └── anomaly_detection.py      # Detección de zonas con precios atípicos
│
├── training_scripts/
│   ├── rf_housing_v1.py          # Entrenamiento Random Forest
│   ├── xgb_housing_v1.py         # Entrenamiento XGBoost
│   ├── mlp_housing_v2.py         # Entrenamiento Red Neuronal (MLP)
│   ├── lr.py                     # Entrenamiento Regresión Ridge
│   ├── diagnostics.py            # Funciones de diagnóstico y visualización
│   └── Descriptive.Rmd           # Análisis descriptivo en R
│
├── models/
│   ├── random_forest.pkl         # Modelo Random Forest serializado
│   ├── random_forest_feature_cols.pkl
│   ├── xgboost.pkl               # Modelo XGBoost serializado
│   ├── xgboost_feature_cols.pkl
│   ├── mlp.pkl                   # Modelo MLP serializado
│   ├── mlp_feature_cols.pkl
│   ├── mlp_scaler_X.pkl          # Scaler de features (MLP)
│   ├── mlp_scaler_y.pkl          # Scaler del target (MLP)
│   └── linear_regression.pkl     # Modelo Ridge serializado
│
├── data/
│   ├── dataset_corregido.xlsx    # Dataset limpio (fuente principal)
│   └── population_departements.csv
│
├── outputs/
│   ├── metrics_rf/xgb/mlp.csv   # Métricas RMSE, MAE, R² por modelo
│   ├── predictions_rf/xgb/mlp.csv  # Predicciones vs valores reales
│   ├── diagnostico_*.png         # Paneles de diagnóstico por modelo
│   └── resultados_*.png          # Gráficas de resultados
│
├── requirements.txt
└── README.md
```

---

## Ejecutar

```bash
git clone https://github.com/Anastasiia127/projet_immo.git
cd projet_immo
pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

Para reentrenar los modelos:

```bash
cd training_scripts
python rf_housing_v1.py
python xgb_housing_v1.py
python mlp_housing_v2.py
python lr.py
```

---

## Stack

Python 3.10 · Streamlit · Plotly · Pandas · Scikit-learn · XGBoost · NumPy · SciPy
