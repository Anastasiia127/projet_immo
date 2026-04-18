# Entre Tendencias y Anomalías
### Análisis del Precio de la Vivienda en Francia mediante Datos

**Proyecto I — Grado en Inteligencia Artificial**  
**Universitat Politècnica de València · 2026**

**Autores:** David Esteban Platero · Vicente Emilio Tralci Sindoni · Laura Muñoz Martínez · Lidia Martínez Bañuls · Anastasiia Nogina

---

## Descripción

Sistema de análisis y predicción del precio de viviendas en Francia (€/m²) basado en un dataset de 37.368 propiedades. El proyecto incluye exploración de datos, entrenamiento de modelos predictivos y detección de anomalías por zona geográfica.

---

## Estructura del repositorio

```
projet_immo/
│
├── app/
│   └── dashboard.py          # Dashboard interactivo (Streamlit)
│
├── src/
│   ├── preprocessing.py      # Carga y limpieza del dataset
│   └── model_loader.py       # Carga de modelos entrenados (.pkl)
│
├── data/
│   └── dataset_corregido.xlsx  # Dataset original (no incluido en el repo)
│
├── models/
│   └── *.pkl                 # Modelos entrenados (generados por el equipo)
│
├── outputs/
│   └── *.csv                 # Resultados exportados
│
├── .streamlit/
│   └── config.toml           # Configuración visual del dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/projet_immo.git
cd projet_immo

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Añadir el dataset
# Copiar dataset_corregido.xlsx dentro de la carpeta data/

# 5. Lanzar el dashboard
streamlit run app/dashboard.py
```

---

## Cómo conectar los modelos entrenados

Cuando los modelos estén entrenados, guardarlos así desde el notebook:

```python
import joblib

# Guardar modelo
joblib.dump(modelo, 'models/linear_regression.pkl')
joblib.dump(modelo, 'models/random_forest.pkl')
joblib.dump(modelo, 'models/mlp.pkl')
```

El dashboard los detecta automáticamente al arrancar.

---

## Modelos implementados

| Modelo | Archivo esperado |
|---|---|
| Regresión Lineal (Ridge) | `models/linear_regression.pkl` |
| Random Forest | `models/random_forest.pkl` |
| MLP (Red Neuronal) | `models/mlp.pkl` |

---

## Métricas de evaluación

- **RMSE** — Error cuadrático medio
- **MAE** — Error absoluto medio  
- **R²** — Coeficiente de determinación

---

## Tecnologías

- Python 3.10+
- Streamlit · Plotly · Pandas · Scikit-learn · Scipy
