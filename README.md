
# OuiPredict — Predicción de precios de vivienda en Francia

**Proyecto I · IA · UPV 2026**  
David Esteban Platero · Vicente Tralci Sindoni · Laura Muñoz · Lidia Martínez · Anastasiia Nogina

---

## Qué es

App interactiva para predecir el precio de viviendas en Francia a partir de sus características, con detección de zonas con precios anómalos. Dataset de 37.368 propiedades.

---

## Demo

[ouipredict.streamlit.app](https://ouipredict.streamlit.app/)

---

## Ejecutar

```bash
git clone https://github.com/Anastasiia127/projet_immo.git
cd projet_immo
pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

## Modelos

| Modelo | Archivo |
|--------|---------|
| Random Forest | `models/random_forest.pkl` |
| MLP | `models/mlp.pkl` |
| XGBoost | `models/xgboost.pkl` |

---

## Stack

Python 3.10 · Streamlit · Plotly · Pandas · Scikit-learn · XGBoost · NumPy · SciPy
```
