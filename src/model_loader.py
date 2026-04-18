import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_PATH = Path(__file__).parent.parent / "models"

# Nombres de archivo esperados para cada modelo
MODEL_FILES = {
    "Regresión Lineal": "linear_regression.pkl",
    "Random Forest":    "random_forest.pkl",
    "MLP":              "mlp.pkl",
}


def load_model(model_name):
    """
    Carga un modelo entrenado desde la carpeta models/.
    Devuelve el modelo si existe, None si no está disponible.

    Uso desde notebook:
        from src.model_loader import load_model
        model = load_model("Random Forest")
        predictions = model.predict(X_test)
    """
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise ValueError(f"Modelo desconocido: '{model_name}'. Opciones: {list(MODEL_FILES.keys())}")

    path = MODELS_PATH / filename
    if path.exists():
        model = joblib.load(path)
        return model
    else:
        return None


def get_available_models():
    """Devuelve lista de modelos que ya tienen archivo .pkl en models/."""
    available = []
    for name, filename in MODEL_FILES.items():
        if (MODELS_PATH / filename).exists():
            available.append(name)
    return available


def get_model_status():
    """
    Devuelve un diccionario con el estado de cada modelo.
    True = archivo .pkl encontrado, False = no encontrado.
    """
    return {
        name: (MODELS_PATH / filename).exists()
        for name, filename in MODEL_FILES.items()
    }


# ── Demo data ─────────────────────────────────────────────────────────────────
# Métricas y predicciones de ejemplo para el dashboard cuando no hay modelos.
# Los compañeros reemplazarán esto con resultados reales al entrenar.

DEMO_METRICS = {
    "Regresión Lineal": {"RMSE": 95000, "MAE": 68000, "R2": 0.61},
    "Random Forest":    {"RMSE": 72000, "MAE": 48000, "R2": 0.78},
    "MLP":              {"RMSE": 78000, "MAE": 53000, "R2": 0.74},
}


def get_demo_predictions(n=300, seed=42):
    """
    Genera predicciones ficticias para mostrar en el dashboard en modo demo.
    Devuelve un DataFrame con columnas: y_real, y_pred, modelo.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(80_000, 600_000, size=n).astype(float)

    rows = []
    noise_levels = {
        "Regresión Lineal": 0.25,
        "Random Forest":    0.15,
        "MLP":              0.18,
    }
    for model_name, noise in noise_levels.items():
        noise_vec = rng.normal(0, noise, size=n)
        y_pred = base * (1 + noise_vec)
        for real, pred in zip(base, y_pred):
            rows.append({"y_real": real, "y_pred": pred, "modelo": model_name})

    return pd.DataFrame(rows)


def get_metrics(model_name, y_real=None, y_pred=None):
    """
    Calcula métricas reales si se pasan y_real e y_pred,
    o devuelve las métricas de demo si no hay datos.

    Uso con modelo real:
        metrics = get_metrics("Random Forest", y_test, predictions)
    """
    if y_real is not None and y_pred is not None:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        mae  = mean_absolute_error(y_real, y_pred)
        r2   = r2_score(y_real, y_pred)
        return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}
    else:
        return DEMO_METRICS.get(model_name, {})
