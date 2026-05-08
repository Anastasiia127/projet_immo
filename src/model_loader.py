import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_PATH  = Path(__file__).parent.parent / "models"
OUTPUTS_PATH = Path(__file__).parent.parent / "outputs"

# Nombres de archivo esperados para cada modelo
MODEL_FILES = {
    "Regresión Lineal": "linear_regression.pkl",
    "Random Forest":    "random_forest.pkl",
    "MLP":              "mlp.pkl",
}

# Archivos CSV con métricas reales (generados desde el notebook)
METRICS_FILES = {
    "Regresión Lineal": "metrics_linear.csv",
    "Random Forest":    "metrics_rf.csv",
    "MLP":              "metrics_mlp.csv",
}

# Archivos CSV con predicciones reales (generados desde el notebook)
PREDICTIONS_FILES = {
    "Regresión Lineal": "predictions_linear.csv",
    "Random Forest":    "predictions_rf.csv",
    "MLP":              "predictions_mlp.csv",
}


def load_model(model_name):
    """Carga un modelo .pkl si existe, None si no."""
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise ValueError(f"Modelo desconocido: '{model_name}'.")
    path = MODELS_PATH / filename
    return joblib.load(path) if path.exists() else None


def get_available_models():
    """Modelos con .pkl disponible."""
    return [name for name, f in MODEL_FILES.items()
            if (MODELS_PATH / f).exists()]


def get_model_status():
    """Estado de cada modelo: pkl, csv_metrics, csv_predictions."""
    status = {}
    for name in MODEL_FILES:
        status[name] = {
            "pkl":             (MODELS_PATH  / MODEL_FILES[name]).exists(),
            "csv_metrics":     (OUTPUTS_PATH / METRICS_FILES[name]).exists(),
            "csv_predictions": (OUTPUTS_PATH / PREDICTIONS_FILES[name]).exists(),
        }
    return status


def is_demo_mode(model_name):
    """True si no hay ni .pkl ni CSV de metricas para ese modelo."""
    s = get_model_status()[model_name]
    return not s["pkl"] and not s["csv_metrics"]


# Metricas demo
DEMO_METRICS = {
    "Regresión Lineal": {"RMSE": 95000, "MAE": 68000, "R2": 0.61},
    "Random Forest":    {"RMSE": 72000, "MAE": 48000, "R2": 0.78},
    "MLP":              {"RMSE": 78000, "MAE": 53000, "R2": 0.74},
}


def get_metrics(model_name, y_real=None, y_pred=None):
    """
    Prioridad:
    1. Si se pasan y_real e y_pred -> calcula metricas en el momento
    2. Si existe outputs/metrics_XX.csv -> lee metricas reales
    3. Si no hay nada -> devuelve metricas de demo

    Como guardar desde el notebook:
        pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2}]).to_csv(
            "../outputs/metrics_rf.csv", index=False)
    """
    if y_real is not None and y_pred is not None:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return {
            "RMSE": round(np.sqrt(mean_squared_error(y_real, y_pred)), 2),
            "MAE":  round(mean_absolute_error(y_real, y_pred), 2),
            "R2":   round(r2_score(y_real, y_pred), 4),
        }

    csv_path = OUTPUTS_PATH / METRICS_FILES.get(model_name, "")
    if csv_path.exists():
        row = pd.read_csv(csv_path).iloc[0]
        return {"RMSE": float(row["RMSE"]), "MAE": float(row["MAE"]), "R2": float(row["R2"])}

    return DEMO_METRICS.get(model_name, {})


def get_predictions(model_name):
    """
    Devuelve DataFrame con columnas y_real, y_pred.
    Prioridad: CSV real > demo generado

    Como guardar desde el notebook:
        pd.DataFrame({"y_real": y_test.values, "y_pred": predictions}).to_csv(
            "../outputs/predictions_rf.csv", index=False)
    """
    csv_path = OUTPUTS_PATH / PREDICTIONS_FILES.get(model_name, "")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["modelo"] = model_name
        return df
    return _demo_predictions_for(model_name)


def get_all_predictions():
    """DataFrame combinado con predicciones de todos los modelos."""
    frames = [get_predictions(name) for name in MODEL_FILES]
    return pd.concat(frames, ignore_index=True)


def _demo_predictions_for(model_name, n=300, seed=42):
    noise_levels = {
        "Regresión Lineal": 0.25,
        "Random Forest":    0.15,
        "MLP":              0.18,
    }
    rng   = np.random.default_rng(seed)
    base  = rng.integers(80_000, 600_000, size=n).astype(float)
    noise = rng.normal(0, noise_levels.get(model_name, 0.2), size=n)
    return pd.DataFrame({
        "y_real": base,
        "y_pred": base * (1 + noise),
        "modelo": model_name,
    })


def get_demo_predictions(n=300, seed=42):
    frames = [_demo_predictions_for(name, n, seed) for name in MODEL_FILES]
    return pd.concat(frames, ignore_index=True)
