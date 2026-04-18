import pandas as pd
import numpy as np
from pathlib import Path

# Ruta por defecto al dataset
DATA_PATH = Path(__file__).parent.parent / "data" / "dataset_corregido.xlsx"

# Variables numéricas que se usan en el modelo
NUMERICAL_FEATURES = [
    "size",
    "nb_rooms",
    "nb_bedrooms",
    "nb_bathrooms",
    "nb_parking_places",
    "nb_boxes",
    "nb_photos",
    "nb_terraces",
    "energy_performance_value",
    "ghg_value",
]

# Variables binarias (0/1) que se usan en el modelo
BINARY_FEATURES = [
    "has_a_balcony",
    "has_a_cellar",
    "has_a_garage",
    "has_air_conditioning",
    "last_floor",
]

# Variables categóricas que se codifican
CATEGORICAL_FEATURES = [
    "property_type",
    "energy_performance_category",
    "ghg_category",
]

# Variable objetivo
TARGET = "price"


def load_data(path=DATA_PATH):
    """Carga el dataset desde Excel."""
    df = pd.read_excel(path)
    return df


def clean_data(df):
    """
    Limpieza básica del dataset:
    - Elimina duplicados
    - Imputa nulos en variables numéricas con la mediana
    - Imputa nulos en variables categóricas con la moda
    - Elimina filas sin precio (target)
    """
    df = df.copy()

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Eliminar filas sin precio
    df = df.dropna(subset=[TARGET])

    # Imputar nulos numéricos con mediana
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Imputar nulos categóricos con moda
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Nota: floor, land_size y exposition tienen >70% nulos
    # Se excluyen del modelo pero se conservan en el DataFrame para el EDA

    return df


def compute_price_per_m2(df):
    """Añade columna precio por m² (€/m²)."""
    df = df.copy()
    df["price_per_m2"] = df[TARGET] / df["size"]
    # Eliminar valores infinitos o nulos que puedan surgir si size == 0
    df["price_per_m2"] = df["price_per_m2"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price_per_m2"])
    return df


def get_features_and_target(df):
    """
    Devuelve X (features) e y (target) listos para entrenar.
    Solo usa columnas numéricas y binarias para evitar problemas de encoding.
    Los compañeros pueden ampliar esto añadiendo encoding de categóricas.
    """
    feature_cols = [c for c in NUMERICAL_FEATURES + BINARY_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y


def load_and_prepare(path=DATA_PATH):
    """
    Pipeline completo: carga, limpia y devuelve el DataFrame listo.
    Uso rápido para el dashboard y los notebooks.
    """
    df = load_data(path)
    df = clean_data(df)
    df = compute_price_per_m2(df)
    return df
