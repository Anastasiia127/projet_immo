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
    "dist_capital_provincia",   # ← nueva variable derivada (recomendación profesor)
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
    "property_type_group",       # ← agrupada (recomendación profesor)
    "energy_performance_category",
    "ghg_category",
]

# Variable objetivo
TARGET = "price"

# ── Agrupación de tipos de propiedad (recomendación profesor) ──────────────────
# De 22 tipos originales → 5 grupos manejables
PROPERTY_TYPE_GROUPS = {
    "appartement": "apartamento",
    "duplex":      "apartamento",
    "loft":        "apartamento",
    "chambre":     "apartamento",
    "péniche":     "apartamento",

    "maison":      "casa",
    "villa":       "casa",
    "chalet":      "casa",
    "ferme":       "casa",
    "gîte":        "casa",
    "moulin":      "casa",
    "manoir":      "casa",
    "château":     "casa",
    "hôtel particulier": "casa",

    "terrain":          "terreno",
    "terrain à bâtir":  "terreno",

    "parking":     "local_comercial",
    "atelier":     "local_comercial",
    "hôtel":       "local_comercial",

    "divers":      "otro",
    "viager":      "otro",
    "propriété":   "otro",
}

# ── Capitales de provincia (lat, lon) para calcular distancia ─────────────────
# Fuente: coordenadas aproximadas de las prefecturas francesas
CAPITALES_PROVINCIA = {
    "01": (46.2044, 5.2264),   # Ain - Bourg-en-Bresse
    "02": (49.8942, 3.2864),   # Aisne - Laon
    "03": (46.5667, 3.3333),   # Allier - Moulins
    "04": (44.0921, 6.2356),   # Alpes-de-Haute-Provence - Digne
    "05": (44.5628, 6.0705),   # Hautes-Alpes - Gap
    "06": (43.7102, 7.2620),   # Alpes-Maritimes - Nice
    "07": (44.9333, 4.8667),   # Ardèche - Privas
    "08": (49.7736, 4.7217),   # Ardennes - Charleville
    "09": (43.0500, 1.6000),   # Ariège - Foix
    "10": (48.2973, 4.0744),   # Aube - Troyes
    "11": (43.2130, 2.3491),   # Aude - Carcassonne
    "12": (44.3500, 2.5667),   # Aveyron - Rodez
    "13": (43.2965, 5.3698),   # Bouches-du-Rhône - Marseille
    "14": (49.1829, -0.3707),  # Calvados - Caen
    "15": (45.0000, 2.6167),   # Cantal - Aurillac
    "16": (45.6500, 0.1500),   # Charente - Angoulême
    "17": (45.7500, -0.6333),  # Charente-Maritime - La Rochelle
    "18": (47.0833, 2.4000),   # Cher - Bourges
    "19": (45.2671, 1.7741),   # Corrèze - Tulle
    "21": (47.3167, 5.0167),   # Côte-d'Or - Dijon
    "22": (48.5137, -2.7653),  # Côtes-d'Armor - Saint-Brieuc
    "23": (46.1667, 1.8667),   # Creuse - Guéret
    "24": (45.1838, 0.7203),   # Dordogne - Périgueux
    "25": (47.2333, 6.0167),   # Doubs - Besançon
    "26": (44.9333, 4.8917),   # Drôme - Valence
    "27": (49.0175, 1.1520),   # Eure - Évreux
    "28": (48.4469, 1.4890),   # Eure-et-Loir - Chartres
    "29": (48.3887, -4.4885),  # Finistère - Quimper
    "30": (43.8370, 4.3601),   # Gard - Nîmes
    "31": (43.6047, 1.4442),   # Haute-Garonne - Toulouse
    "32": (43.6500, 0.5833),   # Gers - Auch
    "33": (44.8378, -0.5792),  # Gironde - Bordeaux
    "34": (43.6119, 3.8772),   # Hérault - Montpellier
    "35": (48.1147, -1.6794),  # Ille-et-Vilaine - Rennes
    "36": (46.8000, 1.6833),   # Indre - Châteauroux
    "37": (47.3936, 0.6892),   # Indre-et-Loire - Tours
    "38": (45.1667, 5.7167),   # Isère - Grenoble
    "39": (46.6667, 5.5500),   # Jura - Lons-le-Saunier
    "40": (43.8939, -0.4993),  # Landes - Mont-de-Marsan
    "41": (47.5833, 1.3333),   # Loir-et-Cher - Blois
    "42": (45.4333, 4.3917),   # Loire - Saint-Étienne
    "43": (45.0433, 3.8836),   # Haute-Loire - Le Puy-en-Velay
    "44": (47.2184, -1.5536),  # Loire-Atlantique - Nantes
    "45": (47.9029, 1.9039),   # Loiret - Orléans
    "46": (44.4500, 1.4333),   # Lot - Cahors
    "47": (44.2000, 0.6167),   # Lot-et-Garonne - Agen
    "48": (44.5167, 3.5000),   # Lozère - Mende
    "49": (47.4736, -0.5543),  # Maine-et-Loire - Angers
    "50": (49.1197, -1.0803),  # Manche - Saint-Lô
    "51": (49.2583, 4.0317),   # Marne - Châlons-en-Champagne
    "52": (48.1114, 5.1375),   # Haute-Marne - Chaumont
    "53": (48.0667, -0.7667),  # Mayenne - Laval
    "54": (48.6921, 6.1844),   # Meurthe-et-Moselle - Nancy
    "55": (49.1622, 5.3836),   # Meuse - Bar-le-Duc
    "56": (47.6594, -2.7597),  # Morbihan - Vannes
    "57": (49.1193, 6.1727),   # Moselle - Metz
    "58": (47.0000, 3.1667),   # Nièvre - Nevers
    "59": (50.6292, 3.0573),   # Nord - Lille
    "60": (49.4197, 2.0825),   # Oise - Beauvais
    "61": (48.4295, 0.0886),   # Orne - Alençon
    "62": (50.2903, 2.7814),   # Pas-de-Calais - Arras
    "63": (45.7831, 3.0824),   # Puy-de-Dôme - Clermont-Ferrand
    "64": (43.2951, -0.3708),  # Pyrénées-Atlantiques - Pau
    "65": (43.2333, 0.0833),   # Hautes-Pyrénées - Tarbes
    "66": (42.6986, 2.8956),   # Pyrénées-Orientales - Perpignan
    "67": (48.5734, 7.7521),   # Bas-Rhin - Strasbourg
    "68": (47.7500, 7.3333),   # Haut-Rhin - Colmar
    "69": (45.7640, 4.8357),   # Rhône - Lyon
    "70": (47.6167, 6.1500),   # Haute-Saône - Vesoul
    "71": (46.3000, 4.8333),   # Saône-et-Loire - Mâcon
    "72": (48.0039, 0.1996),   # Sarthe - Le Mans
    "73": (45.5744, 5.9178),   # Savoie - Chambéry
    "74": (45.8992, 6.1294),   # Haute-Savoie - Annecy
    "75": (48.8566, 2.3522),   # Paris
    "76": (49.4432, 1.0993),   # Seine-Maritime - Rouen
    "77": (48.5400, 2.6600),   # Seine-et-Marne - Melun
    "78": (48.8014, 2.1301),   # Yvelines - Versailles
    "79": (46.3167, -0.4667),  # Deux-Sèvres - Niort
    "80": (49.8942, 2.2958),   # Somme - Amiens
    "81": (43.9275, 2.1479),   # Tarn - Albi
    "82": (44.0167, 1.3500),   # Tarn-et-Garonne - Montauban
    "83": (43.1242, 5.9280),   # Var - Toulon
    "84": (43.9493, 4.8055),   # Vaucluse - Avignon
    "85": (46.6706, -1.4264),  # Vendée - La Roche-sur-Yon
    "86": (46.5833, 0.3333),   # Vienne - Poitiers
    "87": (45.8333, 1.2500),   # Haute-Vienne - Limoges
    "88": (48.1667, 6.4500),   # Vosges - Épinal
    "89": (47.7975, 3.5672),   # Yonne - Auxerre
    "90": (47.6333, 6.8667),   # Territoire de Belfort - Belfort
    "91": (48.6264, 2.4286),   # Essonne - Évry
    "92": (48.8924, 2.2353),   # Hauts-de-Seine - Nanterre
    "93": (48.9356, 2.3539),   # Seine-Saint-Denis - Bobigny
    "94": (48.7833, 2.4667),   # Val-de-Marne - Créteil
    "95": (49.0333, 2.0833),   # Val-d'Oise - Cergy
}


def load_data(path=DATA_PATH):
    """Carga el dataset desde Excel."""
    df = pd.read_excel(path)
    return df


def group_property_types(df):
    """
    Agrupa los 22 tipos de propiedad en 5 categorías manejables.
    Recomendación del profesor: reducir dimensionalidad del encoding.

    Grupos: apartamento, casa, terreno, local_comercial, otro
    """
    df = df.copy()
    df["property_type_group"] = df["property_type"].map(PROPERTY_TYPE_GROUPS).fillna("otro")
    return df


def compute_distance_to_capital(df):
    """
    Calcula la distancia en km entre cada vivienda y la capital
    de su provincia (departamento), usando el código postal.

    Recomendación del profesor: usar distancia a capital en lugar
    de lat/lon directamente, ya que captura mejor la urbanización.
    """
    df = df.copy()

    def haversine(lat1, lon1, lat2, lon2):
        """Distancia en km entre dos puntos geográficos."""
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def get_departement(postal_code):
        """Extrae el código de departamento del código postal."""
        code = str(int(postal_code)).zfill(5)[:2]
        return code

    distances = []
    for _, row in df.iterrows():
        try:
            dep = get_departement(row["postal_code"])
            if dep in CAPITALES_PROVINCIA:
                cap_lat, cap_lon = CAPITALES_PROVINCIA[dep]
                dist = haversine(row["approximate_latitude"], row["approximate_longitude"], cap_lat, cap_lon)
            else:
                dist = np.nan
        except Exception:
            dist = np.nan
        distances.append(dist)

    df["dist_capital_provincia"] = distances
    df["dist_capital_provincia"] = df["dist_capital_provincia"].fillna(df["dist_capital_provincia"].median())
    return df


def add_provincia(df):
    """
    Extrae la provincia (departamento) del código postal.
    Los primeros 2 dígitos del código postal = código de departamento.
    Reduce las 8.643 ciudades a 96 provincias — más manejable para el modelo.
    """
    df = df.copy()
    df["provincia"] = df["postal_code"].apply(
        lambda x: str(int(x)).zfill(5)[:2]
    )
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
    df["price_per_m2"] = df["price_per_m2"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price_per_m2"])
    return df


def get_features_and_target(df):
    """
    Devuelve X (features) e y (target) listos para entrenar.
    Incluye variables numéricas, binarias y encoding de categóricas (one-hot).
    """
    feature_cols = [c for c in NUMERICAL_FEATURES + BINARY_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

    # One-hot encoding de variables categóricas agrupadas
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    y = df[TARGET]
    return X, y


def load_and_prepare(path=DATA_PATH):
    """
    Pipeline completo: carga, limpia y devuelve el DataFrame listo.
    Uso rápido para el dashboard y los notebooks.
    """
    df = load_data(path)
    df = group_property_types(df)       # agrupa 22 tipos → 5 grupos
    df = add_provincia(df)               # extrae provincia del código postal
    df = compute_distance_to_capital(df) # distancia a capital de provincia
    df = clean_data(df)
    df = compute_price_per_m2(df)
    return df
