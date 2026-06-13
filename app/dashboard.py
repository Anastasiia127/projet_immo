import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

from src.model_loader import get_model_status, get_available_models, load_model, get_metrics, get_predictions, OUTPUTS_PATH, PREDICTIONS_FILES, METRICS_FILES, build_input_for_model
from src.preprocessing import load_and_prepare, NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OuiPredict",
    page_icon="🏠",
    layout="wide",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    * { font-size: 1.02em; }
    .stMarkdown p { font-size: 1.05rem; }
    .stCaption { font-size: 0.95rem; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        border-left: 4px solid #0d9488;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando mapa de departamentos...")
def get_geojson():
    """Descarga el GeoJSON de departamentos de Francia (se cachea automáticamente)."""
    import urllib.request
    import json
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None

@st.cache_data(show_spinner="Cargando dataset...")
def get_data():
    return load_and_prepare()

df = get_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 OuiPredict</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tu guía inteligente para comprar vivienda en Francia · UPV · 2026</div>', unsafe_allow_html=True)

# ── Landing hero ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #f0fdfc 0%, #e0f2fe 100%); border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1rem;">
    <p style="font-size: 1.15rem; color: #1e293b; margin-bottom: 1rem;">
        Introduce las características de la vivienda que buscas y te diremos si el precio es justo, 
        cómo se compara con la zona y qué esperar del mercado.
    </p>
    <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">🎯</span>
            <div><strong>Predice el precio justo</strong><br><small>Modelos entrenados con 37.000+ viviendas reales</small></div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">📍</span>
            <div><strong>Analiza la zona</strong><br><small>Precios medianos, mínimos y máximos por departamento</small></div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">📊</span>
            <div><strong>Compara con el mercado</strong><br><small>Ve dónde se sitúa tu vivienda respecto a la competencia</small></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Métricas globales ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total viviendas", f"{len(df):,}".replace(",", "."))
with c2:
    st.metric("Precio mediano", f"{int(df[TARGET].median()):,} €".replace(",", "."))
with c3:
    st.metric("€/m² mediano", f"{int(df['price_per_m2'].median()):,} €".replace(",", "."))
with c4:
    st.metric("Ciudades", f"{df['city'].nunique():,}".replace(",", "."))

st.divider()

with st.expander("❓ ¿Cómo usar OuiPredict?"):
    st.markdown("""
    
    **🏠 Predecir precio** 
    1. Elige el **modelo** — XGBoost es el más preciso (R²=81%), pero todos son válidos
    2. Selecciona el **departamento** donde quieres comprar — los más populares están arriba con iconos
    3. Elige el **tipo de propiedad** (apartamento, casa, terreno...)
    4. Rellena la **superficie** y las características físicas
    5. Marca los **extras** que necesitas (balcón, garaje, A/C...)
    6. Pulsa **"Calcular precio estimado"**
    7. Obtendrás el precio estimado, el contexto de mercado con viviendas similares, y un gauge que te dice si es barato o caro para la zona

    ---

    **📊 Exploración de datos**
    - Usa el **filtro de tipo** para ver solo apartamentos o solo casas
    - Los mapas son interactivos — haz zoom y pasa el ratón para ver detalles
    - Los gráficos técnicos (correlación, varianza...) están plegados al final

    ---

    **🤖 Modelos**
    - Compara la precisión de los tres modelos entrenados
    - El gráfico "Predicción vs Realidad" muestra qué tan bien predice cada modelo — cuanto más cerca de la línea roja, mejor

    ---

    **🔍 Anomalías por zona**
    - Las zonas en **rojo** tienen precios significativamente distintos de lo esperado — pueden estar sobrevaloradas o infravaloradas
    - Usa el slider para ver más o menos ciudades

    ---

    **⚙️ Preprocesamiento**
    - Documentación técnica del pipeline de datos — útil si quieres entender cómo se prepararon los datos para el modelo
    """)


# ── Nombres de los departamentos franceses ────────────────────────────────────
DEPT_NOMBRES = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "21": "Côte-d'Or",
    "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs",
    "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde",
    "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre", "37": "Indre-et-Loire",
    "38": "Isère", "39": "Jura", "40": "Landes", "41": "Loir-et-Cher",
    "42": "Loire", "43": "Haute-Loire", "44": "Loire-Atlantique", "45": "Loiret",
    "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère", "49": "Maine-et-Loire",
    "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan", "57": "Moselle",
    "58": "Nièvre", "59": "Nord", "60": "Oise", "61": "Orne",
    "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie", "75": "Paris",
    "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}

# ── Pestañas ──────────────────────────────────────────────────────────────────
tab5, tab1, tab3, tab4, tab2 = st.tabs([
    "🏠 Predecir precio",
    "📊 Exploración de datos",
    "🤖 Modelos",
    "🔍 Anomalías por zona",
    "⚙️ Preprocesamiento",
])

# ------------------------------------------------------------------------------
# TAB 1 · EDA
# ------------------------------------------------------------------------------
with tab1:

    # ── Filtro por tipo de propiedad ──────────────────────────────────────────
    st.markdown("**Filtro de tipo de propiedad** — afecta a todos los gráficos de precios y mapas de abajo:")
    tipos_disponibles = sorted(df["property_type_group"].unique().tolist())
    tipos_sel = st.multiselect(
        "Selecciona tipos de propiedad",
        options=tipos_disponibles,
        default=["apartamento", "casa"],
    )
    df_filtrado = df[df["property_type_group"].isin(tipos_sel)] if tipos_sel else df
    
    # ── Contador por tipo ─────────────────────────────────────────────────────
    if tipos_sel:
        counts = df_filtrado.groupby("property_type_group").size()
        tipo_icons = {"apartamento": "🏢", "casa": "🏘️", "terreno": "🌿", "local_comercial": "🏪", "otro": "📦"}
        parts = [f"{tipo_icons.get(t, '📍')} **{counts.get(t, 0):,}** {t}".replace(",", ".") for t in tipos_sel if t in counts]
        total = len(df_filtrado)
        st.markdown(f"**{total:,}** viviendas seleccionadas: {' · '.join(parts)}".replace(",", "."))
    else:
        st.markdown(f"**{len(df_filtrado):,}** viviendas (todos los tipos)".replace(",", "."))

    with st.expander("ℹ️ ¿Qué incluye cada grupo?"):
        st.markdown("""
        | Grupo | Tipos originales (francés) |
        |---|---|
        | **apartamento** | appartement, duplex, loft, chambre, péniche |
        | **casa** | maison, villa, chalet, ferme, gîte, moulin, manoir, château, hôtel particulier |
        | **terreno** | terrain, terrain à bâtir |
        | **local_comercial** | parking, atelier, hôtel |
        | **otro** | divers, viager, propriété |

        *Los tipos están en francés porque el dataset proviene del Institut Louis Bachelier (Francia).*
        """)

    # ── Mapa choroplético (primero — más visual) ──────────────────────────────
    st.markdown('<div class="section-title">Precio mediano por departamento</div>', unsafe_allow_html=True)
    st.caption("Cada departamento está coloreado según el precio mediano de sus viviendas. 🟢 Verde = más barato · 🔴 Rojo = más caro. Pasa el ratón para ver el nombre y precio.")

    geojson = get_geojson()
    if geojson:
        dept_stats = (
            df_filtrado.groupby("provincia")
            .agg(precio_mediano=("price", "median"), n=("price", "count"))
            .reset_index()
        )
        dept_stats["nombre"] = dept_stats["provincia"].map(DEPT_NOMBRES).fillna(dept_stats["provincia"])

        fig = px.choropleth_mapbox(
            dept_stats,
            geojson=geojson,
            locations="provincia",
            featureidkey="properties.code",
            color="precio_mediano",
            color_continuous_scale="RdYlGn_r",
            mapbox_style="carto-positron",
            zoom=4.5,
            center={"lat": 46.5, "lon": 2.5},
            opacity=0.75,
            hover_name="nombre",
            hover_data={"precio_mediano": ":,.0f", "n": True, "provincia": False},
            labels={"precio_mediano": "Precio mediano (€)", "n": "Viviendas"},
            title="Precio mediano por departamento — 🟢 barato · 🔴 caro",
            height=600,
        )
        fig.update_layout(
            paper_bgcolor="white",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Precio (€)", tickformat=",.0f"),
        )
        st.plotly_chart(fig, use_container_width=True, key="choropleth_map")
    else:
        st.warning("No se pudo cargar el mapa de departamentos. Verifica tu conexión a internet.")

    # ── Mapa de puntos ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Mapa de precios por ubicación</div>', unsafe_allow_html=True)
    st.caption("🟢 Verde = precio bajo · 🟡 Amarillo = precio medio · 🔴 Rojo = precio alto. El **tamaño** de cada punto indica cuántas viviendas hay en esa zona. Pasa el ratón por encima para ver el detalle.")

    mapa_df = (
        df_filtrado
        .assign(
            lat=df_filtrado["approximate_latitude"].round(1),
            lon=df_filtrado["approximate_longitude"].round(1),
        )
        .groupby(["lat", "lon"])
        .agg(
            precio_mediano=(TARGET, "median"),
            n=(TARGET, "count"),
            precio_m2=("price_per_m2", "median"),
        )
        .reset_index()
    )

    fig = px.scatter_mapbox(
        mapa_df,
        lat="lat", lon="lon",
        color="precio_mediano",
        size="n",
        size_max=35,
        zoom=5,
        center={"lat": 46.5, "lon": 2.5},
        color_continuous_scale="RdYlGn_r",
        range_color=[mapa_df["precio_mediano"].quantile(0.05), mapa_df["precio_mediano"].quantile(0.95)],
        hover_data={"precio_mediano": ":,.0f", "precio_m2": ":,.0f", "n": True, "lat": False, "lon": False},
        labels={"precio_mediano": "Precio mediano (€)", "precio_m2": "€/m²", "n": "Viviendas"},
        title="Distribución geográfica del precio mediano — 🟢 barato · 🟡 medio · 🔴 caro",
        mapbox_style="carto-positron",
        height=600,
        opacity=0.85,
    )
    fig.update_layout(
        paper_bgcolor="white",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Precio (€)",
            tickformat=",.0f",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Distribución de precios ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Distribución de precios</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_filtrado, x=TARGET, nbins=80,
            title="Distribución del precio (€)",
            labels={TARGET: "Precio (€)"},
            color_discrete_sequence=["#0d9488"],
        )
        fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Cada barra representa un rango de precios. El pico más alto indica el precio más frecuente. La cola larga a la derecha revela la presencia de viviendas muy caras (outliers).")

    with col2:
        fig = px.box(
            df_filtrado, x="property_type_group", y=TARGET,
            title="Precio por tipo de propiedad",
            labels={TARGET: "Precio (€)", "property_type_group": "Tipo"},
            color_discrete_sequence=["#818cf8"],
        )
        fig.update_layout(xaxis_tickangle=-35, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🖱️ *Pasa el ratón sobre el gráfico para ver estadísticas detalladas.* · 📦 La línea central es la mediana. La caja abarca el 50% central de los datos. Los puntos fuera son outliers — viviendas con precios muy alejados de lo habitual.")

    # ── Top ciudades ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Precio por ciudad (top 20)</div>', unsafe_allow_html=True)
    st.caption("Se usa la **mediana** y no la media porque es más robusta frente a precios extremos. Las ciudades de la costa (Cannes, Niza) y París suelen liderar este ranking.")

    city_agg = (
        df_filtrado.groupby(["city", "provincia"])
        .agg(precio_mediano=(TARGET, "median"), n=(TARGET, "count"))
        .reset_index()
    )
    top_cities = (
        city_agg[city_agg["n"] >= 5]
        .sort_values("precio_mediano", ascending=False)
        .head(20)
    )
    top_cities["ciudad_provincia"] = top_cities["city"] + " (" + top_cities["provincia"].map(DEPT_NOMBRES).fillna(top_cities["provincia"]) + ")"

    fig = px.bar(
        top_cities, x="ciudad_provincia", y="precio_mediano",
        title="Precio mediano por ciudad — Top 20 (mín. 5 viviendas)",
        labels={"precio_mediano": "Precio mediano (€)", "ciudad_provincia": "Ciudad"},
        color="precio_mediano",
        color_continuous_scale=["#99f6e4", "#0d9488"],
        hover_data={"n": True, "provincia": True, "ciudad_provincia": False},
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # ── Precio vs Superficie ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Precio vs Superficie</div>', unsafe_allow_html=True)
    st.caption("Relación entre el tamaño de la vivienda (m²) y su precio (€). Idealmente deberíamos ver una tendencia positiva: a mayor superficie, mayor precio. Los puntos muy separados de esa tendencia son anomalías interesantes.")

    sample = df_filtrado.sample(min(3000, len(df_filtrado)), random_state=42)
    fig = px.scatter(
        sample, x="size", y=TARGET,
        color="property_type_group",
        title="Precio vs Superficie (muestra de hasta 3.000 viviendas)",
        labels={"size": "Superficie (m²)", TARGET: "Precio (€)", "property_type_group": "Tipo"},
        opacity=0.5,
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🔬 Análisis técnico del dataset")
    st.caption("Las siguientes secciones muestran el análisis estadístico del dataset completo (sin filtro de tipo de propiedad).")

    # ── Correlación ───────────────────────────────────────────────────────────
    with st.expander("📊 Correlación entre variables numéricas"):
        st.caption("Valores cercanos a **+1**: correlación positiva fuerte (ambas variables suben juntas). Cercanos a **-1**: correlación negativa. Cercanos a **0**: sin relación. Las variables con mayor correlación con `price` son las más útiles para el modelo.")
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns] + [TARGET]
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=["#f0fdfc", "#0d9488"],
            title="Matriz de correlación",
            aspect="auto",
        )
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Valores nulos ─────────────────────────────────────────────────────────
    with st.expander("🕳️ Valores nulos por columna"):
        st.caption("Las columnas con muchos nulos son problemáticas para el modelo. Las que superan el 50% (floor, land_size, ghg_value, ghg_category, exposition) se excluyen directamente. Las filas con nulos restantes se eliminan, sin imputación.")
        nulls = df.isnull().sum().reset_index()
        nulls.columns = ["columna", "nulos"]
        nulls["porcentaje"] = (nulls["nulos"] / len(df) * 100).round(1)
        nulls = nulls[nulls["nulos"] > 0].sort_values("nulos", ascending=False)
        if nulls.empty:
            st.success("No hay valores nulos en el dataset procesado.")
        else:
            fig = px.bar(
                nulls, x="columna", y="porcentaje",
                title="% de valores nulos por columna",
                labels={"porcentaje": "% nulos", "columna": "Columna"},
                color="porcentaje",
                color_continuous_scale=["#f0fdfc", "#818cf8"],
                text="porcentaje",
            )
            fig.update_traces(texttemplate="%{text}%")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # ── Análisis de listings ──────────────────────────────────────────────────
    with st.expander("📋 Análisis de listings por ciudad y tipo"):
        st.caption("Distribución de anuncios por ciudad, departamento y tipo de propiedad.")

        col1, col2 = st.columns(2)
        with col1:
            top_listings = (
                df.groupby("city").size().reset_index(name="listings")
                .sort_values("listings", ascending=False).head(20)
            )
            fig = px.bar(
                top_listings, x="city", y="listings",
                title="Top 20 ciudades por número de anuncios",
                labels={"listings": "Nº anuncios", "city": "Ciudad"},
                color="listings", color_continuous_scale=["#99f6e4", "#0d9488"],
                text="listings",
            )
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Ciudades con más anuncios tienen mayor representación — sus precios son más fiables para el modelo.")

        with col2:
            proptype_counts = (
                df.groupby("property_type").size().reset_index(name="listings")
                .sort_values("listings", ascending=False)
            )
            proptype_counts["sparse"] = proptype_counts["listings"].apply(
                lambda x: "< 30 (escaso)" if x < 30 else ">= 30"
            )
            fig = px.bar(
                proptype_counts, x="property_type", y="listings",
                color="sparse",
                color_discrete_map={"< 30 (escaso)": "#f59e0b", ">= 30": "#0d9488"},
                title="Anuncios por tipo de propiedad",
                labels={"listings": "Nº anuncios", "property_type": "Tipo", "sparse": ""},
                text="listings",
            )
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Tipos en naranja tienen menos de 30 anuncios — pueden causar problemas en el modelo.")

        dept_listings = (
            df.groupby("provincia").size().reset_index(name="listings")
            .sort_values("listings", ascending=False)
        )
        dept_listings["nombre"] = dept_listings["provincia"].map(DEPT_NOMBRES).fillna(dept_listings["provincia"])
        dept_listings["sparse"] = dept_listings["listings"].apply(
            lambda x: "< 30 (escaso)" if x < 30 else ">= 30"
        )
        fig = px.bar(
            dept_listings, x="nombre", y="listings",
            color="sparse",
            color_discrete_map={"< 30 (escaso)": "#f59e0b", ">= 30": "#0d9488"},
            title="Número de anuncios por departamento",
            labels={"listings": "Nº anuncios", "nombre": "Departamento", "sparse": ""},
            text="listings",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)

        sparse_cities = (
            df.groupby("city").size().reset_index(name="listings")
            .query("listings < 30").sort_values("listings")
        )
        n_sparse = len(sparse_cities)
        total_sparse = sparse_cities["listings"].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Ciudades con < 30 anuncios", n_sparse)
        col2.metric("Anuncios en esas ciudades", f"{total_sparse:,}".replace(",", "."))
        col3.metric("% del dataset", f"{total_sparse/len(df)*100:.1f}%")
        st.caption(f"Hay {n_sparse} ciudades subrepresentadas. Se recomienda agruparlas por departamento en el modelo.")

    # ── Varianza explicada ────────────────────────────────────────────────────
    with st.expander("📈 Varianza del precio explicada por cada variable"):
        st.caption("Cuánto explica cada variable la variacion del precio. Mayor valor = mas util para el modelo.")

        @st.cache_data(show_spinner="Calculando varianza explicada...")
        def compute_variance_explained(_df):
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import LabelEncoder
            import warnings
            warnings.filterwarnings("ignore")
            df_var = _df[_df["price"] > 0].copy()
            df_var["log_price"] = np.log(df_var["price"])
            results = []
            skip_cols = {"price", "log_price", "price_per_m2", "id_annonce",
                         "approximate_latitude", "approximate_longitude", "postal_code"}
            for col in df_var.columns:
                if col in skip_cols:
                    continue
                series = df_var[col].dropna()
                if len(series) < 100 or series.nunique() < 2:
                    continue
                try:
                    y = df_var.loc[series.index, "log_price"]
                    if pd.api.types.is_numeric_dtype(series):
                        X_col = series.values.reshape(-1, 1)
                        r2 = LinearRegression().fit(X_col, y).score(X_col, y)
                        results.append({"Variable": col, "Tipo": "numerica", "Eta2": round(r2, 4)})
                    else:
                        le = LabelEncoder()
                        X_col = le.fit_transform(series.astype(str)).reshape(-1, 1)
                        r2 = LinearRegression().fit(X_col, y).score(X_col, y)
                        results.append({"Variable": col, "Tipo": "categorica", "Eta2": round(r2, 4)})
                except Exception:
                    continue
            VAR_NAMES = {
            "nb_rooms": "Nº habitaciones", "nb_bathrooms": "Nº baños",
            "nb_bedrooms": "Nº dormitorios", "nb_parking_places": "Plazas parking",
            "nb_boxes": "Nº trasteros", "nb_photos": "Nº fotos",
            "nb_terraces": "Nº terrazas", "size": "Superficie (m²)",
            "energy_performance_value": "Valor energético",
            "ghg_value": "Valor GHG", "dist_capital_provincia": "Dist. capital dpto.",
            "anuncios_por_100k_hab": "Anuncios/100k hab.",
            "has_a_balcony": "Tiene balcón", "has_a_cellar": "Tiene sótano",
            "has_a_garage": "Tiene garaje", "has_air_conditioning": "Aire acond.",
            "last_floor": "Última planta", "property_type_group": "Tipo propiedad",
            "property_type": "Tipo propiedad (detalle)", "provincia": "Departamento",
            "energy_performance_category": "Categoría energética",
            "ghg_category": "Categoría GHG", "floor": "Planta",
            "price_per_m2": "Precio/m²", "city": "Ciudad",
        }
            df_result = pd.DataFrame(results).sort_values("Eta2", ascending=False).head(20)
            df_result["Variable"] = df_result["Variable"].map(lambda x: VAR_NAMES.get(x, x))
            return df_result

    var_df = compute_variance_explained(df)
    var_df["Efecto"] = var_df["Eta2"].apply(
        lambda x: "Grande" if x >= 0.14 else ("Medio" if x >= 0.06 else ("Pequeno" if x >= 0.01 else "Negligible"))
    )
    fig = px.bar(
        var_df.sort_values("Eta2"), x="Eta2", y="Variable",
        color="Efecto",
        color_discrete_map={"Grande": "#0d9488", "Medio": "#818cf8", "Pequeno": "#f59e0b", "Negligible": "#e74c3c"},
        orientation="h",
        title="Top 20 variables por varianza explicada en log(precio)",
        labels={"Eta2": "R2 / eta2", "Variable": "Variable", "Efecto": "Efecto"},
        text="Eta2",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=550)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2 · PREPROCESAMIENTO ──────────────────────────────────────────────────
with tab2:
    # Variables del modelo divididas en numéricas, binarias y categóricas.
    # ✅/❌ indica si la columna existe en el df cargado.
    st.markdown('<div class="section-title">Variables utilizadas en los modelos</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔢 Numéricas**")
        for f in NUMERICAL_FEATURES:
            available = "✅" if f in df.columns else "❌"
            st.markdown(f"{available} `{f}`")
    with col2:
        st.markdown("**🔘 Binarias (0/1)**")
        for f in BINARY_FEATURES:
            available = "✅" if f in df.columns else "❌"
            st.markdown(f"{available} `{f}`")
    with col3:
        st.markdown("**🏷️ Categóricas**")
        for f in CATEGORICAL_FEATURES:
            available = "✅" if f in df.columns else "❌"
            st.markdown(f"{available} `{f}`")

    # Estrategia aplicada a cada tipo de variable con nulos.
    st.markdown('<div class="section-title">Tratamiento de valores nulos</div>', unsafe_allow_html=True)

    tratamiento = pd.DataFrame([
        {"Variable": "floor, land_size, ghg_value, ghg_category, exposition", "Estrategia": "Excluidas del modelo", "Motivo": ">50% de valores nulos"},
        {"Variable": "energy_performance_category", "Estrategia": "Convertida a has_energy_cert (0/1)", "Motivo": "Preserva señal de ausencia sin imputar"},
        {"Variable": "Filas con nulos restantes", "Estrategia": "Eliminación de fila", "Motivo": "Sin imputación para evitar sesgos artificiales"},
        {"Variable": "price (target)", "Estrategia": "Eliminación de fila", "Motivo": "Sin target no hay entrenamiento"},
    ])
    st.dataframe(tratamiento, use_container_width=True, hide_index=True)

    # División hold-out 80/20 con random_state=42.
    st.markdown('<div class="section-title">División de datos</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de pastel que visualiza la proporción 80/20 del split
        fig = px.pie(
            values=[80, 20],
            names=["Entrenamiento (80%)", "Validación (20%)"],
            color_discrete_sequence=["#0d9488", "#e8e0d5"],
            title="Split Train / Test",
        )
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Cálculo dinámico del nº de filas reales en cada partición
        n_train = int(len(df) * 0.8)
        n_test  = len(df) - n_train
        st.markdown("**Parámetros del split:**")
        st.markdown("- Método: `train_test_split` (scikit-learn)")
        st.markdown("- `random_state = 42`")
        st.markdown("- `test_size = 0.2`")
        st.markdown(f"- Filas entrenamiento: **{n_train:,}**".replace(",", "."))
        st.markdown(f"- Filas validación: **{n_test:,}**".replace(",", "."))

    # Random Forest usa además 10-fold CV sobre train para un R² más robusto.
    st.info(
        "**Random Forest** utiliza adicionalmente **validación cruzada de 10 folds** (10-fold cross-validation) "
        "sobre el conjunto de entrenamiento para obtener una estimación más robusta del R². "
        "El conjunto de test (20%) se reserva y no se toca hasta la evaluación final, "
        "evitando así cualquier fuga de datos (*data leakage*).",
        icon="ℹ️"
    )

    # Diagrama visual del flujo de preparación de datos.
    st.markdown('<div class="section-title">Pipeline de preprocesamiento</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .pl-wrap { font-family: "Segoe UI", sans-serif; padding: 0.5rem 0 1rem 0; }
    .pl-step { display: flex; align-items: flex-start; gap: 1.2rem; }
    .pl-box {
        border-radius: 8px; padding: 0.45rem 1rem; font-weight: 700;
        font-size: 0.88rem; min-width: 210px; max-width: 210px;
        text-align: center; border: 2px solid; white-space: nowrap;
        line-height: 1.4;
    }
    .pl-desc {
        display: flex; flex-wrap: wrap; align-items: center;
        gap: 0.3rem; padding-top: 0.4rem;
    }
    .pl-tag {
        display: inline-block; border-radius: 5px; padding: 0.15rem 0.55rem;
        font-size: 0.78rem; font-weight: 500; white-space: nowrap;
    }
    .pl-arrow { margin-left: 105px; color: #94a3b8; line-height: 0.9;
                font-size: 1.05rem; padding: 1px 0; }
    /* colores por tipo */
    .c-src  { background:#dbeafe; border-color:#3b82f6; color:#1d4ed8; }
    .c-load { background:#f1f5f9; border-color:#94a3b8; color:#334155; }
    .c-cln  { background:#fff7ed; border-color:#f97316; color:#c2410c; }
    .c-eng  { background:#faf5ff; border-color:#a855f7; color:#7e22ce; }
    .c-ml   { background:#f0fdfa; border-color:#0d9488; color:#0f766e; }
    .t-cln  { background:#fff7ed; color:#c2410c; border:1px solid #fdba74; }
    .t-eng  { background:#faf5ff; color:#7e22ce; border:1px solid #d8b4fe; }
    .t-ml   { background:#f0fdfa; color:#0f766e; border:1px solid #5eead4; }
    .t-gray { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }
    </style>

    <div class="pl-wrap">

      <div class="pl-step">
        <div class="pl-box c-src">📁 dataset_corregido.xlsx</div>
        <div class="pl-desc">
          <span class="pl-tag t-gray">Fuente de datos original</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-load">load_data()</div>
        <div class="pl-desc">
          <span class="pl-tag t-gray">Lee el Excel con pandas</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-cln">clean_data()</div>
        <div class="pl-desc">
          <span class="pl-tag t-cln">Agrupa property_type → 3 grupos</span>
          <span class="pl-tag t-cln">Elimina outliers precio (p99)</span>
          <span class="pl-tag t-cln">Filtra size [10–5.000 m²]</span>
          <span class="pl-tag t-cln">Elimina nb_bedrooms &gt; nb_rooms</span>
          <span class="pl-tag t-cln">Excluye columnas con &gt;50% nulos</span>
          <span class="pl-tag t-cln">Elimina filas con nulos restantes</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-eng">feature_engineering()</div>
        <div class="pl-desc">
          <span class="pl-tag t-eng">log_size = log(1 + size)</span>
          <span class="pl-tag t-eng">log1p(price) como target</span>
          <span class="pl-tag t-eng">Departamento ← código postal</span>
          <span class="pl-tag t-eng">has_energy_cert (0/1)</span>
          <span class="pl-tag t-eng">Distancias haversine a 8 ciudades</span>
          <span class="pl-tag t-eng">dist_min_ciudad</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-load">compute_price_per_m2()</div>
        <div class="pl-desc">
          <span class="pl-tag t-gray">Añade columna €/m²</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-ml">get_features_and_target()</div>
        <div class="pl-desc">
          <span class="pl-tag t-ml">Separa X (features) e y (log1p_price)</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-ml">train_test_split()</div>
        <div class="pl-desc">
          <span class="pl-tag t-ml">80% entrenamiento · 20% test</span>
          <span class="pl-tag t-ml">random_state = 42</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-ml">cross_val_score()</div>
        <div class="pl-desc">
          <span class="pl-tag t-ml">10-fold CV sobre train</span>
          <span class="pl-tag t-gray">solo Random Forest</span>
        </div>
      </div>
      <div class="pl-arrow">│<br>▼</div>

      <div class="pl-step">
        <div class="pl-box c-ml">🤖 Modelo</div>
        <div class="pl-desc">
          <span class="pl-tag t-ml">Predicciones invertidas con expm1()</span>
          <span class="pl-tag t-ml">→ precio final en €</span>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

# ── TAB 3 · MODELOS ───────────────────────────────────────────────────────────
with tab3:
    # Carga estado de modelos: ✅ pkl+métricas · ⚠️ solo pkl · ⏳ sin entrenar
    model_status = get_model_status()
    available    = get_available_models()
    demo_mode    = len(available) == 0

    st.markdown('<div class="section-title">Estado de los modelos</div>', unsafe_allow_html=True)
    MODELOS_DISPLAY = ["MLP", "Random Forest", "XGBoost"]
    cols = st.columns(3)
    for i, name in enumerate(MODELOS_DISPLAY):
        status = model_status.get(name, {"pkl": False, "metrics": False})
        with cols[i]:
            if status["pkl"] and status["metrics"]:
                st.metric(name, "✅ Completo")
            elif status["pkl"]:
                st.metric(name, "⚠️ Sin métricas")
            else:
                st.metric(name, "⏳ Pendiente")

    # Métricas sobre el conjunto de test (20%). "Demo" si no hay archivo real.
    st.markdown('<div class="section-title">Métricas de evaluación</div>', unsafe_allow_html=True)

    metrics_data = []
    for name in MODELOS_DISPLAY:
        m = get_metrics(name)
        metrics_data.append({
            "Modelo": name,
            "RMSE (€)": f"{m['RMSE']:,.0f}".replace(",", "."),
            "MAE (€)":  f"{m['MAE']:,.0f}".replace(",", "."),
            "R²":       m["R2"],
            "Fuente":   "Real" if (OUTPUTS_PATH / METRICS_FILES.get(name, "")).exists() else "Demo",
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # Comparativa de R² entre modelos
    st.markdown('<div class="section-title">Comparación de R²</div>', unsafe_allow_html=True)

    r2_data = pd.DataFrame([
        {"Modelo": name, "R²": get_metrics(name)["R2"]}
        for name in MODELOS_DISPLAY
    ])
    fig = px.bar(
        r2_data, x="Modelo", y="R²",
        title="R² por modelo (mayor es mejor · máximo = 1.0)",
        color="Modelo",
        color_discrete_sequence=["#5eead4", "#0d9488", "#0f766e"],
        text="R²",
    )
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textfont=dict(size=20, color="white", family="Segoe UI, sans-serif", weight="bold"),
        textposition="inside",
    )
    fig.update_layout(
        yaxis_range=[0, 1],
        plot_bgcolor="#f5f0eb",
        paper_bgcolor="white",
        showlegend=False,
        title_font=dict(size=14, family="Segoe UI, sans-serif"),
        font=dict(family="Segoe UI, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter precio real vs predicho. Cuanto más cerca de la línea roja, mejor.
    st.markdown('<div class="section-title">Predicción vs Realidad</div>', unsafe_allow_html=True)

    model_sel = st.selectbox("Selecciona modelo", MODELOS_DISPLAY)
    subset = get_predictions(model_sel)
    is_real = (OUTPUTS_PATH / PREDICTIONS_FILES.get(model_sel, "")).exists()

    if not is_real:
        st.caption("*Gráfico generado con datos ficticios. Ejecuta los scripts de entrenamiento para ver predicciones reales.*")

    fig = go.Figure()
    # Puntos del conjunto de test
    fig.add_trace(go.Scatter(
        x=subset["y_real"], y=subset["y_pred"],
        mode="markers",
        marker=dict(color="#0d9488", opacity=0.6, size=5),
        name="Predicciones",
    ))
    # Línea de predicción perfecta (y = x)
    max_val = max(subset["y_real"].max(), subset["y_pred"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=2),
        name="Predicción perfecta",
    ))
    fig.update_layout(
        title=f"Predicción vs Realidad — {model_sel} {'(real)' if is_real else '(demo)'}",
        xaxis_title="Precio real (€)",
        yaxis_title="Precio predicho (€)",
        plot_bgcolor="#f5f0eb",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    
# ── TAB 4 · ANOMALÍAS ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Análisis de anomalías por zona</div>', unsafe_allow_html=True)

    st.info(
        "Esta sección clasifica cada departamento como **tendencial** o **atípico** "
        "según si el diferencial entre precio real y predicho por XGBoost supera el intervalo de confianza del 95%.",
        icon="ℹ️"
    )

    # Carga predicciones reales de XGBoost; si no existen, usa datos simulados.
    pred_path = OUTPUTS_PATH / "predictions_xgb.csv"
    if pred_path.exists() and "dept" in pd.read_csv(pred_path, nrows=1).columns:
        preds_df = pd.read_csv(pred_path)
        # Agrega por departamento: mediana real, mediana predicha, n viviendas
        dept_stats = (
            preds_df.groupby("dept")
            .agg(
                precio_real_mediano=("y_real", "median"),
                precio_predicho_mediano=("y_pred", "median"),
                n_viviendas=("y_real", "count"),
            )
            .reset_index()
        )
        dept_stats["dept"] = dept_stats["dept"].astype(str).str.zfill(2)
        dept_stats["nombre"] = dept_stats["dept"].map(DEPT_NOMBRES).fillna(dept_stats["dept"])
        dept_stats["diferencial"] = dept_stats["precio_real_mediano"] - dept_stats["precio_predicho_mediano"]
        dept_stats["diferencial_pct"] = (dept_stats["diferencial"] / dept_stats["precio_predicho_mediano"] * 100).round(1)
        is_real = True
    else:
        # ── Fallback: simulado por ciudad ──
        dept_stats_raw = (
            df.groupby("provincia")
            .agg(
                precio_real_mediano=(TARGET, "median"),
                n_viviendas=(TARGET, "count"),
            )
            .reset_index()
            .rename(columns={"provincia": "dept"})
        )
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.12, size=len(dept_stats_raw))
        dept_stats_raw["precio_predicho_mediano"] = dept_stats_raw["precio_real_mediano"] * (1 + noise)
        dept_stats_raw["diferencial"] = dept_stats_raw["precio_real_mediano"] - dept_stats_raw["precio_predicho_mediano"]
        dept_stats_raw["diferencial_pct"] = (dept_stats_raw["diferencial"] / dept_stats_raw["precio_predicho_mediano"] * 100).round(1)
        dept_stats_raw["nombre"] = dept_stats_raw["dept"].map(DEPT_NOMBRES).fillna(dept_stats_raw["dept"])
        dept_stats = dept_stats_raw
        is_real = False

    mean_diff = dept_stats["diferencial_pct"].mean()
    std_diff  = dept_stats["diferencial_pct"].std()
    umbral    = 1.96 * std_diff

    dept_stats["clasificacion"] = dept_stats["diferencial_pct"].apply(
        lambda x: "Atípico" if abs(x - mean_diff) > umbral else "Tendencial"
    )

    n_atipicos     = (dept_stats["clasificacion"] == "Atípico").sum()
    n_tendenciales = (dept_stats["clasificacion"] == "Tendencial").sum()

    if not is_real:
        st.caption("*Predicciones simuladas — se actualizarán cuando se suban los datos reales.*")

    col1, col2, col3 = st.columns(3)
    col1.metric("Departamentos analizados", len(dept_stats))
    col2.metric("Tendenciales", n_tendenciales)
    col3.metric("Atípicos", f"⚠️ {n_atipicos}")

    st.markdown('<div class="section-title">Diferencial por departamento (%)</div>', unsafe_allow_html=True)

    fig = px.bar(
        dept_stats.sort_values("diferencial_pct"),
        x="diferencial_pct", y="nombre",
        color="clasificacion",
        color_discrete_map={"Tendencial": "#0d9488", "Atípico": "#e74c3c"},
        orientation="h",
        title=f"Diferencial precio real vs predicho (XGBoost) — por departamento",
        labels={"diferencial_pct": "Diferencial (%)", "nombre": "Departamento", "clasificacion": "Clasificación"},
        hover_data={"n_viviendas": True, "precio_real_mediano": ":,.0f", "precio_predicho_mediano": ":,.0f"},
    )
    fig.add_vline(x=mean_diff + umbral, line_dash="dash", line_color="#818cf8", annotation_text="IC 95%+")
    fig.add_vline(x=mean_diff - umbral, line_dash="dash", line_color="#818cf8", annotation_text="IC 95%-")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Departamentos atípicos detectados</div>', unsafe_allow_html=True)

    atipicos = dept_stats[dept_stats["clasificacion"] == "Atípico"].sort_values(
        "diferencial_pct", key=abs, ascending=False
    )[["nombre", "precio_real_mediano", "precio_predicho_mediano", "diferencial_pct", "n_viviendas"]]
    atipicos.columns = ["Departamento", "Precio real mediano (€)", "Precio predicho (XGBoost) (€)", "Diferencial (%)", "N viviendas"]
    st.dataframe(atipicos, use_container_width=True, hide_index=True)
    
# ── TAB 5 · PREDICTOR DE PRECIO ───────────────────────────────────────────────

with tab5:
    st.markdown('<div class="section-title">Predictor de precio de vivienda</div>', unsafe_allow_html=True)

    available_models = get_available_models()
    has_model = len(available_models) > 0

    if not has_model:
        st.info("No hay modelos entrenados disponibles. Las predicciones se basaran en la mediana del dataset filtrado por zona y tipo.", icon="ℹ️")

    st.markdown("Introduce las caracteristicas de la vivienda para obtener una estimacion del precio.")

    # Descripciones de cada modelo para mostrar al usuario
    MODEL_DESCRIPTIONS = {
        "XGBoost":       ("XGBoost — árboles de decisión encadenados",        "Combina cientos de árboles simples aprendiendo de sus errores. Muy preciso con datos tabulares."),
        "Random Forest": ("Random Forest — bosque de árboles independientes",  "Entrena muchos árboles en paralelo y promedia sus predicciones. Robusto y estable."),
        "MLP":           ("MLP — red neuronal multicapa",                       "Red neuronal artificial que aprende patrones complejos. Flexible pero requiere más datos."),
        "Estimación estadística": ("Estimación estadística — mediana del mercado", "Sin IA: usa la mediana de precios de viviendas similares en la zona. Referencia básica."),
    }

    # Ordena modelos por R² descendente; estimación estadística siempre al final
    def get_r2(name):
        if name == "Estimación estadística":
            return -1
        return get_metrics(name).get("R2", 0)

    sorted_models = sorted(available_models, key=get_r2, reverse=True)
    all_options = sorted_models + ["Estimación estadística"]

    def format_model(name):
        if name == "Estimación estadística":
            return "📊 Estimación estadística (sin IA)"
        r2 = get_metrics(name).get("R2", 0)
        icons = {"XGBoost": "🥇", "Random Forest": "🥈", "MLP": "🥉"}
        return f"{icons.get(name, '🤖')} {name} — precisión {r2:.0%}"

    modelo_elegido = st.selectbox(
        "Modelo de predicción",
        options=all_options,
        format_func=format_model,
        index=0,
    )

    # ── Mostrar descripción del modelo seleccionado ───────────────────────────
    if modelo_elegido in MODEL_DESCRIPTIONS:
        title, desc = MODEL_DESCRIPTIONS[modelo_elegido]
        st.caption(f"ℹ️ **{title}** — {desc}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ubicacion**")
        # Departamentos prioritarios (grandes ciudades) aparecen primero
        PRIORITY_DEPTS = ["75", "69", "13", "33", "06"]  # París, Lyon, Marsella, Burdeos, Niza
        all_depts = sorted(df["provincia"].dropna().unique().tolist())
        other_depts = [p for p in all_depts if p not in PRIORITY_DEPTS]
        provincia_options = ["ALL"] + [p for p in PRIORITY_DEPTS if p in all_depts] + other_depts
        
        def format_dept(x):
            if x == "ALL":
                return "🇫🇷 Toda Francia"
            name = DEPT_NOMBRES.get(x, x)
            if x in PRIORITY_DEPTS:
                icons = {"75": "🗼", "69": "🦁", "13": "⛵", "33": "🍷", "06": "🌊"}
                return f"{icons.get(x, '⭐')} {name} ({x})"
            return f"{name} ({x})"
        
        provincia_sel = st.selectbox(
            "Departamento (código = nº del departamento francés)",
            options=provincia_options,
            format_func=format_dept,
        )
        tipo_sel = st.selectbox(
            "Tipo de propiedad",
            options=["apartamento", "casa", "terreno", "local_comercial", "otro"],
            format_func=lambda x: {
                "apartamento": "🏢 Apartamento (piso, dúplex, loft...)",
                "casa": "🏘️ Casa (maison, villa, chalet...)",
                "terreno": "🌿 Terreno",
                "local_comercial": "🏪 Local comercial / Parking",
                "otro": "📦 Otro (viager, propriété mixta...)",
            }.get(x, x),
        )

    with col2:
        st.markdown("**Características físicas**")
        size_sel = st.number_input("Superficie (m²)", min_value=10, max_value=1000, value=80, step=5)
        
        # ── Ocultar habitaciones para terreno/local/otro ──────────────────────
        if tipo_sel in ["terreno", "local_comercial", "otro"]:
            rooms_sel     = 0
            bedrooms_sel  = 0
            bathrooms_sel = 0
            st.info("Para este tipo de propiedad no aplican habitaciones ni baños.")
        else:
            rooms_sel     = st.number_input("Nº total de piezas (habitaciones + salón)", min_value=1, max_value=20, value=3,
                                            help="Cuenta todas las piezas habitables: salón, dormitorios, etc.")
            bedrooms_sel  = st.number_input("Nº de dormitorios", min_value=0, max_value=15, value=2,
                                            help="Solo dormitorios. 0 = estudio/loft.")
            bathrooms_sel = st.number_input("Nº de baños/aseos", min_value=1, max_value=10, value=1,
                                            help="Incluye baños completos y aseos.")

    with col3:
        st.markdown("**Extras**")
        parking_sel = st.number_input("Plazas de parking", min_value=0, max_value=5, value=0)
        balcony_sel = st.checkbox("Tiene balcón", value=False)
        cellar_sel  = st.checkbox("Tiene sótano", value=False)
        garage_sel  = st.checkbox("Tiene garaje", value=False)
        ac_sel      = st.checkbox("Tiene aire acondicionado", value=False)
        energy_sel  = st.selectbox("Clase energética", options=["No importa", "A", "B", "C", "D", "E", "F", "G"], index=0,
                                   help="A = más eficiente, G = menos eficiente. 'No importa' usa valor medio.")

    # ── Resumen de lo seleccionado ────────────────────────────────────────────
    extras = []
    if balcony_sel: extras.append("balcón")
    if cellar_sel:  extras.append("sótano")
    if garage_sel:  extras.append("garaje")
    if ac_sel:      extras.append("A/C")
    if parking_sel: extras.append(f"{parking_sel} parking")
    extras_txt = " · ".join(extras) if extras else "sin extras"
    energy_txt = f"clase {energy_sel}" if energy_sel != "No importa" else "clase energética indiferente"
    tipo_icons2 = {"apartamento": "🏢", "casa": "🏘️", "terreno": "🌿", "local_comercial": "🏪", "otro": "📦"}
    dept_display = format_dept(provincia_sel)
    
    if tipo_sel not in ["terreno", "local_comercial"]:
        st.info(f"{tipo_icons2.get(tipo_sel,'')} **{tipo_sel}** · {size_sel} m² · {rooms_sel} piezas · {bedrooms_sel} dorm. · {bathrooms_sel} baños · {extras_txt} · {energy_txt} · 📍 {dept_display}")
    else:
        st.info(f"{tipo_icons2.get(tipo_sel,'')} **{tipo_sel}** · {size_sel} m² · {extras_txt} · 📍 {dept_display}")

    st.divider()

    if st.button("Calcular precio estimado", type="primary", use_container_width=True):

        dept       = provincia_sel
        if dept == "ALL":
            dept_lat = df["approximate_latitude"].median()
            dept_lon = df["approximate_longitude"].median()
            dept     = df["provincia"].mode()[0]  # departamento más frecuente como fallback
        else:
            dept_lat = df[df["provincia"] == dept]["approximate_latitude"].median()
            dept_lon = df[df["provincia"] == dept]["approximate_longitude"].median()
        tipo_to_group = {"apartamento": "appartement", "casa": "maison", "lujo": "lujo"}
        property_group = tipo_to_group.get(tipo_sel, "appartement")

        input_dict = {
            "dept":                 dept,
            "lat":                  dept_lat,
            "lon":                  dept_lon,
            "size":                 size_sel,
            "nb_rooms":             rooms_sel,
            "nb_bedrooms":          bedrooms_sel,
            "nb_bathrooms":         bathrooms_sel,
            "nb_parking_places":    parking_sel,
            "has_a_balcony":        int(balcony_sel),
            "has_a_cellar":         int(cellar_sel),
            "has_a_garage":         int(garage_sel),
            "has_air_conditioning": int(ac_sel),
            "has_energy_cert":      0 if energy_sel == "No importa" else 1,
            "has_ghg_value":        0,
            "property_group":       property_group,
        }

        precio_pred = None
        fuente      = None
        modelo_usar = modelo_elegido if modelo_elegido != "Estimación estadística" else None

        if modelo_usar:
            X_input, arts = build_input_for_model(modelo_usar, input_dict, df)

            if X_input is not None:
                modelo = load_model(modelo_usar)
                try:
                    if modelo_usar == "MLP" and arts is not None:
                        y_sc        = modelo.predict(arts["scaler_X"].transform(X_input))
                        y_log       = arts["scaler_y"].inverse_transform(y_sc.reshape(-1, 1)).ravel()
                        precio_pred = float(np.expm1(y_log)[0])
                    else:
                        y_log       = modelo.predict(X_input)
                        precio_pred = float(np.expm1(y_log)[0])
                    fuente = f"Modelo: {modelo_usar}"
                except Exception as e:
                    st.warning(f"Error en predicción del modelo: {e}. Usando estimación estadística.")

        if precio_pred is None:
            mask        = (df["provincia"] == provincia_sel) & (df["property_type_group"] == tipo_sel)
            subset_pred = df[mask] if len(df[mask]) >= 5 else df[df["provincia"] == provincia_sel]
            subset_pred = subset_pred if len(subset_pred) >= 5 else df
            precio_pred = float(subset_pred["price_per_m2"].median() * size_sel)
            fuente      = "Estimación estadística (mediana)"

        precio_m2  = precio_pred / size_sel
        dept_name  = "Toda Francia" if provincia_sel == "ALL" else DEPT_NOMBRES.get(provincia_sel, provincia_sel)

        # Viviendas similares: mismo tipo, mismo dpto, superficie ±20%
        size_min = size_sel * 0.8
        size_max = size_sel * 1.2

        dept_filter = (df["provincia"] == provincia_sel) if provincia_sel != "ALL" else pd.Series([True] * len(df), index=df.index)

        similares = df[
            dept_filter &
            (df["property_type_group"] == tipo_sel) &
            (df["size"] >= size_min) &
            (df["size"] <= size_max)
        ]
        if len(similares) < 5:
            similares = df[dept_filter & (df["property_type_group"] == tipo_sel)]
        if len(similares) < 5:
            similares = df[dept_filter]

        zona_stats = similares["price"].describe()
        n_similares = len(similares)

        st.success("Estimación calculada")
        r1, r2, r3 = st.columns(3)
        r1.metric("Precio estimado",  f"{int(precio_pred):,} €".replace(",", "."))
        r2.metric("Precio por m²",    f"{int(precio_m2):,} €/m²".replace(",", "."))
        r3.metric("Superficie",       f"{size_sel} m²")
        st.caption(f"Fuente: {fuente}")

        tipo_label = {"apartamento": "apartamentos", "casa": "casas", "terreno": "terrenos",
                      "local_comercial": "locales", "otro": "otros"}.get(tipo_sel, tipo_sel)
        st.markdown(f"**Contexto: {tipo_label} de {int(size_min)}–{int(size_max)} m² en {dept_name}** *(basado en {n_similares} viviendas similares)*")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio mínimo",  f"{int(zona_stats['min']):,} €".replace(",", "."))
        c2.metric("Precio mediano", f"{int(zona_stats['50%']):,} €".replace(",", "."))
        c3.metric("Precio medio",   f"{int(zona_stats['mean']):,} €".replace(",", "."))
        c4.metric("Precio máximo",  f"{int(zona_stats['max']):,} €".replace(",", "."))

        pct = (precio_pred - zona_stats["min"]) / (zona_stats["max"] - zona_stats["min"]) * 100
        pct = max(0, min(100, pct))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": f"Posición relativa entre viviendas similares en {dept_name}"},
            gauge={
                "axis":  {"range": [0, 100], "ticksuffix": "%"},
                "bar":   {"color": "#000000"},
                "steps": [
                    {"range": [0,  33],  "color": "#2ca02c"},
                    {"range": [33, 66],  "color": "#ff7f0e"},
                    {"range": [66, 100], "color": "#d62728"},
                ],
            },
            number={"suffix": "%", "font": {"size": 28}},
        ))
        fig.update_layout(paper_bgcolor="white", height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"0% = más barato · 100% = más caro entre {tipo_label} similares en {dept_name}")

        # Distancias al centroide del departamento seleccionado
        def haversine_simple(lat1, lon1, lat2, lon2):
            R = 6371
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            return round(R * 2 * math.asin(math.sqrt(a)))

        CIUDADES_DIST = {
            "🗼 París":    (48.8566,  2.3522),
            "🦁 Lyon":     (45.7640,  4.8357),
            "⛵ Marsella": (43.2965,  5.3698),
            "🍷 Burdeos":  (44.8378, -0.5792),
            "🌊 Niza":     (43.7102,  7.2620),
        }
        dists_display = {ciudad: haversine_simple(dept_lat, dept_lon, clat, clon) 
                        for ciudad, (clat, clon) in CIUDADES_DIST.items()}
        
        st.markdown("**🗺️ Distancias desde este departamento:**")
        dist_cols = st.columns(5)
        for i, (ciudad, km) in enumerate(dists_display.items()):
            dist_cols[i].metric(ciudad, f"{km} km")

        # Mini mapa con la ubicación del departamento
        st.markdown(f"**📍 Ubicación: {dept_name}**")
        dept_df = pd.DataFrame([{"lat": dept_lat, "lon": dept_lon, "name": dept_name}])
        fig_map = px.scatter_mapbox(
            dept_df, lat="lat", lon="lon",
            hover_name="name",
            zoom=6,
            center={"lat": dept_lat, "lon": dept_lon},
            mapbox_style="carto-positron",
            height=300,
            size_max=20,
            color_discrete_sequence=["#e74c3c"],
        )
        fig_map.update_traces(marker=dict(size=15))
        fig_map.update_layout(
            paper_bgcolor="white",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        st.plotly_chart(fig_map, use_container_width=True)
