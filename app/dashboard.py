import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.preprocessing import load_and_prepare, NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET
from src.model_loader import get_model_status, get_available_models, load_model, get_metrics, get_demo_predictions, DEMO_METRICS

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis del Precio de la Vivienda en Francia",
    page_icon="🏠",
    layout="wide",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 4px solid #1a1a1a;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos (cacheada) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset...")
def get_data():
    return load_and_prepare()

df = get_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 Entre Tendencias y Anomalías</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis del precio de la vivienda en Francia · UPV · 2026</div>', unsafe_allow_html=True)

# ── Métricas globales ──────────────────────────────────────────────────────────
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

# ── Pestañas ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploración de datos",
    "⚙️ Preprocesamiento",
    "🤖 Modelos",
    "🔍 Anomalías por zona",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Filtro por tipo de propiedad ──────────────────────────────────────────
    tipos_disponibles = sorted(df["property_type_group"].unique().tolist())
    tipos_sel = st.multiselect(
        "Filtrar por tipo de propiedad",
        options=tipos_disponibles,
        default=["apartamento", "casa"],
    )
    df_filtrado = df[df["property_type_group"].isin(tipos_sel)] if tipos_sel else df
    st.caption(f"Mostrando {len(df_filtrado):,} viviendas".replace(",", "."))

    st.markdown('<div class="section-title">Distribución de precios</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_filtrado, x=TARGET, nbins=80,
            title="Distribución del precio (€)",
            labels={TARGET: "Precio (€)"},
            color_discrete_sequence=["#1a1a1a"],
        )
        fig.update_layout(bargap=0.05, plot_bgcolor="#f5f0eb", paper_bgcolor="#f5f0eb")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_filtrado, x="property_type_group", y=TARGET,
            title="Precio por tipo de propiedad",
            labels={TARGET: "Precio (€)", "property_type_group": "Tipo"},
            color_discrete_sequence=["#1a1a1a"],
        )
        fig.update_layout(xaxis_tickangle=-35, plot_bgcolor="#f5f0eb", paper_bgcolor="#f5f0eb")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Precio por ciudad (top 20)</div>', unsafe_allow_html=True)

    top_cities = (
        df_filtrado.groupby("city")[TARGET]
        .median()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    fig = px.bar(
        top_cities, x="city", y=TARGET,
        title="Precio mediano por ciudad — Top 20",
        labels={TARGET: "Precio mediano (€)", "city": "Ciudad"},
        color=TARGET,
        color_continuous_scale=["#e8e0d5", "#1a1a1a"],
    )
    fig.update_layout(plot_bgcolor="#f5f0eb", paper_bgcolor="#f5f0eb", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # ── Mapa de Francia ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Mapa de precios por ubicación</div>', unsafe_allow_html=True)

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
        color_continuous_scale="RdYlGn_r",  # verde=barato, amarillo=medio, rojo=caro
        range_color=[mapa_df["precio_mediano"].quantile(0.05), mapa_df["precio_mediano"].quantile(0.95)],
        hover_data={"precio_mediano": ":,.0f", "precio_m2": ":,.0f", "n": True, "lat": False, "lon": False},
        labels={"precio_mediano": "Precio mediano (€)", "precio_m2": "€/m²", "n": "Viviendas"},
        title="Distribución geográfica del precio mediano — 🟢 barato · 🟡 medio · 🔴 caro",
        mapbox_style="carto-positron",
        height=600,
        opacity=0.85,
    )
    fig.update_layout(
        paper_bgcolor="#f5f0eb",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Precio (€)",
            tickformat=",.0f",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlación entre variables numéricas</div>', unsafe_allow_html=True)

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns] + [TARGET]
    corr = df[num_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#f5f0eb", "#1a1a1a"],
        title="Matriz de correlación",
        aspect="auto",
    )
    fig.update_layout(paper_bgcolor="#f5f0eb")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Valores nulos por columna</div>', unsafe_allow_html=True)

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
            color_continuous_scale=["#e8e0d5", "#e74c3c"],
            text="porcentaje",
        )
        fig.update_traces(texttemplate="%{text}%")
        fig.update_layout(plot_bgcolor="#f5f0eb", paper_bgcolor="#f5f0eb")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Precio vs Superficie</div>', unsafe_allow_html=True)

    sample = df_filtrado.sample(min(3000, len(df_filtrado)), random_state=42)
    fig = px.scatter(
        sample, x="size", y=TARGET,
        color="property_type_group",
        title="Precio vs Superficie (muestra de hasta 3.000 viviendas)",
        labels={"size": "Superficie (m²)", TARGET: "Precio (€)", "property_type_group": "Tipo"},
        opacity=0.5,
    )
    fig.update_layout(plot_bgcolor="#f5f0eb", paper_bgcolor="#f5f0eb")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
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

    st.markdown('<div class="section-title">Tratamiento de valores nulos</div>', unsafe_allow_html=True)

    tratamiento = pd.DataFrame([
        {"Variable": "Numéricas (size, nb_rooms…)", "Estrategia": "Imputación con mediana", "Motivo": "Robusta frente a outliers"},
        {"Variable": "Categóricas (property_type…)", "Estrategia": "Imputación con moda", "Motivo": "Valor más frecuente"},
        {"Variable": "floor, land_size, exposition", "Estrategia": "Excluidas del modelo", "Motivo": ">70% de valores nulos"},
        {"Variable": "price (target)", "Estrategia": "Eliminación de fila", "Motivo": "Sin target no hay entrenamiento"},
    ])
    st.dataframe(tratamiento, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">División de datos</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=[80, 20],
            names=["Entrenamiento (80%)", "Validación (20%)"],
            color_discrete_sequence=["#1a1a1a", "#e8e0d5"],
            title="Split Train / Test",
        )
        fig.update_layout(paper_bgcolor="#f5f0eb")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        n_train = int(len(df) * 0.8)
        n_test  = len(df) - n_train
        st.markdown("**Parámetros del split:**")
        st.markdown(f"- Método: `train_test_split` (scikit-learn)")
        st.markdown(f"- `random_state = 42`")
        st.markdown(f"- `test_size = 0.2`")
        st.markdown(f"- Filas entrenamiento: **{n_train:,}**".replace(",", "."))
        st.markdown(f"- Filas validación: **{n_test:,}**".replace(",", "."))

    st.markdown('<div class="section-title">Pipeline de preprocesamiento</div>', unsafe_allow_html=True)

    st.markdown("""
    ```
    dataset_corregido.xlsx
            │
            ▼
    load_data()          ← Lee el Excel con pandas
            │
            ▼
    clean_data()         ← Elimina duplicados · Imputa nulos · Elimina filas sin precio
            │
            ▼
    compute_price_per_m2()  ← Añade columna €/m²
            │
            ▼
    get_features_and_target()  ← Separa X (features) e y (price)
            │
            ▼
    train_test_split()   ← 80% train / 20% test
            │
            ▼
        Modelo
    ```
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    model_status = get_model_status()
    available    = get_available_models()
    demo_mode    = len(available) == 0

    if demo_mode:
        st.info("⚠️ No se han encontrado modelos entrenados en `models/`. Mostrando datos de demostración.", icon="ℹ️")
    else:
        st.success(f"✅ Modelos cargados: {', '.join(available)}")

    st.markdown('<div class="section-title">Estado de los modelos</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, loaded) in enumerate(model_status.items()):
        with cols[i]:
            status = "✅ Listo" if loaded else "⏳ Pendiente"
            st.metric(name, status)

    st.markdown('<div class="section-title">Métricas de evaluación</div>', unsafe_allow_html=True)

    metrics_data = []
    for name in ["Regresión Lineal", "Random Forest", "MLP"]:
        m = get_metrics(name)
        metrics_data.append({
            "Modelo": name,
            "RMSE (€)": f"{m['RMSE']:,}".replace(",", "."),
            "MAE (€)":  f"{m['MAE']:,}".replace(",", "."),
            "R²":       m["R2"],
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    if demo_mode:
        st.caption("*Valores de demostración — se actualizarán automáticamente cuando se añadan los modelos entrenados.*")

    st.markdown('<div class="section-title">Comparación de R²</div>', unsafe_allow_html=True)

    r2_data = pd.DataFrame([
        {"Modelo": name, "R²": get_metrics(name)["R2"]}
        for name in ["Regresión Lineal", "Random Forest", "MLP"]
    ])
    fig = px.bar(
        r2_data, x="Modelo", y="R²",
        title="R² por modelo (mayor es mejor · máximo = 1.0)",
        color="Modelo",
        color_discrete_sequence=["#888", "#1a1a1a", "#555"],
        text="R²",
    )
    fig.update_traces(texttemplate="%{text:.2f}")
    fig.update_layout(
        yaxis_range=[0, 1],
        plot_bgcolor="#f5f0eb",
        paper_bgcolor="#f5f0eb",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Predicción vs Realidad</div>', unsafe_allow_html=True)

    pred_df = get_demo_predictions()
    model_sel = st.selectbox("Selecciona modelo", ["Regresión Lineal", "Random Forest", "MLP"])
    subset = pred_df[pred_df["modelo"] == model_sel]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset["y_real"], y=subset["y_pred"],
        mode="markers",
        marker=dict(color="#1a1a1a", opacity=0.5, size=5),
        name="Predicciones",
    ))
    max_val = max(subset["y_real"].max(), subset["y_pred"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=2),
        name="Predicción perfecta",
    ))
    fig.update_layout(
        title=f"Predicción vs Realidad — {model_sel}",
        xaxis_title="Precio real (€)",
        yaxis_title="Precio predicho (€)",
        plot_bgcolor="#f5f0eb",
        paper_bgcolor="#f5f0eb",
    )
    st.plotly_chart(fig, use_container_width=True)

    if demo_mode:
        st.caption("*Gráfico generado con datos ficticios. Se reemplazará con predicciones reales al añadir los modelos.*")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · ANOMALÍAS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Análisis de anomalías por zona</div>', unsafe_allow_html=True)

    st.info(
        "Esta sección clasifica cada zona como **tendencial** o **atípica** "
        "según si el diferencial entre precio real y predicho supera el intervalo de confianza del 95%.",
        icon="ℹ️"
    )

    city_stats = (
        df.groupby("city")
        .agg(
            precio_real_mediano=(TARGET, "median"),
            precio_m2_mediano=("price_per_m2", "median"),
            n_viviendas=(TARGET, "count"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.12, size=len(city_stats))
    city_stats["precio_predicho_mediano"] = city_stats["precio_real_mediano"] * (1 + noise)
    city_stats["diferencial"] = city_stats["precio_real_mediano"] - city_stats["precio_predicho_mediano"]
    city_stats["diferencial_pct"] = (city_stats["diferencial"] / city_stats["precio_predicho_mediano"] * 100).round(1)

    mean_diff = city_stats["diferencial_pct"].mean()
    std_diff  = city_stats["diferencial_pct"].std()
    umbral    = 1.96 * std_diff

    city_stats["clasificacion"] = city_stats["diferencial_pct"].apply(
        lambda x: "Atípica" if abs(x - mean_diff) > umbral else "Tendencial"
    )

    n_atipicas     = (city_stats["clasificacion"] == "Atípica").sum()
    n_tendenciales = (city_stats["clasificacion"] == "Tendencial").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Zonas analizadas", len(city_stats))
    col2.metric("Zonas tendenciales", n_tendenciales)
    col3.metric("Zonas atípicas", f"⚠️ {n_atipicas}")

    st.markdown('<div class="section-title">Diferencial por zona (%)</div>', unsafe_allow_html=True)

    top_n = st.slider("Mostrar top N ciudades por número de viviendas", 10, 80, 30)
    top_cities_data = city_stats.nlargest(top_n, "n_viviendas")

    fig = px.bar(
        top_cities_data.sort_values("diferencial_pct"),
        x="diferencial_pct", y="city",
        color="clasificacion",
        color_discrete_map={"Tendencial": "#1a1a1a", "Atípica": "#e74c3c"},
        orientation="h",
        title=f"Diferencial precio real vs predicho — Top {top_n} ciudades",
        labels={"diferencial_pct": "Diferencial (%)", "city": "Ciudad", "clasificacion": "Clasificación"},
    )
    fig.add_vline(x=mean_diff + umbral, line_dash="dash", line_color="#e74c3c", annotation_text="IC 95%+")
    fig.add_vline(x=mean_diff - umbral, line_dash="dash", line_color="#e74c3c", annotation_text="IC 95%-")
    fig.update_layout(plot_bgcolor="#f5f0eb", paper_bgcolor="#f5f0eb", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Zonas atípicas detectadas</div>', unsafe_allow_html=True)

    atipicas = city_stats[city_stats["clasificacion"] == "Atípica"].sort_values(
        "diferencial_pct", key=abs, ascending=False
    )[["city", "precio_real_mediano", "precio_predicho_mediano", "diferencial_pct", "n_viviendas"]]
    atipicas.columns = ["Ciudad", "Precio real mediano (€)", "Precio predicho mediano (€)", "Diferencial (%)", "N viviendas"]
    st.dataframe(atipicas, use_container_width=True, hide_index=True)

    st.caption("*Los precios predichos son simulados hasta que se añadan los modelos entrenados.*")
