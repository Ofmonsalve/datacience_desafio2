"""
Dashboard Agro Colombia
=======================
Dashboard interactivo en Streamlit para el dataset de fincas agrícolas colombianas.

Uso:
    pip install -r requirements.txt
    streamlit run app.py

El CSV se busca en la misma carpeta del script (agro_colombia.csv). También se puede
subir otro archivo desde la barra lateral.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Agro Colombia",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_DEFAULT = Path(__file__).parent / "agro_colombia.csv"

COLUMNAS_ESPERADAS = [
    "ID_Finca",
    "Departamento",
    "Tipo_Cultivo",
    "Area_Hectareas",
    "Produccion_Anual_Ton",
    "Sistema_Riego_Tecnificado",
    "Nivel_Tecnificacion",
    "Precio_Venta_Por_Ton_COP",
    "Tipo_Suelo",
    "Fecha_Ultima_Auditoria",
]

ORDEN_TECNIFICACION = ["Bajo", "Medio", "Alto", "Muy Alto"]
PALETA = px.colors.qualitative.Set2

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Carga y preparación de datos
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(fuente) -> pd.DataFrame:
    df = pd.read_csv(fuente)

    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"El archivo no tiene las columnas: {', '.join(faltantes)}")

    # Tipos
    df["Fecha_Ultima_Auditoria"] = pd.to_datetime(
        df["Fecha_Ultima_Auditoria"], errors="coerce"
    )
    if df["Sistema_Riego_Tecnificado"].dtype == object:
        df["Sistema_Riego_Tecnificado"] = (
            df["Sistema_Riego_Tecnificado"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "sí": True, "si": True, "no": False})
        )
    df["Sistema_Riego_Tecnificado"] = df["Sistema_Riego_Tecnificado"].astype(bool)

    df["Nivel_Tecnificacion"] = pd.Categorical(
        df["Nivel_Tecnificacion"], categories=ORDEN_TECNIFICACION, ordered=True
    )

    # Métricas derivadas
    df["Rendimiento_Ton_Ha"] = df["Produccion_Anual_Ton"] / df["Area_Hectareas"]
    df["Ingreso_Estimado_COP"] = (
        df["Produccion_Anual_Ton"] * df["Precio_Venta_Por_Ton_COP"]
    )
    df["Ingreso_Por_Hectarea_COP"] = df["Ingreso_Estimado_COP"] / df["Area_Hectareas"]
    df["Riego"] = np.where(df["Sistema_Riego_Tecnificado"], "Con riego tecnificado", "Sin riego tecnificado")
    df["Mes_Auditoria"] = df["Fecha_Ultima_Auditoria"].dt.to_period("M").dt.to_timestamp()
    df["Dias_Desde_Auditoria"] = (
        pd.Timestamp.today().normalize() - df["Fecha_Ultima_Auditoria"]
    ).dt.days

    return df


def fmt_cop(valor: float) -> str:
    """Formatea pesos colombianos de forma compacta."""
    if pd.isna(valor):
        return "—"
    for limite, sufijo in ((1e12, "B"), (1e9, "MM"), (1e6, "M"), (1e3, "K")):
        if abs(valor) >= limite:
            return f"${valor / limite:,.2f} {sufijo}"
    return f"${valor:,.0f}"


def fmt_num(valor: float, dec: int = 1) -> str:
    if pd.isna(valor):
        return "—"
    return f"{valor:,.{dec}f}"


# --------------------------------------------------------------------------------------
# Barra lateral: fuente de datos y filtros
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Datos y filtros")

subida = st.sidebar.file_uploader("Cargar otro CSV (opcional)", type=["csv"])

try:
    if subida is not None:
        df = cargar_datos(subida)
        st.sidebar.success("Archivo cargado.")
    elif CSV_DEFAULT.exists():
        df = cargar_datos(CSV_DEFAULT)
    else:
        st.error(
            f"No se encontró `{CSV_DEFAULT.name}` junto al script. "
            "Súbelo desde la barra lateral para continuar."
        )
        st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    departamentos = st.multiselect(
        "Departamento",
        sorted(df["Departamento"].unique()),
        default=sorted(df["Departamento"].unique()),
    )
    cultivos = st.multiselect(
        "Tipo de cultivo",
        sorted(df["Tipo_Cultivo"].unique()),
        default=sorted(df["Tipo_Cultivo"].unique()),
    )
    niveles = st.multiselect(
        "Nivel de tecnificación",
        [n for n in ORDEN_TECNIFICACION if n in df["Nivel_Tecnificacion"].unique()],
        default=[n for n in ORDEN_TECNIFICACION if n in df["Nivel_Tecnificacion"].unique()],
    )
    suelos = st.multiselect(
        "Tipo de suelo",
        sorted(df["Tipo_Suelo"].unique()),
        default=sorted(df["Tipo_Suelo"].unique()),
    )
    riego = st.radio(
        "Riego tecnificado",
        ["Todos", "Solo con riego", "Solo sin riego"],
        horizontal=False,
    )
    area_min, area_max = float(df["Area_Hectareas"].min()), float(df["Area_Hectareas"].max())
    rango_area = st.slider(
        "Área (hectáreas)",
        area_min,
        area_max,
        (area_min, area_max),
        step=0.5,
    )
    dias_alerta = st.number_input(
        "Alerta de auditoría: días desde la última",
        min_value=30,
        max_value=730,
        value=180,
        step=30,
        help="Fincas cuya última auditoría supere este umbral se marcan como pendientes.",
    )

mask = (
    df["Departamento"].isin(departamentos)
    & df["Tipo_Cultivo"].isin(cultivos)
    & df["Nivel_Tecnificacion"].isin(niveles)
    & df["Tipo_Suelo"].isin(suelos)
    & df["Area_Hectareas"].between(*rango_area)
)
if riego == "Solo con riego":
    mask &= df["Sistema_Riego_Tecnificado"]
elif riego == "Solo sin riego":
    mask &= ~df["Sistema_Riego_Tecnificado"]

d = df[mask].copy()

st.sidebar.markdown(f"**{len(d):,} de {len(df):,} fincas** seleccionadas")
if d.empty:
    st.warning("Ningún registro cumple los filtros seleccionados. Ajusta los filtros.")
    st.stop()


# --------------------------------------------------------------------------------------
# Encabezado y KPIs
# --------------------------------------------------------------------------------------
st.title("🌱 Dashboard Agro Colombia")
st.caption(
    f"{len(d):,} fincas · {d['Departamento'].nunique()} departamentos · "
    f"{d['Tipo_Cultivo'].nunique()} cultivos · auditorías entre "
    f"{d['Fecha_Ultima_Auditoria'].min():%b %Y} y {d['Fecha_Ultima_Auditoria'].max():%b %Y}"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fincas", f"{len(d):,}")
k2.metric("Área total", f"{fmt_num(d['Area_Hectareas'].sum())} ha")
k3.metric("Producción anual", f"{fmt_num(d['Produccion_Anual_Ton'].sum())} ton")
k4.metric(
    "Rendimiento medio",
    f"{fmt_num(d['Produccion_Anual_Ton'].sum() / d['Area_Hectareas'].sum(), 2)} ton/ha",
)
k5.metric("Ingreso estimado", fmt_cop(d["Ingreso_Estimado_COP"].sum()))

k6, k7, k8, k9 = st.columns(4)
k6.metric("Precio medio", f"{fmt_cop(d['Precio_Venta_Por_Ton_COP'].mean())} /ton")
k7.metric("Riego tecnificado", f"{d['Sistema_Riego_Tecnificado'].mean() * 100:.1f}%")
k8.metric(
    "Tecnificación alta o muy alta",
    f"{d['Nivel_Tecnificacion'].isin(['Alto', 'Muy Alto']).mean() * 100:.1f}%",
)
pendientes = int((d["Dias_Desde_Auditoria"] > dias_alerta).sum())
k9.metric(
    f"Auditoría > {dias_alerta} días",
    f"{pendientes:,}",
    delta=f"{pendientes / len(d) * 100:.1f}% del total",
    delta_color="inverse",
)

st.divider()


# --------------------------------------------------------------------------------------
# Pestañas
# --------------------------------------------------------------------------------------
tab_geo, tab_rend, tab_com, tab_tec, tab_datos = st.tabs(
    ["🗺️ Geografía y cultivos", "📈 Rendimiento", "💰 Comercial", "🔧 Tecnificación y suelo", "📋 Datos"]
)

# ---------------------------------- Geografía -----------------------------------------
with tab_geo:
    c1, c2 = st.columns(2)

    por_depto = (
        d.groupby("Departamento", as_index=False)
        .agg(
            Produccion=("Produccion_Anual_Ton", "sum"),
            Area=("Area_Hectareas", "sum"),
            Fincas=("ID_Finca", "count"),
        )
        .sort_values("Produccion", ascending=True)
    )
    por_depto["Rendimiento"] = por_depto["Produccion"] / por_depto["Area"]

    fig = px.bar(
        por_depto,
        x="Produccion",
        y="Departamento",
        orientation="h",
        text=por_depto["Produccion"].map(lambda v: f"{v:,.0f}"),
        color="Rendimiento",
        color_continuous_scale="Greens",
        labels={"Produccion": "Producción anual (ton)", "Rendimiento": "ton/ha"},
        title="Producción por
