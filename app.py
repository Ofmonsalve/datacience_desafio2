"""
Dashboard Inteligente Agro Colombia
===================================
Dashboard interactivo + analista IA (Llama 3.3 70B vía Groq) que interpreta y explica
los resultados de los datos filtrados en la conversación.

Uso:
    pip install -r requirements.txt
    streamlit run app.py

Necesita `agro_colombia.csv` en la misma carpeta y una GROQ API Key (se escribe en la
barra lateral; no se guarda en ningún archivo).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ======================================================================================
# Configuración
# ======================================================================================
st.set_page_config(page_title="Agro Colombia · Dashboard Inteligente", page_icon="🌱", layout="wide")

CSV_DEFAULT = Path(__file__).parent / "agro_colombia.csv"

COLUMNAS_ESPERADAS = [
    "ID_Finca", "Departamento", "Tipo_Cultivo", "Area_Hectareas", "Produccion_Anual_Ton",
    "Sistema_Riego_Tecnificado", "Nivel_Tecnificacion", "Precio_Venta_Por_Ton_COP",
    "Tipo_Suelo", "Fecha_Ultima_Auditoria",
]

ORDEN_TEC = ["Bajo", "Medio", "Alto", "Muy Alto"]
PALETA = px.colors.qualitative.Set2

MODELOS = {
    "Llama 3.3 70B (llama-3.3-70b-versatile)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (openai/gpt-oss-120b)": "openai/gpt-oss-120b",
    "Qwen 3.6 27B (qwen/qwen3.6-27b)": "qwen/qwen3.6-27b",
}

SYSTEM_BASE = """Eres un analista de datos agrícolas senior especializado en el agro colombiano.
Interpretas los resultados de un dashboard y los explicas a un usuario de negocio en español.

Reglas estrictas:
- Usa ÚNICAMENTE las cifras del RESUMEN DE DATOS que se te entrega. No inventes números.
- Si te preguntan algo que el resumen no permite responder, dilo y explica qué dato haría falta.
- Cita las cifras concretas que respaldan cada afirmación (valor y unidad).
- Distingue con claridad entre correlación y causalidad; señala cuando una explicación es hipótesis.
- Respuestas concisas y accionables: máximo 4 párrafos cortos o una lista de 5 viñetas.
- Advierte cuando una diferencia entre grupos sea pequeña o se base en pocas fincas (n bajo).
- El "ingreso estimado" es producción × precio: no descuenta costos. Recuérdalo al hablar de rentabilidad.
- Los datos son de una muestra de fincas: no extrapoles al total nacional.
"""

SUGERENCIAS = [
    "Explícame los resultados principales de la selección actual",
    "¿El riego tecnificado se traduce en mayor rendimiento?",
    "¿Qué departamento conviene priorizar y por qué?",
    "¿Qué combinación de cultivo y suelo rinde mejor?",
    "Detecta datos raros o inconsistentes en la selección",
    "Dame 3 acciones concretas basadas en estos números",
]


# ======================================================================================
# Datos
# ======================================================================================
@st.cache_data(show_spinner=False)
def cargar_datos(fuente) -> pd.DataFrame:
    df = pd.read_csv(fuente)

    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"El archivo no tiene las columnas: {', '.join(faltantes)}")

    df["Fecha_Ultima_Auditoria"] = pd.to_datetime(df["Fecha_Ultima_Auditoria"], errors="coerce")

    if df["Sistema_Riego_Tecnificado"].dtype == object:
        df["Sistema_Riego_Tecnificado"] = (
            df["Sistema_Riego_Tecnificado"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False,
                  "sí": True, "si": True, "no": False})
        )
    df["Sistema_Riego_Tecnificado"] = df["Sistema_Riego_Tecnificado"].astype(bool)

    df["Nivel_Tecnificacion"] = pd.Categorical(df["Nivel_Tecnificacion"], categories=ORDEN_TEC, ordered=True)

    df["Rendimiento_Ton_Ha"] = df["Produccion_Anual_Ton"] / df["Area_Hectareas"]
    df["Ingreso_Estimado_COP"] = df["Produccion_Anual_Ton"] * df["Precio_Venta_Por_Ton_COP"]
    df["Ingreso_Por_Hectarea_COP"] = df["Ingreso_Estimado_COP"] / df["Area_Hectareas"]
    df["Riego"] = np.where(df["Sistema_Riego_Tecnificado"], "Con riego", "Sin riego")
    df["Mes_Auditoria"] = df["Fecha_Ultima_Auditoria"].dt.to_period("M").dt.to_timestamp()
    df["Dias_Desde_Auditoria"] = (pd.Timestamp.today().normalize() - df["Fecha_Ultima_Auditoria"]).dt.days
    return df


def fmt_cop(v: float) -> str:
    if pd.isna(v):
        return "—"
    for lim, suf in ((1e12, "B"), (1e9, "MM"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= lim:
            return f"${v / lim:,.2f} {suf}"
    return f"${v:,.0f}"


# ======================================================================================
# Resumen de datos para el modelo (esto es lo que "ve" la IA)
# ======================================================================================
def construir_resumen(d: pd.DataFrame, dias_alerta: int) -> str:
    """Serializa los agregados de la selección actual en texto compacto para el LLM."""
    partes = []

    partes.append(
        "### Contexto\n"
        f"- Fincas en la selección: {len(d)}\n"
        f"- Departamentos: {', '.join(sorted(d['Departamento'].unique()))}\n"
        f"- Cultivos: {', '.join(sorted(d['Tipo_Cultivo'].unique()))}\n"
        f"- Área total: {d['Area_Hectareas'].sum():,.1f} ha "
        f"(media {d['Area_Hectareas'].mean():,.1f}; rango {d['Area_Hectareas'].min():,.2f}–{d['Area_Hectareas'].max():,.2f})\n"
        f"- Producción anual total: {d['Produccion_Anual_Ton'].sum():,.1f} ton\n"
        f"- Rendimiento agregado: {d['Produccion_Anual_Ton'].sum() / d['Area_Hectareas'].sum():,.2f} ton/ha\n"
        f"- Rendimiento por finca: media {d['Rendimiento_Ton_Ha'].mean():,.2f}, "
        f"mediana {d['Rendimiento_Ton_Ha'].median():,.2f}, "
        f"p10 {d['Rendimiento_Ton_Ha'].quantile(.1):,.2f}, p90 {d['Rendimiento_Ton_Ha'].quantile(.9):,.2f} ton/ha\n"
        f"- Precio de venta: media {d['Precio_Venta_Por_Ton_COP'].mean():,.0f} COP/ton "
        f"(min {d['Precio_Venta_Por_Ton_COP'].min():,.0f}; max {d['Precio_Venta_Por_Ton_COP'].max():,.0f})\n"
        f"- Ingreso estimado total: {d['Ingreso_Estimado_COP'].sum():,.0f} COP\n"
        f"- Fincas con riego tecnificado: {int(d['Sistema_Riego_Tecnificado'].sum())} "
        f"({d['Sistema_Riego_Tecnificado'].mean() * 100:.1f}%)\n"
        f"- Última auditoría: entre {d['Fecha_Ultima_Auditoria'].min():%Y-%m-%d} y "
        f"{d['Fecha_Ultima_Auditoria'].max():%Y-%m-%d}; "
        f"{int((d['Dias_Desde_Auditoria'] > dias_alerta).sum())} fincas superan {dias_alerta} días"
    )

    def tabla(g: pd.DataFrame, titulo: str) -> str:
        return f"\n### {titulo}\n{g.round(2).to_markdown()}"

    por_depto = (
        d.groupby("Departamento", observed=True)
        .agg(Fincas=("ID_Finca", "count"), Area_ha=("Area_Hectareas", "sum"),
             Produccion_ton=("Produccion_Anual_Ton", "sum"),
             Rend_medio_ton_ha=("Rendimiento_Ton_Ha", "mean"),
             Precio_medio_COP=("Precio_Venta_Por_Ton_COP", "mean"),
             Ingreso_MM_COP=("Ingreso_Estimado_COP", lambda s: s.sum() / 1e6))
        .sort_values("Produccion_ton", ascending=False)
    )
    partes.append(tabla(por_depto, "Por departamento"))

    por_cultivo = (
        d.groupby("Tipo_Cultivo", observed=True)
        .agg(Fincas=("ID_Finca", "count"), Area_ha=("Area_Hectareas", "sum"),
             Produccion_ton=("Produccion_Anual_Ton", "sum"),
             Rend_medio_ton_ha=("Rendimiento_Ton_Ha", "mean"),
             Precio_medio_COP=("Precio_Venta_Por_Ton_COP", "mean"),
             Ingreso_ha_COP=("Ingreso_Por_Hectarea_COP", "mean"))
        .sort_values("Produccion_ton", ascending=False)
    )
    partes.append(tabla(por_cultivo, "Por tipo de cultivo"))

    por_tec = (
        d.groupby(["Nivel_Tecnificacion", "Riego"], observed=True)
        .agg(Fincas=("ID_Finca", "count"),
             Rend_medio_ton_ha=("Rendimiento_Ton_Ha", "mean"),
             Ingreso_ha_COP=("Ingreso_Por_Hectarea_COP", "mean"))
    )
    partes.append(tabla(por_tec, "Por nivel de tecnificación y riego"))

    por_suelo = (
        d.groupby("Tipo_Suelo", observed=True)
        .agg(Fincas=("ID_Finca", "count"),
             Rend_medio_ton_ha=("Rendimiento_Ton_Ha", "mean"),
             Area_media_ha=("Area_Hectareas", "mean"))
        .sort_values("Rend_medio_ton_ha", ascending=False)
    )
    partes.append(tabla(por_suelo, "Por tipo de suelo"))

    num = ["Area_Hectareas", "Produccion_Anual_Ton", "Rendimiento_Ton_Ha",
           "Precio_Venta_Por_Ton_COP", "Ingreso_Estimado_COP"]
    partes.append(tabla(d[num].corr(numeric_only=True), "Correlaciones de Pearson"))

    extremos = pd.concat([
        d.nlargest(3, "Rendimiento_Ton_Ha").assign(Grupo="Top rendimiento"),
        d.nsmallest(3, "Rendimiento_Ton_Ha").assign(Grupo="Peor rendimiento"),
    ])[["Grupo", "ID_Finca", "Departamento", "Tipo_Cultivo", "Area_Hectareas",
        "Produccion_Anual_Ton", "Rendimiento_Ton_Ha", "Nivel_Tecnificacion", "Riego"]]
    partes.append(tabla(extremos.set_index("Grupo"), "Fincas extremas por rendimiento"))

    return "\n".join(partes)


# ======================================================================================
# Barra lateral
# ======================================================================================
with st.sidebar:
    st.header("🤖 Analista IA")
    api_key = st.text_input("GROQ API Key", type="password", placeholder="gsk_...",
                            help="Solo se usa en esta sesión; no se guarda en ningún archivo.")
    modelo = MODELOS[st.selectbox("Modelo", list(MODELOS.keys()), index=0)]
    temperatura = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.1,
                            help="Baja = interpretaciones más literales y sobrias.")
    st.caption(
        "Clave gratuita en console.groq.com/keys. Groq retira "
        "`llama-3.3-70b-versatile` el 16/08/2026 en los planes gratuito y developer: "
        "si falla, cambia de modelo aquí."
    )

    st.divider()
    st.header("⚙️ Datos y filtros")
    subida = st.file_uploader("Cargar otro CSV (opcional)", type=["csv"])

try:
    if subida is not None:
        df = cargar_datos(subida)
    elif CSV_DEFAULT.exists():
        df = cargar_datos(CSV_DEFAULT)
    else:
        st.error(f"No se encontró `{CSV_DEFAULT.name}` junto al script. Súbelo desde la barra lateral.")
        st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    departamentos = st.multiselect("Departamento", sorted(df["Departamento"].unique()),
                                  default=sorted(df["Departamento"].unique()))
    cultivos = st.multiselect("Tipo de cultivo", sorted(df["Tipo_Cultivo"].unique()),
                              default=sorted(df["Tipo_Cultivo"].unique()))
    niveles_disp = [n for n in ORDEN_TEC if n in set(df["Nivel_Tecnificacion"].dropna())]
    niveles = st.multiselect("Nivel de tecnificación", niveles_disp, default=niveles_disp)
    suelos = st.multiselect("Tipo de suelo", sorted(df["Tipo_Suelo"].unique()),
                            default=sorted(df["Tipo_Suelo"].unique()))
    riego_sel = st.radio("Riego tecnificado", ["Todos", "Solo con riego", "Solo sin riego"])
    a_min, a_max = float(df["Area_Hectareas"].min()), float(df["Area_Hectareas"].max())
    rango_area = st.slider("Área (hectáreas)", a_min, a_max, (a_min, a_max), 0.5)
    dias_alerta = st.number_input("Alerta de auditoría (días)", 30, 730, 180, 30)

mask = (
    df["Departamento"].isin(departamentos)
    & df["Tipo_Cultivo"].isin(cultivos)
    & df["Nivel_Tecnificacion"].isin(niveles)
    & df["Tipo_Suelo"].isin(suelos)
    & df["Area_Hectareas"].between(*rango_area)
)
if riego_sel == "Solo con riego":
    mask &= df["Sistema_Riego_Tecnificado"]
elif riego_sel == "Solo sin riego":
    mask &= ~df["Sistema_Riego_Tecnificado"]

d = df[mask].copy()
st.sidebar.markdown(f"**{len(d):,} de {len(df):,} fincas** seleccionadas")

if d.empty:
    st.warning("Ningún registro cumple los filtros. Ajusta la selección.")
    st.stop()

resumen = construir_resumen(d, dias_alerta)

# Si cambian los filtros, avisamos al chat que el contexto cambió
firma = (len(d), tuple(sorted(departamentos)), tuple(sorted(cultivos)), tuple(sorted(niveles)),
         tuple(sorted(suelos)), riego_sel, rango_area, dias_alerta)
if st.session_state.get("firma_filtros") != firma:
    st.session_state.firma_filtros = firma
    st.session_state.contexto_nuevo = True


# ======================================================================================
# Encabezado y KPIs
# ======================================================================================
st.title("🌱 Agro Colombia · Dashboard Inteligente")
st.caption(
    f"{len(d):,} fincas · {d['Departamento'].nunique()} departamentos · "
    f"{d['Tipo_Cultivo'].nunique()} cultivos · interpretaciones generadas con `{modelo}`"
)

k = st.columns(5)
k[0].metric("Fincas", f"{len(d):,}")
k[1].metric("Área total", f"{d['Area_Hectareas'].sum():,.1f} ha")
k[2].metric("Producción", f"{d['Produccion_Anual_Ton'].sum():,.1f} ton")
k[3].metric("Rendimiento", f"{d['Produccion_Anual_Ton'].sum() / d['Area_Hectareas'].sum():,.2f} ton/ha")
k[4].metric("Ingreso estimado", fmt_cop(d["Ingreso_Estimado_COP"].sum()))

st.divider()

tab_dash, tab_chat, tab_datos = st.tabs(["📊 Dashboard", "💬 Analista IA", "📋 Datos y contexto"])

# ======================================================================================
# Pestaña 1 · Dashboard
# ======================================================================================
with tab_dash:
    c1, c2 = st.columns(2)

    g = (d.groupby("Departamento", as_index=False, observed=True)
           .agg(Produccion=("Produccion_Anual_Ton", "sum"), Area=("Area_Hectareas", "sum"))
           .sort_values("Produccion"))
    g["Rendimiento"] = g["Produccion"] / g["Area"]
    fig = px.bar(g, x="Produccion", y="Departamento", orientation="h", color="Rendimiento",
                 color_continuous_scale="Greens", text=g["Produccion"].map(lambda v: f"{v:,.0f}"),
                 labels={"Produccion": "Producción (ton)", "Rendimiento": "ton/ha"},
                 title="Producción por departamento")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=420, margin=dict(t=60, r=20))
    c1.plotly_chart(fig, width="stretch")

    g = d.groupby("Tipo_Cultivo", as_index=False, observed=True)["Produccion_Anual_Ton"].sum()
    fig = px.pie(g, names="Tipo_Cultivo", values="Produccion_Anual_Ton", hole=0.45,
                 color_discrete_sequence=PALETA, title="Participación por cultivo")
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(height=420, showlegend=False, margin=dict(t=60))
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)

    fig = px.box(d, x="Tipo_Cultivo", y="Rendimiento_Ton_Ha", color="Tipo_Cultivo",
                 points="outliers", color_discrete_sequence=PALETA,
                 labels={"Rendimiento_Ton_Ha": "Rendimiento (ton/ha)", "Tipo_Cultivo": ""},
                 title="Rendimiento por cultivo")
    fig.update_layout(height=420, showlegend=False, margin=dict(t=60))
    c3.plotly_chart(fig, width="stretch")

    g = (d.groupby(["Nivel_Tecnificacion", "Riego"], as_index=False, observed=True)
           .agg(Rendimiento=("Rendimiento_Ton_Ha", "mean"), Fincas=("ID_Finca", "count")))
    fig = px.bar(g, x="Nivel_Tecnificacion", y="Rendimiento", color="Riego", barmode="group",
                 color_discrete_sequence=PALETA, hover_data=["Fincas"],
                 category_orders={"Nivel_Tecnificacion": ORDEN_TEC},
                 labels={"Nivel_Tecnificacion": "Tecnificación", "Rendimiento": "Rend. medio (ton/ha)"},
                 title="Rendimiento por tecnificación y riego")
    fig.update_layout(height=420, margin=dict(t=60))
    c4.plotly_chart(fig, width="stretch")

    c5, c6 = st.columns(2)

    fig = px.scatter(d, x="Area_Hectareas", y="Produccion_Anual_Ton", color="Nivel_Tecnificacion",
                     symbol="Riego", size="Ingreso_Estimado_COP", size_max=18, opacity=0.75,
                     hover_name="ID_Finca", category_orders={"Nivel_Tecnificacion": ORDEN_TEC},
                     labels={"Area_Hectareas": "Área (ha)", "Produccion_Anual_Ton": "Producción (ton)",
                             "Nivel_Tecnificacion": "Tecnificación"},
                     title="Área vs. producción por finca")
    fig.update_layout(height=420, margin=dict(t=60))
    c5.plotly_chart(fig, width="stretch")

    g = (d.groupby("Tipo_Suelo", as_index=False, observed=True)
           .agg(Rendimiento=("Rendimiento_Ton_Ha", "mean"), Fincas=("ID_Finca", "count"))
           .sort_values("Rendimiento", ascending=False))
    fig = px.bar(g, x="Tipo_Suelo", y="Rendimiento", color="Fincas", color_continuous_scale="Oranges",
                 text=g["Rendimiento"].map(lambda v: f"{v:.2f}"),
                 labels={"Rendimiento": "Rend. medio (ton/ha)", "Tipo_Suelo": ""},
                 title="Rendimiento por tipo de suelo")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=420, coloraxis_showscale=False, margin=dict(t=60))
    c6.plotly_chart(fig, width="stretch")

# ======================================================================================
# Pestaña 2 · Analista IA
# ======================================================================================
def responder(pregunta: str):
    """Llama a Groq con el resumen de la selección como contexto y hace streaming."""
    historial = st.session_state.chat[-10:]
    payload = (
        [{"role": "system", "content": SYSTEM_BASE + "\n\n## RESUMEN DE DATOS (selección actual)\n" + resumen}]
        + historial
        + [{"role": "user", "content": pregunta}]
    )
    cliente = Groq(api_key=api_key)
    stream = cliente.chat.completions.create(
        model=modelo, messages=payload, temperature=temperatura, max_tokens=1400, stream=True,
    )
    return st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)


with tab_chat:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if not api_key:
        st.info("Escribe tu **GROQ API Key** en la barra lateral para activar al analista IA.", icon="🔑")

    if st.session_state.get("contexto_nuevo") and st.session_state.chat:
        st.warning(
            "Los filtros cambiaron: las próximas respuestas usarán la nueva selección "
            f"({len(d):,} fincas).", icon="🔄",
        )

    cs = st.columns([3, 1])
    cs[0].markdown("**Pregúntale al analista sobre los resultados de la selección actual**")
    if cs[1].button("🗑️ Limpiar chat", width="stretch"):
        st.session_state.chat = []
        st.rerun()

    pregunta_click = None
    if not st.session_state.chat:
        cols = st.columns(2)
        for i, s in enumerate(SUGERENCIAS):
            if cols[i % 2].button(s, key=f"sug_{i}", width="stretch"):
                pregunta_click = s

    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])

    entrada = st.chat_input("Ej.: ¿por qué Quindío rinde más que Huila?")
    pregunta = entrada or pregunta_click

    if pregunta:
        if not api_key:
            st.warning("Falta la API Key en la barra lateral.", icon="⚠️")
            st.stop()

        with st.chat_message("user", avatar="🧑"):
            st.markdown(pregunta)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                texto = responder(pregunta)
                st.session_state.chat.append({"role": "user", "content": pregunta})
                st.session_state.chat.append({"role": "assistant", "content": texto})
                st.session_state.contexto_nuevo = False
            except Exception as e:
                msg = str(e)
                if "401" in msg or "authentication" in msg.lower() or "invalid_api_key" in msg:
                    st.error("API Key inválida. Revísala en console.groq.com/keys.", icon="🔑")
                elif "429" in msg or "rate limit" in msg.lower():
                    st.error("Límite de peticiones alcanzado. Espera unos segundos.", icon="⏳")
                elif "decommissioned" in msg.lower() or "model_not_found" in msg or "404" in msg:
                    st.error(f"El modelo `{modelo}` no está disponible en tu cuenta. "
                             "Elige otro en la barra lateral.", icon="🚫")
                else:
                    st.error(f"Error al llamar a Groq: {msg}", icon="❌")

    if st.session_state.chat:
        st.download_button(
            "⬇️ Descargar análisis",
            "\n\n".join(f"{'Tú' if m['role'] == 'user' else 'Analista'}: {m['content']}"
                        for m in st.session_state.chat).encode("utf-8"),
            file_name="analisis_agro.md", mime="text/markdown",
        )

# ======================================================================================
# Pestaña 3 · Datos y contexto
# ======================================================================================
with tab_datos:
    st.markdown("**Detalle de fincas filtradas**")
    st.dataframe(d.drop(columns=["Mes_Auditoria"]), width="stretch", hide_index=True, height=380)
    st.download_button("⬇️ Descargar selección en CSV",
                       d.to_csv(index=False).encode("utf-8-sig"),
                       file_name="agro_colombia_filtrado.csv", mime="text/csv")

    with st.expander("🔍 Ver exactamente qué datos recibe la IA"):
        st.caption(
            "El modelo no ve las 500 filas: recibe este resumen agregado de la selección "
            "actual. Así las respuestas son reproducibles y no se inventan cifras."
        )
        st.markdown(resumen)
