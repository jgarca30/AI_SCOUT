"""Aplicación web de AI Scout (Streamlit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_scout import config as C  # noqa: E402
from ai_scout import datos, llm, orquestador  # noqa: E402
from ai_scout.director import fmt  # noqa: E402
from ai_scout.modelos import Consulta  # noqa: E402

st.set_page_config(page_title="AI Scout", page_icon="⚽", layout="wide")

# Datos y proveedores (se cargan una vez)
@st.cache_data(show_spinner="Preparando el dataset…")
def cargar_datos():
    return datos.cargar_o_preparar()


@st.cache_resource
def cargar_proveedores():
    return llm.crear_proveedor("analista"), llm.crear_proveedor("director")


try:
    df, manifiesto = cargar_datos()
except datos.ErrorDatos as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()

prov_analista, prov_director = cargar_proveedores()
llm_disponible = prov_analista is not None or prov_director is not None

ICONO_VEREDICTO = {"candidato_para_revisar": "🟢", "candidato_con_reservas": "🟡", "evidencia_insuficiente": "⚪"}


def edad_txt(edad) -> str:
    return "edad no disponible" if edad is None or pd.isna(edad) else f"{int(edad)} años"


# Barra lateral: consulta
with st.sidebar:
    st.title("⚽ AI Scout")
    st.caption("Recomendador de sustitutos por similitud estadística · Temporada 2024-2025")

    jugadores = df[~df["es_portero"] & df["valido"]].sort_values(["nombre", "minutos"], ascending=[True, False])
    etiquetas = {r.registro_id: datos.etiqueta_registro(r._asdict()) for r in jugadores.itertuples()}
    objetivo_id = st.selectbox("Jugador de referencia", options=list(etiquetas), format_func=etiquetas.get,
                               index=None, placeholder="Escribe un nombre…",
                               help="Se muestran equipo, liga, posición y minutos para distinguir homónimos y "
                                    "jugadores con varios equipos.")
    ligas = st.multiselect("Ligas permitidas", C.LIGAS_NORMALIZADAS, default=C.LIGAS_NORMALIZADAS)
    limitar_edad = st.checkbox("Limitar edad máxima")
    edad_maxima = st.slider("Edad máxima", 16, 40, 25, disabled=not limitar_edad,
                            help="Si se activa, se excluyen los jugadores sin edad registrada.")
    minutos_minimos = st.number_input("Minutos mínimos", min_value=90, max_value=3420,
                                      value=C.MINUTOS_MINIMOS_DEFECTO, step=90)
    usar_ia = st.toggle("Usar análisis con IA", value=llm_disponible, disabled=not llm_disponible,
                        help="Desactívalo para comparar con el ranking puramente numérico.")
    st.caption(llm.descripcion_configuracion())
    buscar = st.button("Buscar sustitutos", type="primary",
                       disabled=objetivo_id is None)

if buscar:
    consulta = Consulta(objetivo_registro_id=objetivo_id, ligas_permitidas=ligas,
                        minutos_minimos=int(minutos_minimos),
                        edad_maxima=int(edad_maxima) if limitar_edad else None, usar_llm=bool(usar_ia))
    with st.spinner("Los agentes están analizando a los candidatos…"):
        st.session_state["resultado"] = orquestador.ejecutar_consulta(
            consulta, df, manifiesto["dataset_version"], prov_analista, prov_director, cache=llm.CacheLLM())

resultado = st.session_state.get("resultado")

# Pantalla inicial
if resultado is None:
    st.header("Encuentra sustitutos con un perfil estadístico parecido")
    st.markdown(
        "1. Elige un **jugador de referencia** en la barra lateral.\n"
        "2. Ajusta las **ligas**, la **edad máxima** y los **minutos mínimos**.\n"
        "3. Pulsa **Buscar sustitutos**.\n\n"
        "El **Agente 1** calcula los 10 perfiles más parecidos (similitud coseno sobre estadísticas por 90 minutos), "
        "el **Agente 2** los analiza con un modelo de lenguaje y el **Agente 3** redacta la ficha de ojeador de los "
        "tres mejores. Cada consulta se guarda en `salidas/` para el cuadro de mando de Power BI.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros en el dataset", f"{manifiesto['filas']:,}".replace(",", "."))
    c2.metric("Jugadores de campo válidos", f"{manifiesto['registros_validos_campo']:,}".replace(",", "."))
    c3.metric("Versión del dataset", manifiesto["dataset_version"])
    st.stop()

# Resultado
q = resultado["consulta"]
if resultado["estado"] == "rechazada":
    st.error("No se ha podido ejecutar la consulta:\n\n" + "\n".join(f"- {e}" for e in resultado["errores"]))
    st.stop()

rec = resultado["recuperacion"]
obj = rec["objetivo"]
st.header(f"Sustitutos para {obj['nombre']}")
st.caption(f"{obj['equipo']} · {obj['liga']} · {obj['posicion_original']} · {edad_txt(obj['edad_registrada'])} · "
           f"{fmt(obj['minutos'], 0)} minutos")

filtros = [", ".join(q["ligas_permitidas"]), f"≥ {fmt(q['minutos_minimos'], 0)} min"]
if q["edad_maxima"] is not None:
    filtros.append(f"≤ {q['edad_maxima']} años")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Grupo de posición", C.GRUPOS[rec["grupo"]])
m2.metric("Jugadores comparados", rec["n_candidatos_validos"])
m3.metric("Modo del ranking", "Híbrido (IA)" if resultado["modo_ranking"] == "hibrido" else "Numérico")
m4.metric("Tiempo", f"{fmt(resultado['duracion_total_ms'] / 1000, 1)} s")
st.caption("Filtros: " + " · ".join(filtros))

for aviso in resultado["avisos"]:
    st.warning(aviso)

if resultado["estado"] == "sin_resultados":
    st.info("No hay recomendaciones porque ningún jugador cumple todas las restricciones. "
            "El sistema no relaja los filtros por su cuenta.")
    st.stop()

if resultado["resumen_consulta"]:
    st.markdown(f"> {resultado['resumen_consulta']}")

# Fichas del Top final
st.subheader("Fichas de ojeador")
columnas = st.columns(len(resultado["top_final"]))
for col, c in zip(columnas, resultado["top_final"]):
    ficha = c["ficha"]
    with col.container(border=True):
        st.markdown(f"#### {c['ranking_final']}. {c['nombre']}")
        st.caption(f"{c['equipo']} · {c['liga']} · {c['posicion_original']} · {edad_txt(c['edad_registrada'])} · "
                   f"{fmt(c['minutos'], 0)} min")
        a, b = st.columns(2)
        a.metric("Índice de similitud", fmt(c["score_numerico"], 1))
        if resultado["modo_ranking"] == "hibrido":
            b.metric("Score final", fmt(c["score_final"], 1),
                     help=f"{C.PESO_NUMERICO} × similitud + {C.PESO_CUALITATIVO} × valoración cualitativa "
                          f"({fmt(c['score_cualitativo'], 1)})")
        st.markdown(f"**{ICONO_VEREDICTO.get(ficha['veredicto'], '')} {C.VEREDICTOS.get(ficha['veredicto'], '')}**")
        st.write(ficha["resumen"])
        for titulo, campo in (("Fortalezas", "fortalezas"), ("Diferencias", "diferencias_relevantes"),
                              ("Reservas", "reservas")):
            if ficha.get(campo):
                st.markdown(f"**{titulo}**\n" + "\n".join(f"- {x}" for x in ficha[campo]))
        st.caption("Ficha redactada por el Agente 3" if ficha["origen"] == "llm"
                   else "Ficha generada por plantilla (sin IA)")

# Comparativa 

st.subheader("Comparativa con el jugador de referencia")
nombres = {c["registro_id"]: f"{c['ranking_final']}. {c['nombre']} ({c['equipo']})" for c in resultado["ranking"]}
elegido_id = st.selectbox("Candidato", list(nombres), format_func=nombres.get)
elegido = next(c for c in resultado["ranking"] if c["registro_id"] == elegido_id)

izq, der = st.columns([3, 2])
with izq:
    try:
        from ai_scout.graficos import radar_percentiles

        st.plotly_chart(radar_percentiles(obj, elegido, rec["variables"]))
        st.caption("Percentiles dentro de su grupo de posición (jugadores con los minutos mínimos).")
    except ImportError:
        st.bar_chart(pd.DataFrame({obj["nombre"]: obj["percentiles"], elegido["nombre"]: elegido["percentiles"]}))
with der:
    tabla = pd.DataFrame([{
        "Variable": C.ETIQUETAS_CORTAS.get(d["variable"], d["variable"]),
        "Unidad": d["unidad"],
        obj["nombre"]: round(d["valor_objetivo"], 2),
        elegido["nombre"]: round(d["valor_candidato"], 2),
        "Diferencia": round(d["diferencia"], 2),
    } for d in elegido["diferencias"]])
    st.dataframe(tabla, hide_index=True)
    if elegido.get("analisis"):
        a = elegido["analisis"]
        st.markdown("**Análisis del Agente 2**")
        st.write(a.get("resumen", ""))
        for d in a.get("dimensiones", []):
            if d.get("puntuacion") is not None:
                st.markdown(f"- {d['nombre'].replace('_', ' ')}: **{d['puntuacion']}/4** · {d.get('justificacion', '')}")

# Top 10
st.subheader("Preselección (Top 10)")
top10 = pd.DataFrame([{
    "#": c["ranking_final"], "Jugador": c["nombre"], "Equipo": c["equipo"], "Liga": c["liga"],
    "Pos.": c["posicion_original"], "Edad": c["edad_registrada"], "Minutos": c["minutos"],
    "Índice similitud": round(c["score_numerico"], 1),
    "Valoración IA": None if c["score_cualitativo"] is None else round(c["score_cualitativo"], 1),
    "Score final": round(c["score_final"], 1), "Rank. numérico": c["ranking_preliminar"],
} for c in resultado["ranking"]])
st.dataframe(top10, hide_index=True)

with st.expander("Limitaciones"):
    for x in resultado["limitaciones_generales"]:
        st.markdown(f"- {x}")
with st.expander("Trazas de los agentes"):
    if resultado["ejecuciones"]:
        st.dataframe(pd.DataFrame(resultado["ejecuciones"]).drop(columns=["consulta_id"]), hide_index=True)
    else:
        st.write("No se ha llamado a ningún modelo de lenguaje en esta consulta.")

st.caption(f"Consulta {q['consulta_id']} guardada en `salidas/` · Dataset {resultado['dataset_version']} · "
           "Los índices miden parecido estadístico, no la probabilidad de éxito de un fichaje.")
