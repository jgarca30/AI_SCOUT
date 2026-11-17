"""Página principal: búsqueda de sustitutos (formulario o lenguaje natural) y resultado."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_scout import config as C
from ai_scout import datos, encaje, graficos, informe, interprete, llm, orquestador
from ai_scout import estilos as E
from ai_scout.modelos import Consulta, nuevo_id_consulta
from ai_scout.redaccion import restaurar_tildes
from ai_scout.similitud import pesos_de_rol

from . import comun
from .comun import fmt

NOMBRE_FASE = {
    "aprendizaje": "Lo aprendido de tus valoraciones", "agente1": "Ojeador de datos",
    "agente2": "Analista de rendimiento", "ranking": "Ranking final",
    "plantilla": "Analista de plantilla", "agente3": "Director deportivo", "agente4": "Jefe de scouting", "guardado": "Consulta guardada",
}
EJEMPLOS = [
    "Sustituto joven de Pedri en la Bundesliga, más creador",
    "Extremo regateador parecido a Lamine Yamal, sub-23, que no sea de La Liga",
    "Relevo para Modric menor de 26 años con al menos 1500 minutos",
]
EDAD_MIN, EDAD_MAX = 16, 40
MINUTOS_MIN, MINUTOS_MAX = 90, 3420
DIVERSIDAD_MAX = 0.6
SIN_DESTINO = "— Sin análisis de plantilla —"
NOMBRE_INDICADOR = {"estilo_equipo": "Juego del equipo", "complemento": "Complemento en la línea",
                    "compatriotas": "Compatriotas", "liga": "Misma liga"}
TEXTO_REVISION = {
    "aprobada": ("green", ":material/verified:", "Revisada por el Jefe de scouting"),
    "corregida": ("blue", ":material/edit_note:", "Corregida tras la revisión"),
    "con_observaciones": ("orange", ":material/report:", "Con observaciones del revisor"),
}


# --------------------------------------------------------------------------
# Estado del formulario
# --------------------------------------------------------------------------
def _valores_iniciales(hay_llm: bool) -> None:
    s = st.session_state
    s.setdefault("f_ligas", list(C.LIGAS_NORMALIZADAS))
    s.setdefault("f_limitar_edad", False)
    s.setdefault("f_edad", 25)
    s.setdefault("f_minutos", C.MINUTOS_MINIMOS_DEFECTO)
    s.setdefault("f_rol", C.ROL_DEFECTO)
    s.setdefault("f_ia", hay_llm)
    s.setdefault("f_aprendizaje", True)
    s.setdefault("f_ocultar", False)
    s.setdefault("f_estilo", False)
    s.setdefault("f_diversidad", 0.0)
    s.setdefault("f_critico", True)
    s.setdefault("f_pesos_mano", False)
    if not hay_llm:
        s["f_ia"] = False


def _volcar_al_formulario(parametros: dict, df: pd.DataFrame) -> bool:
    """Pone en los controles de la barra lateral los parámetros de una consulta (antes de crearlos).

    Devuelve False si el jugador de referencia no está en el dataset actual.
    """
    s = st.session_state
    registro = parametros.get("objetivo_registro_id")
    fila = df[df["registro_id"] == registro]
    if fila.empty or fila.iloc[0]["es_portero"] or not fila.iloc[0]["valido"]:
        return False
    grupo = fila.iloc[0]["grupo_posicion"]
    s["f_objetivo"] = registro
    s["f_ligas"] = [l for l in C.LIGAS_NORMALIZADAS if l in (parametros.get("ligas_permitidas") or [])] \
        or list(C.LIGAS_NORMALIZADAS)
    edad = parametros.get("edad_maxima")
    s["f_limitar_edad"] = edad is not None
    if edad is not None:
        s["f_edad"] = int(min(EDAD_MAX, max(EDAD_MIN, edad)))
    s["f_minutos"] = int(min(MINUTOS_MAX, max(MINUTOS_MIN, parametros.get("minutos_minimos")
                                              or C.MINUTOS_MINIMOS_DEFECTO)))
    rol = parametros.get("rol") or C.ROL_DEFECTO
    s["f_rol"] = rol if rol in C.ROLES[grupo] else C.ROL_DEFECTO
    for clave, campo, defecto in (("f_aprendizaje", "usar_aprendizaje", True), ("f_ocultar", "ocultar_valorados", False),
                                  ("f_estilo", "solo_mismo_estilo", False), ("f_critico", "usar_critico", True)):
        s[clave] = bool(parametros.get(campo, defecto))
    if "usar_llm" in parametros:
        s["f_ia"] = bool(parametros["usar_llm"])
    s["f_diversidad"] = float(min(DIVERSIDAD_MAX, max(0.0, parametros.get("diversidad") or 0.0)))
    pesos = parametros.get("pesos_personalizados")
    s["f_pesos_mano"] = pesos is not None
    base = pesos if pesos is not None else pesos_de_rol(grupo, s["f_rol"])
    for v in C.PERFILES[grupo]:
        s[f"f_peso_{grupo}_{v}"] = float(min(C.PESO_MAXIMO_VARIABLE, max(0.0, base.get(v, 1.0))))
    s["pesos_de"] = (grupo, s["f_rol"])
    s["f_destino"] = parametros.get("equipo_destino") or SIN_DESTINO
    s["destino_de"] = registro
    return True


def _pedir_interpretacion(texto: str | None = None) -> None:
    if texto is not None:
        st.session_state["texto_nl"] = texto
    st.session_state["nl_pendiente"] = (st.session_state.get("texto_nl") or "").strip()


def _pedir_recalculo() -> None:
    st.session_state["formulario_pendiente"] = st.session_state["resultado"]["consulta"]
    st.session_state["ejecutar_pendiente"] = {}


def _interpretar_peticion(df: pd.DataFrame, proveedores: dict) -> None:
    """Secretario técnico: convierte la petición en lenguaje natural en valores del formulario y lanza la búsqueda."""
    texto = st.session_state.pop("nl_pendiente", None)
    if not texto:
        return
    consulta_id = nuevo_id_consulta()
    usar_ia = st.session_state.get("f_ia", True)
    with st.spinner("El Secretario técnico está interpretando tu petición…"):
        interpretacion, ejecuciones = interprete.interpretar(
            texto, df, proveedores.get("interprete") if usar_ia else None, cache=llm.CacheLLM(),
            consulta_id=consulta_id)
    st.session_state["interpretacion"] = {
        "consulta_id": consulta_id, "texto": texto, "origen": interpretacion.origen,
        "resumen": interpretacion.resumen(df), "notas": list(interpretacion.notas),
        "alternativas": [a["etiqueta"] for a in interpretacion.alternativas],
        "encontrado": interpretacion.objetivo_registro_id is not None,
    }
    if interpretacion.objetivo_registro_id is None:
        return
    _volcar_al_formulario({"objetivo_registro_id": interpretacion.objetivo_registro_id,
                           "usar_llm": usar_ia, "usar_aprendizaje": st.session_state.get("f_aprendizaje", True),
                           "ocultar_valorados": st.session_state.get("f_ocultar", False),
                           "usar_critico": st.session_state.get("f_critico", True),
                           "equipo_destino": df.loc[df["registro_id"] == interpretacion.objetivo_registro_id,
                                                    "equipo"].iloc[0],
                           "diversidad": st.session_state.get("f_diversidad", 0.0),
                           **{k: v for k, v in interpretacion.parametros().items() if k != "texto_original"}}, df)
    st.session_state["ejecutar_pendiente"] = {"consulta_id": consulta_id, "texto_original": texto,
                                              "ejecuciones": ejecuciones}


@st.cache_data(show_spinner=False)
def _etiquetas(dataset_version: str) -> dict[str, str]:
    df, _ = comun.cargar_datos()
    jugadores = df[~df["es_portero"] & df["valido"]].sort_values(["nombre", "minutos"], ascending=[True, False])
    return {r.registro_id: datos.etiqueta_registro(r._asdict()) for r in jugadores.itertuples()}


def _descripcion_modelos(proveedores: dict) -> str:
    activos = {rol: p for rol, p in proveedores.items() if p is not None}
    if not activos:
        return "Sin modelo de lenguaje configurado (modo numérico)."
    modelos = {f"{p.nombre}/{p.modelo}" for p in activos.values()}
    if len(modelos) == 1:
        return f"Modelo de los agentes: {modelos.pop()}"
    return " · ".join(f"{rol.capitalize()}: {p.nombre}/{p.modelo}" for rol, p in activos.items())


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
def _barra_lateral(df: pd.DataFrame, manifiesto: dict, proveedores: dict, hay_llm: bool) -> bool:
    """Dibuja el formulario. Devuelve True si se ha pulsado «Buscar sustitutos»."""
    s = st.session_state
    with st.sidebar:
        st.caption(f"Recomendador de sustitutos por similitud estadística · Temporada {C.TEMPORADA}")
        st.text_area("Describe lo que buscas", key="texto_nl", height=92,
                     placeholder="Ej.: sustituto joven de Pedri en la Bundesliga, más creador",
                     help="El Secretario técnico interpreta la petición, rellena el formulario de abajo y lanza la búsqueda. "
                          "Sin modelo de lenguaje se interpreta con reglas.")
        st.button("Interpretar y buscar", on_click=_pedir_interpretacion, icon=":material/chat:", width="stretch")
        st.divider()

        etiquetas = _etiquetas(manifiesto["dataset_version"])
        objetivo_id = st.selectbox("Jugador de referencia", options=list(etiquetas), format_func=etiquetas.get,
                                   index=None, placeholder="Escribe un nombre…", key="f_objetivo",
                                   help="Se muestran equipo, liga, posición y minutos para distinguir homónimos y "
                                        "jugadores con varios equipos.")
        grupo = df.loc[df["registro_id"] == objetivo_id, "grupo_posicion"].iloc[0] if objetivo_id else None
        roles = C.ROLES[grupo] if grupo else {C.ROL_DEFECTO: C.ROLES["MF"][C.ROL_DEFECTO]}
        if s.get("f_rol") not in roles:
            s["f_rol"] = C.ROL_DEFECTO
        rol = st.selectbox("Rol buscado", list(roles), format_func=lambda r: roles[r]["nombre"], key="f_rol",
                           disabled=grupo is None,
                           help="Cada rol da más peso a unas variables al calcular el parecido. Los roles "
                                "dependen de la posición del jugador de referencia.")
        if grupo:
            st.caption(roles[rol]["descripcion"])
        if s.get("destino_de") != objetivo_id:        # al cambiar de jugador, el destino es su club
            s["f_destino"] = df.loc[df["registro_id"] == objetivo_id, "equipo"].iloc[0] if objetivo_id else SIN_DESTINO
            s["destino_de"] = objetivo_id
        st.selectbox("Club de destino", [SIN_DESTINO] + encaje.clubes(df), key="f_destino",
                     help="Plantilla en la que tendría que encajar el sustituto. El Analista de plantilla valora "
                          "el encaje (juego del equipo, hueco que cubre, compatriotas y liga) y esa nota pondera "
                          f"la puntuación final con un peso de {fmt(C.PESO_PLANTILLA, 2)}.")
        st.multiselect("Ligas permitidas", C.LIGAS_NORMALIZADAS, key="f_ligas")
        st.checkbox("Limitar edad máxima", key="f_limitar_edad")
        st.slider("Edad máxima", EDAD_MIN, EDAD_MAX, key="f_edad", disabled=not s["f_limitar_edad"],
                  help="Si se activa, se excluyen los jugadores sin edad registrada.")
        st.number_input("Minutos mínimos", min_value=MINUTOS_MIN, max_value=MINUTOS_MAX, step=90, key="f_minutos")
        st.toggle("Usar análisis con IA", key="f_ia", disabled=not hay_llm,
                  help="Desactívalo para comparar con el ranking puramente numérico.")

        with st.expander("Opciones avanzadas"):
            st.toggle("Aprender de mis valoraciones", key="f_aprendizaje",
                      help="Aplica lo aprendido de tus estrellas: pesos de las variables, adaptación de la "
                           "búsqueda a lo que te gustó para esta referencia y descarte de los que puntuaste bajo.")
            st.toggle("Ocultar candidatos ya valorados", key="f_ocultar", disabled=not s["f_aprendizaje"],
                      help="No vuelve a proponer a quienes ya has puntuado para esta referencia.")
            st.toggle("Solo el mismo estilo de juego", key="f_estilo",
                      help="Restringe los candidatos al estilo de juego (agrupamiento k-means) de la referencia.")
            st.slider("Diversidad de la preselección", 0.0, DIVERSIDAD_MAX, step=0.1, key="f_diversidad",
                      help="0 = los 10 más parecidos. Con más diversidad se evitan candidatos casi idénticos "
                           "entre sí (selección MMR), a costa de algo de parecido.")
            st.toggle("Revisión de las fichas (Jefe de scouting)", key="f_critico", disabled=not (hay_llm and s["f_ia"]),
                      help="Un cuarto agente comprueba las fichas del director y, si encuentra errores, le pide "
                           "una corrección. Añade una o dos llamadas al modelo.")
            st.checkbox("Ajustar los pesos a mano", key="f_pesos_mano", disabled=grupo is None)
            if grupo and s["f_pesos_mano"]:
                if s.get("pesos_de") != (grupo, rol):          # al cambiar de rol se parte de sus pesos
                    for v, peso in pesos_de_rol(grupo, rol).items():
                        s[f"f_peso_{grupo}_{v}"] = float(min(C.PESO_MAXIMO_VARIABLE, peso))
                    s["pesos_de"] = (grupo, rol)
                for v in C.PERFILES[grupo]:
                    st.slider(C.ETIQUETAS_CORTAS.get(v, v), 0.0, C.PESO_MAXIMO_VARIABLE, step=0.25,
                              key=f"f_peso_{grupo}_{v}", help=C.VARIABLES[v][0])

        buscar = st.button("Buscar sustitutos", type="primary", disabled=objetivo_id is None, width="stretch",
                           icon=":material/search:")
        st.caption(_descripcion_modelos(proveedores))
    return buscar


def _consulta_del_formulario(df: pd.DataFrame, **extras) -> Consulta:
    s = st.session_state
    grupo = df.loc[df["registro_id"] == s["f_objetivo"], "grupo_posicion"].iloc[0]
    pesos = None
    if s.get("f_pesos_mano"):
        pesos = {v: float(s.get(f"f_peso_{grupo}_{v}", 1.0)) for v in C.PERFILES[grupo]}
    return Consulta(
        objetivo_registro_id=s["f_objetivo"], ligas_permitidas=list(s["f_ligas"]),
        minutos_minimos=int(s["f_minutos"]), edad_maxima=int(s["f_edad"]) if s["f_limitar_edad"] else None,
        usar_llm=bool(s["f_ia"]), rol=s["f_rol"], pesos_personalizados=pesos,
        usar_aprendizaje=bool(s["f_aprendizaje"]),
        ocultar_valorados=bool(s["f_ocultar"]) and bool(s["f_aprendizaje"]),
        solo_mismo_estilo=bool(s["f_estilo"]), diversidad=float(s["f_diversidad"]),
        usar_critico=bool(s["f_critico"]),
        equipo_destino=None if s.get("f_destino", SIN_DESTINO) == SIN_DESTINO else s["f_destino"], **extras)


# --------------------------------------------------------------------------
# Ejecución con avance por agente
# --------------------------------------------------------------------------
def _ejecutar(consulta: Consulta, df: pd.DataFrame, manifiesto: dict, modelos: dict, proveedores: dict,
              ejecuciones_previas: list[dict] | None = None) -> None:
    with st.status("Buscando sustitutos…", expanded=True) as estado:
        def progreso(fase: str, paso: str, detalle: str) -> None:
            nombre = NOMBRE_FASE.get(fase, fase)
            if paso == "inicio":
                estado.update(label=f"{nombre}{' · ' + detalle if detalle else ''}…")
            elif paso == "fin":
                estado.write(f":material/check_circle: **{nombre}** · {detalle}")
            else:
                estado.write(f":material/skip_next: {nombre}: {detalle}")

        resultado = orquestador.ejecutar_consulta(
            consulta, df, manifiesto["dataset_version"], proveedores.get("analista"), proveedores.get("director"),
            cache=llm.CacheLLM(), proveedor_critico=proveedores.get("critico"),
            proveedor_plantilla=proveedores.get("plantilla"), estilos=modelos, progreso=progreso,
            ejecuciones_previas=ejecuciones_previas)
        segundos = fmt(resultado["duracion_total_ms"] / 1000, 1)
        if resultado["estado"] == "rechazada":
            estado.update(label="La consulta no se ha podido ejecutar", state="error", expanded=False)
        else:
            modo = "ranking híbrido con IA" if resultado["modo_ranking"] == "hibrido" else "ranking numérico"
            estado.update(label=f"Búsqueda terminada en {segundos} s · {modo}", state="complete", expanded=False)
    st.session_state["resultado"] = resultado
    st.session_state["valoraciones_pendientes"] = False


# --------------------------------------------------------------------------
# Portada
# --------------------------------------------------------------------------
def _portada(manifiesto: dict, modelos: dict) -> None:
    st.header("Encuentra sustitutos con un perfil estadístico parecido")
    st.markdown(
        "1. Elige un **jugador de referencia** en la barra lateral (o descríbelo con tus palabras).\n"
        "2. Elige el **rol** que buscas y ajusta las ligas, la edad máxima y los minutos mínimos.\n"
        "3. Pulsa **Buscar sustitutos** y **valora con estrellas** a los candidatos: el sistema aprende de ti.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros en el dataset", fmt(manifiesto["filas"], 0))
    c2.metric("Jugadores de campo válidos", fmt(manifiesto["registros_validos_campo"], 0))
    c3.metric("Estilos de juego", sum(m.k for m in modelos.values()))
    c4.metric("Versión del dataset", manifiesto["dataset_version"])
    st.subheader("Prueba con una petición en lenguaje natural")
    columnas = st.columns(len(EJEMPLOS))
    for columna, ejemplo in zip(columnas, EJEMPLOS):
        columna.button(ejemplo, on_click=_pedir_interpretacion, args=(ejemplo,), width="stretch", wrap=True)
    st.subheader("Cómo funciona")
    st.markdown(
        "- **Secretario técnico**: convierte una petición en lenguaje natural en los filtros de la búsqueda.\n"
        "- **Ojeador de datos**: similitud coseno ponderada por el rol sobre las estadísticas por "
        "90 minutos → preselección de 10 candidatos. Aplica lo aprendido de tus valoraciones.\n"
        "- **Analista de rendimiento**: valora a los 10 candidatos con un modelo de lenguaje, usando "
        "solo los datos (anonimizados).\n"
        "- **Analista de plantilla**: valora cómo encajaría cada candidato en la plantilla del club de destino "
        "y esa nota pondera la puntuación final.\n"
        "- **Director deportivo**: redacta la ficha de ojeador de los tres mejores.\n"
        "- **Jefe de scouting**: comprueba las fichas y, si hay errores, pide una corrección.\n\n"
        "Si un modelo de lenguaje falla, la consulta termina igualmente en **modo numérico**. "
        "Cada consulta se guarda en `salidas/` para el cuadro de mando de Power BI y en el **Historial**.")


# --------------------------------------------------------------------------
# Resultado
# --------------------------------------------------------------------------
def _al_valorar(clave: str, registro_id: str) -> None:
    valor = st.session_state.get(clave)
    comun.guardar_valoracion(st.session_state["resultado"], registro_id, 0 if valor is None else int(valor) + 1)


def _al_editar_tabla(clave: str, registros: list[str]) -> None:
    for fila, estrellas in comun.estrellas_editadas(st.session_state.get(clave), "Tu valoración"):
        comun.guardar_valoracion(st.session_state["resultado"], registros[fila], estrellas)


def _estrellas_de(resultado: dict) -> dict[str, int]:
    """Estrellas vigentes de los candidatos de esta consulta, por registro_id."""
    guardadas = comun.estrellas_guardadas(resultado["dataset_version"])
    persona_ref = resultado["recuperacion"]["objetivo"]["persona_id"]
    return {c["registro_id"]: guardadas[(persona_ref, c["persona_id"])] for c in resultado["ranking"]
            if (persona_ref, c["persona_id"]) in guardadas}


def _pestana_fichas(resultado: dict, estrellas: dict[str, int]) -> None:
    consulta_id = resultado["consulta"]["consulta_id"]
    version = comun.version_valoraciones()
    hibrido = resultado["modo_ranking"] == "hibrido"
    if resultado["resumen_consulta"]:
        st.markdown(f"> {resultado['resumen_consulta']}")
    columnas = st.columns(len(resultado["top_final"]))
    for col, c in zip(columnas, resultado["top_final"]):
        ficha = c["ficha"]
        with col.container(border=True):
            st.markdown(f"#### {c['ranking_final']}. {c['nombre']}")
            st.caption(comun.ficha_jugador(c))
            a, b = st.columns(2)
            a.metric("Índice de similitud", fmt(c["score_numerico"], 1),
                     help="Similitud coseno ponderada (0-100) con el perfil buscado. Similitud base, sin pesos "
                          f"ni personalización: {fmt(c.get('score_base', c['score_numerico']), 1)}.")
            if hibrido or c.get("plantilla"):
                partes = []
                if hibrido:
                    partes.append(f"{fmt(C.PESO_NUMERICO, 1)} × similitud + {fmt(C.PESO_CUALITATIVO, 1)} × valoración "
                                  f"del Analista de rendimiento ({fmt(c['score_cualitativo'], 1)})")
                if c.get("plantilla"):
                    partes.append(f"ponderado con el encaje en la plantilla ({fmt(c['score_plantilla'], 1)}, "
                                  f"peso {fmt(C.PESO_PLANTILLA, 2)})")
                b.metric("Puntuación final", fmt(c["score_final"], 1), help="; ".join(partes))
            if c.get("plantilla"):
                st.caption(f"Encaje en la plantilla: **{fmt(c['score_plantilla'], 1)}** · {c['plantilla']['comentario']}")
            st.markdown(f"**{comun.ICONO_VEREDICTO.get(ficha['veredicto'], '')} "
                        f"{C.VEREDICTOS.get(ficha['veredicto'], '')}**")
            st.write(ficha["resumen"])
            for titulo, campo in (("Fortalezas", "fortalezas"), ("Diferencias", "diferencias_relevantes"),
                                  ("Reservas", "reservas")):
                if ficha.get(campo):
                    st.markdown(f"**{titulo}**\n" + "\n".join(f"- {x}" for x in ficha[campo]))
            revision = ficha.get("revision") or {}
            if revision.get("estado") in TEXTO_REVISION:
                color, icono, texto = TEXTO_REVISION[revision["estado"]]
                st.badge(texto, icon=icono, color=color)
                if revision.get("problemas"):
                    with st.expander("Observaciones del revisor"):
                        for problema in revision["problemas"]:
                            st.markdown(f"- {problema}")
            st.caption("Ficha redactada por el Director deportivo" if ficha["origen"] == "llm"
                       else "Ficha generada automáticamente (sin IA)")
            st.divider()
            st.markdown("**¿Encaja como sustituto?**")
            clave = f"val_{consulta_id}_{c['registro_id']}_{version}"
            actual = estrellas.get(c["registro_id"])
            st.feedback("stars", key=clave, default=None if not actual else actual - 1,
                        on_change=_al_valorar, args=(clave, c["registro_id"]))
            if actual and actual <= C.ESTRELLAS_NEGATIVA:
                st.caption("No volverá a proponerse para esta referencia.")
    st.caption("Tus estrellas se guardan al momento. En la pestaña **Preselección** puedes valorar a los 10 "
               "candidatos: cuantos más valores (también los que no te convencen), más aprende el sistema.")


def _pestana_plantilla(resultado: dict) -> None:
    p = resultado.get("plantilla")
    if not p:
        st.info("Esta búsqueda no tiene club de destino. Elige uno en la barra lateral para que el Analista de "
                "plantilla valore el encaje de los candidatos.")
        return
    st.markdown(f"**Plantilla de destino: {p['equipo']}** ({p['liga']}) · {p['jugadores']} jugadores de campo"
                + (f" · edad media {fmt(p['edad_media'], 1)} años" if p.get("edad_media") else ""))
    if p.get("diagnostico"):
        st.markdown(f"> {p['diagnostico']}")
    a, b = st.columns(2)
    if p.get("estilos_de_la_linea"):
        a.markdown("**Estilos entre los compañeros de línea** (sin la referencia)\n" + "\n".join(
            f"- {estilo}: {fmt(parte * 100, 0)} % de los minutos" for estilo, parte in p["estilos_de_la_linea"].items()))
        if p.get("estilo_que_deja_la_referencia"):
            a.caption(f"La referencia deja libre el estilo «{p['estilo_que_deja_la_referencia']}».")
    if p.get("nacionalidades"):
        b.markdown("**Nacionalidades más presentes**\n" + "\n".join(
            f"- {pais.split()[-1]}: {n}" for pais, n in p["nacionalidades"].items()))
    tabla = pd.DataFrame([{
        "#": c["ranking_final"], "Jugador": c["nombre"], "Equipo": c["equipo"],
        **{NOMBRE_INDICADOR[k]: v for k, v in c["plantilla"]["indicadores"].items()},
        "Valoración del analista (0-4)": c["plantilla"]["encaje_llm"],
        "Nota de plantilla": round(c["score_plantilla"], 1),
        "Sin plantilla": round(c["score_sin_plantilla"], 1), "Puntuación final": round(c["score_final"], 1),
        "Comentario": c["plantilla"]["comentario"],
    } for c in resultado["ranking"]])
    st.dataframe(tabla, hide_index=True)
    origen = "el Analista de plantilla (modelo de lenguaje) y los indicadores" if p["origen"] == "llm" \
        else "los indicadores (sin modelo de lenguaje)"
    st.caption(f"Nota de plantilla calculada con {origen}. Puntuación final = {fmt(1 - p['peso'], 2)} × puntuación "
               f"sin plantilla + {fmt(p['peso'], 2)} × nota de plantilla. Los indicadores van de 0 a 100: parecido "
               "entre el juego de su equipo actual y el del club de destino; si cubre el estilo que deja la "
               "referencia o aporta uno escaso en la línea; compatriotas en la plantilla (compañeros de selección "
               "potenciales); y si ya juega en la misma liga. El dataset no tiene datos directos de química, así "
               "que es una aproximación.")


def _pestana_comparador(resultado: dict) -> None:
    rec, obj = resultado["recuperacion"], resultado["recuperacion"]["objetivo"]
    nombres = {c["registro_id"]: f"{c['ranking_final']}. {c['nombre']} ({c['equipo']})" for c in resultado["ranking"]}
    elegidos_id = st.multiselect(
        "Candidatos a comparar (hasta 3)", list(nombres), format_func=nombres.get, max_selections=3,
        default=[c["registro_id"] for c in resultado["top_final"]][:3],
        key=f"comparar_{resultado['consulta']['consulta_id']}")
    elegidos = [c for c in resultado["ranking"] if c["registro_id"] in elegidos_id]
    if not elegidos:
        st.info("Elige al menos un candidato para compararlo con la referencia.")
        return
    izq, der = st.columns([3, 2])
    with izq:
        st.plotly_chart(graficos.radar_percentiles(obj, elegidos, rec["variables"]))
        st.caption("Percentiles dentro de su grupo de posición (jugadores con los minutos mínimos).")
    with der:
        filas = []
        for v in rec["variables"]:
            fila = {"Variable": C.ETIQUETAS_CORTAS.get(v, v), "Peso": round(rec["pesos"].get(v, 1.0), 2),
                    obj["nombre"]: round(obj["valores"][v], 2)}
            for c in elegidos:
                fila[c["nombre"]] = round(c["valores"][v], 2)
            filas.append(fila)
        st.dataframe(pd.DataFrame(filas), hide_index=True)
        st.caption("Valores por 90 minutos (la precisión de pase, en %). «Peso» es la importancia de la "
                   "variable en esta búsqueda.")
    con_analisis = [c for c in elegidos if c.get("analisis")]
    if con_analisis:
        st.markdown("**Análisis del Analista de rendimiento**")
        for c in con_analisis:
            a = c["analisis"]
            with st.expander(f"{c['nombre']} · valoración {fmt(c['score_cualitativo'], 1)}", expanded=len(con_analisis) == 1):
                st.write(a.get("resumen", ""))
                for d in a.get("dimensiones", []):
                    if d.get("puntuacion") is not None:
                        st.markdown(f"- {restaurar_tildes(d['nombre'].replace('_', ' '))}: **{d['puntuacion']}/4** · "
                                    f"{d.get('justificacion', '')}")
                for titulo, campo in (("Semejanzas", "semejanzas"), ("Diferencias", "diferencias")):
                    if a.get(campo):
                        st.markdown(f"**{titulo}:** " + " ".join(a[campo]))


def _pestana_mapa(resultado: dict, modelos: dict, tabla_estilos: pd.DataFrame) -> None:
    rec, consulta = resultado["recuperacion"], resultado["consulta"]
    modelo = modelos.get(rec["grupo"])
    if modelo is None:
        st.info("No hay modelo de estilos de juego para este grupo de posición.")
        return
    visibles = {c["registro_id"] for c in resultado["ranking"]} | {rec["objetivo"]["registro_id"]}
    tabla = tabla_estilos[(tabla_estilos["grupo_posicion"] == rec["grupo"])
                          & ((tabla_estilos["minutos"] >= consulta["minutos_minimos"])
                             | tabla_estilos["registro_id"].isin(visibles))]
    st.plotly_chart(graficos.mapa_estilos(tabla, modelo.nombres, E.eje_pca(modelo, 0), E.eje_pca(modelo, 1),
                                          rec["objetivo"]["registro_id"], resultado["ranking"]))
    st.caption(f"Cada punto es un jugador del grupo {C.GRUPOS[rec['grupo']].lower()}, situado según las dos "
               "primeras componentes principales de sus estadísticas y coloreado por su estilo de juego "
               "(k-means). La estrella es la referencia y los círculos numerados, los candidatos por puesto. "
               "La similitud se calcula con todas las variables, no solo con las dos del mapa.")
    if rec["objetivo"].get("estilo"):
        i = modelo.nombres.index(rec["objetivo"]["estilo"]) if rec["objetivo"]["estilo"] in modelo.nombres else None
        mismos = sum(1 for c in resultado["ranking"] if c.get("estilo") == rec["objetivo"]["estilo"])
        st.markdown(f"**Estilo de {rec['objetivo']['nombre']}: {rec['objetivo']['estilo']}.** "
                    + (modelo.descripciones[i] if i is not None else "")
                    + f" {mismos} de los {len(resultado['ranking'])} candidatos comparten ese estilo.")


def _pestana_preseleccion(resultado: dict, estrellas: dict[str, int]) -> None:
    rec = resultado["recuperacion"]
    hibrido = resultado["modo_ranking"] == "hibrido"
    tabla = pd.DataFrame([{
        "#": c["ranking_final"], "Jugador": c["nombre"], "Equipo": c["equipo"], "Liga": c["liga"],
        "Pos.": c["posicion_original"], "Edad": c["edad_registrada"], "Minutos": c["minutos"],
        "Estilo": c.get("estilo"), "Similitud": round(c["score_numerico"], 1),
        "Similitud base": round(c.get("score_base", c["score_numerico"]), 1),
        "Valoración IA": None if c["score_cualitativo"] is None else round(c["score_cualitativo"], 1),
        "Plantilla": None if c.get("score_plantilla") is None else round(c["score_plantilla"], 1),
        "Puntuación final": round(c["score_final"], 1), "Puesto numérico": c["ranking_preliminar"],
        "Tu valoración": estrellas.get(c["registro_id"]),
    } for c in resultado["ranking"]])
    con_plantilla = resultado.get("plantilla") is not None
    if not con_plantilla:
        tabla = tabla.drop(columns=["Plantilla"])
    if not hibrido:
        tabla = tabla.drop(columns=["Valoración IA"] + ([] if con_plantilla else ["Puntuación final",
                                                                                 "Puesto numérico"]))
    if (tabla["Similitud"] == tabla["Similitud base"]).all():
        tabla = tabla.drop(columns=["Similitud base"])
    clave = f"tabla_{resultado['consulta']['consulta_id']}_{comun.version_valoraciones()}"
    st.data_editor(
        tabla, hide_index=True, key=clave, disabled=[c for c in tabla.columns if c != "Tu valoración"],
        on_change=_al_editar_tabla, args=(clave, [c["registro_id"] for c in resultado["ranking"]]),
        column_config={"Tu valoración": st.column_config.NumberColumn(
            "Tu valoración ★", min_value=1, max_value=5, step=1, format="%d ★",
            help="Escribe de 1 a 5 estrellas (o borra la celda para retirar la valoración).")})
    st.caption("Escribe de 1 a 5 en **Tu valoración** para puntuar a cualquier candidato. "
               "«Similitud base» es el coseno clásico con la referencia, sin pesos ni personalización.")
    exclusiones = {k.replace("_", " "): v for k, v in rec["exclusiones"].items() if v}
    if exclusiones:
        st.caption("Jugadores excluidos antes de calcular la similitud: "
                   + " · ".join(f"{k}: {v}" for k, v in exclusiones.items()) + ".")


def _pestana_calculo(resultado: dict) -> None:
    rec, consulta = resultado["recuperacion"], resultado["consulta"]
    p = resultado.get("personalizacion") or {}
    izq, der = st.columns(2)
    with izq:
        st.markdown(f"**Importancia de cada variable** · rol: {rec['rol_nombre']}")
        st.plotly_chart(graficos.barras_pesos(rec["pesos_rol"], rec["pesos"]))
        st.caption("La similitud es un coseno ponderado: las variables con más peso cuentan más. "
                   "«Aplicado» = pesos del rol × lo aprendido de tus valoraciones.")
    with der:
        st.markdown("**Lo aprendido de tus valoraciones**")
        if not consulta.get("usar_aprendizaje"):
            st.info("La personalización está desactivada en esta búsqueda.")
        elif not p.get("valoraciones_grupo"):
            st.info("Todavía no has valorado candidatos de este grupo de posición. Puntúa con estrellas y la "
                    "siguiente búsqueda se adaptará.")
        else:
            aprendizaje = p.get("aprendizaje")
            if aprendizaje:
                st.plotly_chart(graficos.barras_multiplicadores(aprendizaje["multiplicadores"]))
                st.caption("Aprendido de "
                           + comun.plural(aprendizaje["n_pares"], "comparación", "comparaciones")
                           + " entre candidatos ("
                           + comun.plural(aprendizaje["n_valoraciones"], "valoración", "valoraciones")
                           + "). Con estos pesos la similitud ordena "
                           f"bien el {fmt(aprendizaje['acierto_despues'] * 100, 0)} % de tus comparaciones "
                           f"(antes: {fmt(aprendizaje['acierto_antes'] * 100, 0)} %).")
            else:
                st.info("Aún no hay comparaciones suficientes para aprender pesos: valora con distinta "
                        "puntuación a dos candidatos de una misma búsqueda.")
    if rec.get("desplazamiento_consulta"):
        st.markdown("**Búsqueda adaptada a esta referencia** · "
                    + comun.plural(p.get("positivas_referencia", 0), "valoración positiva", "valoraciones positivas")
                    + " y " + comun.plural(p.get("negativas_referencia", 0), "negativa"))
        st.plotly_chart(graficos.barras_desplazamiento(rec["desplazamiento_consulta"]))
        st.caption("Realimentación de Rocchio: el perfil buscado se acerca a los candidatos que te gustaron y se "
                   "aleja de los que no. En verde, las variables en las que ahora se busca más que en la "
                   "referencia; en rojo, menos.")
    st.markdown("**Fórmulas**")
    st.markdown(
        f"- Índice de similitud = 100 × (coseno ponderado + 1) / 2, sobre variables estandarizadas con los "
        f"{rec['poblacion_referencia']} jugadores del grupo con al menos {fmt(consulta['minutos_minimos'], 0)} minutos.\n"
        f"- Puntuación final (modo híbrido) = {fmt(consulta['peso_numerico'], 1)} × índice de similitud + "
        f"{fmt(consulta['peso_cualitativo'], 1)} × valoración del Analista de rendimiento.\n"
        + (f"- Con club de destino: puntuación final = {fmt(1 - C.PESO_PLANTILLA, 2)} × puntuación anterior + "
           f"{fmt(C.PESO_PLANTILLA, 2)} × nota de plantilla.\n" if resultado.get("plantilla") else "")
        + (f"- Diversidad (MMR) = {fmt(consulta['diversidad'], 1)}: la preselección evita candidatos casi "
           "idénticos entre sí.\n" if consulta.get("diversidad") else ""))


def _pestana_trazas(resultado: dict) -> None:
    if resultado["ejecuciones"]:
        trazas = pd.DataFrame(resultado["ejecuciones"]).drop(columns=["consulta_id"], errors="ignore")
        trazas["agente"] = trazas["agente"].map(lambda a: C.NOMBRES_AGENTES.get(a, a))
        st.dataframe(trazas, hide_index=True)
        llamadas = [e for e in resultado["ejecuciones"] if e["estado"] != "cache"]
        tokens = sum((e.get("tokens_entrada") or 0) + (e.get("tokens_salida") or 0) for e in llamadas)
        st.caption(f"{len(llamadas)} llamadas a modelos de lenguaje · {fmt(tokens, 0)} tokens · "
                   f"{sum(1 for e in resultado['ejecuciones'] if e['estado'] == 'cache')} respuestas de la caché.")
    else:
        st.write("No se ha llamado a ningún modelo de lenguaje en esta consulta.")
    revision = resultado.get("revision")
    if revision and revision.get("valoracion_global"):
        st.markdown(f"**Revisión del Jefe de scouting:** {revision['valoracion_global']}")
    if resultado["incidencias"]:
        st.markdown("**Incidencias**")
        st.dataframe(pd.DataFrame(resultado["incidencias"]), hide_index=True)
    st.markdown("**Limitaciones**")
    for x in resultado["limitaciones_generales"]:
        st.markdown(f"- {x}")


def _descargas(resultado: dict, estrellas: dict[str, int]) -> None:
    nombre = "".join(ch if ch.isalnum() else "_" for ch in resultado["recuperacion"]["objetivo"]["nombre"])
    a, b, _ = st.columns([1, 1, 3])
    try:
        import fpdf  # noqa: F401
        a.download_button("Informe en PDF", data=lambda: informe.generar_pdf(resultado, estrellas),
                          file_name=f"ai_scout_{nombre}.pdf", mime="application/pdf", on_click="ignore",
                          icon=":material/picture_as_pdf:", width="stretch")
    except ImportError:
        a.caption("Instala fpdf2 para descargar el informe en PDF.")
    b.download_button("Top 10 en CSV", data=informe.csv_preseleccion(resultado),
                      file_name=f"ai_scout_{nombre}.csv", mime="text/csv", on_click="ignore",
                      icon=":material/download:", width="stretch")


def _mostrar_interpretacion(resultado: dict | None) -> None:
    info = st.session_state.get("interpretacion")
    if not info:
        return
    if not info["encontrado"]:
        st.error("No he podido lanzar la búsqueda a partir de tu petición:\n\n"
                 + "\n".join(f"- {n}" for n in info["notas"])
                 + "\n\nEscribe el nombre del jugador de referencia o elígelo en la barra lateral.")
        return
    if resultado is None or resultado["consulta"]["consulta_id"] != info["consulta_id"]:
        return
    origen = "Secretario técnico, con modelo de lenguaje" if info["origen"] == "llm" else "intérprete por reglas"
    texto = f"**He entendido** ({origen}): {info['resumen']}."
    if info["notas"]:
        texto += "\n\n" + "\n".join(f"- {n}" for n in info["notas"])
    if info["alternativas"]:
        texto += "\n\nOtros jugadores que encajan con el nombre: " + "; ".join(info["alternativas"][:3]) + "."
    st.info(texto, icon=":material/chat:")


def _mostrar(resultado: dict, modelos: dict, tabla_estilos: pd.DataFrame) -> None:
    consulta = resultado["consulta"]
    if resultado["estado"] == "rechazada":
        st.error("No se ha podido ejecutar la consulta:\n\n" + "\n".join(f"- {e}" for e in resultado["errores"]))
        return
    rec = resultado["recuperacion"]
    obj = rec["objetivo"]
    st.header(f"Sustitutos para {obj['nombre']}")
    st.caption(comun.ficha_jugador(obj))

    filtros = [", ".join(consulta["ligas_permitidas"]), f"≥ {fmt(consulta['minutos_minimos'], 0)} min"]
    if consulta["edad_maxima"] is not None:
        filtros.append(f"≤ {consulta['edad_maxima']} años")
    if consulta.get("solo_mismo_estilo"):
        filtros.append("mismo estilo de juego")
    st.markdown(f"**Grupo:** {C.GRUPOS[rec['grupo']]} · **Rol buscado:** {rec['rol_nombre']} · **Filtros:** "
                + " · ".join(filtros))
    m1, m2, m3 = st.columns(3)
    m1.metric("Jugadores comparados", rec["n_candidatos_validos"])
    m2.metric("Modo del ranking", "Híbrido (IA)" if resultado["modo_ranking"] == "hibrido" else "Numérico")
    m3.metric("Tiempo", f"{fmt(resultado['duracion_total_ms'] / 1000, 1)} s")

    p = resultado.get("personalizacion") or {}
    if p.get("valoraciones_grupo"):
        partes = []
        if p.get("pesos_aprendidos"):
            partes.append("pesos aprendidos de " + comun.plural(p["aprendizaje"]["n_pares"], "comparación",
                                                                "comparaciones"))
        if p.get("rocchio_aplicado"):
            partes.append("búsqueda adaptada con "
                          + comun.plural(p["valoraciones_referencia"], "valoración", "valoraciones")
                          + " de esta referencia")
        if p.get("descartados"):
            partes.append(comun.plural(p["descartados"], "candidato descartado", "candidatos descartados")
                          + " por ti")
        if partes:
            st.success("Búsqueda personalizada con tus valoraciones: " + " · ".join(partes) + ".",
                       icon=":material/psychology:")
    for aviso in resultado["avisos"]:
        st.warning(aviso)
    if resultado["estado"] == "sin_resultados":
        st.info("No hay recomendaciones porque ningún jugador cumple todas las restricciones. "
                "El sistema no relaja los filtros por su cuenta.")
        return

    if st.session_state.get("valoraciones_pendientes"):
        aviso, boton = st.columns([4, 1], vertical_alignment="center")
        aviso.info("Has valorado candidatos después de calcular esta lista. Recalcula para que el sistema "
                   "aplique lo que ha aprendido.", icon=":material/star:")
        boton.button("Recalcular", on_click=_pedir_recalculo, type="primary", width="stretch",
                     icon=":material/refresh:")

    estrellas = _estrellas_de(resultado)
    fichas, comparador, plantilla, mapa, preseleccion, calculo, trazas = st.tabs(
        ["Fichas de ojeador", "Comparador", "Encaje en la plantilla", "Mapa de estilos", "Preselección (Top 10)",
         "Cómo se ha calculado", "Trazas de los agentes"])
    with plantilla:
        _pestana_plantilla(resultado)
    with fichas:
        _pestana_fichas(resultado, estrellas)
        _descargas(resultado, estrellas)
    with comparador:
        _pestana_comparador(resultado)
    with mapa:
        _pestana_mapa(resultado, modelos, tabla_estilos)
    with preseleccion:
        _pestana_preseleccion(resultado, estrellas)
    with calculo:
        _pestana_calculo(resultado)
    with trazas:
        _pestana_trazas(resultado)
    st.caption(f"Consulta {consulta['consulta_id']} guardada en `salidas/` · Dataset {resultado['dataset_version']} · "
               "Los índices miden parecido estadístico, no la probabilidad de éxito de un fichaje.")


# --------------------------------------------------------------------------
# Página
# --------------------------------------------------------------------------
def pagina() -> None:
    df, manifiesto, modelos, tabla_estilos, proveedores = comun.contexto()
    hay_llm = proveedores.get("analista") is not None or proveedores.get("director") is not None
    _valores_iniciales(hay_llm)

    # Peticiones pendientes: se resuelven ANTES de crear los controles para poder rellenarlos
    _interpretar_peticion(df, proveedores)
    pendiente = st.session_state.pop("formulario_pendiente", None)
    if pendiente and not _volcar_al_formulario(pendiente, df):
        st.session_state.pop("ejecutar_pendiente", None)
        st.warning("El jugador de referencia de esa consulta no está en el dataset actual.")

    buscar = _barra_lateral(df, manifiesto, proveedores, hay_llm)
    extras = st.session_state.pop("ejecutar_pendiente", None)
    if (buscar or extras is not None) and st.session_state.get("f_objetivo"):
        extras = dict(extras or {})
        previas = extras.pop("ejecuciones", None)
        if buscar:
            st.session_state.pop("interpretacion", None)
        _ejecutar(_consulta_del_formulario(df, **extras), df, manifiesto, modelos, proveedores, previas)

    resultado = st.session_state.get("resultado")
    _mostrar_interpretacion(resultado)
    if resultado is None:
        _portada(manifiesto, modelos)
    else:
        _mostrar(resultado, modelos, tabla_estilos)
