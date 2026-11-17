"""Página «Evaluación»: resultados de evaluar_sistema.py con gráficos, y botón para volver a ejecutarla."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from ai_scout import config as C
from ai_scout import evaluacion as EV
from ai_scout import graficos

from . import comun

NOMBRES = {
    "grupo": "Grupo", "medida": "Medida", "consultas": "Consultas", "candidatos": "Candidatos", "mrr": "MRR",
    "acierto_1": "Acierto Top 1", "acierto_5": "Acierto Top 5", "acierto_10": "Acierto Top 10",
    "rango_mediano": "Puesto mediano", "jugadores": "Jugadores", "posiciones_distintas": "Posiciones distintas",
    "misma_posicion_top10": "Misma posición en el Top 10", "esperado_por_azar": "Esperado por azar",
    "mejora_sobre_azar": "Veces mejor que el azar", "estilos": "Estilos", "silueta": "Silueta",
    "varianza_mapa_2d": "Varianza del mapa", "mismo_estilo_top10": "Mismo estilo en el Top 10",
    "diversidad": "Diversidad pedida", "indice_similitud_medio": "Índice de similitud medio",
    "diversidad_intralista": "Diversidad dentro de la lista", "cobertura_catalogo": "Cobertura del catálogo",
    "gini": "Gini", "prueba": "Prueba", "detalle": "Detalle", "solapamiento_top10": "Top 10 en común",
    "candidatos_que_cambian": "Candidatos que cambian", "tipo_de_usuario": "Tipo de usuario simulado",
    "candidatos_valorados": "Candidatos valorados", "busquedas": "Búsquedas",
    "relevantes_sin_realimentacion": "Relevantes sin realimentación", "relevantes_con_rocchio": "Relevantes con Rocchio",
    "diferencia": "Diferencia", "escenario": "Escenario", "referencias_valoradas": "Referencias valoradas",
    "ndcg_10": "nDCG@10", "precision_10": "Precisión@10", "agente": "Agente", "tareas": "Tareas",
    "llamadas_al_modelo": "Llamadas al modelo", "respuestas_de_cache": "Respuestas de caché",
    "validas_al_primer_intento": "Válidas al primer intento", "tareas_resueltas": "Tareas resueltas",
    "duracion_media_s": "Duración media (s)", "tokens_medios": "Tokens medios", "estado": "Estado",
    "codigo_error": "Código de error", "veces": "Veces", "estado_consulta": "Estado de la consulta",
    "modo_ranking": "Modo del ranking", "consultas_hibridas": "Consultas híbridas",
    "tau_de_kendall_medio": "Tau de Kendall medio", "puestos_que_se_mueve_un_candidato": "Puestos que se mueve un candidato",
    "consultas_con_otro_top3": "Consultas con otro Top 3", "top3_en_comun": "Top 3 en común", "tramo": "Tramo",
    "valoraciones": "Valoraciones", "estrellas_medias": "Estrellas medias", "positivas": "Positivas",
    "negativas": "Negativas", "llamadas": "Llamadas", "reintentos": "Reintentos", "errores_de_api": "Errores de API",
    "referencia": "Referencia", "repeticiones": "Repeticiones",
    "top3_en_comun_entre_repeticiones": "Top 3 en común entre repeticiones", "mismo_top3": "Mismo Top 3",
    "desviacion_valoracion_ia": "Desviación de la valoración IA", "puestos_que_mueve_la_ia": "Puestos que mueve la IA",
    "k": "k", "inercia": "Inercia", "elegido": "Elegido",
}


def _tabla(tabla: pd.DataFrame | None, columnas: list[str] | None = None) -> None:
    if tabla is None or len(tabla) == 0:
        st.caption("Sin datos.")
        return
    t = tabla[columnas] if columnas else tabla
    numericas = t.select_dtypes("float").columns
    t = t.copy()
    t[numericas] = t[numericas].round(3)
    st.dataframe(t.rename(columns=lambda c: NOMBRES.get(c, c.replace("_", " ").capitalize())), hide_index=True)


def _ejecutar(df: pd.DataFrame, dataset_version: str, rapido: bool) -> None:
    with st.status("Evaluando el sistema…", expanded=True) as estado:
        def avisar(texto: str) -> None:
            estado.update(label=f"{texto}…")
            estado.write(f":material/check_circle: {texto}")

        tablas = EV.evaluar_todo(df, rapido=rapido, avisar=avisar)
        EV.escribir_resultados(tablas, dataset_version=dataset_version)
        estado.update(label="Evaluación terminada", state="complete", expanded=False)


def pagina() -> None:
    df, manifiesto, _, _, _ = comun.contexto()
    st.header("Evaluación del sistema")
    st.markdown(
        "No existe una «verdad» sobre qué jugador sustituye a cuál, así que el recomendador se evalúa desde "
        "varios ángulos: verdad-terreno aproximada, coherencia con los estilos, diversidad, sensibilidad a los "
        "parámetros, aprendizaje con usuarios simulados y registro de uso real. No se llama a ningún modelo de "
        "lenguaje (para eso: `python src/evaluar_sistema.py --llm 3`).")
    a, b, _ = st.columns([1, 1, 2], vertical_alignment="center")
    rapido = b.toggle("Versión rápida", help="Simulaciones más cortas (unos 30 segundos en lugar de 1-2 minutos).")
    if a.button("Ejecutar la evaluación", type="primary", icon=":material/monitoring:", width="stretch"):
        _ejecutar(df, manifiesto["dataset_version"], rapido)
    tablas = EV.leer_resultados()
    if not tablas:
        st.info("Todavía no hay resultados. Pulsa «Ejecutar la evaluación».", icon=":material/monitoring:")
        return
    informe = C.DIR_SALIDAS / C.SUBDIR_EVALUACION / "informe.md"
    if informe.exists():
        fecha = datetime.fromtimestamp(informe.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        st.caption(f"Última evaluación: {fecha} · tablas e informe en `salidas/evaluacion/`.")

    similitud, estilos, listas, aprendizaje, uso, texto = st.tabs(
        ["Calidad de la similitud", "Estilos de juego", "Diversidad y sensibilidad", "Aprendizaje", "Uso real",
         "Informe"])

    with similitud:
        st.markdown("**Autorrecuperación.** Un jugador que cambió de equipo aparece dos veces en el dataset. "
                    "Se busca con uno de sus registros y se mira en qué puesto sale el otro.")
        auto = tablas.get("autorrecuperacion")
        if auto is not None and len(auto):
            todos = auto[auto["grupo"] == "Todos"]
            st.plotly_chart(graficos.barras_comparacion(todos, "medida", {"mrr": "MRR", "acierto_10": "Acierto en el Top 10"}))
            _tabla(auto)
            st.caption("Hay pocos casos y un mismo jugador puede cambiar de papel al cambiar de equipo, así que "
                       "el resultado es orientativo: lo importante es la distancia respecto al azar.")
        st.markdown("**Posición detallada.** Etiqueta que la similitud no usa (por ejemplo «MF,FW»): ¿los "
                    "candidatos del Top 10 la comparten con la referencia más que por azar?")
        _tabla(tablas.get("posicion_detallada"))

    with estilos:
        siluetas = tablas.get("estilos_siluetas")
        if siluetas is not None and len(siluetas):
            st.plotly_chart(graficos.curvas_silueta(siluetas))
            st.caption("La silueta es baja para cualquier k: los estilos son una partición orientativa de un "
                       "continuo. El rombo marca el k elegido (interpretabilidad con una silueta casi igual).")
        st.markdown("**Coherencia entre similitud y estilos.** Proporción de candidatos del Top 10 que comparten "
                    "estilo con la referencia.")
        _tabla(tablas.get("estilos_coherencia"))

    with listas:
        st.markdown("**Cobertura y diversidad.** Cada jugador hace de referencia una vez. Con más diversidad "
                    "(MMR) las listas son más variadas y las recomendaciones se concentran menos.")
        _tabla(tablas.get("diversidad"))
        st.markdown("**Sensibilidad.** Cuánto cambia el Top 10 al elegir un rol o al escalar con otro umbral "
                    "de minutos.")
        _tabla(tablas.get("sensibilidad"))

    with aprendizaje:
        curvas = tablas.get("aprendizaje")
        if curvas is not None and len(curvas):
            st.markdown("**Aprendizaje de los pesos con usuarios simulados.** El usuario simulado prefiere a los "
                        "candidatos según los pesos de un rol que el sistema desconoce. Se mide, en referencias "
                        "que no ha valorado, cuánto se parece el Top 10 del sistema a su Top 10 ideal.")
            medida = st.segmented_control("Medida", ["precision_10", "ndcg_10"], format_func=NOMBRES.get,
                                          default="precision_10", key="evaluacion_medida") or "precision_10"
            st.plotly_chart(graficos.curvas_aprendizaje(curvas, medida))
            st.caption("El sistema aprende rápido si el usuario valora la lista entera, apenas cambia si solo "
                       "valora el Top 3 (casi no hay comparaciones) y no empeora con valoraciones ruidosas.")
        rocchio = tablas.get("rocchio")
        if rocchio is not None and len(rocchio):
            st.markdown("**Realimentación de Rocchio.** Proporción de los 10 candidatos siguientes (aún no "
                        "vistos) que están entre los 30 preferidos del usuario simulado.")
            t = rocchio.assign(caso=rocchio["tipo_de_usuario"] + " · valora " + rocchio["candidatos_valorados"].astype(str))
            st.plotly_chart(graficos.barras_comparacion(t, "caso", {"relevantes_sin_realimentacion": "Sin realimentación",
                                                                    "relevantes_con_rocchio": "Con Rocchio"}))
            _tabla(rocchio)

    with uso:
        st.markdown("Indicadores calculados con tus consultas reales guardadas en `salidas/`.")
        hay = False
        for nombre, titulo in (("uso_agentes", "Agentes LLM"), ("uso_errores", "Fallos de los agentes"),
                               ("uso_consultas", "Consultas por estado y modo"),
                               ("uso_numerico_vs_hibrido", "Ranking numérico frente a híbrido"),
                               ("uso_valoraciones", "Valoraciones del usuario"),
                               ("llm_agentes", "Agentes LLM en vivo (--llm)"),
                               ("llm_estabilidad", "Estabilidad entre repeticiones (--llm)")):
            if nombre in tablas:
                hay = True
                st.markdown(f"**{titulo}**")
                _tabla(tablas[nombre])
        if not hay:
            st.caption("Todavía no hay consultas guardadas con las que calcular estos indicadores.")

    with texto:
        if informe.exists():
            contenido = informe.read_text(encoding="utf-8")
            st.download_button("Descargar informe (Markdown)", contenido, file_name="evaluacion_ai_scout.md",
                               mime="text/markdown", on_click="ignore", icon=":material/download:")
            st.markdown(contenido)
