"""Página «Historial»: consultas guardadas, que se pueden reabrir sin recalcular o repetir."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_scout import exportador

from . import comun

MODOS = {"hibrido": "Híbrido (IA)", "numerico": "Numérico"}
ESTADOS = {"completada": "Completada", "parcial": "Parcial", "sin_resultados": "Sin resultados"}


def _abrir(consulta_id: str, repetir: bool, dataset_version: str) -> None:
    resultado = exportador.cargar_historial(consulta_id)
    if resultado is None:
        st.session_state["aviso_historial"] = "No se ha podido leer esa consulta del historial."
        return
    if repetir and resultado["dataset_version"] != dataset_version:
        st.session_state["aviso_historial"] = ("Esa consulta se hizo con otra versión del dataset y no se puede "
                                               "repetir tal cual; ábrela para verla.")
        return
    st.session_state.pop("interpretacion", None)
    st.session_state["formulario_pendiente"] = resultado["consulta"]
    if repetir:
        st.session_state["ejecutar_pendiente"] = {}
    else:
        st.session_state["resultado"] = resultado
        st.session_state["valoraciones_pendientes"] = False
    st.session_state["ir_a_buscar"] = True


def pagina() -> None:
    _, manifiesto, _, _, _ = comun.contexto()
    if st.session_state.pop("ir_a_buscar", False):
        st.switch_page(st.session_state["paginas"]["buscar"])
    st.header("Historial de consultas")
    st.markdown("Cada búsqueda se guarda completa. Puedes **abrirla** tal como quedó, sin volver a llamar a los "
                "modelos de lenguaje, o **repetirla** para que se aplique lo que el sistema ha aprendido desde "
                "entonces.")
    aviso = st.session_state.pop("aviso_historial", None)
    if aviso:
        st.warning(aviso)
    lista = exportador.listar_historial()
    if lista.empty:
        st.info("Todavía no hay consultas guardadas. Haz una búsqueda y aparecerá aquí.", icon=":material/history:")
        return

    fechas = pd.to_datetime(lista["fecha_utc"], errors="coerce", utc=True)
    tabla = pd.DataFrame({
        "Fecha": fechas.dt.tz_convert(None).dt.strftime("%d/%m/%Y %H:%M"), "Referencia": lista["referencia"],
        "Equipo": lista["equipo"], "Rol": lista["rol"], "Filtros": lista["filtros"],
        "Ranking": lista["modo_ranking"].map(MODOS).fillna(lista["modo_ranking"]),
        "Estado": lista["estado"].map(ESTADOS).fillna(lista["estado"]), "Top 3": lista["top"],
        "Segundos": lista["duracion_s"], "Petición en lenguaje natural": lista["texto_original"],
    })
    st.caption(f"{len(tabla)} consultas guardadas (horas en UTC). Selecciona una fila.")
    evento = st.dataframe(tabla, hide_index=True, on_select="rerun", selection_mode="single-row",
                          key="tabla_historial", height=min(460, 60 + 35 * len(tabla)))
    filas = evento.selection.rows
    a, b, _ = st.columns([1, 1, 3])
    consulta_id = lista.iloc[filas[0]]["consulta_id"] if filas else None
    a.button("Abrir resultado", type="primary", disabled=consulta_id is None, width="stretch",
             icon=":material/open_in_new:", on_click=_abrir, args=(consulta_id, False, manifiesto["dataset_version"]))
    b.button("Repetir la búsqueda", disabled=consulta_id is None, width="stretch", icon=":material/refresh:",
             on_click=_abrir, args=(consulta_id, True, manifiesto["dataset_version"]))
    if consulta_id:
        st.caption(f"Consulta seleccionada: {consulta_id}")
