"""Elementos compartidos por las páginas: carga de datos, modelos, valoraciones y formato."""
from __future__ import annotations

from dataclasses import fields

import pandas as pd
import streamlit as st

from ai_scout import config as C
from ai_scout import datos, estilos, exportador, llm
from ai_scout import valoraciones as V
from ai_scout.director import fmt
from ai_scout.modelos import Consulta

__all__ = ["ICONO_VEREDICTO", "contexto", "fmt", "edad_txt", "ficha_jugador", "plural", "estrellas_guardadas",
           "guardar_valoracion", "estrellas_editadas", "consulta_desde", "version_valoraciones"]

ICONO_VEREDICTO = {"candidato_para_revisar": "🟢", "candidato_con_reservas": "🟡", "evidencia_insuficiente": "⚪"}


# --------------------------------------------------------------------------
# Datos, estilos y proveedores (se cargan una vez)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparando el dataset…")
def cargar_datos():
    return datos.cargar_o_preparar()


@st.cache_resource(show_spinner="Calculando los estilos de juego…")
def cargar_estilos(dataset_version: str):
    """Modelos de estilos y tabla de jugadores con su estilo (también se exporta para Power BI)."""
    df, _ = cargar_datos()
    modelos = estilos.calcular_estilos(df)
    tabla = estilos.tabla_estilos(modelos, df)
    try:
        exportador.exportar_estilos(tabla, dataset_version)
    except OSError:
        pass  # si no se puede escribir en salidas/, la aplicación funciona igual
    return modelos, tabla


@st.cache_resource
def cargar_proveedores() -> dict:
    return {rol: llm.crear_proveedor(rol) for rol in C.ROLES_LLM}


def contexto() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, dict]:
    """(datos, manifiesto, modelos de estilos, tabla de estilos, proveedores LLM). Detiene la página si falla."""
    try:
        df, manifiesto = cargar_datos()
    except datos.ErrorDatos as e:
        st.error(f"No se pudo cargar el dataset: {e}")
        st.stop()
    modelos, tabla = cargar_estilos(manifiesto["dataset_version"])
    return df, manifiesto, modelos, tabla, cargar_proveedores()


# --------------------------------------------------------------------------
# Formato
# --------------------------------------------------------------------------
def edad_txt(edad) -> str:
    return "edad no disponible" if edad is None or pd.isna(edad) else f"{int(edad)} años"


def plural(n: int, singular: str, plural_: str | None = None) -> str:
    """«1 valoración», «3 valoraciones»."""
    return f"{n} {singular if n == 1 else (plural_ or singular + 's')}"


def ficha_jugador(j: dict, con_estilo: bool = True) -> str:
    partes = [j["equipo"], j["liga"], j["posicion_original"], edad_txt(j.get("edad_registrada")),
              f"{fmt(j['minutos'], 0)} min"]
    if con_estilo and j.get("estilo"):
        partes.append(f"estilo: {j['estilo']}")
    return " · ".join(partes)


# --------------------------------------------------------------------------
# Valoraciones del usuario
# --------------------------------------------------------------------------
def version_valoraciones() -> int:
    """Contador que cambia cada vez que se guarda una valoración (sirve para refrescar los controles)."""
    return st.session_state.setdefault("version_valoraciones", 0)


def estrellas_guardadas(dataset_version: str) -> dict[tuple[str, str], int]:
    """Estrellas vigentes por (persona de referencia, persona candidata)."""
    try:
        vigentes = V.vigentes(dataset_version=dataset_version)
    except (OSError, ValueError, KeyError):
        return {}
    return {(v.objetivo_persona_id, v.candidato_persona_id): int(v.estrellas) for v in vigentes.itertuples()}


def guardar_valoracion(resultado: dict, registro_id: str, estrellas: int) -> None:
    """Guarda (o retira, con 0) la valoración de un candidato de la consulta mostrada."""
    V.registrar(resultado, registro_id, int(estrellas))
    st.session_state["version_valoraciones"] = version_valoraciones() + 1
    st.session_state["valoraciones_pendientes"] = True


def estrellas_editadas(estado_editor: dict | None, columna: str) -> list[tuple[int, int]]:
    """Cambios hechos en una tabla editable: lista de (fila, estrellas); 0 si se ha borrado la celda."""
    cambios = []
    for fila, valores in ((estado_editor or {}).get("edited_rows") or {}).items():
        if columna in valores:
            valor = valores[columna]
            cambios.append((int(fila), 0 if valor is None or pd.isna(valor) else int(valor)))
    return cambios


# --------------------------------------------------------------------------
# Consultas
# --------------------------------------------------------------------------
def consulta_desde(parametros: dict, **cambios) -> Consulta:
    """Nueva consulta con los mismos parámetros que una anterior (con otro identificador y fecha)."""
    validos = {f.name for f in fields(Consulta)} - {"consulta_id", "fecha_utc"}
    datos_consulta = {k: v for k, v in parametros.items() if k in validos}
    datos_consulta.update(cambios)
    return Consulta(**datos_consulta)
