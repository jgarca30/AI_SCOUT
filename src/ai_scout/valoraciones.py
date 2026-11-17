"""Valoraciones del usuario (realimentación explícita).

El usuario puntúa de 1 a 5 estrellas a los candidatos recomendados. Cada valoración se
guarda en salidas/valoraciones.csv.

La tabla es un registro histórico: si el usuario cambia de opinión se añade una fila
nueva y la anterior deja de estar vigente. 0 estrellas significa «valoración retirada».
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config as C

FICHERO = "valoraciones.csv"
COLUMNAS = ["valoracion_id", "fecha_utc", "consulta_id", "dataset_version", "objetivo_registro_id",
            "objetivo_persona_id", "objetivo_nombre", "candidato_registro_id", "candidato_persona_id",
            "candidato_nombre", "grupo_posicion", "rol", "pesos_base_json", "estrellas", "relevancia",
            "ranking_final", "score_numerico", "score_final", "modo_ranking", "vigente"]
CLAVE = ["objetivo_persona_id", "candidato_persona_id"]  # una valoración vigente por pareja


def relevancia(estrellas: int) -> float:
    """Relevancia en [-1, 1]: 5★ = +1, 4★ = +0,5, 3★ = 0, 2★ = −0,5, 1★ = −1."""
    return (int(estrellas) - 3) / 2.0


def _ruta(directorio: Path | str | None) -> Path:
    return Path(directorio or C.DIR_SALIDAS) / FICHERO


def cargar(directorio: Path | str | None = None) -> pd.DataFrame:
    """Todas las valoraciones registradas (también las que ya no están vigentes)."""
    ruta = _ruta(directorio)
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    tabla = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    for col in COLUMNAS:
        if col not in tabla.columns:
            tabla[col] = ""
    tabla["estrellas"] = pd.to_numeric(tabla["estrellas"], errors="coerce").fillna(0).astype(int)
    for col in ("relevancia", "ranking_final", "score_numerico", "score_final"):
        tabla[col] = pd.to_numeric(tabla[col], errors="coerce")
    # La vigencia se recalcula siempre: la última fila de cada pareja referencia-candidato
    ultima = ~tabla.duplicated(CLAVE, keep="last")
    tabla["vigente"] = ultima & (tabla["estrellas"] > 0)
    return tabla[COLUMNAS]


def vigentes(tabla: pd.DataFrame | None = None, dataset_version: str | None = None,
             directorio: Path | str | None = None) -> pd.DataFrame:
    """Valoraciones en vigor: la última de cada pareja, sin las retiradas.

    Si se indica dataset_version solo se devuelven las de esa versión del dataset (los
    identificadores de registro no son comparables entre versiones).
    """
    if tabla is None:
        tabla = cargar(directorio)
    v = tabla[tabla["vigente"].astype(bool)] if len(tabla) else tabla
    if dataset_version is not None and len(v):
        v = v[v["dataset_version"] == dataset_version]
    return v.reset_index(drop=True)


def registrar_fila(fila: dict, directorio: Path | str | None = None) -> dict:
    """Añade una valoración y actualiza la columna «vigente» de las anteriores."""
    estrellas = int(fila["estrellas"])
    if not 0 <= estrellas <= 5:
        raise ValueError("La valoración debe ser de 1 a 5 estrellas (0 para retirarla).")
    ruta = _ruta(directorio)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nueva = {c: fila.get(c) for c in COLUMNAS}
    nueva["valoracion_id"] = fila.get("valoracion_id") or (
        datetime.now(timezone.utc).strftime("V%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    nueva["fecha_utc"] = fila.get("fecha_utc") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    nueva["estrellas"] = estrellas
    nueva["relevancia"] = relevancia(estrellas) if estrellas else None
    tabla = pd.concat([cargar(directorio), pd.DataFrame([nueva], columns=COLUMNAS)], ignore_index=True)
    ultima = ~tabla.duplicated(CLAVE, keep="last")
    tabla["vigente"] = ultima & (tabla["estrellas"].astype(int) > 0)
    salida = tabla.copy()
    salida["vigente"] = salida["vigente"].map({True: "true", False: "false"})
    salida.to_csv(ruta, index=False, encoding="utf-8")
    return nueva


def registrar(resultado: dict, candidato_registro_id: str, estrellas: int,
              directorio: Path | str | None = None) -> dict:
    """Guarda la valoración de un candidato del resultado de una consulta."""
    candidato = next((c for c in resultado["ranking"] if c["registro_id"] == candidato_registro_id), None)
    if candidato is None:
        raise KeyError(f"El candidato {candidato_registro_id} no pertenece a esta consulta.")
    objetivo = resultado["recuperacion"]["objetivo"]
    consulta = resultado["consulta"]
    return registrar_fila({
        "consulta_id": consulta["consulta_id"], "dataset_version": resultado["dataset_version"],
        "objetivo_registro_id": objetivo["registro_id"], "objetivo_persona_id": objetivo["persona_id"],
        "objetivo_nombre": objetivo["nombre"], "candidato_registro_id": candidato["registro_id"],
        "candidato_persona_id": candidato["persona_id"], "candidato_nombre": candidato["nombre"],
        "grupo_posicion": candidato["grupo_posicion"], "rol": consulta.get("rol"),
        # Pesos de la búsqueda (rol o ajuste manual), sin lo aprendido: el aprendizaje parte de ellos
        "pesos_base_json": json.dumps(resultado["recuperacion"].get("pesos_rol") or {}),
        "estrellas": estrellas,
        "ranking_final": candidato.get("ranking_final"), "score_numerico": candidato.get("score_numerico"),
        "score_final": candidato.get("score_final"), "modo_ranking": resultado.get("modo_ranking"),
    }, directorio)


def borrar_todas(directorio: Path | str | None = None) -> int:
    """Olvida todas las valoraciones. Devuelve cuántas filas había."""
    ruta = _ruta(directorio)
    if not ruta.exists():
        return 0
    n = len(cargar(directorio))
    ruta.unlink()
    return n
