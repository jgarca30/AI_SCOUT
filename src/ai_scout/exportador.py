"""Exportación de cada consulta a tablas CSV para Power BI Desktop (CU-04).
Las consultas anteriores nunca se sobrescriben: cada consulta añade filas con su propio consulta_id.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C

COLUMNAS = {
    "consultas": ["consulta_id", "dataset_version", "objetivo_registro_id", "fecha_utc", "minutos_minimos",
                  "edad_maxima", "ligas_permitidas_json", "usar_llm", "perfil_version", "peso_numerico",
                  "peso_cualitativo", "modo_ranking", "estado_consulta", "n_poblacion_referencia",
                  "n_candidatos_validos", "n_candidatos", "duracion_total_ms", "avisos"],
    "resultados": ["resultado_id", "consulta_id", "candidato_registro_id", "candidato_nombre", "equipo", "liga",
                   "posicion_original", "grupo_posicion", "edad_registrada", "minutos", "coseno_original",
                   "score_numerico", "score_cualitativo", "score_final", "ranking_preliminar", "ranking_final",
                   "es_top_final", "cumple_restricciones", "estado_analisis", "estado_economico", "origen_ficha",
                   "resumen", "fortalezas", "diferencias", "reservas", "veredicto"],
    "registros": ["registro_id", "persona_id", "dataset_version", "nombre", "nacionalidad", "ano_nacimiento",
                  "equipo", "liga", "temporada_asignada", "posicion_original", "grupo_posicion",
                  "edad_registrada", "minutos"],
    "evidencias": ["evidencia_id", "resultado_id", "consulta_id", "variable", "descripcion", "unidad",
                   "valor_objetivo", "valor_candidato", "diferencia", "z_diferencia"],
    "ejecuciones_agentes": ["ejecucion_id", "consulta_id", "agente", "intento", "proveedor", "modelo",
                            "prompt_version", "estado", "duracion_ms", "tokens_entrada", "tokens_salida",
                            "coste_usd", "codigo_error", "detalle"],
    "incidencias": ["incidencia_id", "dataset_version", "consulta_id", "registro_id", "fase", "severidad",
                    "codigo", "descripcion"],
}


def _texto_lista(valor) -> str:
    if not valor:
        return ""
    return " | ".join(valor) if isinstance(valor, list) else str(valor)


def _bool(x) -> str:
    return "true" if x else "false"


def _anexar(nombre: str, filas: list[dict], directorio: Path) -> Path:
    ruta = directorio / f"{nombre}.csv"
    tabla = pd.DataFrame(filas, columns=COLUMNAS[nombre])
    escribir_cabecera = not ruta.exists()
    tabla.to_csv(ruta, mode="a", header=escribir_cabecera, index=False, encoding="utf-8")
    return ruta


def _actualizar_registros(filas: list[dict], directorio: Path) -> Path:
    ruta = directorio / "registros.csv"
    nueva = pd.DataFrame(filas, columns=COLUMNAS["registros"])
    if ruta.exists():
        anterior = pd.read_csv(ruta, dtype=str, keep_default_na=False)
        nueva = pd.concat([anterior, nueva.astype(str).replace({"None": "", "nan": ""})], ignore_index=True)
    nueva = nueva.drop_duplicates(["registro_id", "dataset_version"], keep="last")
    nueva.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


def _registro(r: dict, dataset_version: str) -> dict:
    return {"registro_id": r["registro_id"], "persona_id": r.get("persona_id"), "dataset_version": dataset_version,
            "nombre": r["nombre"], "nacionalidad": r.get("nacionalidad"), "ano_nacimiento": r.get("ano_nacimiento"),
            "equipo": r["equipo"], "liga": r["liga"], "temporada_asignada": C.TEMPORADA,
            "posicion_original": r["posicion_original"], "grupo_posicion": r["grupo_posicion"],
            "edad_registrada": r.get("edad_registrada"), "minutos": r["minutos"]}


def exportar_consulta(resultado: dict, directorio: Path = C.DIR_SALIDAS) -> dict:
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    q = resultado["consulta"]
    qid = q["consulta_id"]
    version = resultado["dataset_version"]
    rec = resultado.get("recuperacion") or {}

    consultas = [{
        "consulta_id": qid, "dataset_version": version, "objetivo_registro_id": q["objetivo_registro_id"],
        "fecha_utc": q["fecha_utc"], "minutos_minimos": q["minutos_minimos"], "edad_maxima": q["edad_maxima"],
        "ligas_permitidas_json": json.dumps(q["ligas_permitidas"], ensure_ascii=False),
        "usar_llm": _bool(q["usar_llm"]), "perfil_version": resultado["perfil_version"],
        "peso_numerico": q["peso_numerico"], "peso_cualitativo": q["peso_cualitativo"],
        "modo_ranking": resultado["modo_ranking"], "estado_consulta": resultado["estado"],
        "n_poblacion_referencia": rec.get("poblacion_referencia"), "n_candidatos_validos": rec.get("n_candidatos_validos"),
        "n_candidatos": len(resultado["ranking"]), "duracion_total_ms": resultado.get("duracion_total_ms"),
        "avisos": _texto_lista(resultado["avisos"] + resultado["errores"]),
    }]

    resultados, evidencias, registros = [], [], []
    if rec.get("objetivo"):
        registros.append(_registro(rec["objetivo"], version))
    for c in resultado["ranking"]:
        rid = f"{qid}-{c['registro_id']}"
        ficha = c.get("ficha") or {}
        resultados.append({
            "resultado_id": rid, "consulta_id": qid, "candidato_registro_id": c["registro_id"],
            "candidato_nombre": c["nombre"], "equipo": c["equipo"], "liga": c["liga"],
            "posicion_original": c["posicion_original"], "grupo_posicion": c["grupo_posicion"],
            "edad_registrada": c["edad_registrada"], "minutos": c["minutos"],
            "coseno_original": round(c["coseno"], 6), "score_numerico": round(c["score_numerico"], 4),
            "score_cualitativo": None if c["score_cualitativo"] is None else round(c["score_cualitativo"], 4),
            "score_final": round(c["score_final"], 4), "ranking_preliminar": c["ranking_preliminar"],
            "ranking_final": c["ranking_final"], "es_top_final": _bool(c["es_top_final"]),
            "cumple_restricciones": "true", "estado_analisis": c["estado_analisis"],
            "estado_economico": "no_evaluado", "origen_ficha": ficha.get("origen"),
            "resumen": ficha.get("resumen"), "fortalezas": _texto_lista(ficha.get("fortalezas")),
            "diferencias": _texto_lista(ficha.get("diferencias_relevantes")),
            "reservas": _texto_lista(ficha.get("reservas")), "veredicto": ficha.get("veredicto"),
        })
        for d in c["diferencias"]:
            evidencias.append({"evidencia_id": f"{rid}-{d['variable']}", "resultado_id": rid, "consulta_id": qid,
                               **{k: d[k] for k in ("variable", "descripcion", "unidad", "valor_objetivo",
                                                    "valor_candidato", "diferencia", "z_diferencia")}})
        registros.append(_registro(c, version))

    ejecuciones = [{"ejecucion_id": f"{qid}-E{i:02d}", **e} for i, e in enumerate(resultado["ejecuciones"], 1)]
    incidencias = [{"incidencia_id": f"{qid}-I{i:02d}", "dataset_version": version, "consulta_id": qid, **inc}
                   for i, inc in enumerate(resultado["incidencias"], 1)]

    rutas = {
        "consultas": _anexar("consultas", consultas, directorio),
        "resultados": _anexar("resultados", resultados, directorio),
        "evidencias": _anexar("evidencias", evidencias, directorio),
        "ejecuciones_agentes": _anexar("ejecuciones_agentes", ejecuciones, directorio),
        "incidencias": _anexar("incidencias", incidencias, directorio),
    }
    if registros:
        rutas["registros"] = _actualizar_registros(registros, directorio)
    return {k: str(v) for k, v in rutas.items()}


def exportar_incidencias_datos(incidencias: list[dict], directorio: Path = C.DIR_SALIDAS) -> Path | None:
    """Añade las incidencias de la preparación de datos (una sola vez por versión del dataset)."""
    if not incidencias:
        return None
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / "incidencias.csv"
    nuevas = pd.DataFrame(incidencias, columns=COLUMNAS["incidencias"])
    if ruta.exists():
        anterior = pd.read_csv(ruta, dtype=str, keep_default_na=False)
        ya = set(anterior["incidencia_id"])
        nuevas = nuevas[~nuevas["incidencia_id"].isin(ya)]
        if nuevas.empty:
            return ruta
    nuevas.to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")
    return ruta
