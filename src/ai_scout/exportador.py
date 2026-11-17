"""Exportación de cada consulta a tablas CSV para Power BI Desktop (CU-04) e historial.

Las consultas anteriores nunca se sobrescriben: cada consulta añade filas con su propio consulta_id.
Además, el resultado completo de cada consulta se guarda en salidas/historial/ para poder
reabrirla en la aplicación sin volver a calcular ni a llamar a los modelos de lenguaje.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

COLUMNAS = {
    "consultas": ["consulta_id", "dataset_version", "objetivo_registro_id", "fecha_utc", "minutos_minimos",
                  "edad_maxima", "ligas_permitidas_json", "usar_llm", "perfil_version", "peso_numerico",
                  "peso_cualitativo", "modo_ranking", "estado_consulta", "n_poblacion_referencia",
                  "n_candidatos_validos", "n_candidatos", "duracion_total_ms", "avisos",
                  # Columnas añadidas con los roles, los estilos, el aprendizaje y el revisor
                  "rol", "rol_nombre", "pesos_json", "estilo_objetivo", "solo_mismo_estilo", "diversidad",
                  "usar_aprendizaje", "valoraciones_grupo", "valoraciones_referencia", "pesos_aprendidos",
                  "consulta_adaptada", "usar_critico", "revision_fichas", "texto_original", "equipo_destino"],
    "resultados": ["resultado_id", "consulta_id", "candidato_registro_id", "candidato_nombre", "equipo", "liga",
                   "posicion_original", "grupo_posicion", "edad_registrada", "minutos", "coseno_original",
                   "score_numerico", "score_cualitativo", "score_final", "ranking_preliminar", "ranking_final",
                   "es_top_final", "cumple_restricciones", "estado_analisis", "estado_economico", "origen_ficha",
                   "resumen", "fortalezas", "diferencias", "reservas", "veredicto",
                   "estilo", "score_base", "revision_ficha", "observaciones_revisor",
                   "score_plantilla", "score_sin_plantilla", "encaje_plantilla"],
    "registros": ["registro_id", "persona_id", "dataset_version", "nombre", "nacionalidad", "ano_nacimiento",
                  "equipo", "liga", "temporada_asignada", "posicion_original", "grupo_posicion",
                  "edad_registrada", "minutos", "estilo"],
    "evidencias": ["evidencia_id", "resultado_id", "consulta_id", "variable", "descripcion", "unidad",
                   "valor_objetivo", "valor_candidato", "diferencia", "z_diferencia", "peso"],
    "ejecuciones_agentes": ["ejecucion_id", "consulta_id", "agente", "intento", "proveedor", "modelo",
                            "prompt_version", "estado", "duracion_ms", "tokens_entrada", "tokens_salida",
                            "coste_usd", "codigo_error", "detalle"],
    "incidencias": ["incidencia_id", "dataset_version", "consulta_id", "registro_id", "fase", "severidad",
                    "codigo", "descripcion"],
    "estilos_jugadores": ["registro_id", "dataset_version", "persona_id", "nombre", "equipo", "liga",
                          "posicion_original", "grupo_posicion", "edad_registrada", "minutos", "estilo_id",
                          "estilo", "pca_x", "pca_y", "en_entrenamiento"],
}


def _texto_lista(valor) -> str:
    if not valor:
        return ""
    return " | ".join(valor) if isinstance(valor, list) else str(valor)


def _bool(x) -> str:
    return "true" if x else "false"


def _anexar(nombre: str, filas: list[dict], directorio: Path) -> Path:
    """Añade filas a la tabla. Si el fichero tiene columnas de una versión anterior, lo actualiza antes."""
    ruta = directorio / f"{nombre}.csv"
    columnas = COLUMNAS[nombre]
    tabla = pd.DataFrame(filas, columns=columnas)
    if ruta.exists():
        cabecera = list(pd.read_csv(ruta, nrows=0, dtype=str).columns)
        if cabecera != columnas:
            anterior = pd.read_csv(ruta, dtype=str, keep_default_na=False)
            anterior.reindex(columns=columnas, fill_value="").to_csv(ruta, index=False, encoding="utf-8")
    tabla.to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")
    return ruta


def _actualizar_registros(filas: list[dict], directorio: Path) -> Path:
    ruta = directorio / "registros.csv"
    nueva = pd.DataFrame(filas, columns=COLUMNAS["registros"])
    if ruta.exists():
        anterior = pd.read_csv(ruta, dtype=str, keep_default_na=False).reindex(columns=COLUMNAS["registros"],
                                                                              fill_value="")
        nueva = pd.concat([anterior, nueva.astype(str).replace({"None": "", "nan": ""})], ignore_index=True)
    nueva = nueva.drop_duplicates(["registro_id", "dataset_version"], keep="last")
    nueva.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


def _registro(r: dict, dataset_version: str) -> dict:
    return {"registro_id": r["registro_id"], "persona_id": r.get("persona_id"), "dataset_version": dataset_version,
            "nombre": r["nombre"], "nacionalidad": r.get("nacionalidad"), "ano_nacimiento": r.get("ano_nacimiento"),
            "equipo": r["equipo"], "liga": r["liga"], "temporada_asignada": C.TEMPORADA,
            "posicion_original": r["posicion_original"], "grupo_posicion": r["grupo_posicion"],
            "edad_registrada": r.get("edad_registrada"), "minutos": r["minutos"], "estilo": r.get("estilo")}


def exportar_consulta(resultado: dict, directorio: Path | str | None = None) -> dict:
    directorio = Path(directorio) if directorio else C.DIR_SALIDAS
    directorio.mkdir(parents=True, exist_ok=True)
    q = resultado["consulta"]
    qid = q["consulta_id"]
    version = resultado["dataset_version"]
    rec = resultado.get("recuperacion") or {}
    personalizacion = resultado.get("personalizacion") or {}
    revision = resultado.get("revision") or {}

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
        "rol": q.get("rol"), "rol_nombre": rec.get("rol_nombre"),
        "pesos_json": json.dumps({v: round(p, 4) for v, p in (rec.get("pesos") or {}).items()}),
        "estilo_objetivo": (rec.get("objetivo") or {}).get("estilo"),
        "solo_mismo_estilo": _bool(q.get("solo_mismo_estilo")), "diversidad": q.get("diversidad"),
        "usar_aprendizaje": _bool(q.get("usar_aprendizaje")),
        "valoraciones_grupo": personalizacion.get("valoraciones_grupo", 0),
        "valoraciones_referencia": personalizacion.get("valoraciones_referencia", 0),
        "pesos_aprendidos": _bool(personalizacion.get("pesos_aprendidos")),
        "consulta_adaptada": _bool(personalizacion.get("rocchio_aplicado")),
        "usar_critico": _bool(q.get("usar_critico")), "revision_fichas": revision.get("estado"),
        "texto_original": q.get("texto_original"), "equipo_destino": q.get("equipo_destino"),
    }]

    resultados, evidencias, registros = [], [], []
    if rec.get("objetivo"):
        registros.append(_registro(rec["objetivo"], version))
    for c in resultado["ranking"]:
        rid = f"{qid}-{c['registro_id']}"
        ficha = c.get("ficha") or {}
        rev = ficha.get("revision") or {}
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
            "estilo": c.get("estilo"),
            "score_base": None if c.get("score_base") is None else round(c["score_base"], 4),
            "revision_ficha": rev.get("estado"), "observaciones_revisor": _texto_lista(rev.get("problemas")),
            "score_plantilla": None if c.get("score_plantilla") is None else round(c["score_plantilla"], 4),
            "score_sin_plantilla": None if c.get("score_sin_plantilla") is None else round(c["score_sin_plantilla"], 4),
            "encaje_plantilla": (c.get("plantilla") or {}).get("comentario"),
        })
        for d in c["diferencias"]:
            evidencias.append({"evidencia_id": f"{rid}-{d['variable']}", "resultado_id": rid, "consulta_id": qid,
                               **{k: d.get(k) for k in ("variable", "descripcion", "unidad", "valor_objetivo",
                                                        "valor_candidato", "diferencia", "z_diferencia", "peso")}})
        registros.append(_registro(c, version))

    ejecuciones = [{"ejecucion_id": f"{qid}-E{i:02d}", **{**e, "consulta_id": qid}}
                   for i, e in enumerate(resultado["ejecuciones"], 1)]
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


def exportar_incidencias_datos(incidencias: list[dict], directorio: Path | str | None = None) -> Path | None:
    """Añade las incidencias de la preparación de datos (una sola vez por versión del dataset)."""
    if not incidencias:
        return None
    directorio = Path(directorio) if directorio else C.DIR_SALIDAS
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


def exportar_estilos(tabla: pd.DataFrame, dataset_version: str, directorio: Path | str | None = None) -> Path:
    """Tabla de todos los jugadores con su estilo de juego y su posición en el mapa PCA.

    Se reescribe entera: depende solo del dataset, no de las consultas.
    """
    directorio = Path(directorio) if directorio else C.DIR_SALIDAS
    directorio.mkdir(parents=True, exist_ok=True)
    salida = tabla.copy()
    salida["dataset_version"] = dataset_version
    salida["en_entrenamiento"] = salida["en_entrenamiento"].map(_bool)
    for col in ("pca_x", "pca_y"):
        salida[col] = salida[col].round(4)
    ruta = directorio / "estilos_jugadores.csv"
    salida.reindex(columns=COLUMNAS["estilos_jugadores"]).to_csv(ruta, index=False, encoding="utf-8")
    return ruta


# --------------------------------------------------------------------------
# Historial de consultas (resultado completo en JSON)
# --------------------------------------------------------------------------
def _a_json(x):
    """Convierte a tipos de Python lo que json no sabe escribir (numpy, conjuntos, rutas…)."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"No se puede guardar en JSON un valor de tipo {type(x).__name__}")


def guardar_historial(resultado: dict, directorio: Path | str | None = None) -> Path:
    carpeta = (Path(directorio) if directorio else C.DIR_SALIDAS) / C.SUBDIR_HISTORIAL
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{resultado['consulta']['consulta_id']}.json"
    ruta.write_text(json.dumps(resultado, ensure_ascii=False, default=_a_json), encoding="utf-8")
    return ruta


def cargar_historial(consulta_id: str, directorio: Path | str | None = None) -> dict | None:
    """Resultado completo de una consulta guardada, o None si no existe o no se puede leer."""
    ruta = (Path(directorio) if directorio else C.DIR_SALIDAS) / C.SUBDIR_HISTORIAL / f"{consulta_id}.json"
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def listar_historial(directorio: Path | str | None = None) -> pd.DataFrame:
    """Resumen de las consultas guardadas, de la más reciente a la más antigua."""
    carpeta = (Path(directorio) if directorio else C.DIR_SALIDAS) / C.SUBDIR_HISTORIAL
    columnas = ["consulta_id", "fecha_utc", "referencia", "equipo", "grupo", "rol", "filtros", "modo_ranking",
                "estado", "top", "duracion_s", "texto_original"]
    filas = []
    for ruta in sorted(carpeta.glob("*.json")) if carpeta.exists() else []:
        try:
            r = json.loads(ruta.read_text(encoding="utf-8"))
            q, rec = r["consulta"], r.get("recuperacion") or {}
            obj = rec.get("objetivo") or {}
            filtros = [f"{len(q['ligas_permitidas'])} ligas", f"≥ {q['minutos_minimos']} min"]
            if q.get("edad_maxima") is not None:
                filtros.append(f"≤ {q['edad_maxima']} años")
            filas.append({
                "consulta_id": q["consulta_id"], "fecha_utc": q["fecha_utc"], "referencia": obj.get("nombre"),
                "equipo": obj.get("equipo"), "grupo": rec.get("grupo"), "rol": rec.get("rol_nombre"),
                "filtros": " · ".join(filtros), "modo_ranking": r.get("modo_ranking"), "estado": r.get("estado"),
                "top": ", ".join(c["nombre"] for c in r.get("top_final", [])),
                "duracion_s": round((r.get("duracion_total_ms") or 0) / 1000, 1),
                "texto_original": q.get("texto_original"),
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue  # un fichero dañado no impide ver el resto del historial
    tabla = pd.DataFrame(filas, columns=columnas)
    return tabla.sort_values("fecha_utc", ascending=False).reset_index(drop=True)
