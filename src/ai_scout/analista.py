"""Agente 2 · Analista de perfiles (LLM).

Compara cada candidato del Top 10 con el jugador de referencia usando SOLO los
datos que recibe. 
Los nombres se anonimizan para que el modelo no use lo que «recuerda» de los jugadores.
 Su salida JSON se valida antes de usarse y el score cualitativo lo calcula el código, nunca el LLM.
"""
from __future__ import annotations

import json
import math

from . import config as C
from .datos import normalizar_texto

TOLERANCIA_VALOR = 0.011  # los valores se envían redondeados a 2 decimales

SISTEMA = f"""Eres el Agente 2 (analista de perfiles) de AI Scout, un recomendador académico de futbolistas.
Recibes las estadísticas de un jugador de REFERENCIA y de varios CANDIDATOS del mismo grupo de posición.
Tu tarea es valorar, para cada candidato, cuánto se parece su perfil estadístico al de la referencia.

Reglas obligatorias:
1. Usa EXCLUSIVAMENTE los datos proporcionados. No uses conocimiento previo sobre jugadores, equipos o ligas.
2. Puntúa cada una de estas dimensiones con un entero de 0 a 4, o null si los datos no permiten evaluarla:
{chr(10).join(f'   - {k}: {v}' for k, v in C.DIMENSIONES.items())}
   Escala: {"; ".join(f"{k} = {v}" for k, v in C.ESCALA_PUNTUACION.items())}.
3. Cada dimensión puntuada debe incluir al menos una evidencia con "variable", "valor_objetivo" y
   "valor_candidato" COPIADOS EXACTAMENTE de los datos de entrada (mismos decimales).
4. No hagas afirmaciones sobre lesiones, estado físico, velocidad, personalidad, contratos, salarios,
   precio, presión del rival, mapas de calor ni sobre el equipo de destino: no están en los datos.
   Si quieres señalar que algo no se puede saber, hazlo solo en "limitaciones".
5. Formula las inferencias como hipótesis compatibles con los indicadores, no como hechos.
6. Incluye exactamente a todos los candidatos recibidos, identificados por "candidato_registro_id".
   No añadas candidatos.
7. Responde ÚNICAMENTE con un objeto JSON con este formato:
{{"candidatos": [{{"candidato_registro_id": "R0000", "estado": "evaluado" | "no_evaluable",
  "dimensiones": [{{"nombre": "progresion_creacion", "puntuacion": 0-4 o null,
     "evidencias": [{{"variable": "PrgP_p90", "valor_objetivo": 0.0, "valor_candidato": 0.0}}],
     "justificacion": "frase breve"}}],
  "semejanzas": ["..."], "diferencias": ["..."], "limitaciones": ["..."], "resumen": "una o dos frases"}}]}}
Escribe los textos en español."""

_LISTA_TEXTOS = {"type": "array", "items": {"type": "string"}}

# Esquema JSON de la respuesta.
ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {"candidatos": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "candidato_registro_id": {"type": "string"},
            "estado": {"type": "string", "enum": ["evaluado", "no_evaluable"]},
            "dimensiones": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string", "enum": list(C.DIMENSIONES)},
                    "puntuacion": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 4}, {"type": "null"}]},
                    "evidencias": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"variable": {"type": "string"}, "valor_objetivo": {"type": "number"},
                                       "valor_candidato": {"type": "number"}},
                        "required": ["variable", "valor_objetivo", "valor_candidato"]}},
                    "justificacion": {"type": "string"},
                },
                "required": ["nombre", "puntuacion", "evidencias", "justificacion"]}},
            "semejanzas": _LISTA_TEXTOS, "diferencias": _LISTA_TEXTOS, "limitaciones": _LISTA_TEXTOS,
            "resumen": {"type": "string"},
        },
        "required": ["candidato_registro_id", "estado", "dimensiones", "semejanzas", "diferencias",
                     "limitaciones", "resumen"]}}},
    "required": ["candidatos"],
}


def r2(x: float) -> float:
    return round(float(x), 2)


def construir_payload(recuperacion: dict) -> dict:
    variables = recuperacion["variables"]
    obj = recuperacion["objetivo"]
    return {
        "grupo_posicion": f"{recuperacion['grupo']} ({C.GRUPOS[recuperacion['grupo']]})",
        "variables": [{"variable": v, "descripcion": C.VARIABLES[v][0], "unidad": C.VARIABLES[v][1]}
                      for v in variables],
        "nota_percentiles": "Percentil dentro de los jugadores del mismo grupo con los minutos mínimos (0-100).",
        "referencia": {
            "id": "REFERENCIA", "minutos": obj["minutos"],
            "valores": {v: r2(obj["valores"][v]) for v in variables},
            "percentiles": {v: round(obj["percentiles"][v]) for v in variables},
        },
        "candidatos": [{
            "candidato_registro_id": c["registro_id"], "minutos": c["minutos"],
            "similitud_coseno": round(c["coseno"], 3),
            "valores": {v: r2(c["valores"][v]) for v in variables},
            "percentiles": {v: round(c["percentiles"].get(v, float("nan"))) for v in variables
                            if not math.isnan(c["percentiles"].get(v, float("nan")))},
        } for c in recuperacion["candidatos"]],
    }


def construir_mensaje(payload: dict) -> str:
    return ("Analiza a estos candidatos frente al jugador de referencia.\n\nDATOS:\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=1) + "\n```")


def _textos_con_terminos_prohibidos(textos: list[str]) -> list[str]:
    encontrados = set()
    for t in textos:
        n = normalizar_texto(t)
        for termino in C.TERMINOS_SIN_RESPALDO:
            if termino in n:
                encontrados.add(termino)
    return sorted(encontrados)


def _es_numero(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validar_respuesta(datos: dict, payload: dict) -> tuple[dict | None, list[str]]:
    """Valida la salida del Agente 2. Devuelve (análisis por candidato, errores)."""
    errores: list[str] = []
    lista = datos.get("candidatos")
    if not isinstance(lista, list):
        return None, ["Falta la lista 'candidatos'."]

    esperados = {c["candidato_registro_id"]: c for c in payload["candidatos"]}
    referencia = payload["referencia"]["valores"]
    variables = set(referencia)
    vistos = set()
    analisis = {}

    for item in lista:
        if not isinstance(item, dict):
            errores.append("Un elemento de 'candidatos' no es un objeto.")
            continue
        cid = item.get("candidato_registro_id")
        if cid not in esperados:
            errores.append(f"Candidato no solicitado o identificador desconocido: {cid!r}.")
            continue
        if cid in vistos:
            errores.append(f"Candidato repetido: {cid}.")
            continue
        vistos.add(cid)
        estado = item.get("estado")
        if estado not in ("evaluado", "no_evaluable"):
            errores.append(f"{cid}: 'estado' debe ser 'evaluado' o 'no_evaluable'.")

        dims = item.get("dimensiones", [])
        if not isinstance(dims, list):
            errores.append(f"{cid}: 'dimensiones' debe ser una lista.")
            dims = []
        nombres = set()
        puntuaciones = []
        textos = []
        for d in dims:
            if not isinstance(d, dict):
                errores.append(f"{cid}: dimensión con formato incorrecto.")
                continue
            nombre = d.get("nombre")
            if nombre not in C.DIMENSIONES:
                errores.append(f"{cid}: dimensión desconocida {nombre!r}.")
                continue
            if nombre in nombres:
                errores.append(f"{cid}: dimensión repetida {nombre}.")
                continue
            nombres.add(nombre)
            p = d.get("puntuacion")
            if p is not None and (not isinstance(p, int) or isinstance(p, bool) or not 0 <= p <= 4):
                errores.append(f"{cid}/{nombre}: la puntuación debe ser un entero de 0 a 4 o null.")
                continue
            evidencias = d.get("evidencias") or []
            if p is not None and not evidencias:
                errores.append(f"{cid}/{nombre}: una dimensión puntuada necesita evidencias.")
            for ev in evidencias if isinstance(evidencias, list) else []:
                var = ev.get("variable") if isinstance(ev, dict) else None
                if var not in variables:
                    errores.append(f"{cid}/{nombre}: variable de evidencia no válida {var!r}.")
                    continue
                vo, vc = ev.get("valor_objetivo"), ev.get("valor_candidato")
                if not (_es_numero(vo) and _es_numero(vc)):
                    errores.append(f"{cid}/{nombre}: los valores de la evidencia {var} deben ser numéricos.")
                    continue
                if abs(vo - referencia[var]) > TOLERANCIA_VALOR:
                    errores.append(f"{cid}/{nombre}: valor_objetivo de {var} = {vo} no coincide con los datos "
                                   f"({referencia[var]}).")
                if abs(vc - esperados[cid]["valores"][var]) > TOLERANCIA_VALOR:
                    errores.append(f"{cid}/{nombre}: valor_candidato de {var} = {vc} no coincide con los datos "
                                   f"({esperados[cid]['valores'][var]}).")
            if p is not None:
                puntuaciones.append(p)
            textos.append(str(d.get("justificacion", "")))

        for campo in ("semejanzas", "diferencias", "limitaciones"):
            valor = item.get(campo, [])
            if not isinstance(valor, list) or not all(isinstance(x, str) for x in valor):
                errores.append(f"{cid}: '{campo}' debe ser una lista de textos.")
                item[campo] = []
        if not isinstance(item.get("resumen", ""), str):
            errores.append(f"{cid}: 'resumen' debe ser un texto.")
            item["resumen"] = ""
        textos += item["semejanzas"] + item["diferencias"] + [item.get("resumen", "")]
        prohibidos = _textos_con_terminos_prohibidos(textos)
        if prohibidos:
            errores.append(f"{cid}: afirma aspectos que no están en los datos ({', '.join(prohibidos)}); "
                           "muévelo a 'limitaciones' o elimínalo.")

        evaluable = estado == "evaluado" and len(puntuaciones) >= C.MIN_DIMENSIONES_EVALUABLES
        score = 100.0 * sum(puntuaciones) / (4 * len(puntuaciones)) if evaluable else None
        analisis[cid] = {
            "estado_analisis": "evaluado" if evaluable else "no_evaluable",
            "score_cualitativo": score,
            "dimensiones": dims,
            "semejanzas": item["semejanzas"], "diferencias": item["diferencias"],
            "limitaciones": item["limitaciones"], "resumen": item.get("resumen", ""),
        }

    faltan = set(esperados) - vistos
    if faltan:
        errores.append(f"Faltan candidatos en la respuesta: {', '.join(sorted(faltan))}.")
    return (analisis if not errores else None), errores
