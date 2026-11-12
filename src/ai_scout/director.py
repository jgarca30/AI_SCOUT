"""Agente 3 · Director deportivo (LLM) y ficha de respaldo por plantilla.

Redacta la ficha de ojeador del Top final a partir del ranking YA calculado.
No puede añadir candidatos, cambiar el orden ni citar cifras que no estén en los datos que recibe. 
Si falla, la ficha se genera con una plantilla.
"""
from __future__ import annotations

import json
import re

from . import config as C
from .analista import _textos_con_terminos_prohibidos, r2

SISTEMA = f"""Eres el Agente 3 (director deportivo) de AI Scout, un recomendador académico de futbolistas.
Recibes un ranking YA CALCULADO de candidatos para sustituir a un jugador de referencia, con sus estadísticas
y el análisis previo de otro agente. Debes redactar una ficha de ojeador breve para cada candidato.

Reglas obligatorias:
1. No cambies el orden ni añadas o quites candidatos. Usa los identificadores "candidato_registro_id" recibidos.
2. Usa solo los datos recibidos. Si citas una cifra, cópiala tal cual aparece en los datos (mismos decimales).
   No calcules porcentajes, medias ni cifras nuevas.
3. No hables de lesiones, estado físico, velocidad, personalidad, contratos, salarios, precio, valor de
   mercado ni encaje táctico en un equipo concreto: no están en los datos. Puedes mencionar que no se conocen
   solo dentro de "reservas" o "limitaciones_generales".
4. Los índices son de similitud estadística, NO probabilidades de éxito.
5. "veredicto" debe ser uno de: {", ".join(C.VEREDICTOS)}.
6. En los textos refiérete a cada jugador por su "nombre", nunca por su identificador (R0000): el
   identificador solo va en el campo "candidato_registro_id".
7. Responde ÚNICAMENTE con un objeto JSON con este formato:
{{"resumen_consulta": "2-3 frases", "limitaciones_generales": ["..."],
  "candidatos": [{{"candidato_registro_id": "R0000", "resumen": "1-2 frases", "fortalezas": ["..."],
     "diferencias_relevantes": ["..."], "reservas": ["..."], "veredicto": "candidato_para_revisar"}}]}}
Escribe en español, con tono profesional y frases cortas."""

_LISTA_TEXTOS = {"type": "array", "items": {"type": "string"}}

# Esquema JSON de la respuesta.
ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "resumen_consulta": {"type": "string"},
        "limitaciones_generales": _LISTA_TEXTOS,
        "candidatos": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "candidato_registro_id": {"type": "string"}, "resumen": {"type": "string"},
                "fortalezas": _LISTA_TEXTOS, "diferencias_relevantes": _LISTA_TEXTOS, "reservas": _LISTA_TEXTOS,
                "veredicto": {"type": "string", "enum": list(C.VEREDICTOS)},
            },
            "required": ["candidato_registro_id", "resumen", "fortalezas", "diferencias_relevantes",
                         "reservas", "veredicto"]}},
    },
    "required": ["resumen_consulta", "limitaciones_generales", "candidatos"],
}


def construir_payload(recuperacion: dict, ranking: list[dict], consulta, modo_ranking: str) -> dict:
    variables = recuperacion["variables"]
    obj = recuperacion["objetivo"]
    return {
        "temporada": C.TEMPORADA,
        "restricciones": {
            "ligas_permitidas": consulta.ligas_permitidas, "edad_maxima": consulta.edad_maxima,
            "minutos_minimos": consulta.minutos_minimos,
        },
        "modo_ranking": modo_ranking,
        "nota_indices": "indice_similitud y score_final van de 0 a 100 y miden parecido estadístico, no éxito.",
        "variables": {v: C.VARIABLES[v][0] for v in variables},
        "referencia": {
            "nombre": obj["nombre"], "equipo": obj["equipo"], "liga": obj["liga"],
            "posicion": obj["posicion_original"], "edad_registrada": obj["edad_registrada"],
            "minutos": obj["minutos"], "valores": {v: r2(obj["valores"][v]) for v in variables},
        },
        "candidatos": [{
            "candidato_registro_id": c["registro_id"], "ranking_final": c["ranking_final"],
            "nombre": c["nombre"], "equipo": c["equipo"], "liga": c["liga"], "posicion": c["posicion_original"],
            "edad_registrada": c["edad_registrada"], "minutos": c["minutos"],
            "indice_similitud": round(c["score_numerico"], 1),
            "score_final": round(c["score_final"], 1),
            "valores": {v: r2(c["valores"][v]) for v in variables},
            "analisis_previo": {k: c.get("analisis", {}).get(k) for k in ("semejanzas", "diferencias", "resumen")}
            if c.get("analisis") else None,
        } for c in ranking],
        "limitaciones_conocidas": C.LIMITACIONES_GENERALES,
    }


def construir_mensaje(payload: dict) -> str:
    return ("Redacta la ficha de ojeador de estos candidatos.\n\nDATOS:\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=1) + "\n```")


# Comprobación de cifras: toda cifra del texto debe aparecer en los datos
_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d+)?)(?![\w])")


def _numeros_en_texto(texto: str) -> list[float]:
    numeros = []
    for token in _NUM.findall(texto):
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):  
            numeros.append(float(token.replace(".", "")))
        else:
            numeros.append(float(token.replace(",", ".")))
    return numeros


def _numeros_permitidos(payload: dict) -> set[float]:
    permitidos = {float(n) for n in range(0, 11)} | {90.0, 100.0}
    pendientes = [payload]
    while pendientes:
        x = pendientes.pop()
        if isinstance(x, dict):
            pendientes.extend(x.values())
        elif isinstance(x, list):
            pendientes.extend(x)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            for dec in (0, 1, 2):
                permitidos.add(round(float(x), dec))
        elif isinstance(x, str):
            permitidos.update(_numeros_en_texto(x))
    return permitidos


def cifras_no_respaldadas(textos: list[str], payload: dict) -> list[float]:
    permitidos = _numeros_permitidos(payload)
    malas = []
    for t in textos:
        for n in _numeros_en_texto(t):
            if not any(abs(n - p) < 1e-9 for p in permitidos):
                malas.append(n)
    return malas


def validar_respuesta(datos: dict, payload: dict) -> tuple[dict | None, list[str]]:
    errores = []
    esperados = [c["candidato_registro_id"] for c in payload["candidatos"]]
    lista = datos.get("candidatos")
    if not isinstance(lista, list):
        return None, ["Falta la lista 'candidatos'."]
    fichas = {}
    textos_globales = []
    for item in lista:
        if not isinstance(item, dict):
            errores.append("Un elemento de 'candidatos' no es un objeto.")
            continue
        cid = item.get("candidato_registro_id")
        if cid not in esperados:
            errores.append(f"Candidato no incluido en el ranking: {cid!r}.")
            continue
        if cid in fichas:
            errores.append(f"Candidato repetido: {cid}.")
            continue
        if item.get("veredicto") not in C.VEREDICTOS:
            errores.append(f"{cid}: veredicto no válido {item.get('veredicto')!r}.")
        for campo in ("fortalezas", "diferencias_relevantes", "reservas"):
            v = item.get(campo, [])
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                errores.append(f"{cid}: '{campo}' debe ser una lista de textos.")
                item[campo] = []
        if not isinstance(item.get("resumen"), str) or not item.get("resumen", "").strip():
            errores.append(f"{cid}: falta 'resumen'.")
            item["resumen"] = ""
        afirmaciones = [item["resumen"]] + item["fortalezas"] + item["diferencias_relevantes"]
        prohibidos = _textos_con_terminos_prohibidos(afirmaciones)
        if prohibidos:
            errores.append(f"{cid}: afirma aspectos que no están en los datos ({', '.join(prohibidos)}).")
        todos = afirmaciones + item["reservas"]
        malas = cifras_no_respaldadas(todos, payload)
        if malas:
            errores.append(f"{cid}: cifras que no aparecen en los datos: {', '.join(str(m) for m in malas[:5])}.")
        textos_globales += todos
        fichas[cid] = {k: item.get(k) for k in ("resumen", "fortalezas", "diferencias_relevantes", "reservas",
                                                   "veredicto")}
        fichas[cid]["origen"] = "llm"
    faltan = [c for c in esperados if c not in fichas]
    if faltan:
        errores.append(f"Faltan fichas de: {', '.join(faltan)}.")
    resumen = datos.get("resumen_consulta", "")
    limitaciones = datos.get("limitaciones_generales", [])
    if not isinstance(resumen, str):
        errores.append("'resumen_consulta' debe ser un texto.")
        resumen = ""
    if not isinstance(limitaciones, list) or not all(isinstance(x, str) for x in limitaciones):
        errores.append("'limitaciones_generales' debe ser una lista de textos.")
        limitaciones = []
    malas = cifras_no_respaldadas([resumen] + limitaciones, payload)
    if malas:
        errores.append(f"resumen_consulta: cifras que no aparecen en los datos: {', '.join(map(str, malas[:5]))}.")
    if _textos_con_terminos_prohibidos([resumen]):
        errores.append("resumen_consulta: afirma aspectos que no están en los datos.")
    if errores:
        return None, errores
    return {"resumen_consulta": resumen, "limitaciones_generales": limitaciones, "fichas": fichas}, []


def fmt(x: float, dec: int = 2) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _texto_variable(d: dict) -> str:
    return (f"{C.ETIQUETAS_CORTAS.get(d['variable'], d['variable']).lower()}: "
            f"{fmt(d['valor_candidato'])} frente a {fmt(d['valor_objetivo'])} ({d['unidad']})")


def ficha_plantilla(candidato: dict) -> dict:
    """Ficha construida solo con las métricas (sin LLM)."""
    difs = sorted(candidato["diferencias"], key=lambda d: abs(d["z_diferencia"]))
    cercanas, lejanas = difs[:2], difs[-2:][::-1]
    score = candidato["score_numerico"]
    # Umbrales orientativos sobre el índice de similitud
    if score >= 85:
        veredicto = "candidato_para_revisar"
    elif score >= 70:
        veredicto = "candidato_con_reservas"
    else:
        veredicto = "evidencia_insuficiente"
    reservas = ["Ficha generada automáticamente a partir de las métricas, sin análisis cualitativo.",
                "El contexto táctico de su equipo no está en los datos."]
    if candidato.get("edad_registrada") is None:
        reservas.append("Edad no disponible en el dataset.")
    return {
        "resumen": (f"Perfil estadístico con un índice de similitud de {fmt(score, 1)} sobre 100 respecto al jugador "
                    f"de referencia, con {fmt(candidato['minutos'], 0)} minutos en {candidato['liga']}."),
        "fortalezas": ["Valor muy cercano en " + _texto_variable(d) for d in cercanas],
        "diferencias_relevantes": ["Mayor diferencia en " + _texto_variable(d) for d in lejanas],
        "reservas": reservas,
        "veredicto": veredicto,
        "origen": "plantilla",
    }


def resumen_plantilla(recuperacion: dict, n_final: int) -> str:
    obj = recuperacion["objetivo"]
    return (f"Se han comparado {recuperacion['n_candidatos_validos']} jugadores del grupo "
            f"{C.GRUPOS[recuperacion['grupo']].lower()} que cumplen las restricciones con {obj['nombre']} "
            f"({obj['equipo']}). Se muestran los {n_final} perfiles estadísticos más parecidos.")
