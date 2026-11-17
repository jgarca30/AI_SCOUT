"""Jefe de scouting de fichas (LLM).

Lee los datos que tenía el director y las fichas que ha redactado, y comprueba lo que el
código no puede validar: que las comparaciones concuerdan con las cifras, que fortalezas y diferencias no se contradicen y que el veredicto es coherente.

Si encuentra problemas, el orquestador pide al director UNA corrección. El revisor no reescribe fichas ni cambia el ranking.
"""
from __future__ import annotations

import json

from . import config as C
from .director import cifras_no_respaldadas
from .redaccion import pulir, pulir_lista

SISTEMA = f"""Eres el Jefe de scouting (revisor de los informes) de AI Scout, un recomendador académico de futbolistas.
Recibes los DATOS con los que trabajó el director deportivo y las FICHAS que ha redactado.
Tu tarea es revisar cada ficha y decir si puede publicarse tal cual.

Comprueba en cada ficha:
a) Que cada comparación con el jugador de referencia (mayor, menor, similar, superior, inferior…) concuerda
   con los valores de "valores" del candidato y de la referencia.
b) Que "fortalezas" y "diferencias_relevantes" no se contradicen entre sí ni con el "analisis_previo".
c) Que el "veredicto" es coherente con el índice de similitud y con las diferencias señaladas
   ({", ".join(C.VEREDICTOS)}).
d) Que no se afirma nada que no se pueda comprobar con los datos.

Reglas obligatorias:
1. Sé estricto pero justo: marca un problema SOLO si puedes señalar el dato que lo demuestra. Cita la
   variable y copia los valores tal como aparecen en los datos (mismos decimales). No inventes cifras.
2. Diferencias de redacción o de estilo no son problemas. Si la ficha es correcta, apruébala.
3. Si "aprobada" es true, "problemas" debe ser una lista vacía. Si es false, explica cada problema en una
   frase breve y concreta, indicando qué habría que corregir.
4. No reescribas las fichas ni propongas otro orden de candidatos.
5. Incluye exactamente una revisión por cada ficha recibida, identificada por "candidato_registro_id".
6. Responde ÚNICAMENTE con un objeto JSON con este formato:
{{"revisiones": [{{"candidato_registro_id": "R0000", "aprobada": true, "problemas": []}}],
  "valoracion_global": "una frase"}}
Escribe en español correcto, con todas sus tildes y la letra ñ."""

ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "revisiones": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "candidato_registro_id": {"type": "string"},
                "aprobada": {"type": "boolean"},
                "problemas": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidato_registro_id", "aprobada", "problemas"]}},
        "valoracion_global": {"type": "string"},
    },
    "required": ["revisiones", "valoracion_global"],
}

CAMPOS_FICHA = ("resumen", "fortalezas", "diferencias_relevantes", "reservas", "veredicto")


def construir_payload(payload_director: dict, fichas: dict) -> dict:
    """Datos del director + las fichas que ha redactado (`fichas` es la salida validada del Director deportivo)."""
    return {
        "datos": payload_director,
        "fichas": {
            "resumen_consulta": fichas["resumen_consulta"],
            "candidatos": [{"candidato_registro_id": cid, **{k: f.get(k) for k in CAMPOS_FICHA}}
                           for cid, f in fichas["fichas"].items()],
        },
    }


def construir_mensaje(payload: dict) -> str:
    return ("Revisa estas fichas de ojeador con los datos que las respaldan.\n\nDATOS Y FICHAS:\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=1) + "\n```")


def validar_respuesta(datos: dict, payload: dict) -> tuple[dict | None, list[str]]:
    """Devuelve ({candidato_id: {"aprobada", "problemas"}, "valoracion_global": …}, errores)."""
    errores: list[str] = []
    lista = datos.get("revisiones")
    if not isinstance(lista, list):
        return None, ["Falta la lista 'revisiones'."]
    esperados = [c["candidato_registro_id"] for c in payload["fichas"]["candidatos"]]
    revisiones: dict[str, dict] = {}
    for item in lista:
        if not isinstance(item, dict):
            errores.append("Un elemento de 'revisiones' no es un objeto.")
            continue
        cid = item.get("candidato_registro_id")
        if cid not in esperados:
            errores.append(f"Revisión de una ficha que no existe: {cid!r}.")
            continue
        if cid in revisiones:
            errores.append(f"Revisión repetida: {cid}.")
            continue
        aprobada, problemas = item.get("aprobada"), item.get("problemas", [])
        if not isinstance(aprobada, bool):
            errores.append(f"{cid}: 'aprobada' debe ser true o false.")
            continue
        if not isinstance(problemas, list) or not all(isinstance(x, str) for x in problemas):
            errores.append(f"{cid}: 'problemas' debe ser una lista de textos.")
            continue
        problemas = [p.strip() for p in problemas if p.strip()]
        if aprobada and problemas:
            errores.append(f"{cid}: una ficha aprobada no puede tener problemas; si los hay, pon 'aprobada': false.")
        if not aprobada and not problemas:
            errores.append(f"{cid}: una ficha no aprobada debe explicar al menos un problema.")
        malas = cifras_no_respaldadas(problemas, payload)
        if malas:
            errores.append(f"{cid}: citas cifras que no aparecen en los datos: "
                           f"{', '.join(str(m) for m in malas[:5])}.")
        revisiones[cid] = {"aprobada": aprobada, "problemas": pulir_lista(problemas)}
    faltan = [c for c in esperados if c not in revisiones]
    if faltan:
        errores.append(f"Faltan revisiones de: {', '.join(faltan)}.")
    global_ = datos.get("valoracion_global", "")
    if not isinstance(global_, str):
        errores.append("'valoracion_global' debe ser un texto.")
        global_ = ""
    if errores:
        return None, errores
    return {"revisiones": revisiones, "valoracion_global": pulir(global_)}, []


def mensaje_de_correccion(mensaje_director: str, fichas: dict, revision: dict) -> str:
    """Mensaje para que el director corrija su versión con las observaciones del revisor."""
    anterior = {
        "resumen_consulta": fichas["resumen_consulta"],
        "limitaciones_generales": fichas["limitaciones_generales"],
        "candidatos": [{"candidato_registro_id": cid, **{k: f.get(k) for k in CAMPOS_FICHA}}
                       for cid, f in fichas["fichas"].items()],
    }
    observaciones = "\n".join(f"- {cid}: {p}" for cid, r in revision["revisiones"].items()
                              if not r["aprobada"] for p in r["problemas"])
    return (mensaje_director
            + "\n\nTU VERSIÓN ANTERIOR:\n" + json.dumps(anterior, ensure_ascii=False)
            + "\n\nUn revisor ha encontrado estos problemas en tu versión anterior:\n" + observaciones
            + "\n\nCorrige SOLO lo necesario para resolverlos, mantén el resto y devuelve de nuevo el JSON "
              "completo con todas las fichas. Recuerda: no cambies el orden ni cites cifras que no estén en los datos.")
