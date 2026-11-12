"""Ejecución controlada de un agente LLM: llamada, validación, reintento y traza.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from . import config as C
from .llm import CacheLLM, ErrorLLM, ProveedorLLM, extraer_json

Validador = Callable[[dict], tuple[Any, list[str]]]


def _ejecucion(consulta_id: str, rol: str, intento: int, proveedor: ProveedorLLM, estado: str,
               duracion_ms: int | None = None, tokens_entrada=None, tokens_salida=None, coste=None,
               codigo_error=None, detalle=None) -> dict:
    return {
        "consulta_id": consulta_id, "agente": rol, "intento": intento,
        "proveedor": proveedor.nombre, "modelo": proveedor.modelo, "prompt_version": C.PROMPT_VERSION,
        "estado": estado, "duracion_ms": duracion_ms, "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida, "coste_usd": coste, "codigo_error": codigo_error, "detalle": detalle,
    }


def ejecutar_agente(rol: str, proveedor: ProveedorLLM, sistema: str, usuario: str, validar: Validador,
                    consulta_id: str, max_tokens: int, cache: CacheLLM | None = None,
                    max_intentos: int = C.LLM_MAX_INTENTOS) -> tuple[Any | None, list[dict]]:
    """Devuelve (resultado validado o None, lista de ejecuciones registradas).

    - Errores transitorios de la API, JSON inválido o validación fallida, se realiza un reintento.
    - En el reintento se indica al modelo qué falló .
    - Error no transitorio (clave inválida), no se reintenta.
    """
    ejecuciones: list[dict] = []
    clave = cache.clave(proveedor, sistema, usuario) if cache else None

    if cache and clave:
        texto = cache.leer(clave)
        if texto:
            try:
                resultado, errores = validar(extraer_json(texto))
                if not errores:
                    ejecuciones.append(_ejecucion(consulta_id, rol, 0, proveedor, "cache", 0, 0, 0, 0.0))
                    return resultado, ejecuciones
            except ValueError:
                pass

    mensaje = usuario
    for intento in range(1, max_intentos + 1):
        inicio = time.perf_counter()
        try:
            resp = proveedor.completar(sistema, mensaje, max_tokens)
        except ErrorLLM as e:
            ejecuciones.append(_ejecucion(consulta_id, rol, intento, proveedor, "error_api",
                                          int((time.perf_counter() - inicio) * 1000), codigo_error=e.codigo))
            if not e.transitorio:
                break
            continue

        base = dict(duracion_ms=resp.duracion_ms, tokens_entrada=resp.tokens_entrada,
                    tokens_salida=resp.tokens_salida, coste=resp.coste_usd)
        try:
            if resp.truncada:
                raise ValueError("respuesta truncada por max_tokens")
            datos = extraer_json(resp.texto)
        except ValueError as e:
            ejecuciones.append(_ejecucion(consulta_id, rol, intento, proveedor, "json_invalido",
                                          codigo_error="json_invalido", detalle=str(e)[:300], **base))
            mensaje = usuario + ("\n\nTu respuesta anterior no era un JSON válido. Responde ÚNICAMENTE con el "
                                 "objeto JSON pedido, sin texto adicional.")
            continue

        resultado, errores = validar(datos)
        if errores:
            ejecuciones.append(_ejecucion(consulta_id, rol, intento, proveedor, "validacion_fallida",
                                          codigo_error="validacion", detalle=" | ".join(errores)[:500], **base))
            lista = "\n".join(f"- {e}" for e in errores[:8])
            mensaje = usuario + ("\n\nTu respuesta anterior no superó la validación por estos motivos:\n"
                                 f"{lista}\nCorrígelos y devuelve de nuevo el JSON completo.")
            continue

        ejecuciones.append(_ejecucion(consulta_id, rol, intento, proveedor, "ok", **base))
        if cache and clave:
            cache.guardar(clave, resp.texto)
        return resultado, ejecuciones

    return None, ejecuciones


def tokens_usados(ejecuciones: list[dict]) -> int:
    return sum((e.get("tokens_entrada") or 0) + (e.get("tokens_salida") or 0) for e in ejecuciones)
