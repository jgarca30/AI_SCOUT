"""Acceso a los modelos de lenguaje con una interfaz común.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from . import config as C


class ErrorLLM(Exception):
    def __init__(self, mensaje: str, codigo: str = "error_api", transitorio: bool = False):
        super().__init__(mensaje)
        self.codigo = codigo
        self.transitorio = transitorio


@dataclass
class RespuestaLLM:
    texto: str
    modelo: str
    tokens_entrada: int | None = None
    tokens_salida: int | None = None
    duracion_ms: int | None = None
    truncada: bool = False

    @property
    def coste_usd(self) -> float | None:
        precios = C.PRECIOS_USD_MTOK.get(self.modelo)
        if not precios or self.tokens_entrada is None or self.tokens_salida is None:
            return None
        return (self.tokens_entrada * precios[0] + self.tokens_salida * precios[1]) / 1_000_000


class ProveedorLLM(Protocol):
    nombre: str
    modelo: str

    def completar(self, sistema: str, usuario: str, max_tokens: int) -> RespuestaLLM: ...


# --------------------------------------------------------------------------
# Proveedores
# --------------------------------------------------------------------------
class ProveedorAnthropic:
    """Cliente de Claude (API de Anthropic)."""

    nombre = "anthropic"

    def __init__(self, modelo: str, api_key: str, timeout: float = C.LLM_TIMEOUT_S,
                 esfuerzo: str | None = None):
        import anthropic  # importación diferida: solo hace falta si se usa

        self._sdk = anthropic
        self.modelo = modelo
        self.esfuerzo = esfuerzo
        self._cliente = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)

    def completar(self, sistema: str, usuario: str, max_tokens: int) -> RespuestaLLM:
        a = self._sdk
        inicio = time.perf_counter()
        # Los modelos Claude 5.x no admiten temperature/top_p: se usan los valores por defecto.
        parametros = dict(model=self.modelo, max_tokens=max_tokens, system=sistema,
                          messages=[{"role": "user", "content": usuario}])
        if self.esfuerzo:
            parametros["output_config"] = {"effort": self.esfuerzo}
        try:
            if self.modelo in C.MODELOS_CON_FALLBACK:
                # Si el filtro de seguridad rechaza la petición, la API la repite en otro modelo.
                resp = self._cliente.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **parametros)
            else:
                resp = self._cliente.messages.create(**parametros)
        except a.APIStatusError as e:
            raise ErrorLLM(f"HTTP {e.status_code}", f"http_{e.status_code}",
                           transitorio=e.status_code in (408, 409, 429, 500, 502, 503, 504, 529)) from e
        except a.APIConnectionError as e:  # incluye timeout
            raise ErrorLLM(type(e).__name__, "conexion_o_timeout", transitorio=True) from e
        if resp.stop_reason == "refusal":
            raise ErrorLLM("el modelo rechazó la petición", "rechazo", transitorio=False)
        texto = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        return RespuestaLLM(texto=texto, modelo=getattr(resp, "model", None) or self.modelo,
                            tokens_entrada=getattr(resp.usage, "input_tokens", None),
                            tokens_salida=getattr(resp.usage, "output_tokens", None),
                            duracion_ms=int((time.perf_counter() - inicio) * 1000),
                            truncada=resp.stop_reason == "max_tokens")


class ProveedorGemini:
    """Cliente de Gemini (API de Google AI Studio). Tiene un nivel gratuito con límites diarios."""

    nombre = "gemini"
    # Motivos de parada por filtros de contenido: se tratan como un rechazo del modelo
    _BLOQUEOS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}

    def __init__(self, modelo: str, api_key: str, timeout: float = C.LLM_TIMEOUT_S, esquema: dict | None = None):
        from google import genai  # importación diferida: solo hace falta si se usa
        from google.genai import errors, types

        self._errores = errors
        self._types = types
        self.modelo = modelo
        self.esquema = esquema  # esquema JSON de la respuesta (salida estructurada)
        # timeout en milisegundos. 
        self._cliente = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=15.0,
                                                 http_status_codes=[503])))

    def completar(self, sistema: str, usuario: str, max_tokens: int) -> RespuestaLLM:
        import httpx2

        inicio = time.perf_counter()
        config = self._types.GenerateContentConfig(
            system_instruction=sistema, max_output_tokens=max_tokens,
            response_mime_type="application/json",  # obliga a responder con JSON…
            response_json_schema=self.esquema)       # …y con la estructura esperada
        try:
            resp = self._cliente.models.generate_content(model=self.modelo, contents=usuario, config=config)
        except self._errores.APIError as e:
            # 429 = límite del nivel gratuito (por minuto o por día)
            raise ErrorLLM(f"HTTP {e.code} {e.status or ''}".strip(), f"http_{e.code}",
                           transitorio=e.code in (408, 429, 500, 502, 503, 504)) from e
        except (httpx2.TransportError, httpx2.TimeoutException) as e:
            raise ErrorLLM(type(e).__name__, "conexion_o_timeout", transitorio=True) from e

        candidato = resp.candidates[0] if resp.candidates else None
        motivo = getattr(getattr(candidato, "finish_reason", None), "name", None)
        if candidato is None or motivo in self._BLOQUEOS:
            raise ErrorLLM(f"respuesta bloqueada ({motivo or 'sin candidatos'})", "rechazo", transitorio=False)
        partes = getattr(getattr(candidato, "content", None), "parts", None) or []
        texto = "".join(p.text for p in partes if getattr(p, "text", None) and not getattr(p, "thought", False))
        uso = resp.usage_metadata
        salida = None
        if uso is not None and uso.candidates_token_count is not None:
            salida = uso.candidates_token_count + (uso.thoughts_token_count or 0)
        return RespuestaLLM(texto=texto, modelo=self.modelo,
                            tokens_entrada=getattr(uso, "prompt_token_count", None),
                            tokens_salida=salida,
                            duracion_ms=int((time.perf_counter() - inicio) * 1000),
                            truncada=motivo == "MAX_TOKENS")


class ProveedorSimulado:
    """Proveedor de pruebas: una función decide la respuesta. No hace llamadas externas."""

    nombre = "simulado"

    def __init__(self, funcion: Callable[[str, str], str], modelo: str = "simulado"):
        self.funcion = funcion
        self.modelo = modelo
        self.llamadas = 0

    def completar(self, sistema: str, usuario: str, max_tokens: int) -> RespuestaLLM:
        self.llamadas += 1
        texto = self.funcion(sistema, usuario)
        return RespuestaLLM(texto=texto, modelo=self.modelo, tokens_entrada=len(usuario) // 4,
                            tokens_salida=len(texto) // 4, duracion_ms=1)


def crear_proveedor(rol: str) -> ProveedorLLM | None:
    """Crea el proveedor configurado para un rol ('analista' o 'director'), o None si no hay."""
    proveedor = C.LLM_PROVEEDOR
    if proveedor in ("", "ninguno", "none"):
        return None
    defecto = C.MODELOS_DEFECTO.get(proveedor, {}).get(rol)
    modelo = C.entorno(f"LLM_MODELO_{rol.upper()}", defecto)
    try:
        if proveedor == "anthropic":
            clave = C.entorno("ANTHROPIC_API_KEY")
            esfuerzo = C.entorno(f"LLM_ESFUERZO_{rol.upper()}")
            return ProveedorAnthropic(modelo, clave, esfuerzo=esfuerzo) if clave else None
        if proveedor == "gemini":
            clave = C.entorno("GEMINI_API_KEY") or C.entorno("GOOGLE_API_KEY")
            if not clave:
                return None
            from . import analista, director  # importación local para evitar ciclos
            esquema = (analista if rol == "analista" else director).ESQUEMA_RESPUESTA
            return ProveedorGemini(modelo, clave, esquema=esquema)
    except ImportError:
        return None
    return None


def descripcion_configuracion() -> str:
    """Texto breve para la app: qué proveedor y modelos se usarán."""
    a, d = crear_proveedor("analista"), crear_proveedor("director")
    if not a and not d:
        return "Sin modelo de lenguaje configurado (modo numérico)."
    partes = []
    if a:
        partes.append(f"Analista: {a.nombre}/{a.modelo}")
    if d:
        partes.append(f"Director: {d.nombre}/{d.modelo}")
    return " · ".join(partes)



def extraer_json(texto: str) -> dict:
    """Extrae el objeto JSON de la respuesta."""
    if not texto or not texto.strip():
        raise ValueError("respuesta vacía")
    t = texto.strip()
    bloque = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if bloque:
        t = bloque.group(1)
    else:
        inicio, fin = t.find("{"), t.rfind("}")
        if inicio == -1 or fin <= inicio:
            raise ValueError("no hay ningún objeto JSON en la respuesta")
        t = t[inicio:fin + 1]
    datos = json.loads(t)
    if not isinstance(datos, dict):
        raise ValueError("el JSON no es un objeto")
    return datos


class CacheLLM:
    """Caché en disco de respuestas YA VALIDADAS. Los fallos nunca se guardan.

    La clave incluye proveedor, modelo, versiones de prompt y esquema, y el
    texto completo enviado, con la versión del dataset y la consulta.
    """

    def __init__(self, directorio: Path = C.DIR_CACHE_LLM, activa: bool = C.LLM_USAR_CACHE):
        self.directorio = Path(directorio)
        self.activa = activa

    def clave(self, proveedor: ProveedorLLM, sistema: str, usuario: str) -> str:
        base = json.dumps([proveedor.nombre, proveedor.modelo, C.PROMPT_VERSION, C.ESQUEMA_VERSION, sistema, usuario],
                          ensure_ascii=False)
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def leer(self, clave: str) -> str | None:
        if not self.activa:
            return None
        ruta = self.directorio / f"{clave}.json"
        if not ruta.exists():
            return None
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))["texto"]
        except (OSError, ValueError, KeyError):
            return None

    def guardar(self, clave: str, texto: str) -> None:
        if not self.activa:
            return
        self.directorio.mkdir(parents=True, exist_ok=True)
        (self.directorio / f"{clave}.json").write_text(json.dumps({"texto": texto}, ensure_ascii=False),
                                                       encoding="utf-8")
