"""Proveedor Gemini sin llamadas reales: se sustituye el cliente por uno falso."""
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")
from google.genai import errors  # noqa: E402

from ai_scout.llm import ErrorLLM, ProveedorGemini  # noqa: E402


def _respuesta(texto, motivo="STOP"):
    partes = [SimpleNamespace(text="pensando…", thought=True), SimpleNamespace(text=texto, thought=False)]
    cand = SimpleNamespace(finish_reason=SimpleNamespace(name=motivo), content=SimpleNamespace(parts=partes))
    uso = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5)
    return SimpleNamespace(candidates=[cand], usage_metadata=uso)


def _proveedor(funcion):
    p = ProveedorGemini("gemini-3.5-flash-lite", "clave-falsa", esquema={"type": "object"})
    p._cliente = SimpleNamespace(models=SimpleNamespace(generate_content=funcion))
    return p


def test_respuesta_ok_ignora_partes_de_razonamiento():
    r = _proveedor(lambda **kw: _respuesta('{"a": 1}')).completar("sis", "usu", 100)
    assert r.texto == '{"a": 1}' and r.tokens_entrada == 100 and r.tokens_salida == 25
    assert not r.truncada and r.coste_usd == 0.0


def test_respuesta_truncada():
    assert _proveedor(lambda **kw: _respuesta("{", "MAX_TOKENS")).completar("s", "u", 10).truncada


def test_bloqueo_por_seguridad_es_rechazo():
    with pytest.raises(ErrorLLM) as e:
        _proveedor(lambda **kw: _respuesta("", "SAFETY")).completar("s", "u", 10)
    assert e.value.codigo == "rechazo" and not e.value.transitorio


@pytest.mark.parametrize("codigo,transitorio", [(429, True), (503, True), (400, False), (403, False)])
def test_errores_http(codigo, transitorio):
    def falla(**kw):
        raise errors.APIError(codigo, {"error": {"message": "x", "status": "X"}})
    with pytest.raises(ErrorLLM) as e:
        _proveedor(falla).completar("s", "u", 10)
    assert e.value.codigo == f"http_{codigo}" and e.value.transitorio is transitorio


def test_envia_esquema_json_y_crear_proveedor_lo_asigna(monkeypatch):
    recibido = {}

    def captura(**kw):
        recibido.update(kw)
        return _respuesta("{}")
    _proveedor(captura).completar("s", "u", 10)
    assert recibido["config"].response_json_schema == {"type": "object"}
    assert recibido["config"].response_mime_type == "application/json"

    from ai_scout import analista, director, llm
    from ai_scout import config as C
    monkeypatch.setattr(C, "LLM_PROVEEDOR", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "clave-falsa")
    assert llm.crear_proveedor("analista").esquema is analista.ESQUEMA_RESPUESTA
    assert llm.crear_proveedor("director").esquema is director.ESQUEMA_RESPUESTA
