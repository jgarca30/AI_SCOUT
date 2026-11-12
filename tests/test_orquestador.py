import pandas as pd

import tests.test_simulador as S
from ai_scout import orquestador
from ai_scout.llm import CacheLLM, ProveedorSimulado
from ai_scout.modelos import Consulta
from conftest import id_de


def _ejecutar(df, analista_fn=None, director_fn=None, tmp_path=None, cache=None, **consulta):
    pa = ProveedorSimulado(analista_fn) if analista_fn else None
    pd_ = ProveedorSimulado(director_fn) if director_fn else None
    c = Consulta(id_de(df, consulta.pop("nombre", "MF Jugador 06")), **consulta)
    r = orquestador.ejecutar_consulta(c, df, "test", pa, pd_, cache=cache, exportar=tmp_path is not None,
                                      dir_salidas=tmp_path)
    return r, pa, pd_


def test_flujo_completo_hibrido(df_sintetico):
    r, pa, pd_ = _ejecutar(df_sintetico, S.analista_valido, S.director_valido)
    assert r["estado"] == "completada" and r["modo_ranking"] == "hibrido"
    assert len(r["top_final"]) == 3 and all(c["ficha"]["origen"] == "llm" for c in r["top_final"])
    assert [e["estado"] for e in r["ejecuciones"]] == ["ok", "ok"]


def test_sin_llm_modo_numerico_y_plantilla(df_sintetico):
    r, _, _ = _ejecutar(df_sintetico)
    assert r["modo_ranking"] == "numerico"
    assert all(c["ficha"]["origen"] == "plantilla" for c in r["top_final"])
    assert any("No hay modelo" in a for a in r["avisos"])


def test_analista_invalido_degrada_toda_la_consulta(df_sintetico):
    r, pa, _ = _ejecutar(df_sintetico, S.analista_valor_inventado, S.director_valido)
    assert r["modo_ranking"] == "numerico" and pa.llamadas == 2
    assert all(c["score_final"] == c["score_numerico"] for c in r["ranking"])
    assert any(i["codigo"] == "degradacion_agente2" for i in r["incidencias"])


def test_reintento_corrige_json(df_sintetico):
    r, pa, _ = _ejecutar(df_sintetico, S.Secuencia(S.texto_no_json, S.analista_valido))
    assert r["modo_ranking"] == "hibrido" and pa.llamadas == 2
    assert [e["estado"] for e in r["ejecuciones"]] == ["json_invalido", "ok"]


def test_error_no_transitorio_no_reintenta(df_sintetico):
    r, pa, _ = _ejecutar(df_sintetico, S.error_api(transitorio=False))
    assert pa.llamadas == 1 and r["modo_ranking"] == "numerico"


def test_error_transitorio_reintenta(df_sintetico):
    r, pa, _ = _ejecutar(df_sintetico, S.Secuencia(S.error_api(True), S.analista_valido))
    assert pa.llamadas == 2 and r["modo_ranking"] == "hibrido"


def test_director_invalido_usa_plantilla(df_sintetico):
    r, _, pd_ = _ejecutar(df_sintetico, S.analista_valido, S.director_cifra_inventada)
    assert pd_.llamadas == 2 and r["modo_ranking"] == "hibrido"
    assert all(c["ficha"]["origen"] == "plantilla" for c in r["top_final"])


def test_consulta_rechazada_no_llama_al_llm(df_sintetico):
    r, pa, _ = _ejecutar(df_sintetico, S.analista_valido, nombre="Poco Minutos")
    assert r["estado"] == "rechazada" and pa.llamadas == 0


def test_sin_resultados(df_sintetico):
    r, pa, _ = _ejecutar(df_sintetico, S.analista_valido, edad_maxima=16)
    assert r["estado"] == "sin_resultados" and pa.llamadas == 0


def test_cache_reutiliza_respuestas_validas(df_sintetico, tmp_path):
    cache = CacheLLM(tmp_path / "cache", activa=True)
    df = df_sintetico
    c = Consulta(id_de(df, "DF Jugador 03"))
    pa = ProveedorSimulado(S.analista_valido)
    orquestador.ejecutar_consulta(c, df, "test", pa, None, cache=cache, exportar=False)
    c2 = Consulta(id_de(df, "DF Jugador 03"))
    r2 = orquestador.ejecutar_consulta(c2, df, "test", pa, None, cache=cache, exportar=False)
    assert pa.llamadas == 1 and r2["ejecuciones"][0]["estado"] == "cache"


def test_exportacion_para_power_bi(df_sintetico, tmp_path):
    r1, _, _ = _ejecutar(df_sintetico, S.analista_valido, S.director_valido, tmp_path=tmp_path)
    r2, _, _ = _ejecutar(df_sintetico, tmp_path=tmp_path, nombre="FW Jugador 05")
    _ejecutar(df_sintetico, tmp_path=tmp_path, nombre="Poco Minutos")
    consultas = pd.read_csv(tmp_path / "consultas.csv", dtype=str)
    resultados = pd.read_csv(tmp_path / "resultados.csv", dtype=str)
    evidencias = pd.read_csv(tmp_path / "evidencias.csv", dtype=str)
    registros = pd.read_csv(tmp_path / "registros.csv", dtype=str)
    ejecuciones = pd.read_csv(tmp_path / "ejecuciones_agentes.csv", dtype=str)
    incidencias = pd.read_csv(tmp_path / "incidencias.csv", dtype=str)
    assert len(consultas) == 3 and consultas["consulta_id"].is_unique
    assert set(consultas["estado_consulta"]) == {"completada", "rechazada"}
    assert resultados["resultado_id"].is_unique and evidencias["evidencia_id"].is_unique
    assert set(resultados["consulta_id"]) <= set(consultas["consulta_id"])
    assert set(evidencias["resultado_id"]) <= set(resultados["resultado_id"])
    assert set(resultados["candidato_registro_id"]) <= set(registros["registro_id"])
    assert registros["registro_id"].is_unique
    assert set(resultados["es_top_final"]) == {"true", "false"}
    assert (resultados["estado_economico"] == "no_evaluado").all()
    assert len(ejecuciones) == 2 and (incidencias["codigo"] == "consulta_rechazada").any()

