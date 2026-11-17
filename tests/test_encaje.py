"""Encaje en la plantilla del club de destino y Analista de plantilla."""
import json

import pandas as pd
import pytest

import tests.test_simulador as S
from ai_scout import config as C
from ai_scout import encaje, estilos, informe, ojeador, orquestador
from ai_scout.llm import ProveedorSimulado
from ai_scout.modelos import Consulta
from conftest import id_de


@pytest.fixture()
def contexto(df_sintetico):
    modelos = estilos.calcular_estilos(df_sintetico)
    rec = ojeador.recuperar_candidatos(Consulta(id_de(df_sintetico, "MF Jugador 06")), df_sintetico, modelos)
    return df_sintetico, modelos, rec


def test_perfiles_de_equipo_estandarizados(df_sintetico):
    perfiles = encaje.perfiles_de_equipo(df_sintetico)
    assert set(perfiles.index) == set(encaje.clubes(df_sintetico)) and list(perfiles.columns) == encaje.VARIABLES_EQUIPO
    assert perfiles.mean().abs().max() < 1e-9
    assert encaje._parecido_de_equipos(perfiles, "Equipo 0", "Equipo 0") == 100.0
    assert 0 <= encaje._parecido_de_equipos(perfiles, "Equipo 0", "Equipo 1") <= 100
    assert encaje._parecido_de_equipos(perfiles, "Equipo 0", "No existe") is None


def test_indicadores_de_encaje(contexto):
    df, modelos, rec = contexto
    destino = rec["objetivo"]["equipo"]
    e = encaje.evaluar(rec, df, destino, modelos)
    assert e["equipo"] == destino and e["referencia_en_el_club"] and set(e["candidatos"]) == {
        c["registro_id"] for c in rec["candidatos"]}
    assert abs(sum(e["estilos_de_la_linea"].values()) - 1) < 0.02 or not e["estilos_de_la_linea"]
    for c in rec["candidatos"]:
        datos = e["candidatos"][c["registro_id"]]
        ind = datos["indicadores"]
        assert set(ind) == set(C.PESOS_ENCAJE) and 0 <= datos["score_indicadores"] <= 100
        assert ind["liga"] == (100.0 if c["liga"] == e["liga"] else 0.0)
        if c["estilo"] == rec["objetivo"]["estilo"]:
            assert ind["complemento"] == 100.0                           # cubre el hueco de la referencia
        if c["equipo"] == destino:
            assert ind["estilo_equipo"] == 100.0 and datos["ya_en_el_club"]
        assert ind["compatriotas"] == encaje.PUNTOS_COMPATRIOTAS.get(datos["n_compatriotas"], 100.0)
    # Sin modelo de estilos el indicador de complemento no se calcula y la nota usa los demás
    sin_estilos = encaje.evaluar(ojeador.recuperar_candidatos(Consulta(rec["objetivo"]["registro_id"]), df), df, destino)
    assert all(d["indicadores"]["complemento"] is None for d in sin_estilos["candidatos"].values())


def test_validacion_del_analista_de_plantilla(contexto):
    df, modelos, rec = contexto
    payload = encaje.construir_payload(rec, encaje.evaluar(rec, df, "Equipo 1", modelos))
    assert all("nombre" not in c for c in payload["candidatos"])
    mensaje = encaje.construir_mensaje(payload)
    valoracion, errores = encaje.validar_respuesta(json.loads(S.plantilla_valida("", mensaje)), payload)
    assert errores == [] and all(v["encaje"] == 3 for v in valoracion["candidatos"].values())
    _, errores = encaje.validar_respuesta(json.loads(S.plantilla_invalida("", mensaje)), payload)
    assert any("entero de 0 a 4" in e for e in errores) and any("Faltan" in e for e in errores)
    datos = json.loads(S.plantilla_valida("", mensaje))
    datos["candidatos"][0]["comentario"] = "Tiene un 987,65 % de química con el vestuario y buen salario."
    _, errores = encaje.validar_respuesta(datos, payload)
    assert any("cifras" in e for e in errores) and any("no están en los datos" in e for e in errores)


def _ejecutar(df, modelos, destino, plantilla_fn=None, **consulta):
    proveedores = dict(proveedor_analista=ProveedorSimulado(S.analista_valido),
                       proveedor_director=ProveedorSimulado(S.director_valido)) if plantilla_fn else {}
    pp = ProveedorSimulado(plantilla_fn) if plantilla_fn else None
    r = orquestador.ejecutar_consulta(Consulta(id_de(df, "MF Jugador 06"), equipo_destino=destino, **consulta), df,
                                      "test", exportar=False, estilos=modelos, proveedor_plantilla=pp, **proveedores)
    return r, pp


def test_la_plantilla_pondera_la_puntuacion_final(contexto):
    df, modelos, _ = contexto
    sin, _ = _ejecutar(df, modelos, None)
    assert sin["plantilla"] is None and all("score_plantilla" not in c for c in sin["ranking"])
    con, _ = _ejecutar(df, modelos, "Equipo 1")
    assert con["plantilla"]["origen"] == "indicadores" and con["plantilla"]["equipo"] == "Equipo 1"
    for c in con["ranking"]:
        esperado = (1 - C.PESO_PLANTILLA) * c["score_sin_plantilla"] + C.PESO_PLANTILLA * c["score_plantilla"]
        assert c["score_final"] == pytest.approx(esperado) and c["plantilla"]["comentario"]
        assert c["score_sin_plantilla"] == pytest.approx(c["score_numerico"])
    finales = [c["score_final"] for c in con["ranking"]]
    assert finales == sorted(finales, reverse=True)
    assert [c["ranking_final"] for c in con["ranking"]] == list(range(1, len(finales) + 1))
    assert {c["registro_id"] for c in con["ranking"]} == {c["registro_id"] for c in sin["ranking"]}


def test_analista_de_plantilla_con_llm_y_degradacion(contexto):
    df, modelos, _ = contexto
    r, pp = _ejecutar(df, modelos, "Equipo 1", S.plantilla_valida)
    assert pp.llamadas == 1 and r["plantilla"]["origen"] == "llm" and r["plantilla"]["diagnostico"]
    assert [e["agente"] for e in r["ejecuciones"]] == ["analista", "plantilla", "director"]
    c = r["ranking"][0]
    assert c["plantilla"]["encaje_llm"] == 3
    assert c["score_plantilla"] == pytest.approx((c["plantilla"]["score_indicadores"] + 75) / 2)
    # Respuesta inválida dos veces: la nota se calcula solo con los indicadores
    r, pp = _ejecutar(df, modelos, "Equipo 1", S.plantilla_invalida)
    assert pp.llamadas == 2 and r["plantilla"]["origen"] == "indicadores"
    assert any("Analista de plantilla" in a for a in r["avisos"])
    assert all(c["score_plantilla"] == pytest.approx(c["plantilla"]["score_indicadores"]) for c in r["ranking"])


def test_club_de_destino_desconocido_y_exportacion(contexto, tmp_path):
    df, modelos, rec = contexto
    c = Consulta(rec["objetivo"]["registro_id"], equipo_destino="Club Inventado")
    assert any("club de destino" in e for e in ojeador.validar_consulta(c, df))
    r = orquestador.ejecutar_consulta(Consulta(rec["objetivo"]["registro_id"], equipo_destino="Equipo 2"), df, "test",
                                      dir_salidas=tmp_path, estilos=modelos)
    resultados = pd.read_csv(tmp_path / "resultados.csv", dtype=str, keep_default_na=False)
    assert resultados["score_plantilla"].astype(float).between(0, 100).all() and resultados["encaje_plantilla"].ne("").all()
    assert pd.read_csv(tmp_path / "consultas.csv", dtype=str)["equipo_destino"].iloc[0] == "Equipo 2"
    assert informe.generar_pdf(r).startswith(b"%PDF")
    assert "Plantilla" in informe.tabla_preseleccion(r).columns
