"""Jefe de scouting (revisor) y bucle de corrección del director."""
import json

import pytest

import tests.test_simulador as S
from ai_scout import critico, director, ojeador, orquestador
from ai_scout.llm import ProveedorSimulado
from ai_scout.modelos import Consulta
from ai_scout.ranking import calcular_ranking
from conftest import id_de


@pytest.fixture()
def payload_c(df_sintetico):
    """Datos del director + fichas ya validadas: lo que recibe el revisor."""
    rec = ojeador.recuperar_candidatos(Consulta(id_de(df_sintetico, "MF Jugador 06")), df_sintetico)
    ranking, modo, _ = calcular_ranking(rec["candidatos"], None, 0.8, 0.2)
    payload_d = director.construir_payload(rec, ranking[:3], Consulta("x"), modo)
    fichas, errores = director.validar_respuesta(
        json.loads(S.director_valido("", director.construir_mensaje(payload_d))), payload_d)
    assert errores == []
    return critico.construir_payload(payload_d, fichas), fichas, director.construir_mensaje(payload_d)


def _revisar(funcion, payload):
    return critico.validar_respuesta(json.loads(funcion("", critico.construir_mensaje(payload))), payload)


def test_el_revisor_recibe_datos_y_fichas(payload_c):
    payload, fichas, _ = payload_c
    assert len(payload["fichas"]["candidatos"]) == 3 and "valores" in payload["datos"]["candidatos"][0]
    assert all(set(critico.CAMPOS_FICHA) <= set(f) for f in payload["fichas"]["candidatos"])


def test_revision_aprobada_y_rechazada(payload_c):
    payload = payload_c[0]
    revision, errores = _revisar(S.critico_aprueba, payload)
    assert errores == [] and all(r["aprobada"] and not r["problemas"] for r in revision["revisiones"].values())
    revision, errores = _revisar(S.critico_rechaza_primera, payload)
    assert errores == []
    assert [r["aprobada"] for r in revision["revisiones"].values()] == [False, True, True]


@pytest.mark.parametrize("funcion, fragmento", [
    (S.critico_cifra_inventada, "cifras que no aparecen"),      # el revisor tampoco puede inventar cifras
    (S.critico_incoherente, "no puede tener problemas"),
    (S.texto_no_json, None),
])
def test_revisiones_invalidas(payload_c, funcion, fragmento):
    payload = payload_c[0]
    if fragmento is None:
        with pytest.raises(ValueError):
            _revisar(funcion, payload)
        return
    revision, errores = _revisar(funcion, payload)
    assert revision is None and any(fragmento in e for e in errores)


def test_revision_incompleta_o_con_fichas_inventadas(payload_c):
    payload = payload_c[0]
    datos = json.loads(S.critico_aprueba("", critico.construir_mensaje(payload)))
    datos["revisiones"].pop()
    assert any("Faltan" in e for e in critico.validar_respuesta(datos, payload)[1])
    datos["revisiones"].append({"candidato_registro_id": "R9999", "aprobada": True, "problemas": []})
    assert any("R9999" in e for e in critico.validar_respuesta(datos, payload)[1])
    datos = json.loads(S.critico_aprueba("", critico.construir_mensaje(payload)))
    datos["revisiones"][0].update(aprobada=False, problemas=[])
    assert any("al menos un problema" in e for e in critico.validar_respuesta(datos, payload)[1])


def test_mensaje_de_correccion(payload_c):
    payload, fichas, mensaje_d = payload_c
    revision, _ = _revisar(S.critico_rechaza_primera, payload)
    mensaje = critico.mensaje_de_correccion(mensaje_d, fichas, revision)
    assert mensaje.startswith(mensaje_d) and "TU VERSIÓN ANTERIOR" in mensaje
    assert "La fortaleza citada no concuerda" in mensaje
    assert S.payload(mensaje) == S.payload(mensaje_d)             # los datos siguen siendo los mismos


# ---- Orquestador con revisor ----
def _ejecutar(df, director_fn=S.director_valido, critico_fn=S.critico_aprueba, **consulta):
    pa, pd_, pc = ProveedorSimulado(S.analista_valido), ProveedorSimulado(director_fn), ProveedorSimulado(critico_fn)
    c = Consulta(id_de(df, "MF Jugador 06"), **consulta)
    r = orquestador.ejecutar_consulta(c, df, "test", pa, pd_, exportar=False, proveedor_critico=pc)
    return r, pd_, pc


def test_fichas_aprobadas(df_sintetico):
    r, pd_, pc = _ejecutar(df_sintetico)
    assert r["revision"]["estado"] == "aprobadas" and r["revision"]["rechazadas"] == 0
    assert all(c["ficha"]["revision"] == {"estado": "aprobada", "problemas": []} for c in r["top_final"])
    assert [e["agente"] for e in r["ejecuciones"]] == ["analista", "director", "critico"]
    assert pd_.llamadas == 1 and pc.llamadas == 1


def test_ficha_rechazada_y_corregida_por_el_director(df_sintetico):
    r, pd_, pc = _ejecutar(df_sintetico, S.Secuencia(S.director_valido, S.director_corregido),
                           S.critico_rechaza_primera)
    assert r["revision"]["estado"] == "corregidas" and r["revision"]["rechazadas"] == 1
    primera, segunda = r["top_final"][0]["ficha"], r["top_final"][1]["ficha"]
    assert primera["revision"]["estado"] == "corregida" and primera["revision"]["problemas"]
    assert segunda["revision"]["estado"] == "aprobada"
    assert primera["fortalezas"] == ["Versión corregida tras la revisión."] and primera["origen"] == "llm"
    assert [e["agente"] for e in r["ejecuciones"]] == ["analista", "director", "critico", "director_correccion"]
    assert pd_.llamadas == 2 and pc.llamadas == 1                 # una sola corrección, sin segunda revisión


def test_correccion_invalida_mantiene_la_ficha_con_observaciones(df_sintetico):
    r, pd_, _ = _ejecutar(df_sintetico, S.Secuencia(S.director_valido, S.director_cifra_inventada),
                          S.critico_rechaza_primera)
    assert r["revision"]["estado"] == "con_observaciones"
    primera = r["top_final"][0]["ficha"]
    assert primera["revision"]["estado"] == "con_observaciones" and primera["revision"]["problemas"]
    assert primera["fortalezas"] == ["Progresión parecida a la referencia."]      # la versión original
    assert any(i["codigo"] == "correccion_fallida" for i in r["incidencias"])
    assert pd_.llamadas == 2                                       # la corrección no se reintenta


def test_si_el_revisor_falla_las_fichas_se_muestran_sin_revisar(df_sintetico):
    r, _, pc = _ejecutar(df_sintetico, critico_fn=S.texto_no_json)
    assert r["revision"]["estado"] == "no_revisada" and pc.llamadas == 2
    assert all(c["ficha"]["origen"] == "llm" and c["ficha"]["revision"]["estado"] == "no_revisada"
               for c in r["top_final"])
    assert any("Jefe de scouting" in a for a in r["avisos"])
    assert any(i["codigo"] == "degradacion_agente4" for i in r["incidencias"])


def test_revisor_desactivado_o_sin_fichas_que_revisar(df_sintetico):
    r, _, pc = _ejecutar(df_sintetico, usar_critico=False)
    assert pc.llamadas == 0 and r["revision"] is None
    r, _, pc = _ejecutar(df_sintetico, director_fn=S.director_cifra_inventada)     # fichas por plantilla
    assert pc.llamadas == 0 and r["revision"] is None
    assert all(c["ficha"]["origen"] == "plantilla" for c in r["top_final"])
