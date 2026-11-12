import json

import pytest

import tests.test_simulador as S
from ai_scout import analista, director, ojeador
from ai_scout.llm import extraer_json
from ai_scout.modelos import Consulta
from ai_scout.ranking import calcular_ranking
from conftest import id_de


@pytest.fixture()
def recuperacion(df_sintetico):
    return ojeador.recuperar_candidatos(Consulta(id_de(df_sintetico, "MF Jugador 06")), df_sintetico)


@pytest.fixture()
def payload_a(recuperacion):
    return analista.construir_payload(recuperacion)


def _validar(funcion, payload):
    return analista.validar_respuesta(json.loads(funcion("", analista.construir_mensaje(payload))), payload)


def test_payload_anonimizado(payload_a, recuperacion):
    texto = json.dumps(payload_a, ensure_ascii=False)
    assert recuperacion["objetivo"]["nombre"] not in texto
    assert all(c["nombre"] not in texto for c in recuperacion["candidatos"])


def test_analisis_valido_y_score_cualitativo(payload_a):
    analisis, errores = _validar(S.analista_valido, payload_a)
    assert errores == []
    a = next(iter(analisis.values()))
    assert a["estado_analisis"] == "evaluado"
    assert a["score_cualitativo"] == pytest.approx(100 * (3 + 2) / 8)


@pytest.mark.parametrize("funcion, fragmento", [
    (S.analista_valor_inventado, "no coincide"),
    (S.analista_candidato_extra, "R9999"),
    (S.analista_termino_prohibido, "no están en los datos"),
])
def test_analisis_invalidos(payload_a, funcion, fragmento):
    analisis, errores = _validar(funcion, payload_a)
    assert analisis is None and any(fragmento in e for e in errores)


def test_falta_un_candidato(payload_a):
    datos = json.loads(S.analista_valido("", analista.construir_mensaje(payload_a)))
    datos["candidatos"].pop()
    analisis, errores = analista.validar_respuesta(datos, payload_a)
    assert analisis is None and any("Faltan" in e for e in errores)


def test_menos_de_dos_dimensiones_es_no_evaluable(payload_a):
    analisis, errores = _validar(S.analista_no_evaluable, payload_a)
    assert errores == []
    assert all(a["score_cualitativo"] is None for a in analisis.values())


def test_ranking_hibrido_y_degradado(recuperacion, payload_a):
    analisis, _ = _validar(S.analista_valido, payload_a)
    ranking, modo, _ = calcular_ranking(recuperacion["candidatos"], analisis, 0.8, 0.2)
    assert modo == "hibrido"
    c = ranking[0]
    assert c["score_final"] == pytest.approx(0.8 * c["score_numerico"] + 0.2 * c["score_cualitativo"])
    # Si hay un solo candidato no evaluable, se realiza toda la consulta en modo numérico
    analisis[next(iter(analisis))]["score_cualitativo"] = None
    ranking, modo, motivo = calcular_ranking(recuperacion["candidatos"], analisis, 0.8, 0.2)
    assert modo == "numerico" and motivo
    assert all(c["score_final"] == c["score_numerico"] for c in ranking)
    assert [c["registro_id"] for c in ranking] == [c["registro_id"] for c in recuperacion["candidatos"]]


def _payload_director(recuperacion):
    ranking, modo, _ = calcular_ranking(recuperacion["candidatos"], None, 0.8, 0.2)
    return director.construir_payload(recuperacion, ranking[:3], Consulta("x"), modo)


def test_director_valido(recuperacion):
    p = _payload_director(recuperacion)
    fichas, errores = director.validar_respuesta(json.loads(S.director_valido("", director.construir_mensaje(p))), p)
    assert errores == [] and len(fichas["fichas"]) == 3


def test_director_rechaza_cifras_inventadas(recuperacion):
    p = _payload_director(recuperacion)
    datos = json.loads(S.director_cifra_inventada("", director.construir_mensaje(p)))
    fichas, errores = director.validar_respuesta(datos, p)
    assert fichas is None and any("73.5" in e for e in errores)


def test_numeros_en_texto():
    assert director._numeros_en_texto("Jugó 1.827 minutos, 7,1 pases y 94.3 de índice") == [1827, 7.1, 94.3]


def test_ficha_plantilla(recuperacion):
    ranking, _, _ = calcular_ranking(recuperacion["candidatos"], None, 0.8, 0.2)
    ficha = director.ficha_plantilla(ranking[0])
    assert ficha["origen"] == "plantilla" and len(ficha["fortalezas"]) == 2
    assert ficha["veredicto"] in ("candidato_para_revisar", "candidato_con_reservas", "evidencia_insuficiente")


def test_extraer_json_tolerante():
    assert extraer_json('Aquí está:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extraer_json('texto {"a": {"b": 2}} fin') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extraer_json("sin json")
