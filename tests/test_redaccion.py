"""Pulido determinista de los textos de los modelos de lenguaje."""
import json

import tests.test_simulador as S
from ai_scout import director, ojeador, redaccion
from ai_scout.modelos import Consulta
from ai_scout.ranking import calcular_ranking
from conftest import id_de


def test_restaura_tildes_y_enes():
    sucio = ("Busqueda hibrida: candidatos de hasta 23 anos con un minimo de participacion. "
             "Tiene mas precision y un indice estadistico alto en el area.")
    limpio = ("Búsqueda híbrida: candidatos de hasta 23 años con un mínimo de participación. "
              "Tiene más precisión y un índice estadístico alto en el área.")
    assert redaccion.restaurar_tildes(sucio) == limpio
    assert redaccion.restaurar_tildes(limpio) == limpio                     # un texto correcto no cambia


def test_no_toca_plurales_nombres_propios_ni_cifras():
    texto = "Las recuperaciones y conducciones de Mas (Sion) en el Area Metropolitana: 2,35 por 90 con xAG."
    assert redaccion.restaurar_tildes(texto) == texto
    assert redaccion.pulir(None) is None and redaccion.pulir_lista(None) is None


def test_los_codigos_de_variable_se_nombran():
    assert redaccion.pulir("Menor valor en Succ_p90 y en Cmp_pct que la referencia; npxG_p90 similar.") == \
        "Menor valor en regates y en precisión de pase que la referencia; npxG similar."
    assert redaccion.nombrar_variables("Codigo_p90 desconocido") == "Codigo_p90 desconocido"


def test_las_fichas_del_director_salen_pulidas(df_sintetico):
    rec = ojeador.recuperar_candidatos(Consulta(id_de(df_sintetico, "MF Jugador 06")), df_sintetico)
    ranking, modo, _ = calcular_ranking(rec["candidatos"], None, 0.8, 0.2)
    payload = director.construir_payload(rec, ranking[:3], Consulta("x"), modo)
    datos = json.loads(S.director_valido("", director.construir_mensaje(payload)))
    datos["resumen_consulta"] = "Busqueda de perfiles con mas participacion."
    datos["candidatos"][0]["reservas"] = ["Diferencia grande en KP_p90 en los ultimos partidos."]
    fichas, errores = director.validar_respuesta(datos, payload)
    assert errores == [] and fichas["resumen_consulta"] == "Búsqueda de perfiles con más participación."
    assert next(iter(fichas["fichas"].values()))["reservas"] == ["Diferencia grande en pases clave en los últimos partidos."]
