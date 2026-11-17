"""Los gráficos se construyen sin errores y con las series esperadas."""
import numpy as np
import pandas as pd
import pytest

from ai_scout import estilos, graficos, ojeador
from ai_scout.modelos import Consulta
from ai_scout.ranking import calcular_ranking
from conftest import id_de

pytest.importorskip("plotly")


@pytest.fixture()
def contexto(df_sintetico):
    modelos = estilos.calcular_estilos(df_sintetico)
    rec = ojeador.recuperar_candidatos(Consulta(id_de(df_sintetico, "MF Jugador 06"), rol="creador"), df_sintetico,
                                       modelos)
    ranking, _, _ = calcular_ranking(rec["candidatos"], None, 0.8, 0.2)
    return df_sintetico, modelos, rec, ranking


def test_radar_con_uno_o_varios_candidatos(contexto):
    _, _, rec, ranking = contexto
    uno = graficos.radar_percentiles(rec["objetivo"], ranking[0], rec["variables"])
    varios = graficos.radar_percentiles(rec["objetivo"], ranking[:3], rec["variables"])
    assert len(uno.data) == 2 and len(varios.data) == 4
    assert varios.data[0].name == rec["objetivo"]["nombre"] and len(varios.data[0].r) == len(rec["variables"]) + 1


def test_mapa_de_estilos_resalta_referencia_y_candidatos(contexto):
    df, modelos, rec, ranking = contexto
    m = modelos["MF"]
    tabla = estilos.tabla_estilos(modelos, df)
    tabla = tabla[tabla["grupo_posicion"] == "MF"]
    fig = graficos.mapa_estilos(tabla, m.nombres, estilos.eje_pca(m, 0), estilos.eje_pca(m, 1),
                                rec["objetivo"]["registro_id"], ranking)
    nombres = [t.name for t in fig.data]
    assert nombres == m.nombres + ["Candidatos", "Referencia"]
    assert list(fig.data[-2].text) == [str(c["ranking_final"]) for c in ranking]
    assert sum(len(t.x) for t in fig.data[:m.k]) == len(tabla)
    solo_mapa = graficos.mapa_estilos(tabla, m.nombres, "x", "y")
    assert len(solo_mapa.data) == m.k


def test_graficos_de_pesos_y_aprendizaje(contexto):
    _, modelos, rec, _ = contexto
    pesos = graficos.barras_pesos(rec["pesos_rol"], rec["pesos"])
    assert len(pesos.data) == 2 and len(pesos.data[0].y) == len(rec["variables"])
    mult = graficos.barras_multiplicadores({"KP_p90": 1.4, "Int_p90": 0.7, "Cmp_pct": 0.9})
    assert list(mult.data[0].y) == ["Intercepciones", "Precisión de pase", "Pases clave"]      # de menor a mayor
    assert list(mult.data[0].text) == ["×0,70", "×0,90", "×1,40"]
    despl = graficos.barras_desplazamiento({"KP_p90": 0.3, "Int_p90": -0.2})
    assert list(despl.data[0].x) == [-0.2, 0.3]
    m = modelos["MF"]
    calor = graficos.mapa_centroides(m.variables, m.centroides, m.nombres)
    assert np.asarray(calor.data[0].z).shape == (m.k, len(m.variables))


def test_graficos_de_la_evaluacion():
    curvas = pd.DataFrame({"escenario": ["A", "A", "B", "B"], "referencias_valoradas": [0, 5, 0, 5],
                           "precision_10": [0.7, 0.9, 0.7, 0.75], "ndcg_10": [0.9, 0.95, 0.9, 0.91]})
    assert [t.name for t in graficos.curvas_aprendizaje(curvas).data] == ["A", "B"]
    siluetas = pd.DataFrame({"grupo": ["DF"] * 3, "k": [2, 3, 4], "silueta": [0.3, 0.25, 0.2],
                             "elegido": [False, False, True]})
    fig = graficos.curvas_silueta(siluetas)
    assert len(fig.data) == 2 and list(fig.data[1].x) == [4]
    barras = graficos.barras_comparacion(pd.DataFrame({"g": ["a", "b"], "x": [1, 2], "y": [2, 3]}), "g",
                                         {"x": "Uno", "y": "Dos"})
    assert [t.name for t in barras.data] == ["Uno", "Dos"]
