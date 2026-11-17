import numpy as np
import pytest

from ai_scout import config as C
from ai_scout import ojeador, similitud
from ai_scout.modelos import Consulta
from conftest import id_de


# ---- Pesos ----
def test_pesos_de_rol_cubren_todas_las_variables():
    for grupo, roles in C.ROLES.items():
        for rol in roles:
            pesos = similitud.pesos_de_rol(grupo, rol)
            assert list(pesos) == C.PERFILES[grupo] and all(p >= 0 for p in pesos.values())
        assert set(similitud.pesos_de_rol(grupo, None).values()) == {1.0}   # equilibrado
    with pytest.raises(KeyError):
        similitud.pesos_de_rol("MF", "portero")


def test_los_roles_solo_citan_variables_de_su_grupo():
    for grupo, roles in C.ROLES.items():
        assert C.ROL_DEFECTO in roles
        for rol in roles.values():
            assert set(rol["pesos"]) <= set(C.PERFILES[grupo])


def test_vector_de_pesos_normalizado_a_media_uno():
    w = similitud.vector_pesos(["a", "b", "c"], {"a": 3.0}, {"b": 2.0})
    assert w.mean() == pytest.approx(1.0) and w[0] / w[2] == pytest.approx(3.0) and w[1] / w[2] == pytest.approx(2.0)
    assert similitud.vector_pesos(["a", "b"], None).tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        similitud.vector_pesos(["a", "b"], {"a": 0, "b": 0})
    with pytest.raises(ValueError):
        similitud.vector_pesos(["a", "b"], {"a": -1})


# ---- Coseno ponderado ----
def test_coseno_ponderado():
    q = np.array([1.0, 2.0, -1.0])
    Z = np.array([[2.0, 4.0, -2.0], [-1.0, -2.0, 1.0], [1.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
    sin_pesos = similitud.cosenos(q, Z)
    assert sin_pesos[0] == pytest.approx(1.0) and sin_pesos[1] == pytest.approx(-1.0)
    assert np.isnan(sin_pesos[3])                                 # norma cero: similitud no definida
    assert similitud.cosenos(q, Z, np.ones(3)) == pytest.approx(sin_pesos, nan_ok=True)
    # Con peso 0 en la tercera variable, el tercer vector pasa a ser idéntico en dirección
    assert similitud.cosenos(q, Z, np.array([1.5, 1.5, 0.0]))[2] == pytest.approx(1.0)


def test_indice_de_similitud():
    assert [similitud.indice(c) for c in (1, 0, -1)] == [100, 50, 0]


# ---- Rocchio ----
def test_rocchio_acerca_a_los_positivos_y_aleja_de_los_negativos():
    ref = np.array([1.0, 0.0])
    positivos, negativos = np.array([[1.0, 1.0]]), np.array([[1.0, -1.0]])
    q = similitud.consulta_rocchio(ref, positivos, np.ones(1), negativos, np.ones(1), alfa=1, beta=0.5, gamma=0.5)
    assert q.tolist() == [1.0, 1.0]
    assert similitud.cosenos(q, positivos)[0] > similitud.cosenos(ref, positivos)[0]
    assert similitud.cosenos(q, negativos)[0] < similitud.cosenos(ref, negativos)[0]
    assert similitud.consulta_rocchio(ref, None, None, None, None).tolist() == ref.tolist()


# ---- Diversidad (MMR) ----
def test_mmr():
    Z = np.array([[1.0, 0.0], [0.99, 0.05], [0.98, 0.1], [0.6, 0.8], [0.5, -0.85]])
    sim = similitud.cosenos(np.array([1.0, 0.0]), Z)
    assert similitud.seleccionar_mmr(sim, Z, 3, 0.0) == [0, 1, 2]              # sin diversidad: los primeros
    diverso = similitud.seleccionar_mmr(sim, Z, 3, 0.7)
    assert diverso[0] == 0 and set(diverso) != {0, 1, 2}                        # entra alguno distinto
    assert similitud.seleccionar_mmr(sim, Z, 10, 0.5) and len(similitud.seleccionar_mmr(sim, Z, 10, 0.5)) == 5
    assert similitud.seleccionar_mmr(sim[:0], Z[:0], 3, 0.5) == []


# ---- Ojeador de datos con roles, pesos y diversidad ----
def test_el_rol_cambia_la_similitud_y_queda_trazado(df_sintetico):
    base = id_de(df_sintetico, "MF Jugador 06")
    equilibrado = ojeador.recuperar_candidatos(Consulta(base, top_preseleccion=50), df_sintetico)
    creador = ojeador.recuperar_candidatos(Consulta(base, rol="creador", top_preseleccion=50), df_sintetico)
    assert equilibrado["rol_nombre"] == "Equilibrado" and set(equilibrado["pesos"].values()) == {1.0}
    assert creador["pesos"]["KP_p90"] > creador["pesos"]["Recov_p90"]
    assert np.mean(list(creador["pesos"].values())) == pytest.approx(1.0)
    cos_e = {c["registro_id"]: c["coseno"] for c in equilibrado["candidatos"]}
    cos_c = {c["registro_id"]: c["coseno"] for c in creador["candidatos"]}
    assert set(cos_e) == set(cos_c) and any(abs(cos_e[r] - cos_c[r]) > 1e-6 for r in cos_e)
    # La similitud base (pesos iguales) no depende del rol y coincide con la del rol equilibrado
    for c in creador["candidatos"]:
        assert c["coseno_base"] == pytest.approx(cos_e[c["registro_id"]])
    assert all(d["peso"] == creador["pesos"][d["variable"]] for d in creador["candidatos"][0]["diferencias"])


def test_pesos_ajustados_a_mano(df_sintetico):
    base = id_de(df_sintetico, "FW Jugador 04")
    solo_goles = {v: (1.0 if v == "npxG_p90" else 0.0) for v in C.PERFILES["FW"]}
    r = ojeador.recuperar_candidatos(Consulta(base, pesos_personalizados=solo_goles), df_sintetico)
    assert r["rol_nombre"] == "Pesos ajustados a mano"
    # Con una sola variable el coseno ponderado solo puede valer +1 o -1
    assert all(abs(abs(c["coseno"]) - 1.0) < 1e-9 for c in r["candidatos"])


@pytest.mark.parametrize("cambios, texto", [
    ({"rol": "finalizador"}, "rol"),                               # rol de delanteros para un centrocampista
    ({"pesos_personalizados": {"npxG_p90": 2.0}}, "no pertenecen"),
    ({"pesos_personalizados": {"KP_p90": 9.0}}, "entre 0 y"),
    ({"pesos_personalizados": {v: 0.0 for v in C.PERFILES["MF"]}}, "mayor que cero"),
    ({"diversidad": 1.5}, "diversidad"),
])
def test_consultas_invalidas_por_rol_pesos_o_diversidad(df_sintetico, cambios, texto):
    c = Consulta(id_de(df_sintetico, "MF Jugador 03"), **cambios)
    assert any(texto in e.lower() for e in ojeador.validar_consulta(c, df_sintetico))


def test_diversidad_cambia_la_preseleccion_y_mantiene_el_orden(df_sintetico):
    base = id_de(df_sintetico, "DF Jugador 02")
    normal = ojeador.recuperar_candidatos(Consulta(base, top_preseleccion=5), df_sintetico)
    diversa = ojeador.recuperar_candidatos(Consulta(base, top_preseleccion=5, diversidad=0.8), df_sintetico)
    assert len(diversa["candidatos"]) == 5
    assert diversa["candidatos"][0]["registro_id"] == normal["candidatos"][0]["registro_id"]
    assert {c["registro_id"] for c in diversa["candidatos"]} != {c["registro_id"] for c in normal["candidatos"]}
    cos = [c["coseno"] for c in diversa["candidatos"]]
    assert cos == sorted(cos, reverse=True)
