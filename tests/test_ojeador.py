import pytest

from ai_scout import ojeador
from ai_scout.modelos import Consulta
from conftest import id_de


def test_indice_de_similitud():
    assert ojeador.score_numerico(1) == 100
    assert ojeador.score_numerico(0) == 50
    assert ojeador.score_numerico(-1) == 0


def test_consulta_valida(df_sintetico):
    assert ojeador.validar_consulta(Consulta(id_de(df_sintetico, "MF Jugador 03")), df_sintetico) == []


@pytest.mark.parametrize("cambios, texto", [
    ({"ligas_permitidas": []}, "liga"),
    ({"ligas_permitidas": ["Eredivisie"]}, "desconocidas"),
    ({"minutos_minimos": 0}, "minutos"),
    ({"edad_maxima": 70}, "edad"),
])
def test_consultas_invalidas(df_sintetico, cambios, texto):
    c = Consulta(id_de(df_sintetico, "MF Jugador 03"), **cambios)
    assert any(texto in e.lower() for e in ojeador.validar_consulta(c, df_sintetico))


def test_rechaza_portero_y_pocos_minutos_y_desconocido(df_sintetico):
    assert ojeador.validar_consulta(Consulta(id_de(df_sintetico, "Portero Uno")), df_sintetico)
    errores = ojeador.validar_consulta(Consulta(id_de(df_sintetico, "Poco Minutos")), df_sintetico)
    assert errores and "73 minutos" in errores[0]
    assert ojeador.validar_consulta(Consulta("R9999"), df_sintetico)


def test_excluye_objetivo_y_misma_persona(df_sintetico):
    c = Consulta(id_de(df_sintetico, "Traspasado", "Equipo A"), minutos_minimos=600)
    r = ojeador.recuperar_candidatos(c, df_sintetico)
    nombres = [x["nombre"] for x in r["candidatos"]]
    assert "Traspasado" not in nombres


def test_una_sola_representacion_por_persona(df_sintetico):
    c = Consulta(id_de(df_sintetico, "MF Jugador 05"), top_preseleccion=50)
    r = ojeador.recuperar_candidatos(c, df_sintetico)
    t = [x for x in r["candidatos"] if x["nombre"] == "Traspasado"]
    assert len(t) == 1 and t[0]["equipo"] == "Equipo B"  # el registro con más minutos


def test_restricciones_se_cumplen(df_sintetico):
    c = Consulta(id_de(df_sintetico, "DF Jugador 04"), ligas_permitidas=["La Liga", "Serie A"], edad_maxima=26,
                 minutos_minimos=900)
    r = ojeador.recuperar_candidatos(c, df_sintetico)
    assert r["candidatos"]
    for x in r["candidatos"]:
        assert x["liga"] in ("La Liga", "Serie A")
        assert x["edad_registrada"] is not None and x["edad_registrada"] <= 26
        assert x["minutos"] >= 900 and x["grupo_posicion"] == "DF"


def test_edad_ausente_solo_sin_filtro_de_edad(df_sintetico):
    base = id_de(df_sintetico, "MF Jugador 02")
    sin_filtro = ojeador.recuperar_candidatos(Consulta(base, top_preseleccion=50), df_sintetico)
    con_filtro = ojeador.recuperar_candidatos(Consulta(base, top_preseleccion=50, edad_maxima=40), df_sintetico)
    assert "Sin Edad" in [x["nombre"] for x in sin_filtro["candidatos"]]
    assert "Sin Edad" not in [x["nombre"] for x in con_filtro["candidatos"]]
    assert con_filtro["exclusiones"]["edad_no_disponible"] == 1


def test_filtra_antes_del_top(df_sintetico):
    c = Consulta(id_de(df_sintetico, "FW Jugador 01"), ligas_permitidas=["Ligue 1"])
    r = ojeador.recuperar_candidatos(c, df_sintetico)
    assert r["candidatos"] and all(x["liga"] == "Ligue 1" for x in r["candidatos"])


def test_orden_y_reproducibilidad(df_sintetico):
    c = Consulta(id_de(df_sintetico, "MF Jugador 07"))
    r1 = ojeador.recuperar_candidatos(c, df_sintetico)
    r2 = ojeador.recuperar_candidatos(c, df_sintetico)
    assert [x["registro_id"] for x in r1["candidatos"]] == [x["registro_id"] for x in r2["candidatos"]]
    cos = [x["coseno"] for x in r1["candidatos"]]
    assert cos == sorted(cos, reverse=True) and len(cos) <= 10
    assert [x["ranking_preliminar"] for x in r1["candidatos"]] == list(range(1, len(cos) + 1))


def test_desempate_por_minutos(df_sintetico):
    df = df_sintetico.copy()
    a, b = id_de(df, "FW Jugador 03"), id_de(df, "FW Jugador 04")
    columnas = [c for c in df.columns if c.endswith(("_p90", "_pct"))]
    df.loc[df["registro_id"] == b, columnas] = df.loc[df["registro_id"] == a, columnas].to_numpy()
    r = ojeador.recuperar_candidatos(Consulta(id_de(df, "FW Jugador 10"), top_preseleccion=50), df)
    ids = [x["registro_id"] for x in r["candidatos"]]
    mas_minutos = max((a, b), key=lambda i: df.loc[df["registro_id"] == i, "minutos"].iloc[0])
    assert ids.index(mas_minutos) < ids.index(a if mas_minutos == b else b)


def test_lista_parcial_y_sin_candidatos(df_sintetico):
    base = id_de(df_sintetico, "FW Jugador 02")
    vacia = ojeador.recuperar_candidatos(Consulta(base, edad_maxima=16), df_sintetico)
    assert vacia["candidatos"] == []
    parcial = ojeador.recuperar_candidatos(Consulta(base, ligas_permitidas=["Premier League"], edad_maxima=20),
                                           df_sintetico)
    assert 0 < len(parcial["candidatos"]) < 3 and parcial["avisos"]


def test_evidencias_trazables(df_sintetico):
    r = ojeador.recuperar_candidatos(Consulta(id_de(df_sintetico, "DF Jugador 08")), df_sintetico)
    c = r["candidatos"][0]
    fila = df_sintetico[df_sintetico["registro_id"] == c["registro_id"]].iloc[0]
    for d in c["diferencias"]:
        assert d["valor_candidato"] == pytest.approx(fila[d["variable"]])
        assert d["diferencia"] == pytest.approx(d["valor_candidato"] - d["valor_objetivo"])


# ---- Dataset real ----
def test_real_relevo_de_modric(datos_reales):
    df = datos_reales[0]
    c = Consulta(id_de(df, "Luka Modrić"), edad_maxima=25)
    assert ojeador.validar_consulta(c, df) == []
    r = ojeador.recuperar_candidatos(c, df)
    assert r["poblacion_referencia"] == 563 and r["n_candidatos_validos"] == 296
    assert len(r["candidatos"]) == 10
    assert all(x["grupo_posicion"] == "MF" and x["edad_registrada"] <= 25 for x in r["candidatos"])


def test_real_rodri_rechazado_por_minutos(datos_reales):
    df = datos_reales[0]
    errores = ojeador.validar_consulta(Consulta(id_de(df, "Rodri", "Manchester City")), df)
    assert errores and "73 minutos" in errores[0]
