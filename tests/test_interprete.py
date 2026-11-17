"""Secretario técnico: intérprete de consultas en lenguaje natural (con LLM y por reglas)."""
import pytest

import tests.test_simulador as S
from ai_scout import config as C
from ai_scout import interprete as I
from ai_scout import ojeador
from ai_scout.llm import ProveedorSimulado
from ai_scout.modelos import Consulta
from conftest import id_de


# ---- Búsqueda de jugadores ----
def test_buscar_jugador_por_nombre_parcial_y_con_erratas(df_sintetico):
    assert I.buscar_jugador("MF Jugador 06", df_sintetico)[0]["registro_id"] == id_de(df_sintetico, "MF Jugador 06")
    assert I.buscar_jugador("traspasado", df_sintetico)[0]["registro_id"] == \
        id_de(df_sintetico, "Traspasado", "Equipo B")                 # el registro con más minutos
    assert I.buscar_jugador("Traspasado", df_sintetico, equipo="Equipo A")[0]["registro_id"] == \
        id_de(df_sintetico, "Traspasado", "Equipo A")
    assert I.buscar_jugador("Traspazado", df_sintetico)[0]["etiqueta"].startswith("Traspasado")   # errata
    assert I.buscar_jugador("Portero Uno", df_sintetico) == []        # los porteros no son candidatos
    assert I.buscar_jugador("Zzzzz", df_sintetico) == [] and I.buscar_jugador("", df_sintetico) == []


# ---- Intérprete por reglas ----
def test_reglas_extraen_todos_los_parametros(df_sintetico):
    r = I.interpretar_por_reglas("Busco un sustituto joven de Traspasado en la Bundesliga o la Serie A, "
                                 "creador y con al menos 900 minutos", df_sintetico)
    assert r.origen == "reglas" and r.objetivo_registro_id == id_de(df_sintetico, "Traspasado", "Equipo B")
    assert r.ligas == ["Serie A", "Bundesliga"] and r.edad_maxima == I.EDAD_JOVEN and r.minutos_minimos == 900
    assert r.rol == "creador" and any("joven" in n for n in r.notas)
    consulta = Consulta(r.objetivo_registro_id, **r.parametros())
    assert ojeador.validar_consulta(consulta, df_sintetico) == [] and consulta.texto_original.startswith("Busco")


@pytest.mark.parametrize("texto, edad", [
    ("sustituto de Traspasado sub-21", 21), ("relevo de Traspasado menor de 25 años", 24),
    ("como Traspasado, máximo 26 años", 26), ("parecido a Traspasado de 22 años o menos", 22),
    ("sustituto de Traspasado", None),
])
def test_reglas_de_edad(df_sintetico, texto, edad):
    assert I.interpretar_por_reglas(texto, df_sintetico).edad_maxima == edad


def test_reglas_de_ligas_excluidas_y_equipo_como_pista(df_sintetico):
    r = I.interpretar_por_reglas("alternativa a Traspasado del Equipo A que no sea de La Liga", df_sintetico)
    assert r.objetivo_registro_id == id_de(df_sintetico, "Traspasado", "Equipo A")
    assert r.ligas == [l for l in C.LIGAS_NORMALIZADAS if l != "La Liga"]
    assert I.interpretar_por_reglas("sustituto de Traspasado", df_sintetico).ligas is None


def test_reglas_sin_jugador_o_con_homonimos(df_sintetico):
    r = I.interpretar_por_reglas("un mediocentro defensivo joven", df_sintetico)
    assert r.objetivo_registro_id is None and any("jugador de referencia" in n for n in r.notas)
    r = I.interpretar_por_reglas("sustituto de Homonimo", df_sintetico)
    assert r.objetivo_registro_id and len(r.alternativas) == 1 and any("varios jugadores" in n for n in r.notas)


def test_real_frases_naturales(datos_reales):
    df = datos_reales[0]

    def nombre(r):
        return df.loc[df["registro_id"] == r.objetivo_registro_id, "nombre"].iloc[0]

    r = I.interpretar_por_reglas("extremo regateador parecido a Lamine Yamal, sub-21, que no sea de La Liga", df)
    assert nombre(r) == "Lamine Yamal" and r.rol == "extremo_regateador" and r.edad_maxima == 21
    assert "La Liga" not in r.ligas and len(r.ligas) == 4
    r = I.interpretar_por_reglas("quiero un central con salida de balón tipo Van Dijk, máximo 26 años", df)
    assert nombre(r) == "Virgil van Dijk" and r.rol == "salida_balon" and r.edad_maxima == 26
    r = I.interpretar_por_reglas("sustituto de Bellingam titular del mismo estilo", df)         # con errata
    assert nombre(r) == "Jude Bellingham" and r.minutos_minimos == I.MINUTOS_TITULAR and r.solo_mismo_estilo
    r = I.interpretar_por_reglas("parecido a Odegaard en la premier", df)                        # Ø → o
    assert nombre(r) == "Martin Ødegaard" and r.ligas == ["Premier League"]
    r = I.interpretar_por_reglas("sustituto de Rodri del Manchester City", df)
    assert df.loc[df["registro_id"] == r.objetivo_registro_id, "equipo"].iloc[0] == "Manchester City"
    r = I.interpretar_por_reglas("Busco un sustituto joven de Pedri en la Bundesliga, más creador", df)
    assert nombre(r) == "Pedri" and r.rol == "creador" and "Creador / mediapunta" in r.resumen(df)


# ---- Intérprete con modelo de lenguaje ----
def test_llm_valido(df_sintetico):
    proveedor = ProveedorSimulado(S.interprete(jugador_referencia="traspasado", equipo_referencia="Equipo A",
                                               ligas=["Bundesliga"], edad_maxima=22, rol="recuperador",
                                               no_aplicable=["que sea zurdo"]))
    r, ejecuciones = I.interpretar("pivote zurdo como Traspasado (Equipo A), sub-22, Bundesliga", df_sintetico,
                                   proveedor, consulta_id="Q1")
    assert r.origen == "llm" and r.objetivo_registro_id == id_de(df_sintetico, "Traspasado", "Equipo A")
    assert (r.ligas, r.edad_maxima, r.rol) == (["Bundesliga"], 22, "recuperador")
    assert any("zurdo" in n for n in r.notas)
    assert [(e["agente"], e["estado"], e["consulta_id"]) for e in ejecuciones] == [("interprete", "ok", "Q1")]


def test_llm_con_rol_de_otro_grupo_o_sin_jugador(df_sintetico):
    proveedor = ProveedorSimulado(S.interprete(jugador_referencia="Traspasado", rol="finalizador"))
    r, _ = I.interpretar("un goleador como Traspasado", df_sintetico, proveedor)
    assert r.rol is None and any("rol" in n for n in r.notas)       # «finalizador» no existe para centrocampistas
    # Si el LLM no ve el nombre, las reglas lo buscan en el texto
    r, _ = I.interpretar("sustituto de Traspasado", df_sintetico, ProveedorSimulado(S.interprete()))
    assert r.origen == "llm" and r.objetivo_registro_id == id_de(df_sintetico, "Traspasado", "Equipo B")
    # Jugador que no está en el dataset
    r, _ = I.interpretar("sustituto de Maradona", df_sintetico,
                         ProveedorSimulado(S.interprete(jugador_referencia="Maradona")))
    assert r.objetivo_registro_id is None and any("Maradona" in n for n in r.notas)


def test_si_el_llm_falla_se_usan_las_reglas(df_sintetico):
    for funcion in (S.interprete(ligas=["Eredivisie"]), S.interprete(edad_maxima=90), S.texto_no_json,
                    S.error_api(transitorio=False)):
        proveedor = ProveedorSimulado(funcion)
        r, ejecuciones = I.interpretar("sustituto de Traspasado sub-23 en la Serie A", df_sintetico, proveedor)
        assert r.origen == "reglas" and r.edad_maxima == 23 and r.ligas == ["Serie A"]
        assert r.objetivo_registro_id == id_de(df_sintetico, "Traspasado", "Equipo B")
        assert ejecuciones and all(e["estado"] != "ok" for e in ejecuciones)
        assert any("reglas" in n for n in r.notas)
    r, ejecuciones = I.interpretar("sustituto de Traspasado", df_sintetico)        # sin proveedor
    assert r.origen == "reglas" and ejecuciones == [] and not any("reglas" in n for n in r.notas)


def test_el_esquema_cubre_todos_los_roles():
    claves = {rol for roles in C.ROLES.values() for rol in roles}
    assert set(I.ESQUEMA_RESPUESTA["properties"]["rol"]["anyOf"][0]["enum"]) == claves
    for grupo, roles in I.PALABRAS_ROL.items():
        assert set(roles) <= set(C.ROLES[grupo])
    for rol in claves - {C.ROL_DEFECTO}:
        assert f'"{rol}"' in I.SISTEMA
