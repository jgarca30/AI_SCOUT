import numpy as np
import pytest

from ai_scout import config as C
from ai_scout import estilos, ojeador
from ai_scout.modelos import Consulta
from conftest import id_de


def test_nombres_de_estilo_por_reglas():
    variables = C.PERFILES["FW"]
    centroides = np.array([
        [1.8, 0.0, 0.0, -0.3, 0.0, -0.4, 0.0],     # destaca en npxG
        [0.1, 0.2, 0.0, 0.1, -0.1, 0.0, 0.2],      # nada destaca y está en la media
        [-0.8, -0.7, -0.9, -0.6, -0.8, -0.9, -0.7],  # todo por debajo
        [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],       # también npxG: hay que distinguirlo
    ])
    nombres, descripciones = estilos.nombrar_estilos("FW", variables, centroides)
    assert nombres[1] == "Perfil equilibrado" and nombres[2] == "Baja participación con balón"
    assert nombres[0].startswith("Finalizador") and nombres[3].startswith("Finalizador")
    assert len(set(nombres)) == 4                                     # sin nombres repetidos
    assert "npxG" in descripciones[0] and "Por debajo de la media" in descripciones[2]


def test_estilos_con_datos_sinteticos(df_sintetico):
    modelos = estilos.calcular_estilos(df_sintetico)
    assert set(modelos) == {"DF", "MF", "FW"}
    for grupo, m in modelos.items():
        validos = df_sintetico[(df_sintetico["grupo_posicion"] == grupo) & df_sintetico["valido"]]
        assert set(m.tabla["registro_id"]) == set(validos["registro_id"])      # todos los válidos tienen estilo
        assert 2 <= m.k <= C.ESTILOS_K[grupo] and sum(m.tamanos) == int(m.tabla["en_entrenamiento"].sum())
        assert m.centroides.shape == (m.k, len(m.variables)) and m.componentes.shape == (2, len(m.variables))
        assert set(m.tabla["estilo_id"]) <= set(range(m.k))
    # El jugador con pocos minutos no entrena el modelo, pero recibe un estilo
    poco = modelos["MF"].tabla[modelos["MF"].tabla["registro_id"] == id_de(df_sintetico, "Poco Minutos")].iloc[0]
    assert not poco["en_entrenamiento"] and poco["estilo"] in modelos["MF"].nombres


def test_estilos_reproducibles(df_sintetico):
    a, b = estilos.calcular_estilos_grupo(df_sintetico, "DF"), estilos.calcular_estilos_grupo(df_sintetico, "DF")
    assert a.nombres == b.nombres and a.tabla.equals(b.tabla)
    assert np.allclose(a.centroides, b.centroides)


def test_grupo_sin_jugadores_suficientes(df_sintetico):
    pocos = df_sintetico[df_sintetico["grupo_posicion"] == "FW"].head(4)
    assert estilos.calcular_estilos_grupo(pocos, "FW") is None
    assert estilos.estilo_de(None, "FW", "R0001") is None


def test_filtro_por_mismo_estilo(df_sintetico):
    modelos = estilos.calcular_estilos(df_sintetico)
    base = id_de(df_sintetico, "MF Jugador 06")
    r = ojeador.recuperar_candidatos(Consulta(base, solo_mismo_estilo=True, top_preseleccion=50), df_sintetico,
                                     modelos)
    assert r["objetivo"]["estilo"] in modelos["MF"].nombres
    assert all(c["estilo"] == r["objetivo"]["estilo"] for c in r["candidatos"])
    assert "otro_estilo_de_juego" in r["exclusiones"]
    assert r["exclusiones"]["otro_estilo_de_juego"] > 0
    # Sin el filtro aparecen más candidatos, y los del mismo estilo siguen estando todos
    libre = ojeador.recuperar_candidatos(Consulta(base, top_preseleccion=50), df_sintetico, modelos)
    del_mismo_estilo = {c["registro_id"] for c in libre["candidatos"] if c["estilo"] == r["objetivo"]["estilo"]}
    assert len(libre["candidatos"]) > len(r["candidatos"])
    assert del_mismo_estilo <= {c["registro_id"] for c in r["candidatos"]}
    # Sin modelo de estilos el filtro no se puede aplicar: se avisa y no se excluye a nadie
    sin_modelo = ojeador.recuperar_candidatos(Consulta(base, solo_mismo_estilo=True), df_sintetico)
    assert any("estilo" in a for a in sin_modelo["avisos"]) and sin_modelo["objetivo"]["estilo"] is None


# ---- Dataset real ----
def test_real_cuatro_estilos_interpretables_por_grupo(datos_reales, estilos_reales):
    df = datos_reales[0]
    for grupo, m in estilos_reales.items():
        assert m.k == 4 and len(set(m.nombres)) == 4
        assert m.silueta > 0.1 and 0.4 < m.varianza_explicada.sum() <= 1.0
        assert min(m.tamanos) >= 30                                   # ningún estilo residual
    tabla = estilos.tabla_estilos(estilos_reales, df)
    assert len(tabla) == int((df["valido"] & ~df["es_portero"]).sum()) and tabla["registro_id"].is_unique

    def estilo(nombre):
        return tabla.loc[tabla["nombre"] == nombre].sort_values("minutos").iloc[-1]["estilo"]

    assert estilo("Erling Haaland") == "Finalizador"
    assert estilo("Lamine Yamal") == "Creador"
    assert estilo("Virgil van Dijk") == "Central de área"
    assert "varianza" in estilos.eje_pca(estilos_reales["MF"], 0)
