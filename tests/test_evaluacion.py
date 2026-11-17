"""Evaluación del recomendador (métricas, simulaciones e informe)."""
import numpy as np
import pandas as pd
import pytest

import tests.test_simulador as S
from ai_scout import config as C
from ai_scout import evaluacion as E
from ai_scout import orquestador
from ai_scout import valoraciones as V
from ai_scout.llm import ProveedorSimulado
from ai_scout.modelos import Consulta
from conftest import id_de


def test_indice_de_gini():
    assert E.gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    assert E.gini(np.array([0, 0, 0, 20])) == pytest.approx(0.75)
    assert E.gini(np.array([])) == 0.0 and E.gini(np.zeros(4)) == 0.0


def test_matriz_de_cosenos():
    Z = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    S_ = E.matriz_cosenos(Z)
    assert S_[0, 1] == pytest.approx(1.0) and S_[0, 2] == pytest.approx(0.0) and np.isneginf(S_[0, 0])


def test_estrellas_del_usuario_simulado():
    assert [E.estrellas_simuladas(p) for p in (0, 9, 10, 24, 25, 49, 50, 99, 100)] == [5, 5, 4, 4, 3, 3, 2, 2, 1]


def test_autorrecuperacion_con_el_jugador_traspasado(df_sintetico):
    tabla = E.evaluar_autorrecuperacion(df_sintetico)
    mf = tabla[(tabla["grupo"] == "MF") & tabla["medida"].str.startswith("Similitud coseno")].iloc[0]
    assert mf["consultas"] == 2                                   # los dos registros de «Traspasado»
    assert 0 < mf["mrr"] <= 1 and mf["rango_mediano"] >= 1
    assert set(tabla["medida"]) == {"Similitud coseno (la del sistema)", "Distancia euclídea", "Azar (valor esperado)"}
    assert "Todos" in set(tabla["grupo"])


def test_tabla_markdown():
    tabla = pd.DataFrame({"grupo": ["DF"], "indice_medio": [0.5], "elegido": [True], "vacio": [np.nan]})
    texto = E.tabla_markdown(tabla)
    assert texto.splitlines()[0] == "| grupo | indice medio | elegido | vacio |"
    assert texto.splitlines()[2] == "| DF | 0,500 | sí | — |"


def test_resumen_del_uso_real(df_sintetico, tmp_path):
    assert E.resumen_del_uso(tmp_path) == {}
    ref = id_de(df_sintetico, "MF Jugador 06")
    r = orquestador.ejecutar_consulta(Consulta(ref), df_sintetico, "test", ProveedorSimulado(S.analista_valido),
                                      ProveedorSimulado(S.director_valido), dir_salidas=tmp_path,
                                      proveedor_critico=ProveedorSimulado(S.critico_aprueba))
    orquestador.ejecutar_consulta(Consulta(ref), df_sintetico, "test",
                                  ProveedorSimulado(S.Secuencia(S.texto_no_json, S.analista_valido)),
                                  dir_salidas=tmp_path)
    orquestador.ejecutar_consulta(Consulta(ref, usar_llm=False), df_sintetico, "test", dir_salidas=tmp_path)
    V.registrar(r, r["top_final"][0]["registro_id"], 5, tmp_path)
    V.registrar(r, r["ranking"][-1]["registro_id"], 2, tmp_path)

    tablas = E.resumen_del_uso(tmp_path)
    agentes = tablas["uso_agentes"].set_index("agente")
    assert agentes.loc["analista", "tareas"] == 2 and agentes.loc["analista", "llamadas_al_modelo"] == 3
    assert agentes.loc["analista", "validas_al_primer_intento"] == pytest.approx(0.5)
    assert agentes.loc["analista", "tareas_resueltas"] == pytest.approx(1.0)
    assert list(tablas["uso_errores"]["estado"]) == ["json_invalido"]
    assert tablas["uso_consultas"]["consultas"].sum() == 3
    assert tablas["uso_numerico_vs_hibrido"]["consultas_hibridas"].iloc[0] == 2
    valoraciones = tablas["uso_valoraciones"].set_index("tramo")
    assert valoraciones.loc[f"Top {C.TOP_FINAL}", "estrellas_medias"] == 5
    assert valoraciones.loc["Resto de la preselección", "negativas"] == 1.0


def test_evaluar_llm_con_proveedores_simulados(df_sintetico):
    df = df_sintetico.copy()
    df["minutos"] = df["minutos"].where(df["minutos"] < 700, 2000)     # hacen falta referencias con muchos minutos
    proveedores = {"analista": ProveedorSimulado(S.analista_valido), "director": ProveedorSimulado(S.director_valido),
                   "critico": ProveedorSimulado(S.critico_aprueba)}
    tablas = E.evaluar_llm(df, "test", proveedores, n_consultas=2, repeticiones=2)
    agentes = tablas["llm_agentes"].set_index("agente")
    assert set(agentes.index) == {"analista", "director", "critico"} and (agentes["llamadas"] == 4).all()
    assert (agentes["validas_al_primer_intento"] == 1.0).all()
    estabilidad = tablas["llm_estabilidad"]
    assert len(estabilidad) == 2 and (estabilidad["mismo_top3"] == 1.0).all()         # el simulador es determinista
    assert (estabilidad["desviacion_valoracion_ia"] == 0).all()


# ---- Dataset real ----
def test_real_la_similitud_supera_al_azar(datos_reales):
    df = datos_reales[0]
    auto = E.evaluar_autorrecuperacion(df)
    todos = auto[auto["grupo"] == "Todos"].set_index("medida")
    assert todos.loc["Similitud coseno (la del sistema)", "consultas"] >= 40
    assert todos.loc["Similitud coseno (la del sistema)", "mrr"] > 2 * todos.loc["Azar (valor esperado)", "mrr"]
    assert todos.loc["Similitud coseno (la del sistema)", "acierto_10"] > 4 * todos.loc["Azar (valor esperado)", "acierto_10"]
    posicion = E.evaluar_posicion_detallada(df)
    # En defensas casi todos comparten etiqueta («DF»), así que apenas hay margen sobre el azar
    mejora = posicion.set_index("grupo")["mejora_sobre_azar"]
    assert len(posicion) == 3 and (mejora > 1.0).all() and mejora["MF"] > 1.2 and mejora["FW"] > 1.2


def test_real_diversidad_y_sensibilidad(datos_reales):
    df = datos_reales[0]
    diversidad = E.evaluar_diversidad(df[df["grupo_posicion"] == "FW"], diversidades=(0.0, 0.6))
    normal, diverso = diversidad.iloc[0], diversidad.iloc[1]
    assert diverso["diversidad_intralista"] > normal["diversidad_intralista"]
    assert diverso["indice_similitud_medio"] < normal["indice_similitud_medio"]
    assert normal["cobertura_catalogo"] > 0.9 and 0 <= normal["gini"] <= 1
    sensibilidad = E.evaluar_sensibilidad(df)
    roles = sensibilidad[sensibilidad["prueba"].str.startswith("Rol")]
    minutos = sensibilidad[sensibilidad["prueba"].str.startswith("Escalado")]
    assert len(roles) == 10 and roles["solapamiento_top10"].between(0.4, 0.95).all()      # los roles cambian la lista
    assert len(minutos) == 3 and (minutos["solapamiento_top10"] > 0.9).all()               # el escalado es estable


def test_real_el_aprendizaje_mejora_con_usuarios_simulados(datos_reales):
    df = datos_reales[0]
    detalle = E.simular_aprendizaje(df[df["grupo_posicion"] == "FW"], referencias_valoradas=5, referencias_prueba=15,
                                    semillas=(0,), puntos=(0, 5),
                                    escenarios={"Valora los 10 candidatos": {"valorados": 10, "ruido": 0.0}})
    curva = detalle.groupby("referencias_valoradas")["precision_10"].mean()
    assert len(detalle) == 6 and curva.loc[5] > curva.loc[0] + 0.03
    rocchio = E.simular_rocchio(df[df["grupo_posicion"] == "FW"], referencias=20)
    con_objetivo = rocchio[rocchio["tipo_de_usuario"].str.startswith("Como")]
    assert (con_objetivo["diferencia"] > 0).all()                 # ayuda a quien busca «como X pero con más de algo»


def test_informe_completo_en_disco(df_sintetico, tmp_path):
    avisos = []
    tablas = E.evaluar_todo(df_sintetico, tmp_path, rapido=True, avisar=avisos.append)
    assert len(avisos) == 7 and "autorrecuperacion" in tablas and "sensibilidad" in tablas
    carpeta = E.escribir_resultados(tablas, tmp_path, "test")
    informe = (carpeta / "informe.md").read_text(encoding="utf-8")
    assert informe.startswith("# Evaluación de AI Scout") and "## 1. Autorrecuperación" in informe
    leidas = E.leer_resultados(tmp_path)
    assert set(leidas) == {n for n, t in tablas.items() if len(t)}
    assert E.leer_resultados(tmp_path / "no-existe") == {}
