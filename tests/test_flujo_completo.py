"""Orquestador con valoraciones, estilos, progreso, historial y exportación ampliada."""
import json

import pandas as pd

import tests.test_simulador as S
from ai_scout import config as C
from ai_scout import estilos, exportador, interprete, orquestador
from ai_scout import valoraciones as V
from ai_scout.llm import ProveedorSimulado
from ai_scout.modelos import Consulta, nuevo_id_consulta
from conftest import id_de


def _consulta(df, nombre="MF Jugador 06", **campos):
    return Consulta(id_de(df, nombre), **campos)


def test_las_valoraciones_cambian_la_siguiente_busqueda(df_sintetico, tmp_path):
    primera = orquestador.ejecutar_consulta(_consulta(df_sintetico), df_sintetico, "test", dir_salidas=tmp_path)
    assert primera["personalizacion"]["valoraciones_grupo"] == 0
    descartado, favorito = primera["top_final"][0], primera["top_final"][1]
    V.registrar(primera, descartado["registro_id"], 1, tmp_path)
    V.registrar(primera, favorito["registro_id"], 5, tmp_path)

    segunda = orquestador.ejecutar_consulta(_consulta(df_sintetico), df_sintetico, "test", dir_salidas=tmp_path)
    ids = [c["registro_id"] for c in segunda["ranking"]]
    assert descartado["registro_id"] not in ids and favorito["registro_id"] in ids
    p = segunda["personalizacion"]
    assert p["valoraciones_referencia"] == 2 and p["descartados"] == 1 and p["rocchio_aplicado"]
    assert p["pesos_aprendidos"] and p["aprendizaje"]["n_pares"] == 1
    assert next(c for c in segunda["ranking"] if c["registro_id"] == favorito["registro_id"])["valoracion_usuario"] == 5

    # Con la personalización desactivada se obtiene el ranking original
    sin = orquestador.ejecutar_consulta(_consulta(df_sintetico, usar_aprendizaje=False), df_sintetico, "test",
                                        dir_salidas=tmp_path)
    assert [c["registro_id"] for c in sin["ranking"]] == [c["registro_id"] for c in primera["ranking"]]
    assert sin["personalizacion"] is None

    # Las valoraciones hechas con otra versión del dataset no se aplican
    otra = orquestador.ejecutar_consulta(_consulta(df_sintetico), df_sintetico, "otra-version", dir_salidas=tmp_path)
    assert otra["personalizacion"]["valoraciones_grupo"] == 0

    consultas = pd.read_csv(tmp_path / "consultas.csv", dtype=str, keep_default_na=False)
    assert list(consultas["valoraciones_referencia"]) == ["0", "2", "0", "0"]
    assert list(consultas["consulta_adaptada"]) == ["false", "true", "false", "false"]
    valoraciones = pd.read_csv(tmp_path / "valoraciones.csv", dtype=str)
    assert set(valoraciones["consulta_id"]) == {primera["consulta"]["consulta_id"]}
    assert json.loads(valoraciones["pesos_base_json"].iloc[0]).keys() == primera["recuperacion"]["pesos"].keys()


def test_valoraciones_ilegibles_no_impiden_la_busqueda(df_sintetico, tmp_path):
    (tmp_path / "valoraciones.csv").write_bytes(b"\xff\xfe esto no es un csv \x00\x00")
    r = orquestador.ejecutar_consulta(_consulta(df_sintetico), df_sintetico, "test", dir_salidas=tmp_path,
                                      exportar=False)
    assert r["estado"] == "completada" and any("valoraciones" in a for a in r["avisos"])
    assert any(i["codigo"] == "valoraciones_ilegibles" for i in r["incidencias"])


def test_estilos_rol_y_progreso(df_sintetico):
    modelos = estilos.calcular_estilos(df_sintetico)
    avisos = []
    r = orquestador.ejecutar_consulta(
        _consulta(df_sintetico, rol="creador"), df_sintetico, "test", ProveedorSimulado(S.analista_valido),
        ProveedorSimulado(S.director_valido), exportar=False, proveedor_critico=ProveedorSimulado(S.critico_aprueba),
        estilos=modelos, progreso=lambda fase, estado, detalle: avisos.append((fase, estado)))
    assert r["recuperacion"]["rol_nombre"] == "Creador / mediapunta"
    assert r["recuperacion"]["objetivo"]["estilo"] and all(c["estilo"] for c in r["ranking"])
    fases = [f for f, estado in avisos if estado == "fin"]
    assert fases == ["aprendizaje", "agente1", "agente2", "ranking", "agente3", "agente4", "guardado"]
    assert ("agente2", "inicio") in avisos

    avisos.clear()
    orquestador.ejecutar_consulta(_consulta(df_sintetico, usar_llm=False, usar_aprendizaje=False), df_sintetico,
                                  "test", exportar=False, progreso=lambda f, e, d: avisos.append((f, e)))
    assert [f for f, e in avisos if e == "omitido"] == ["aprendizaje", "agente2", "plantilla", "agente3", "agente4"]


def test_avisos_explican_la_causa_del_fallo(df_sintetico):
    def cuota(sistema, usuario):
        from ai_scout.llm import ErrorLLM
        raise ErrorLLM("HTTP 429", "http_429", transitorio=True)

    r = orquestador.ejecutar_consulta(_consulta(df_sintetico), df_sintetico, "test", ProveedorSimulado(cuota),
                                      exportar=False)
    assert r["modo_ranking"] == "numerico" and any("cupo gratuito" in a for a in r["avisos"])
    assert orquestador.causa_del_fallo([]) == "fallo desconocido"
    assert orquestador.causa_del_fallo([{"codigo_error": "otro"}]) == "otro"


def test_consulta_en_lenguaje_natural_de_principio_a_fin(df_sintetico, tmp_path):
    texto = "sustituto de MF Jugador 06 sub-30 en la Serie A, creador"
    qid = nuevo_id_consulta()
    proveedor = ProveedorSimulado(S.interprete(jugador_referencia="MF Jugador 06", ligas=["Serie A"],
                                               edad_maxima=30, rol="creador"))
    interpretacion, ejecuciones = interprete.interpretar(texto, df_sintetico, proveedor, consulta_id=qid)
    consulta = Consulta(interpretacion.objetivo_registro_id, consulta_id=qid, **interpretacion.parametros())
    r = orquestador.ejecutar_consulta(consulta, df_sintetico, "test", dir_salidas=tmp_path,
                                      ejecuciones_previas=ejecuciones)
    assert r["consulta"]["texto_original"] == texto and r["consulta"]["rol"] == "creador"
    assert all(c["liga"] == "Serie A" and c["edad_registrada"] <= 30 for c in r["ranking"])
    assert r["ejecuciones"][0]["agente"] == "interprete"
    trazas = pd.read_csv(tmp_path / "ejecuciones_agentes.csv", dtype=str)
    assert list(trazas["agente"]) == ["interprete"] and list(trazas["consulta_id"]) == [qid]
    assert pd.read_csv(tmp_path / "consultas.csv", dtype=str)["texto_original"].iloc[0] == texto


# ---- Historial ----
def test_historial_guarda_y_recupera_la_consulta_completa(df_sintetico, tmp_path):
    assert exportador.listar_historial(tmp_path).empty
    r1 = orquestador.ejecutar_consulta(
        _consulta(df_sintetico), df_sintetico, "test", ProveedorSimulado(S.analista_valido),
        ProveedorSimulado(S.director_valido), dir_salidas=tmp_path,
        proveedor_critico=ProveedorSimulado(S.critico_aprueba), estilos=estilos.calcular_estilos(df_sintetico))
    r2 = orquestador.ejecutar_consulta(_consulta(df_sintetico, "FW Jugador 05", edad_maxima=30), df_sintetico,
                                       "test", dir_salidas=tmp_path)
    orquestador.ejecutar_consulta(_consulta(df_sintetico, "Poco Minutos"), df_sintetico, "test",
                                  dir_salidas=tmp_path)                       # rechazada: no se guarda

    lista = exportador.listar_historial(tmp_path)
    assert set(lista["consulta_id"]) == {r1["consulta"]["consulta_id"], r2["consulta"]["consulta_id"]}
    assert list(lista["fecha_utc"]) == sorted(lista["fecha_utc"], reverse=True)
    fila = lista[lista["consulta_id"] == r2["consulta"]["consulta_id"]].iloc[0]
    assert fila["referencia"] == "FW Jugador 05" and "≤ 30 años" in fila["filtros"] and fila["modo_ranking"] == "numerico"

    recuperada = exportador.cargar_historial(r1["consulta"]["consulta_id"], tmp_path)
    assert recuperada["modo_ranking"] == "hibrido" and recuperada["revision"]["estado"] == "aprobadas"
    assert [c["registro_id"] for c in recuperada["ranking"]] == [c["registro_id"] for c in r1["ranking"]]
    assert recuperada["top_final"][0]["ficha"] == r1["top_final"][0]["ficha"]
    assert recuperada["recuperacion"]["pesos"] == r1["recuperacion"]["pesos"]
    assert exportador.cargar_historial("no-existe", tmp_path) is None

    # Un fichero dañado no impide ver el resto
    (tmp_path / C.SUBDIR_HISTORIAL / "roto.json").write_text("{esto no es json", encoding="utf-8")
    assert len(exportador.listar_historial(tmp_path)) == 2
    assert exportador.cargar_historial("roto", tmp_path) is None


# ---- Exportación ----
def test_las_tablas_antiguas_se_actualizan_sin_perder_filas(df_sintetico, tmp_path):
    antiguas = ["consulta_id", "dataset_version", "objetivo_registro_id", "fecha_utc", "modo_ranking"]
    pd.DataFrame([{"consulta_id": "Q-antigua", "dataset_version": "v0", "objetivo_registro_id": "R0001",
                   "fecha_utc": "2026-01-01T00:00:00+00:00", "modo_ranking": "numerico"}],
                 columns=antiguas).to_csv(tmp_path / "consultas.csv", index=False)
    orquestador.ejecutar_consulta(_consulta(df_sintetico, rol="organizador"), df_sintetico, "test",
                                  dir_salidas=tmp_path)
    consultas = pd.read_csv(tmp_path / "consultas.csv", dtype=str, keep_default_na=False)
    assert list(consultas.columns) == exportador.COLUMNAS["consultas"]
    assert list(consultas["consulta_id"])[0] == "Q-antigua" and len(consultas) == 2
    assert list(consultas["rol"]) == ["", "organizador"] and consultas["rol_nombre"].iloc[1] == "Mediocentro organizador"
    assert abs(sum(json.loads(consultas["pesos_json"].iloc[1]).values()) - len(C.PERFILES["MF"])) < 0.01


def test_columnas_nuevas_de_resultados_y_evidencias(df_sintetico, tmp_path):
    modelos = estilos.calcular_estilos(df_sintetico)
    orquestador.ejecutar_consulta(
        _consulta(df_sintetico), df_sintetico, "test", ProveedorSimulado(S.analista_valido),
        ProveedorSimulado(S.Secuencia(S.director_valido, S.director_corregido)), dir_salidas=tmp_path,
        proveedor_critico=ProveedorSimulado(S.critico_rechaza_primera), estilos=modelos)
    resultados = pd.read_csv(tmp_path / "resultados.csv", dtype=str, keep_default_na=False)
    top = resultados[resultados["es_top_final"] == "true"]
    assert list(top["revision_ficha"]) == ["corregida", "aprobada", "aprobada"]
    assert top["observaciones_revisor"].iloc[0] and resultados["estilo"].ne("").all()
    assert resultados["score_base"].astype(float).between(0, 100).all()
    evidencias = pd.read_csv(tmp_path / "evidencias.csv", dtype=str)
    assert evidencias["peso"].astype(float).eq(1.0).all()                      # rol equilibrado
    registros = pd.read_csv(tmp_path / "registros.csv", dtype=str, keep_default_na=False)
    assert registros["estilo"].ne("").all()
    assert pd.read_csv(tmp_path / "consultas.csv", dtype=str)["revision_fichas"].iloc[0] == "corregidas"


def test_exportar_estilos_para_power_bi(df_sintetico, tmp_path):
    modelos = estilos.calcular_estilos(df_sintetico)
    ruta = exportador.exportar_estilos(estilos.tabla_estilos(modelos, df_sintetico), "test", tmp_path)
    tabla = pd.read_csv(ruta, dtype=str)
    assert list(tabla.columns) == exportador.COLUMNAS["estilos_jugadores"] and tabla["registro_id"].is_unique
    assert len(tabla) == int((df_sintetico["valido"] & ~df_sintetico["es_portero"]).sum())
    assert set(tabla["en_entrenamiento"]) == {"true", "false"} and set(tabla["dataset_version"]) == {"test"}
