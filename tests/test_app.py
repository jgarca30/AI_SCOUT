"""Pruebas de la aplicación web con el simulador de Streamlit (AppTest): sin navegador ni llamadas externas."""
import pandas as pd
import pytest

import tests.test_simulador as S
from ai_scout import config as C
from ai_scout import evaluacion, exportador, llm
from ai_scout import valoraciones as V
from ai_scout.llm import ProveedorSimulado

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from vistas import comun  # noqa: E402

ESPERA = 180  # segundos: la primera ejecución calcula los estilos de juego


def pagina_buscar():
    from vistas import buscar
    buscar.pagina()


def pagina_estilos():
    from vistas import estilos_de_juego
    estilos_de_juego.pagina()


def pagina_aprendizaje():
    from vistas import aprendizaje
    aprendizaje.pagina()


def pagina_historial():
    from vistas import historial
    historial.pagina()


def pagina_evaluacion():
    from vistas import evaluacion as pagina
    pagina.pagina()


@pytest.fixture()
def datos_app():
    if not C.RUTA_CSV.exists():
        pytest.skip("No está el CSV real en dataset/")
    return comun.cargar_datos()[0]


@pytest.fixture()
def sin_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(comun, "cargar_proveedores", lambda: {rol: None for rol in C.ROLES_LLM})
    monkeypatch.setattr(llm, "CacheLLM", lambda: None)


@pytest.fixture()
def con_llm_simulado(monkeypatch):
    proveedores = {
        "interprete": ProveedorSimulado(S.interprete(jugador_referencia="Pedri", ligas=["Bundesliga", "Serie A"],
                                                     edad_maxima=27, rol="creador", no_aplicable=["que sea zurdo"])),
        "analista": ProveedorSimulado(S.analista_valido), "director": ProveedorSimulado(S.director_valido),
        "critico": ProveedorSimulado(S.critico_aprueba), "plantilla": ProveedorSimulado(S.plantilla_valida),
    }
    monkeypatch.setattr(comun, "cargar_proveedores", lambda: proveedores)
    monkeypatch.setattr(llm, "CacheLLM", lambda: None)
    return proveedores


def registro(df, nombre):
    return df[(df["nombre"] == nombre) & df["valido"]].sort_values("minutos").iloc[-1]["registro_id"]


def boton(at, etiqueta):
    return next(b for b in at.button if b.label == etiqueta)


def buscar(at, df, nombre, **controles):
    at.selectbox(key="f_objetivo").set_value(registro(df, nombre)).run()
    for clave, valor in controles.items():
        at.session_state[clave] = valor
    at.run()
    boton(at, "Buscar sustitutos").click().run()
    assert not at.exception, at.exception
    return at.session_state["resultado"]


# ---- Página principal ----
def test_portada_sin_modelo_de_lenguaje(datos_app, sin_llm):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    assert not at.exception
    assert at.header[0].value == "Encuentra sustitutos con un perfil estadístico parecido"
    assert at.toggle(key="f_ia").value is False and at.toggle(key="f_ia").disabled
    assert boton(at, "Buscar sustitutos").disabled                      # falta elegir jugador
    assert any("modo numérico" in c.value for c in at.sidebar.caption)


def test_cromo_de_la_portada_lanza_la_busqueda(datos_app, sin_llm):
    from vistas.buscar import CROMOS
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    assert len([b for b in at.button if b.label == "Ver sustitutos"]) == len(CROMOS)
    at.button(key=f"cromo_boton_{CROMOS.index('Erling Haaland')}").click().run()
    assert not at.exception, at.exception
    assert at.header[0].value == "Sustitutos para Erling Haaland"
    assert at.selectbox(key="f_destino").value == "Manchester City"      # su club, como destino


def test_busqueda_numerica_con_rol_del_grupo(datos_app, sin_llm):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    at.selectbox(key="f_objetivo").set_value(registro(datos_app, "Erling Haaland")).run()
    assert at.selectbox(key="f_rol").options == [r["nombre"] for r in C.ROLES["FW"].values()]
    resultado = buscar(at, datos_app, "Erling Haaland", f_rol="finalizador", f_limitar_edad=True, f_edad=24,
                       f_ligas=["Serie A", "Bundesliga"])
    consulta = resultado["consulta"]
    assert (consulta["rol"], consulta["edad_maxima"], consulta["usar_llm"]) == ("finalizador", 24, False)
    assert resultado["modo_ranking"] == "numerico" and len(resultado["top_final"]) == 3
    assert all(c["liga"] in ("Serie A", "Bundesliga") and c["edad_registrada"] <= 24 for c in resultado["ranking"])
    assert at.header[0].value == "Sustitutos para Erling Haaland"
    assert [t.label for t in at.tabs][:2] == ["Fichas de ojeador", "Comparador"]
    assert len(at.feedback) == 3 and len(at.download_button) == 2
    # Al cambiar a un jugador de otro grupo, el rol vuelve a «equilibrado»
    at.selectbox(key="f_objetivo").set_value(registro(datos_app, "Pedri")).run()
    assert at.selectbox(key="f_rol").value == C.ROL_DEFECTO and not at.exception


def test_pesos_a_mano_y_opciones_avanzadas(datos_app, sin_llm):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    at.selectbox(key="f_objetivo").set_value(registro(datos_app, "Pedri")).run()
    at.session_state["f_rol"] = "creador"
    at.checkbox(key="f_pesos_mano").check().run()
    assert at.slider(key="f_peso_MF_KP_p90").value == 2.5               # parte de los pesos del rol
    at.slider(key="f_peso_MF_KP_p90").set_value(3.0).run()
    resultado = buscar(at, datos_app, "Pedri", f_diversidad=0.3, f_estilo=True)
    consulta = resultado["consulta"]
    assert consulta["pesos_personalizados"]["KP_p90"] == 3.0 and consulta["diversidad"] == 0.3
    assert resultado["recuperacion"]["rol_nombre"] == "Pesos ajustados a mano"
    assert all(c["estilo"] == resultado["recuperacion"]["objetivo"]["estilo"] for c in resultado["ranking"])


def test_busqueda_rechazada_muestra_el_motivo(datos_app, sin_llm):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    at.selectbox(key="f_objetivo").set_value(registro(datos_app, "Pedri")).run()
    at.session_state["f_ligas"] = []
    at.run()
    boton(at, "Buscar sustitutos").click().run()
    assert not at.exception and any("Selecciona al menos una liga" in e.value for e in at.error)


def test_peticion_en_lenguaje_natural_con_ia(datos_app, con_llm_simulado):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    at.text_area(key="texto_nl").set_value("un creador zurdo como Pedri, sub-27, en Alemania o Italia").run()
    boton(at, "Interpretar y buscar").click().run()
    assert not at.exception, at.exception
    resultado = at.session_state["resultado"]
    consulta = resultado["consulta"]
    assert consulta["texto_original"].startswith("un creador zurdo") and consulta["rol"] == "creador"
    assert consulta["ligas_permitidas"] == ["Serie A", "Bundesliga"] and consulta["edad_maxima"] == 27
    assert resultado["modo_ranking"] == "hibrido" and resultado["revision"]["estado"] == "aprobadas"
    assert [e["agente"] for e in resultado["ejecuciones"]] == ["interprete", "analista", "plantilla", "director", "critico"]
    assert resultado["plantilla"]["equipo"] == "Barcelona" and resultado["plantilla"]["origen"] == "llm"
    assert at.selectbox(key="f_destino").value == "Barcelona" and "Encaje en la plantilla" in [t.label for t in at.tabs]
    # El formulario queda relleno con lo interpretado y se explica lo que se ha entendido
    assert at.selectbox(key="f_rol").value == "creador" and at.checkbox(key="f_limitar_edad").value
    assert not any("He entendido" in i.value for i in at.info)            # sin mensajes en primera persona
    peticion = [c.value for c in at.caption if c.value.startswith("Petición:")]
    assert peticion and "un creador zurdo" in peticion[0] and "zurdo" in peticion[0].split("»", 1)[1]
    assert any("Creador / mediapunta" in m.value for m in at.markdown)
    assert all(p.llamadas == 1 for p in con_llm_simulado.values())


def test_peticion_sin_jugador_no_lanza_la_busqueda(datos_app, sin_llm):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    at.text_area(key="texto_nl").set_value("un mediocentro defensivo joven").run()
    boton(at, "Interpretar y buscar").click().run()
    assert not at.exception and "resultado" not in at.session_state
    assert any("jugador de referencia" in e.value for e in at.error)


def test_valorar_con_estrellas_y_recalcular(datos_app, sin_llm):
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    primero = buscar(at, datos_app, "Pedri")
    top = [c["registro_id"] for c in primero["top_final"]]
    at.feedback[0].set_value(4).run()                                   # 5 estrellas al primero
    at.feedback[1].set_value(0).run()                                   # 1 estrella al segundo
    assert not at.exception
    vigentes = V.vigentes()
    assert dict(zip(vigentes["candidato_registro_id"], vigentes["estrellas"])) == {top[0]: 5, top[1]: 1}
    assert at.session_state["valoraciones_pendientes"] and at.feedback[0].value == 4
    assert any("No volverá a proponerse" in c.value for c in at.caption)

    boton(at, "Recalcular").click().run()
    assert not at.exception
    segundo = at.session_state["resultado"]
    assert segundo["consulta"]["consulta_id"] != primero["consulta"]["consulta_id"]
    ids = [c["registro_id"] for c in segundo["ranking"]]
    assert top[1] not in ids and top[0] in ids
    assert segundo["personalizacion"]["rocchio_aplicado"] and not at.session_state["valoraciones_pendientes"]
    assert any("Búsqueda personalizada" in s.value for s in at.success)
    assert at.feedback[0].value == 4                                    # el favorito conserva sus estrellas


# ---- Otras páginas ----
def test_pagina_de_estilos(datos_app, sin_llm):
    at = AppTest.from_function(pagina_estilos, default_timeout=ESPERA).run()
    assert not at.exception and at.header[0].value == "Estilos de juego"
    assert at.metric[0].value == "4"
    at.segmented_control(key="estilos_grupo").set_value("FW").run()
    at.selectbox(key="estilos_resaltar_FW").set_value(registro(datos_app, "Lamine Yamal")).run()
    assert not at.exception and any("estilo: **Creador**" in m.value for m in at.markdown)


def test_pagina_de_aprendizaje(datos_app, sin_llm):
    at = AppTest.from_function(pagina_aprendizaje, default_timeout=ESPERA).run()
    assert not at.exception and any("Todavía no has valorado" in i.value for i in at.info)

    busqueda = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    resultado = buscar(busqueda, datos_app, "Pedri")
    for candidato, estrellas in zip(resultado["ranking"], (5, 1, 4, 2, 3)):
        V.registrar(resultado, candidato["registro_id"], estrellas)
    at = AppTest.from_function(pagina_aprendizaje, default_timeout=ESPERA).run()
    assert not at.exception
    valores = {m.label: m.value for m in at.metric}
    assert valores["Valoraciones vigentes"] == "5" and valores["Candidatos descartados"] == "2"
    assert valores["Comparaciones aprendidas"] == "10"
    # Borrar todo exige confirmar
    assert boton(at, "Borrar valoraciones").disabled
    at.checkbox(key="confirmar_olvido").check().run()
    boton(at, "Borrar valoraciones").click().run()
    assert not at.exception and len(V.cargar()) == 0


def test_pagina_de_historial(datos_app, sin_llm):
    at = AppTest.from_function(pagina_historial, default_timeout=ESPERA).run()
    assert not at.exception and any("Todavía no hay consultas" in i.value for i in at.info)
    busqueda = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    buscar(busqueda, datos_app, "Pedri")
    buscar(busqueda, datos_app, "Erling Haaland")
    at = AppTest.from_function(pagina_historial, default_timeout=ESPERA).run()
    assert not at.exception
    tabla = at.dataframe[0].value
    assert list(tabla["Referencia"]) == ["Erling Haaland", "Pedri"]     # la más reciente, primero
    assert boton(at, "Abrir resultado").disabled                        # hasta que se seleccione una fila
    assert len(exportador.listar_historial()) == 2


def test_pagina_de_evaluacion(datos_app, sin_llm):
    at = AppTest.from_function(pagina_evaluacion, default_timeout=ESPERA).run()
    assert not at.exception and any("Todavía no hay resultados" in i.value for i in at.info)
    tablas = {
        "autorrecuperacion": pd.DataFrame({"grupo": ["Todos"] * 2, "medida": ["Similitud coseno (la del sistema)",
                                                                            "Azar (valor esperado)"],
                                           "consultas": [58, 58], "candidatos": [1800, 1800], "mrr": [0.04, 0.01],
                                           "acierto_1": [0.0, 0.0], "acierto_5": [0.09, 0.01],
                                           "acierto_10": [0.14, 0.02], "rango_mediano": [124.0, 330.0]}),
        "estilos_siluetas": pd.DataFrame({"grupo": ["DF"] * 2, "k": [3, 4], "silueta": [0.22, 0.21],
                                          "inercia": [3245.0, 2934.0], "elegido": [False, True]}),
        "aprendizaje": pd.DataFrame({"escenario": ["Valora los 10 candidatos"] * 2, "referencias_valoradas": [0, 5],
                                     "ndcg_10": [0.94, 0.97], "precision_10": [0.78, 0.88]}),
        "rocchio": evaluacion.simular_rocchio(datos_app[datos_app["grupo_posicion"] == "FW"], referencias=5),
    }
    evaluacion.escribir_resultados(tablas, dataset_version="test")
    at = AppTest.from_function(pagina_evaluacion, default_timeout=ESPERA).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Calidad de la similitud", "Estilos de juego", "Diversidad y sensibilidad",
                                          "Aprendizaje", "Uso real", "Informe"]
    assert any("Última evaluación" in c.value for c in at.caption)


# ---- Utilidades y flujo historial → búsqueda ----
def test_utilidades_de_la_interfaz():
    assert comun.plural(1, "valoración", "valoraciones") == "1 valoración"
    assert comun.plural(3, "valoración", "valoraciones") == "3 valoraciones" and comun.plural(2, "negativa") == "2 negativas"
    estado = {"edited_rows": {"0": {"Tu valoración": 4}, 2: {"Tu valoración": None}, 1: {"Otra columna": 1}}}
    assert comun.estrellas_editadas(estado, "Tu valoración") == [(0, 4), (2, 0)]
    assert comun.estrellas_editadas(None, "Tu valoración") == []
    consulta = comun.consulta_desde({"objetivo_registro_id": "R0001", "rol": "creador", "consulta_id": "vieja",
                                     "fecha_utc": "2020", "campo_desconocido": 1}, diversidad=0.2)
    assert (consulta.rol, consulta.diversidad) == ("creador", 0.2) and consulta.consulta_id != "vieja"


def pagina_abrir_ultima():
    """Simula pulsar «Abrir resultado» (o «Repetir la búsqueda») sobre la consulta más reciente."""
    import streamlit as st

    from ai_scout import exportador as exp
    from vistas import comun as c, historial

    version = c.contexto()[1]["dataset_version"]
    historial._abrir(exp.listar_historial().iloc[0]["consulta_id"], st.session_state.get("repetir", False), version)


def test_abrir_y_repetir_desde_el_historial(datos_app, sin_llm):
    busqueda = AppTest.from_function(pagina_buscar, default_timeout=ESPERA).run()
    original = buscar(busqueda, datos_app, "Erling Haaland", f_rol="finalizador", f_limitar_edad=True, f_edad=26,
                      f_diversidad=0.2)

    # Abrir: se muestra la consulta guardada sin recalcular y el formulario queda como estaba
    abrir = AppTest.from_function(pagina_abrir_ultima, default_timeout=ESPERA).run()
    assert not abrir.exception and abrir.session_state["ir_a_buscar"]
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA)
    at.session_state["resultado"] = abrir.session_state["resultado"]
    at.session_state["formulario_pendiente"] = abrir.session_state["formulario_pendiente"]
    at.run()
    assert not at.exception and at.header[0].value == "Sustitutos para Erling Haaland"
    assert at.session_state["resultado"]["consulta"]["consulta_id"] == original["consulta"]["consulta_id"]
    assert at.selectbox(key="f_rol").value == "finalizador" and at.slider(key="f_edad").value == 26
    assert at.slider(key="f_diversidad").value == 0.2 and len(exportador.listar_historial()) == 1

    # Repetir: se ejecuta de nuevo con los mismos parámetros
    repetir = AppTest.from_function(pagina_abrir_ultima, default_timeout=ESPERA)
    repetir.session_state["repetir"] = True
    repetir.run()
    at = AppTest.from_function(pagina_buscar, default_timeout=ESPERA)
    at.session_state["formulario_pendiente"] = repetir.session_state["formulario_pendiente"]
    at.session_state["ejecutar_pendiente"] = repetir.session_state["ejecutar_pendiente"]
    at.run()
    assert not at.exception
    nuevo = at.session_state["resultado"]
    assert nuevo["consulta"]["consulta_id"] != original["consulta"]["consulta_id"]
    assert [c["registro_id"] for c in nuevo["ranking"]] == [c["registro_id"] for c in original["ranking"]]
    assert len(exportador.listar_historial()) == 2
