import json

import numpy as np
import pandas as pd
import pytest

from ai_scout import aprendizaje, ojeador
from ai_scout import config as C
from ai_scout import valoraciones as V
from ai_scout.modelos import Consulta
from ai_scout.similitud import construir_espacio, cosenos, pesos_de_rol, vector_pesos
from conftest import id_de


def valorar(df, referencia_id, candidato_id, estrellas, directorio=None, version="test", pesos=None):
    ref = df[df["registro_id"] == referencia_id].iloc[0]
    cand = df[df["registro_id"] == candidato_id].iloc[0]
    return V.registrar_fila({
        "dataset_version": version, "objetivo_registro_id": referencia_id, "objetivo_persona_id": ref["persona_id"],
        "objetivo_nombre": ref["nombre"], "candidato_registro_id": candidato_id,
        "candidato_persona_id": cand["persona_id"], "candidato_nombre": cand["nombre"],
        "grupo_posicion": cand["grupo_posicion"], "estrellas": estrellas,
        "pesos_base_json": json.dumps(pesos) if pesos else "",
    }, directorio)


def personalizacion_de(df, referencia_id, version="test"):
    objetivo = df[df["registro_id"] == referencia_id].iloc[0]
    return aprendizaje.construir_personalizacion(V.vigentes(dataset_version=version), df, objetivo)


# ---- Registro de valoraciones ----
def test_relevancia():
    assert [V.relevancia(e) for e in (1, 2, 3, 4, 5)] == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_la_ultima_valoracion_de_cada_pareja_es_la_vigente(df_sintetico):
    ref, a, b = (id_de(df_sintetico, f"MF Jugador {n:02d}") for n in (6, 1, 2))
    assert len(V.cargar()) == 0 and len(V.vigentes()) == 0
    valorar(df_sintetico, ref, a, 5)
    valorar(df_sintetico, ref, b, 2)
    valorar(df_sintetico, ref, a, 1)                    # cambia de opinión
    todas, vig = V.cargar(), V.vigentes()
    assert len(todas) == 3 and todas["valoracion_id"].is_unique
    assert dict(zip(vig["candidato_registro_id"], vig["estrellas"])) == {a: 1, b: 2}
    valorar(df_sintetico, ref, b, 0)                    # retira la valoración
    assert list(V.vigentes()["candidato_registro_id"]) == [a]
    en_disco = pd.read_csv(C.DIR_SALIDAS / V.FICHERO, dtype=str)
    assert list(en_disco["vigente"]) == ["false", "false", "true", "false"]
    assert V.borrar_todas() == 4 and len(V.cargar()) == 0 and V.borrar_todas() == 0


def test_valoraciones_de_otra_version_del_dataset_no_cuentan(df_sintetico):
    ref, a = id_de(df_sintetico, "MF Jugador 06"), id_de(df_sintetico, "MF Jugador 01")
    valorar(df_sintetico, ref, a, 5, version="antigua")
    assert len(V.vigentes(dataset_version="antigua")) == 1 and len(V.vigentes(dataset_version="nueva")) == 0


def test_estrellas_fuera_de_rango(df_sintetico):
    ref, a = id_de(df_sintetico, "MF Jugador 06"), id_de(df_sintetico, "MF Jugador 01")
    with pytest.raises(ValueError):
        valorar(df_sintetico, ref, a, 6)


# ---- Efecto de las valoraciones en el Ojeador de datos ----
def test_descartados_desaparecen_y_los_que_gustan_se_quedan(df_sintetico):
    ref = id_de(df_sintetico, "MF Jugador 06")
    antes = ojeador.recuperar_candidatos(Consulta(ref, top_preseleccion=5), df_sintetico)
    gusta, no_gusta, neutro = (antes["candidatos"][i]["registro_id"] for i in (2, 0, 1))
    valorar(df_sintetico, ref, gusta, 5)
    valorar(df_sintetico, ref, no_gusta, 1)
    valorar(df_sintetico, ref, neutro, 3)
    p = personalizacion_de(df_sintetico, ref)
    assert p.hay_realimentacion and len(p.positivos) == 1 and len(p.negativos) == 1 and len(p.descartados) == 1
    despues = ojeador.recuperar_candidatos(Consulta(ref, top_preseleccion=5), df_sintetico, personalizacion=p)
    ids = [c["registro_id"] for c in despues["candidatos"]]
    assert no_gusta not in ids and gusta in ids and neutro in ids
    assert despues["exclusiones"]["descartados_por_el_usuario"] == 1
    assert next(c for c in despues["candidatos"] if c["registro_id"] == gusta)["valoracion_usuario"] == 5
    assert despues["personalizacion"]["rocchio_aplicado"] and despues["desplazamiento_consulta"]
    # La consulta se ha movido hacia el candidato que gustó
    cos_antes = next(c["coseno"] for c in antes["candidatos"] if c["registro_id"] == gusta)
    cos_despues = next(c["coseno"] for c in despues["candidatos"] if c["registro_id"] == gusta)
    assert cos_despues > cos_antes
    # La similitud base no cambia: sigue siendo el coseno clásico con la referencia
    assert next(c["coseno_base"] for c in despues["candidatos"] if c["registro_id"] == gusta) == pytest.approx(cos_antes)


def test_personalizacion_desactivada_no_cambia_nada(df_sintetico):
    ref = id_de(df_sintetico, "MF Jugador 06")
    antes = ojeador.recuperar_candidatos(Consulta(ref), df_sintetico)
    valorar(df_sintetico, ref, antes["candidatos"][0]["registro_id"], 1)
    valorar(df_sintetico, ref, antes["candidatos"][1]["registro_id"], 5)
    p = personalizacion_de(df_sintetico, ref)
    igual = ojeador.recuperar_candidatos(Consulta(ref, usar_aprendizaje=False), df_sintetico, personalizacion=p)
    assert [c["registro_id"] for c in igual["candidatos"]] == [c["registro_id"] for c in antes["candidatos"]]
    assert igual["personalizacion"] is None and igual["desplazamiento_consulta"] == {}


def test_ocultar_los_ya_valorados(df_sintetico):
    ref = id_de(df_sintetico, "MF Jugador 06")
    antes = ojeador.recuperar_candidatos(Consulta(ref, top_preseleccion=4), df_sintetico)
    for c, estrellas in zip(antes["candidatos"], (5, 4, 3, 5)):
        valorar(df_sintetico, ref, c["registro_id"], estrellas)
    p = personalizacion_de(df_sintetico, ref)
    r = ojeador.recuperar_candidatos(Consulta(ref, top_preseleccion=4, ocultar_valorados=True), df_sintetico,
                                     personalizacion=p)
    assert not {c["registro_id"] for c in r["candidatos"]} & {c["registro_id"] for c in antes["candidatos"]}
    assert r["exclusiones"]["ya_valorados"] == 4


def test_las_valoraciones_de_una_referencia_no_descartan_en_otra(df_sintetico):
    ref, otra = id_de(df_sintetico, "MF Jugador 06"), id_de(df_sintetico, "MF Jugador 07")
    candidato = ojeador.recuperar_candidatos(Consulta(otra), df_sintetico)["candidatos"][0]["registro_id"]
    if candidato == ref:
        candidato = ojeador.recuperar_candidatos(Consulta(otra), df_sintetico)["candidatos"][1]["registro_id"]
    valorar(df_sintetico, ref, candidato, 1)
    p = personalizacion_de(df_sintetico, otra)
    assert not p.descartados and not p.hay_realimentacion and p.n_grupo == 1


# ---- Aprendizaje de la importancia de las variables ----
def test_sin_comparaciones_no_hay_pesos_aprendidos(df_sintetico):
    ref = id_de(df_sintetico, "MF Jugador 06")
    lista = ojeador.recuperar_candidatos(Consulta(ref), df_sintetico)["candidatos"]
    valorar(df_sintetico, ref, lista[0]["registro_id"], 5)
    valorar(df_sintetico, ref, lista[1]["registro_id"], 5)        # mismas estrellas: no hay preferencia
    p = personalizacion_de(df_sintetico, ref)
    assert p.multiplicadores == {} and p.aprendizaje is None
    valorar(df_sintetico, ref, lista[2]["registro_id"], 2)        # ahora sí hay comparaciones
    p = personalizacion_de(df_sintetico, ref)
    assert p.aprendizaje["n_pares"] == 2 and p.aprendizaje["n_valoraciones"] == 3
    m = np.array(list(p.multiplicadores.values()))
    assert m.mean() == pytest.approx(1.0) and (m > 0).all()


def test_gradiente_del_aprendizaje(df_sintetico):
    """La derivada analítica de la pérdida coincide con la numérica."""
    ref = id_de(df_sintetico, "DF Jugador 05")
    lista = ojeador.recuperar_candidatos(Consulta(ref, top_preseleccion=6), df_sintetico)["candidatos"]
    for c, estrellas in zip(lista, (5, 1, 4, 2, 3, 5)):
        valorar(df_sintetico, ref, c["registro_id"], estrellas)
    espacio = construir_espacio(df_sintetico, "DF", C.MINUTOS_MINIMOS_DEFECTO)
    pares = aprendizaje.pares_de_entrenamiento(V.vigentes(), df_sintetico, espacio)
    # Estrellas 5, 1, 4, 2, 3, 5 → 14 parejas con preferencia; la de 5 frente a 1 pesa el máximo
    assert len(pares["omega"]) == 14 and pares["omega"].max() == pytest.approx(1.0)

    def perdida(theta):
        d = C.APRENDIZAJE_NITIDEZ * aprendizaje.diferencias_de_similitud(pares, np.exp(theta))
        return float(np.sum(pares["omega"] * np.logaddexp(0, -d)))

    theta = np.random.default_rng(1).normal(0, 0.3, len(espacio.variables))
    W = pares["W"] * np.exp(theta)
    s_a, g_a = aprendizaje._similitud_y_gradiente(pares["R"], pares["A"], W)
    s_b, g_b = aprendizaje._similitud_y_gradiente(pares["R"], pares["B"], W)
    sigma = 1 / (1 + np.exp(C.APRENDIZAJE_NITIDEZ * (s_a - s_b)))
    analitico = -C.APRENDIZAJE_NITIDEZ * ((pares["omega"] * sigma)[:, None] * ((g_a - g_b) * W)).sum(axis=0)
    numerico = np.array([(perdida(theta + h) - perdida(theta - h)) / 2e-6
                         for h in np.eye(len(theta)) * 1e-6])
    assert analitico == pytest.approx(numerico, rel=1e-4, abs=1e-6)


def test_real_aprende_los_pesos_de_un_usuario_simulado(datos_reales):
    """Un usuario que valora según el rol «creador» hace que el sistema descubra esos pesos."""
    df = datos_reales[0]
    espacio = construir_espacio(df, "MF", C.MINUTOS_MINIMOS_DEFECTO)
    poblacion = espacio.poblacion.sort_values(["persona_id", "minutos"], ascending=[True, False]) \
        .drop_duplicates("persona_id").reset_index(drop=True)
    Z = espacio.z(poblacion)
    w_usuario = vector_pesos(espacio.variables, pesos_de_rol("MF", "creador"))
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(poblacion))[:8]:
        gusto = cosenos(Z[i], Z, w_usuario)
        gusto[i] = -np.inf
        posicion = np.empty(len(gusto), dtype=int)
        posicion[np.argsort(-gusto, kind="stable")] = np.arange(len(gusto))
        sistema = cosenos(Z[i], Z)
        sistema[i] = -np.inf
        for j in np.argsort(-sistema, kind="stable")[:10]:           # el Top 10 que le enseña el sistema
            estrellas = 5 if posicion[j] < 10 else 4 if posicion[j] < 25 else 3 if posicion[j] < 50 else \
                2 if posicion[j] < 100 else 1
            valorar(df, poblacion.loc[i, "registro_id"], poblacion.loc[j, "registro_id"], int(estrellas))
    aprendido = aprendizaje.aprender_multiplicadores(V.vigentes(), df, "MF")
    m = np.array([aprendido["multiplicadores"][v] for v in espacio.variables])
    assert np.corrcoef(m, w_usuario)[0, 1] > 0.8
    assert aprendido["acierto_despues"] > aprendido["acierto_antes"]
    limite_inferior, limite_superior = C.APRENDIZAJE_LIMITES
    assert m.mean() == pytest.approx(1.0) and m.min() > limite_inferior / 2 and m.max() < limite_superior * 2
    # Las variables que definen al creador son las que más peso ganan
    mas_peso = {espacio.variables[i] for i in np.argsort(-m)[:2]}
    assert mas_peso == {"KP_p90", "PPA_p90"}


def test_los_pesos_aprendidos_se_aplican_sobre_los_del_rol(df_sintetico):
    ref = id_de(df_sintetico, "MF Jugador 06")
    lista = ojeador.recuperar_candidatos(Consulta(ref, top_preseleccion=6), df_sintetico)["candidatos"]
    for c, estrellas in zip(lista, (1, 5, 2, 4, 5, 1)):
        valorar(df_sintetico, ref, c["registro_id"], estrellas)
    otra = id_de(df_sintetico, "MF Jugador 09")                    # otra referencia del mismo grupo
    p = personalizacion_de(df_sintetico, otra)
    assert p.multiplicadores and not p.hay_realimentacion
    r = ojeador.recuperar_candidatos(Consulta(otra, rol="organizador"), df_sintetico, personalizacion=p)
    esperado = vector_pesos(r["variables"], pesos_de_rol("MF", "organizador"), p.multiplicadores)
    assert list(r["pesos"].values()) == pytest.approx(list(esperado))
    assert list(r["pesos_rol"].values()) == pytest.approx(list(vector_pesos(r["variables"],
                                                                           pesos_de_rol("MF", "organizador"))))
    # En otro grupo de posición no se aplica nada
    assert not personalizacion_de(df_sintetico, id_de(df_sintetico, "FW Jugador 03")).multiplicadores
