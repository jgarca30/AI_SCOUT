"""Orquestador de AI Scout (CU-02 Buscar sustituto).

Valida la consulta, llama a los agentes en orden, valida sus respuestas,
calcula el ranking final, gestiona los fallos y exporta el resultado.
"""
from __future__ import annotations

import time

import pandas as pd

from . import analista, director, exportador
from . import config as C
from .ejecutor_llm import ejecutar_agente, tokens_usados
from .llm import CacheLLM, ProveedorLLM
from .modelos import Consulta
from .ojeador import recuperar_candidatos, validar_consulta
from .ranking import calcular_ranking


def ejecutar_consulta(consulta: Consulta, df: pd.DataFrame, dataset_version: str,
                      proveedor_analista: ProveedorLLM | None = None,
                      proveedor_director: ProveedorLLM | None = None,
                      cache: CacheLLM | None = None, exportar: bool = True,
                      dir_salidas=C.DIR_SALIDAS) -> dict:
    inicio = time.perf_counter()
    resultado = {
        "consulta": consulta.como_dict(), "dataset_version": dataset_version,
        "perfil_version": C.PERFIL_VERSION, "estado": None, "errores": [], "avisos": [],
        "modo_ranking": "numerico", "recuperacion": None, "ranking": [], "top_final": [],
        "resumen_consulta": "", "limitaciones_generales": list(C.LIMITACIONES_GENERALES),
        "ejecuciones": [], "incidencias": [],
    }

    def incidencia(codigo, descripcion, severidad="aviso", registro_id=None, fase="consulta"):
        resultado["incidencias"].append({"fase": fase, "severidad": severidad, "codigo": codigo,
                                         "registro_id": registro_id, "descripcion": descripcion})

    def terminar(estado: str) -> dict:
        resultado["estado"] = estado
        resultado["duracion_total_ms"] = int((time.perf_counter() - inicio) * 1000)
        if exportar:
            resultado["rutas_exportacion"] = exportador.exportar_consulta(resultado, dir_salidas)
        return resultado

    # 1. Validación de la consulta 
    errores = validar_consulta(consulta, df)
    if errores:
        resultado["errores"] = errores
        for e in errores:
            incidencia("consulta_rechazada", e, "error", consulta.objetivo_registro_id)
        return terminar("rechazada")

    # 2. Agente 1 · Ojeador estadístico
    t = time.perf_counter()
    try:
        rec = recuperar_candidatos(consulta, df)
    except ValueError as e:
        resultado["errores"] = [str(e)]
        incidencia("similitud_no_calculable", str(e), "error", consulta.objetivo_registro_id)
        return terminar("rechazada")
    resultado["duracion_agente1_ms"] = int((time.perf_counter() - t) * 1000)
    resultado["recuperacion"] = rec
    resultado["avisos"].extend(rec["avisos"])
    if not rec["candidatos"]:
        resultado["avisos"].append("Ningún jugador cumple todas las restricciones. Prueba con más ligas, "
                                   "otra edad máxima o menos minutos mínimos.")
        incidencia("sin_candidatos", "Ningún candidato cumple las restricciones.", "info")
        return terminar("sin_resultados")

    # 3. Agente 2 · Analista de perfiles 
    analisis = None
    if consulta.usar_llm and proveedor_analista is None:
        resultado["avisos"].append("No hay modelo de lenguaje configurado: se usa el ranking numérico y fichas "
                                   "por plantilla.")
    elif consulta.usar_llm:
        payload_a = analista.construir_payload(rec)
        analisis, ejec = ejecutar_agente(
            "analista", proveedor_analista, analista.SISTEMA, analista.construir_mensaje(payload_a),
            lambda datos: analista.validar_respuesta(datos, payload_a), consulta.consulta_id,
            max_tokens=12000, cache=cache)
        resultado["ejecuciones"].extend(ejec)
        if analisis is None:
            resultado["avisos"].append("El análisis del Agente 2 no superó la validación; la consulta usa el "
                                       "ranking numérico (modo degradado).")
            incidencia("degradacion_agente2", ejec[-1].get("detalle") or ejec[-1].get("codigo_error") or "fallo",
                       "aviso")

    # 4. Ranking final
    ranking, modo, motivo = calcular_ranking(rec["candidatos"], analisis, consulta.peso_numerico,
                                             consulta.peso_cualitativo)
    resultado["modo_ranking"] = modo
    if analisis is not None and modo == "numerico":
        resultado["avisos"].append(f"Ranking numérico para toda la consulta: {motivo}.")
        incidencia("degradacion_ranking", motivo, "aviso")
    top = ranking[:consulta.top_final]

    # 5. Agente 3 · Director deportivo o plantilla
    fichas = None
    if consulta.usar_llm and proveedor_director is not None:
        if tokens_usados(resultado["ejecuciones"]) > C.LLM_TOPE_TOKENS_CONSULTA:
            resultado["avisos"].append("Se ha alcanzado el tope de tokens de la consulta; fichas por plantilla.")
            incidencia("tope_tokens", "Tope de tokens alcanzado antes del Agente 3.", "aviso")
        else:
            payload_d = director.construir_payload(rec, top, consulta, modo)
            fichas, ejec = ejecutar_agente(
                "director", proveedor_director, director.SISTEMA, director.construir_mensaje(payload_d),
                lambda datos: director.validar_respuesta(datos, payload_d), consulta.consulta_id,
                max_tokens=8000, cache=cache)
            resultado["ejecuciones"].extend(ejec)
            if fichas is None:
                resultado["avisos"].append("La ficha del Agente 3 no superó la validación; se usa la ficha por "
                                           "plantilla.")
                incidencia("degradacion_agente3", ejec[-1].get("detalle") or ejec[-1].get("codigo_error") or "fallo",
                           "aviso")

    for c in ranking:
        c["es_top_final"] = c["ranking_final"] <= consulta.top_final
        if c["es_top_final"]:
            c["ficha"] = fichas["fichas"][c["registro_id"]] if fichas else director.ficha_plantilla(c)
    if fichas:
        resultado["resumen_consulta"] = fichas["resumen_consulta"]
        if fichas["limitaciones_generales"]:
            resultado["limitaciones_generales"] = fichas["limitaciones_generales"]
    else:
        resultado["resumen_consulta"] = director.resumen_plantilla(rec, len(top))

    resultado["ranking"] = ranking
    resultado["top_final"] = [c for c in ranking if c["es_top_final"]]
    return terminar("completada" if len(top) >= consulta.top_final else "parcial")
