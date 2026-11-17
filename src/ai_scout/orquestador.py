"""Orquestador de AI Scout (CU-02 Buscar sustituto).

Valida la consulta, recupera lo aprendido de las valoraciones del usuario, llama a los
agentes en orden, valida sus respuestas, calcula el ranking final, gestiona los fallos y
guarda el resultado.


Si un agente LLM falla, la consulta termina igualmente en modo degradado.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd

from . import analista, aprendizaje, critico, director, encaje, exportador, valoraciones
from . import config as C
from .ejecutor_llm import ejecutar_agente, tokens_usados
from .llm import CacheLLM, ProveedorLLM
from .modelos import Consulta
from .ojeador import recuperar_candidatos, validar_consulta
from .ranking import calcular_ranking

# Aviso de progreso: (fase, estado, detalle). Estado: "inicio", "fin" u "omitido".
Progreso = Callable[[str, str, str], None]

# Explicación breve de los fallos más habituales de un agente LLM (para los avisos)
CAUSAS_FALLO = {
    "http_429": "se ha agotado el cupo gratuito del modelo (error 429); prueba más tarde o cambia de modelo",
    "http_503": "el modelo está saturado (error 503)",
    "http_401": "la clave de la API no es válida (error 401)",
    "http_403": "la clave de la API no tiene permiso para usar el modelo (error 403)",
    "http_404": "el modelo configurado no existe (error 404)",
    "http_400": "la API ha rechazado la petición (error 400); revisa la clave y el modelo",
    "conexion_o_timeout": "no hay conexión con el proveedor o ha tardado demasiado",
    "rechazo": "el modelo ha rechazado la petición",
    "json_invalido": "la respuesta no era un JSON válido",
    "validacion": "la respuesta no ha superado la validación",
}


def causa_del_fallo(ejecuciones: list[dict]) -> str:
    """Texto legible con el motivo del último intento fallido de un agente."""
    if not ejecuciones:
        return "fallo desconocido"
    ultimo = ejecuciones[-1]
    codigo = ultimo.get("codigo_error") or "fallo"
    return CAUSAS_FALLO.get(codigo, codigo)


def ejecutar_consulta(consulta: Consulta, df: pd.DataFrame, dataset_version: str,
                      proveedor_analista: ProveedorLLM | None = None,
                      proveedor_director: ProveedorLLM | None = None,
                      cache: CacheLLM | None = None, exportar: bool = True,
                      dir_salidas: Path | str | None = None, *,
                      proveedor_critico: ProveedorLLM | None = None,
                      proveedor_plantilla: ProveedorLLM | None = None,
                      estilos: dict | None = None,
                      progreso: Progreso | None = None,
                      ejecuciones_previas: list[dict] | None = None) -> dict:
    """Ejecuta una consulta completa y devuelve su resultado.

    - `estilos`: modelos de estilos de juego (estilos.calcular_estilos). Sin ellos la consulta
      funciona igual, pero sin etiquetas de estilo ni filtro por estilo.
    - `progreso`: función a la que se avisa del avance (la app la usa para mostrar cada paso).
    - `ejecuciones_previas`: trazas de agentes que han trabajado antes (el Secretario técnico).
    """
    inicio = time.perf_counter()
    dir_salidas = Path(dir_salidas) if dir_salidas else C.DIR_SALIDAS
    resultado = {
        "consulta": consulta.como_dict(), "dataset_version": dataset_version,
        "perfil_version": C.PERFIL_VERSION, "estado": None, "errores": [], "avisos": [],
        "modo_ranking": "numerico", "recuperacion": None, "ranking": [], "top_final": [],
        "resumen_consulta": "", "limitaciones_generales": list(C.LIMITACIONES_GENERALES),
        "personalizacion": None, "revision": None, "plantilla": None,
        "ejecuciones": list(ejecuciones_previas or []), "incidencias": [],
    }

    def avisar(fase: str, estado: str, detalle: str = "") -> None:
        if progreso is not None:
            progreso(fase, estado, detalle)

    def incidencia(codigo, descripcion, severidad="aviso", registro_id=None, fase="consulta"):
        resultado["incidencias"].append({"fase": fase, "severidad": severidad, "codigo": codigo,
                                         "registro_id": registro_id, "descripcion": descripcion})

    def terminar(estado: str) -> dict:
        resultado["estado"] = estado
        resultado["duracion_total_ms"] = int((time.perf_counter() - inicio) * 1000)
        if exportar:
            resultado["rutas_exportacion"] = exportador.exportar_consulta(resultado, dir_salidas)
            if estado != "rechazada":
                exportador.guardar_historial(resultado, dir_salidas)
        return resultado

    # 1. Validación de la consulta
    errores = validar_consulta(consulta, df)
    if errores:
        resultado["errores"] = errores
        for e in errores:
            incidencia("consulta_rechazada", e, "error", consulta.objetivo_registro_id)
        return terminar("rechazada")
    objetivo = df[df["registro_id"] == consulta.objetivo_registro_id].iloc[0]

    # 2. Lo aprendido de las valoraciones del usuario
    personalizacion = None
    if consulta.usar_aprendizaje:
        avisar("aprendizaje", "inicio")
        try:
            vigentes = valoraciones.vigentes(dataset_version=dataset_version, directorio=dir_salidas)
            personalizacion = aprendizaje.construir_personalizacion(vigentes, df, objetivo)
        except (OSError, ValueError, KeyError) as e:
            resultado["avisos"].append("No se han podido leer las valoraciones guardadas; la búsqueda no está "
                                       "personalizada.")
            incidencia("valoraciones_ilegibles", str(e)[:300], "aviso")
        if personalizacion is None or personalizacion.n_grupo == 0:
            avisar("aprendizaje", "fin", "todavía no hay valoraciones para este grupo de posición")
        else:
            partes = [f"{personalizacion.n_grupo} valoraciones del grupo"]
            if personalizacion.multiplicadores:
                partes.append("pesos aprendidos aplicados")
            if personalizacion.hay_realimentacion:
                partes.append(f"consulta adaptada con {len(personalizacion.valorados)} valoraciones de esta referencia")
            avisar("aprendizaje", "fin", " · ".join(partes))
    else:
        avisar("aprendizaje", "omitido", "personalización desactivada")

    # 3. Ojeador de datos
    avisar("agente1", "inicio")
    t = time.perf_counter()
    try:
        rec = recuperar_candidatos(consulta, df, estilos, personalizacion)
    except ValueError as e:
        resultado["errores"] = [str(e)]
        incidencia("similitud_no_calculable", str(e), "error", consulta.objetivo_registro_id)
        return terminar("rechazada")
    resultado["duracion_agente1_ms"] = int((time.perf_counter() - t) * 1000)
    resultado["recuperacion"] = rec
    resultado["personalizacion"] = rec["personalizacion"]
    resultado["avisos"].extend(rec["avisos"])
    avisar("agente1", "fin", f"{rec['n_candidatos_validos']} jugadores comparados · rol: {rec['rol_nombre']}")
    if not rec["candidatos"]:
        resultado["avisos"].append("Ningún jugador cumple todas las restricciones. Prueba con más ligas, "
                                   "otra edad máxima o menos minutos mínimos.")
        incidencia("sin_candidatos", "Ningún candidato cumple las restricciones.", "info")
        return terminar("sin_resultados")

    # 4. Analista de rendimiento
    analisis = None
    if not consulta.usar_llm:
        avisar("agente2", "omitido", "análisis con IA desactivado")
    elif proveedor_analista is None:
        resultado["avisos"].append("No hay modelo de lenguaje configurado: se usa el ranking numérico y fichas "
                                   "automáticas.")
        avisar("agente2", "omitido", "sin modelo de lenguaje configurado")
    else:
        avisar("agente2", "inicio")
        payload_a = analista.construir_payload(rec)
        analisis, ejec = ejecutar_agente(
            "analista", proveedor_analista, analista.SISTEMA, analista.construir_mensaje(payload_a),
            lambda datos: analista.validar_respuesta(datos, payload_a), consulta.consulta_id,
            max_tokens=12000, cache=cache)
        resultado["ejecuciones"].extend(ejec)
        if analisis is None:
            causa = causa_del_fallo(ejec)
            resultado["avisos"].append(f"El Analista de rendimiento no ha dado un análisis válido ({causa}); la consulta usa "
                                       "el ranking numérico (modo degradado).")
            incidencia("degradacion_agente2", ejec[-1].get("detalle") or ejec[-1].get("codigo_error") or "fallo",
                       "aviso")
            avisar("agente2", "fin", f"sin análisis válido: {causa}")
        else:
            avisar("agente2", "fin", f"{len(analisis)} candidatos analizados")

    # 5. Ranking final
    ranking, modo, motivo = calcular_ranking(rec["candidatos"], analisis, consulta.peso_numerico,
                                             consulta.peso_cualitativo)
    resultado["modo_ranking"] = modo
    if analisis is not None and modo == "numerico":
        resultado["avisos"].append(f"Ranking numérico para toda la consulta: {motivo}.")
        incidencia("degradacion_ranking", motivo, "aviso")
    avisar("ranking", "fin", "híbrido (similitud + análisis)" if modo == "hibrido" else "numérico")

    # 5b. Analista de plantilla: encaje en el club de destino, que pondera la puntuación final
    if consulta.equipo_destino is None:
        avisar("plantilla", "omitido", "sin club de destino")
    else:
        avisar("plantilla", "inicio")
        evaluacion = encaje.evaluar(rec, df, consulta.equipo_destino, estilos)
        valoracion = None
        if consulta.usar_llm and proveedor_plantilla is not None \
                and tokens_usados(resultado["ejecuciones"]) <= C.LLM_TOPE_TOKENS_CONSULTA:
            payload_p = encaje.construir_payload(rec, evaluacion)
            valoracion, ejec = ejecutar_agente(
                "plantilla", proveedor_plantilla, encaje.SISTEMA, encaje.construir_mensaje(payload_p),
                lambda datos: encaje.validar_respuesta(datos, payload_p), consulta.consulta_id,
                max_tokens=6000, cache=cache)
            resultado["ejecuciones"].extend(ejec)
            if valoracion is None:
                causa = causa_del_fallo(ejec)
                resultado["avisos"].append(f"El Analista de plantilla no ha dado una valoración válida ({causa}); "
                                           "el encaje se calcula solo con los indicadores.")
                incidencia("degradacion_plantilla", ejec[-1].get("detalle") or ejec[-1].get("codigo_error")
                           or "fallo", "aviso")
        ranking = encaje.ponderar(ranking, evaluacion, valoracion)
        resultado["plantilla"] = {
            **{k: v for k, v in evaluacion.items() if k != "candidatos"}, "peso": C.PESO_PLANTILLA,
            "diagnostico": valoracion["diagnostico"] if valoracion else "",
            "origen": "llm" if valoracion else "indicadores"}
        avisar("plantilla", "fin", f"encaje en {consulta.equipo_destino} valorado "
                                   + ("por el modelo y los indicadores" if valoracion else "con los indicadores"))
    top = ranking[:consulta.top_final]

    # 6. Director deportivo o plantilla
    fichas = None
    payload_d = mensaje_d = None
    if consulta.usar_llm and proveedor_director is not None:
        if tokens_usados(resultado["ejecuciones"]) > C.LLM_TOPE_TOKENS_CONSULTA:
            resultado["avisos"].append("Se ha alcanzado el tope de tokens de la consulta; fichas automáticas.")
            incidencia("tope_tokens", "Tope de tokens alcanzado antes del Director deportivo.", "aviso")
            avisar("agente3", "omitido", "tope de tokens alcanzado")
        else:
            avisar("agente3", "inicio")
            payload_d = director.construir_payload(rec, top, consulta, modo)
            mensaje_d = director.construir_mensaje(payload_d)
            fichas, ejec = ejecutar_agente(
                "director", proveedor_director, director.SISTEMA, mensaje_d,
                lambda datos: director.validar_respuesta(datos, payload_d), consulta.consulta_id,
                max_tokens=8000, cache=cache)
            resultado["ejecuciones"].extend(ejec)
            if fichas is None:
                causa = causa_del_fallo(ejec)
                resultado["avisos"].append(f"El Director deportivo no ha dado fichas válidas ({causa}); se usa la ficha "
                                           "automática.")
                incidencia("degradacion_agente3", ejec[-1].get("detalle") or ejec[-1].get("codigo_error") or "fallo",
                           "aviso")
                avisar("agente3", "fin", f"sin fichas válidas: {causa}")
            else:
                avisar("agente3", "fin", f"{len(fichas['fichas'])} fichas redactadas")
    else:
        avisar("agente3", "omitido", "fichas automáticas")

    # 7. Jefe de scouting de las fichas (y una posible corrección del director)
    revisiones: dict[str, dict] = {}
    if fichas is None or not consulta.usar_critico or proveedor_critico is None:
        avisar("agente4", "omitido", "sin fichas que revisar" if fichas is None else "revisión desactivada")
    elif tokens_usados(resultado["ejecuciones"]) > C.LLM_TOPE_TOKENS_CONSULTA:
        incidencia("tope_tokens", "Tope de tokens alcanzado antes del Jefe de scouting.", "aviso")
        avisar("agente4", "omitido", "tope de tokens alcanzado")
    else:
        avisar("agente4", "inicio")
        payload_c = critico.construir_payload(payload_d, fichas)
        revision, ejec = ejecutar_agente(
            "critico", proveedor_critico, critico.SISTEMA, critico.construir_mensaje(payload_c),
            lambda datos: critico.validar_respuesta(datos, payload_c), consulta.consulta_id,
            max_tokens=4000, cache=cache)
        resultado["ejecuciones"].extend(ejec)
        if revision is None:
            causa = causa_del_fallo(ejec)
            resultado["avisos"].append(f"El Jefe de scouting no ha podido revisar las fichas ({causa}); se muestran sin "
                                       "revisar.")
            incidencia("degradacion_agente4", ejec[-1].get("detalle") or ejec[-1].get("codigo_error") or "fallo",
                       "aviso")
            resultado["revision"] = {"estado": "no_revisada", "valoracion_global": "", "rechazadas": 0}
            avisar("agente4", "fin", f"sin revisión: {causa}")
        else:
            rechazadas = [cid for cid, r in revision["revisiones"].items() if not r["aprobada"]]
            revisiones = {cid: {"estado": "aprobada", "problemas": []} for cid in revision["revisiones"]}
            estado_revision = "aprobadas"
            if rechazadas:
                # Patrón evaluador-optimizador: el director corrige una sola vez con las observaciones
                avisar("agente4", "fin", f"{len(rechazadas)} ficha(s) con observaciones: se pide una corrección")
                avisar("agente3", "inicio", "corrigiendo las fichas")
                corregidas, ejec = ejecutar_agente(
                    "director_correccion", proveedor_director, director.SISTEMA,
                    critico.mensaje_de_correccion(mensaje_d, fichas, revision),
                    lambda datos: director.validar_respuesta(datos, payload_d), consulta.consulta_id,
                    max_tokens=8000, cache=cache, max_intentos=1)
                resultado["ejecuciones"].extend(ejec)
                for cid in rechazadas:
                    revisiones[cid] = {"estado": "corregida" if corregidas else "con_observaciones",
                                       "problemas": revision["revisiones"][cid]["problemas"]}
                if corregidas:
                    fichas = corregidas
                    estado_revision = "corregidas"
                    avisar("agente3", "fin", f"{len(rechazadas)} ficha(s) corregida(s)")
                else:
                    estado_revision = "con_observaciones"
                    incidencia("correccion_fallida", "La corrección del director no superó la validación; se "
                               "mantienen las fichas originales con las observaciones del revisor.", "aviso")
                    avisar("agente3", "fin", "la corrección no ha superado la validación; se mantiene la ficha "
                                             "original con las observaciones")
            else:
                avisar("agente4", "fin", "todas las fichas aprobadas")
            resultado["revision"] = {"estado": estado_revision, "valoracion_global": revision["valoracion_global"],
                                     "rechazadas": len(rechazadas)}

    for c in ranking:
        c["es_top_final"] = c["ranking_final"] <= consulta.top_final
        if c["es_top_final"]:
            c["ficha"] = dict(fichas["fichas"][c["registro_id"]]) if fichas else director.ficha_plantilla(c)
            c["ficha"]["revision"] = revisiones.get(c["registro_id"], {"estado": "no_revisada", "problemas": []})
    if fichas:
        resultado["resumen_consulta"] = fichas["resumen_consulta"]
        if fichas["limitaciones_generales"]:
            resultado["limitaciones_generales"] = fichas["limitaciones_generales"]
    else:
        resultado["resumen_consulta"] = director.resumen_plantilla(rec, len(top))

    resultado["ranking"] = ranking
    resultado["top_final"] = [c for c in ranking if c["es_top_final"]]
    final = terminar("completada" if len(top) >= consulta.top_final else "parcial")
    avisar("guardado", "fin", f"consulta {consulta.consulta_id}")
    return final
