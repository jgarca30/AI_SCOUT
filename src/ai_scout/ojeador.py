"""Ojeador de datos.

1. Valida la consulta.
2. Estandariza las variables del perfil con media y desviación de toda la
   población elegible del grupo.
3. Fija la importancia de cada variable: pesos del rol buscado (o ajustados a mano)
   multiplicados por lo aprendido de las valoraciones del usuario.
4. Adapta la consulta con las valoraciones de esa referencia (Rocchio).
5. Filtra la población ANTES de calcular la similitud (grupo, minutos, ligas, edad,
   descartados por el usuario y, si se pide, mismo estilo de juego).
6. Calcula la similitud coseno ponderada y el índice 0-100.
7. Ordena con desempates reproducibles (o diversifica con MMR) y devuelve el Top 10
   con evidencias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .modelos import Consulta
from .similitud import (construir_espacio, consulta_rocchio, cosenos, indice, pesos_de_rol,
                        poblacion_grupo, seleccionar_mmr, vector_pesos)

__all__ = ["score_numerico", "poblacion_grupo", "validar_consulta", "recuperar_candidatos"]

DIVERSIDAD_MAXIMA = 0.9


def score_numerico(coseno: float) -> float:
    """Transforma la similitud coseno (-1..1) en un índice 0..100. No es una probabilidad."""
    return indice(coseno)


def validar_consulta(consulta: Consulta, df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de errores; vacía si la consulta es válida."""
    errores = []
    if not consulta.ligas_permitidas:
        errores.append("Selecciona al menos una liga.")
    else:
        desconocidas = [l for l in consulta.ligas_permitidas if l not in C.LIGAS_NORMALIZADAS]
        if desconocidas:
            errores.append(f"Ligas desconocidas: {', '.join(desconocidas)}.")
    if consulta.minutos_minimos is None or consulta.minutos_minimos <= 0:
        errores.append("Los minutos mínimos deben ser un número positivo.")
    if consulta.edad_maxima is not None and not (15 <= consulta.edad_maxima <= 45):
        errores.append("La edad máxima debe estar entre 15 y 45 años.")
    if consulta.top_preseleccion < 1 or consulta.top_final < 1 or consulta.top_final > consulta.top_preseleccion:
        errores.append("Los tamaños del Top no son coherentes.")
    if not 0 <= consulta.diversidad <= DIVERSIDAD_MAXIMA:
        errores.append(f"La diversidad debe estar entre 0 y {DIVERSIDAD_MAXIMA}.")

    if consulta.equipo_destino is not None and consulta.equipo_destino not in set(df["equipo"]):
        errores.append(f"El club de destino «{consulta.equipo_destino}» no está en el dataset.")

    fila = df[df["registro_id"] == consulta.objetivo_registro_id]
    if fila.empty:
        errores.append(f"El jugador de referencia ({consulta.objetivo_registro_id}) no existe en el dataset.")
        return errores
    fila = fila.iloc[0]
    if fila["es_portero"]:
        errores.append(f"{fila['nombre']} es portero; los porteros están fuera del alcance de AI Scout.")
    elif not fila["valido"]:
        errores.append(f"{fila['nombre']} no se puede usar como referencia: {fila['motivo_invalido']}.")
    elif consulta.minutos_minimos and fila["minutos"] < consulta.minutos_minimos:
        errores.append(
            f"{fila['nombre']} solo tiene {int(fila['minutos'])} minutos ({fila['equipo']}), por debajo del mínimo de "
            f"{consulta.minutos_minimos}. Con tan pocos minutos sus estadísticas por 90 no son fiables.")

    # Rol y pesos: dependen del grupo de posición del jugador de referencia
    grupo = fila["grupo_posicion"]
    if grupo in C.ROLES:
        if consulta.rol not in C.ROLES[grupo]:
            errores.append(f"El rol «{consulta.rol}» no existe para {C.GRUPOS[grupo].lower()}. "
                           f"Roles disponibles: {', '.join(C.ROLES[grupo])}.")
        pesos = consulta.pesos_personalizados
        if pesos is not None:
            ajenas = [v for v in pesos if v not in C.PERFILES[grupo]]
            if ajenas:
                errores.append(f"Pesos de variables que no pertenecen al perfil de {C.GRUPOS[grupo].lower()}: "
                               f"{', '.join(ajenas)}.")
            valores = [pesos.get(v, 1.0) for v in C.PERFILES[grupo]]
            if any((not isinstance(x, (int, float))) or x < 0 or x > C.PESO_MAXIMO_VARIABLE for x in valores):
                errores.append(f"Los pesos deben estar entre 0 y {C.PESO_MAXIMO_VARIABLE:g}.")
            elif not sum(valores) > 0:
                errores.append("Al menos una variable debe tener un peso mayor que cero.")
    return errores


def _fila_a_registro(fila: pd.Series) -> dict:
    edad = fila["edad_registrada"]
    return {
        "registro_id": fila["registro_id"], "persona_id": fila["persona_id"], "nombre": fila["nombre"],
        "equipo": fila["equipo"], "liga": fila["liga"], "posicion_original": fila["posicion_original"],
        "grupo_posicion": fila["grupo_posicion"], "nacionalidad": fila["nacionalidad"],
        "ano_nacimiento": None if pd.isna(fila["ano_nacimiento"]) else int(fila["ano_nacimiento"]),
        "edad_registrada": None if pd.isna(edad) else int(edad), "minutos": int(fila["minutos"]),
    }


def _desplazamiento(variables: list[str], z_obj: np.ndarray, q: np.ndarray) -> dict:
    """Cambio del perfil buscado tras aplicar Rocchio, en unidades z.

    La consulta adaptada se reescala a la norma de la referencia: al coseno solo le importa
    la dirección del vector, así que se compara dirección con dirección.
    """
    norma_q = float(np.linalg.norm(q))
    if norma_q == 0:
        return {}
    q_escalada = q * float(np.linalg.norm(z_obj)) / norma_q
    return {v: float(x) for v, x in zip(variables, q_escalada - z_obj)}


def recuperar_candidatos(consulta: Consulta, df: pd.DataFrame, estilos: dict | None = None,
                         personalizacion=None) -> dict:
    """Ejecuta el Ojeador de datos. Supone una consulta ya validada.

    - `estilos`: modelos de estilos de juego (estilos.calcular_estilos); opcional.
    - `personalizacion`: lo aprendido de las valoraciones (aprendizaje.Personalizacion); solo se
      aplica si la consulta tiene `usar_aprendizaje`.
    """
    objetivo = df[df["registro_id"] == consulta.objetivo_registro_id].iloc[0]
    grupo = objetivo["grupo_posicion"]
    avisos: list[str] = []
    if not consulta.usar_aprendizaje:
        personalizacion = None

    # ---- Espacio estandarizado del grupo (no depende de ligas ni edad) ----
    espacio = construir_espacio(df, grupo, consulta.minutos_minimos)
    poblacion, variables, descartadas = espacio.poblacion, espacio.variables, espacio.descartadas
    if descartadas:
        avisos.append(f"Variables sin varianza retiradas del cálculo: {', '.join(descartadas)}.")
    if not variables:
        raise ValueError("No quedan variables utilizables para calcular la similitud.")

    # ---- Importancia de cada variable: rol (o pesos a mano) × aprendido del usuario ----
    pesos_base = consulta.pesos_personalizados if consulta.pesos_personalizados is not None \
        else pesos_de_rol(grupo, consulta.rol)
    multiplicadores = dict(personalizacion.multiplicadores) if personalizacion else {}
    w_rol = vector_pesos(variables, pesos_base)
    w = vector_pesos(variables, pesos_base, multiplicadores)

    z_obj = espacio.z_fila(objetivo)
    if float(np.linalg.norm(z_obj * np.sqrt(w))) == 0:
        raise ValueError("El perfil del jugador de referencia coincide exactamente con la media; "
                         "la similitud coseno no está definida.")

    # ---- Consulta adaptada con las valoraciones de esta referencia (Rocchio) ----
    q = z_obj
    if personalizacion and personalizacion.hay_realimentacion:
        del_grupo = df[(df["grupo_posicion"] == grupo) & df["valido"]].set_index("registro_id")

        def vectores(pares):
            presentes = [(rid, peso) for rid, peso in pares if rid in del_grupo.index]
            if not presentes:
                return None, None
            return (espacio.z(del_grupo.loc[[rid for rid, _ in presentes]]),
                    np.array([peso for _, peso in presentes], dtype=float))

        z_pos, p_pos = vectores(personalizacion.positivos)
        z_neg, p_neg = vectores(personalizacion.negativos)
        q = consulta_rocchio(z_obj, z_pos, p_pos, z_neg, p_neg)
        if float(np.linalg.norm(q * np.sqrt(w))) == 0:
            q = z_obj

    # Percentiles dentro del grupo (para gráficos e interpretación)
    percentiles = poblacion[variables].rank(pct=True) * 100.0
    if objetivo["registro_id"] in set(poblacion["registro_id"]):
        perc_obj = percentiles.loc[poblacion["registro_id"] == objetivo["registro_id"]].iloc[0]
    else:  # no debería ocurrir tras validar, pero se calcula igualmente
        perc_obj = pd.Series({v: float((poblacion[v] <= objetivo[v]).mean() * 100) for v in variables})

    # Estilos de juego
    modelo_estilos = (estilos or {}).get(grupo)
    estilo_por_registro = {}
    if modelo_estilos is not None:
        t = modelo_estilos.tabla
        estilo_por_registro = {rid: {"id": int(eid), "nombre": nombre}
                               for rid, eid, nombre in zip(t["registro_id"], t["estilo_id"], t["estilo"])}
    estilo_obj = estilo_por_registro.get(objetivo["registro_id"])

    # ---- Restricciones duras ANTES de calcular el Top ----
    exclusiones = {}
    cand = poblacion
    n_inicial = len(cand)
    cand = cand[cand["liga"].isin(consulta.ligas_permitidas)]
    exclusiones["liga"] = n_inicial - len(cand)
    if consulta.edad_maxima is not None:
        sin_edad = cand["edad_registrada"].isna()
        exclusiones["edad_no_disponible"] = int(sin_edad.sum())
        cand = cand[~sin_edad]
        n = len(cand)
        cand = cand[cand["edad_registrada"] <= consulta.edad_maxima]
        exclusiones["edad"] = n - len(cand)
    n = len(cand)
    cand = cand[(cand["registro_id"] != objetivo["registro_id"]) & (cand["persona_id"] != objetivo["persona_id"])]
    exclusiones["objetivo_y_misma_persona"] = n - len(cand)
    if personalizacion and personalizacion.descartados:
        n = len(cand)
        cand = cand[~cand["persona_id"].isin(personalizacion.descartados)]
        exclusiones["descartados_por_el_usuario"] = n - len(cand)
    if personalizacion and consulta.ocultar_valorados and personalizacion.valorados:
        n = len(cand)
        cand = cand[~cand["persona_id"].isin(set(personalizacion.valorados))]
        exclusiones["ya_valorados"] = n - len(cand)
    if consulta.solo_mismo_estilo:
        if estilo_obj is None:
            avisos.append("No se puede filtrar por estilo de juego: no hay modelo de estilos para este grupo.")
        else:
            n = len(cand)
            mismo = cand["registro_id"].map(lambda r: estilo_por_registro.get(r, {}).get("id") == estilo_obj["id"])
            cand = cand[mismo.astype(bool)] if len(cand) else cand
            exclusiones["otro_estilo_de_juego"] = n - len(cand)
    # Una sola representación por persona: el registro con más minutos; empate por registro_id
    n = len(cand)
    cand = cand.sort_values(["persona_id", "minutos", "registro_id"], ascending=[True, False, True])
    cand = cand.drop_duplicates("persona_id", keep="first")
    exclusiones["duplicados_de_persona"] = n - len(cand)

    # ---- Similitud coseno ponderada ----
    resultados = []
    if len(cand):
        z_cand = espacio.z(cand)
        sim = cosenos(q, z_cand, w)                  # la que ordena: pesos + consulta adaptada
        sim_base = cosenos(z_obj, z_cand)            # coseno clásico con la referencia (para comparar)
        exclusiones["norma_cero"] = int(np.isnan(sim).sum())
        cand = cand.assign(_coseno=sim, _coseno_base=sim_base, _idx=range(len(cand)))
        cand = cand[cand["_coseno"].notna()]
        cand = cand.sort_values(["_coseno", "minutos", "registro_id"], ascending=[False, False, True])
        if consulta.diversidad > 0 and len(cand) > consulta.top_preseleccion:
            bolsa = cand.head(max(C.MMR_BOLSA, consulta.top_preseleccion))
            elegidos = seleccionar_mmr(bolsa["_coseno"].to_numpy(), z_cand[bolsa["_idx"].to_numpy()],
                                       consulta.top_preseleccion, consulta.diversidad, w)
            top = bolsa.iloc[sorted(elegidos)]       # se mantiene el orden por similitud
        else:
            top = cand.head(consulta.top_preseleccion)

        valorados = personalizacion.valorados if personalizacion else {}
        for posicion, (_, fila) in enumerate(top.iterrows(), start=1):
            z_c = z_cand[int(fila["_idx"])]
            perc_c = percentiles.loc[fila.name] if fila.name in percentiles.index else None
            diferencias = []
            for j, v in enumerate(variables):
                diferencias.append({
                    "variable": v, "descripcion": C.VARIABLES[v][0], "unidad": C.VARIABLES[v][1],
                    "valor_objetivo": float(objetivo[v]), "valor_candidato": float(fila[v]),
                    "diferencia": float(fila[v] - objetivo[v]),
                    "z_diferencia": float(z_c[j] - z_obj[j]),
                    "peso": float(w[j]),
                })
            registro = _fila_a_registro(fila)
            coseno_base = float(fila["_coseno_base"]) if not pd.isna(fila["_coseno_base"]) else float(fila["_coseno"])
            registro.update({
                "coseno": float(fila["_coseno"]),
                "score_numerico": score_numerico(float(fila["_coseno"])),
                "coseno_base": coseno_base,
                "score_base": score_numerico(coseno_base),
                "ranking_preliminar": posicion,
                "estilo": estilo_por_registro.get(fila["registro_id"], {}).get("nombre"),
                "valoracion_usuario": valorados.get(fila["persona_id"]),
                "valores": {v: float(fila[v]) for v in variables},
                "percentiles": {v: float(perc_c[v]) for v in variables} if perc_c is not None else {},
                "diferencias": diferencias,
            })
            resultados.append(registro)

    if 0 < len(resultados) < consulta.top_final:
        avisos.append(f"Solo {len(resultados)} candidato(s) cumplen todas las restricciones; "
                      "no se relajan los filtros para completar la lista.")

    objetivo_dict = _fila_a_registro(objetivo)
    objetivo_dict.update({
        "valores": {v: float(objetivo[v]) for v in variables},
        "percentiles": {v: float(perc_obj[v]) for v in variables},
        "estilo": estilo_obj["nombre"] if estilo_obj else None,
    })
    rol = C.ROLES[grupo].get(consulta.rol, C.ROLES[grupo][C.ROL_DEFECTO])
    return {
        "grupo": grupo,
        "variables": variables,
        "variables_descartadas": descartadas,
        "escalado": {"medias": espacio.medias[variables].to_dict(),
                     "desviaciones": espacio.desviaciones[variables].to_dict()},
        "rol": consulta.rol,
        "rol_nombre": "Pesos ajustados a mano" if consulta.pesos_personalizados is not None else rol["nombre"],
        "pesos_rol": {v: float(x) for v, x in zip(variables, w_rol)},
        "multiplicadores": {v: float(multiplicadores.get(v, 1.0)) for v in variables} if multiplicadores else {},
        "pesos": {v: float(x) for v, x in zip(variables, w)},
        "personalizacion": personalizacion.resumen() if personalizacion else None,
        "desplazamiento_consulta": _desplazamiento(variables, z_obj, q)
        if personalizacion and personalizacion.hay_realimentacion else {},
        "diversidad": consulta.diversidad,
        "poblacion_referencia": int(len(poblacion)),
        "n_candidatos_validos": int(len(cand)),
        "exclusiones": exclusiones,
        "objetivo": objetivo_dict,
        "candidatos": resultados,
        "avisos": avisos,
    }
