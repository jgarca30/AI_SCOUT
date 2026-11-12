"""Agente 1 · Ojeador estadístico.

1. Valida la consulta.
2. Filtra la población ANTES de calcular la similitud (grupo, minutos, ligas, edad).
3. Estandariza las variables del perfil con media y desviación de toda la
   población elegible del grupo.
4. Calcula la similitud coseno y el índice 0-100.
5. Ordena con desempates reproducibles y devuelve el Top 10 con evidencias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .modelos import Consulta


def score_numerico(coseno: float) -> float:
    """Transforma la similitud coseno (-1..1) en un índice 0..100. No es una probabilidad."""
    return 100.0 * (coseno + 1.0) / 2.0


def poblacion_grupo(df: pd.DataFrame, grupo: str, minutos_minimos: float) -> pd.DataFrame:
    """Registros válidos del grupo con los minutos mínimos (población de referencia del escalado)."""
    return df[(df["grupo_posicion"] == grupo) & df["valido"] & (df["minutos"] >= minutos_minimos)].copy()


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


def recuperar_candidatos(consulta: Consulta, df: pd.DataFrame) -> dict:
    """Ejecuta el Agente 1. Supone una consulta ya validada."""
    objetivo = df[df["registro_id"] == consulta.objetivo_registro_id].iloc[0]
    grupo = objetivo["grupo_posicion"]
    variables = list(C.PERFILES[grupo])
    avisos: list[str] = []

    poblacion = poblacion_grupo(df, grupo, consulta.minutos_minimos)

    # Escalado sobre toda la población elegible del grupo (no depende de ligas ni edad)
    medias = poblacion[variables].mean()
    desviaciones = poblacion[variables].std(ddof=0)
    descartadas = [v for v in variables if not desviaciones[v] > 0]
    if descartadas:
        avisos.append(f"Variables sin varianza retiradas del cálculo: {', '.join(descartadas)}.")
        variables = [v for v in variables if v not in descartadas]
    if not variables:
        raise ValueError("No quedan variables utilizables para calcular la similitud.")

    def estandarizar(tabla: pd.DataFrame) -> np.ndarray:
        return ((tabla[variables] - medias[variables]) / desviaciones[variables]).to_numpy(dtype=float)

    z_obj = estandarizar(objetivo.to_frame().T)[0]
    norma_obj = float(np.linalg.norm(z_obj))
    if norma_obj == 0:
        raise ValueError("El perfil del jugador de referencia coincide exactamente con la media; "
                         "la similitud coseno no está definida.")

    # Percentiles dentro del grupo (para gráficos e interpretación)
    percentiles = poblacion[variables].rank(pct=True) * 100.0
    if objetivo["registro_id"] in set(poblacion["registro_id"]):
        perc_obj = percentiles.loc[poblacion["registro_id"] == objetivo["registro_id"]].iloc[0]
    else:  # no debería ocurrir tras validar, pero se calcula igualmente
        perc_obj = pd.Series({v: float((poblacion[v] <= objetivo[v]).mean() * 100) for v in variables})

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
    # Una sola representación por persona: el registro con más minutos; empate por registro_id
    n = len(cand)
    cand = cand.sort_values(["persona_id", "minutos", "registro_id"], ascending=[True, False, True])
    cand = cand.drop_duplicates("persona_id", keep="first")
    exclusiones["duplicados_de_persona"] = n - len(cand)

    # ---- Similitud coseno ----
    resultados = []
    if len(cand):
        z_cand = estandarizar(cand)
        normas = np.linalg.norm(z_cand, axis=1)
        validos = normas > 0
        exclusiones["norma_cero"] = int((~validos).sum())
        cosenos = np.full(len(cand), np.nan)
        cosenos[validos] = (z_cand[validos] @ z_obj) / (normas[validos] * norma_obj)
        cand = cand.assign(_coseno=np.clip(cosenos, -1.0, 1.0), _idx=range(len(cand)))
        cand = cand[cand["_coseno"].notna()]
        cand = cand.sort_values(["_coseno", "minutos", "registro_id"], ascending=[False, False, True])
        top = cand.head(consulta.top_preseleccion)

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
                })
            registro = _fila_a_registro(fila)
            registro.update({
                "coseno": float(fila["_coseno"]),
                "score_numerico": score_numerico(float(fila["_coseno"])),
                "ranking_preliminar": posicion,
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
    })
    return {
        "grupo": grupo,
        "variables": variables,
        "variables_descartadas": descartadas,
        "escalado": {"medias": medias[variables].to_dict(), "desviaciones": desviaciones[variables].to_dict()},
        "poblacion_referencia": int(len(poblacion)),
        "n_candidatos_validos": int(len(cand)),
        "exclusiones": exclusiones,
        "objetivo": objetivo_dict,
        "candidatos": resultados,
        "avisos": avisos,
    }
