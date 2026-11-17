"""Espacio vectorial del recomendador y medidas de similitud.

Cada jugador es un vector con las variables de su grupo de posición, estandarizadas.
Sobre ese espacio se definen la similitud coseno ponderada (los pesos expresan la importancia de cada
variable), la realimentación de Rocchio y la selección diversificada (MMR).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class EspacioGrupo:
    """Variables, escalado y población de referencia de un grupo de posición."""

    grupo: str
    variables: list[str]         
    descartadas: list[str]
    medias: pd.Series
    desviaciones: pd.Series
    poblacion: pd.DataFrame     

    def z(self, tabla: pd.DataFrame) -> np.ndarray:
        """Matriz estandarizada (filas = registros, columnas = variables)."""
        return ((tabla[self.variables] - self.medias[self.variables])
                / self.desviaciones[self.variables]).to_numpy(dtype=float)

    def z_fila(self, fila: pd.Series) -> np.ndarray:
        return self.z(fila.to_frame().T)[0]


def poblacion_grupo(df: pd.DataFrame, grupo: str, minutos_minimos: float) -> pd.DataFrame:
    """Registros válidos del grupo con los minutos mínimos (población de referencia del escalado)."""
    return df[(df["grupo_posicion"] == grupo) & df["valido"] & (df["minutos"] >= minutos_minimos)].copy()


def construir_espacio(df: pd.DataFrame, grupo: str, minutos_minimos: float) -> EspacioGrupo:
    """Escalado sobre toda la población elegible del grupo (no depende de ligas ni de edad)."""
    poblacion = poblacion_grupo(df, grupo, minutos_minimos)
    variables = list(C.PERFILES[grupo])
    medias = poblacion[variables].mean()
    desviaciones = poblacion[variables].std(ddof=0)
    descartadas = [v for v in variables if not desviaciones[v] > 0]
    variables = [v for v in variables if v not in descartadas]
    return EspacioGrupo(grupo, variables, descartadas, medias, desviaciones, poblacion)


# Pesos
def pesos_de_rol(grupo: str, rol: str | None) -> dict[str, float]:
    """Pesos relativos del rol para todas las variables del grupo (las no citadas valen 1)."""
    definicion = C.ROLES[grupo].get(rol or C.ROL_DEFECTO)
    if definicion is None:
        raise KeyError(f"Rol desconocido para {grupo}: {rol!r}")
    return {v: float(definicion["pesos"].get(v, 1.0)) for v in C.PERFILES[grupo]}


def vector_pesos(variables: list[str], *pesos: dict[str, float] | None) -> np.ndarray:
    """Combina (multiplicando) varios diccionarios de pesos y normaliza a media 1.

    Un peso ausente vale 1. Lanza ValueError si hay pesos negativos o si todos son 0.
    """
    w = np.ones(len(variables), dtype=float)
    for p in pesos:
        if p:
            w *= np.array([float(p.get(v, 1.0)) for v in variables], dtype=float)
    if not np.all(np.isfinite(w)) or (w < 0).any():
        raise ValueError("Los pesos deben ser números no negativos.")
    if not w.sum() > 0:
        raise ValueError("Al menos una variable debe tener peso mayor que cero.")
    return w * len(w) / w.sum()


# Similitud
def cosenos(q: np.ndarray, Z: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Similitud coseno ponderada entre el vector q y cada fila de Z.

    cos_w(q, z) = Σ w·q·z / (√Σ w·q² · √Σ w·z²).
    Devuelve NaN donde alguna norma es cero (similitud no definida).
    """
    Z = np.atleast_2d(Z)
    raiz = np.ones(Z.shape[1]) if w is None else np.sqrt(w)
    qw, Zw = q * raiz, Z * raiz
    norma_q = float(np.linalg.norm(qw))
    normas = np.linalg.norm(Zw, axis=1)
    resultado = np.full(Z.shape[0], np.nan)
    if norma_q > 0:
        validos = normas > 0
        resultado[validos] = (Zw[validos] @ qw) / (normas[validos] * norma_q)
    return np.clip(resultado, -1.0, 1.0)


def indice(coseno: float) -> float:
    """Transforma la similitud coseno (-1..1) en un índice 0..100. No es una probabilidad."""
    return 100.0 * (coseno + 1.0) / 2.0


def consulta_rocchio(z_referencia: np.ndarray, positivos: np.ndarray | None, pesos_pos: np.ndarray | None,
                     negativos: np.ndarray | None, pesos_neg: np.ndarray | None,
                     alfa: float = C.ROCCHIO_ALFA, beta: float = C.ROCCHIO_BETA,
                     gamma: float = C.ROCCHIO_GAMMA) -> np.ndarray:
    """Realimentación de relevancia de Rocchio.

    Desplaza el vector de la referencia hacia el centro de los candidatos valorados en
    positivo y lo aleja del centro de los valorados en negativo (medias ponderadas por
    la intensidad de la valoración).
    """
    q = alfa * np.asarray(z_referencia, dtype=float)
    if positivos is not None and len(positivos):
        q = q + beta * np.average(positivos, axis=0, weights=pesos_pos)
    if negativos is not None and len(negativos):
        q = q - gamma * np.average(negativos, axis=0, weights=pesos_neg)
    return q


def seleccionar_mmr(similitudes: np.ndarray, Z: np.ndarray, k: int, diversidad: float,
                    w: np.ndarray | None = None) -> list[int]:
    """Selección diversificada (Maximal Marginal Relevance).

    `similitudes` y `Z` deben venir ordenados de mayor a menor similitud con la consulta.
    En cada paso elige el candidato que maximiza
        (1 − diversidad)·sim(consulta, c) − diversidad·max sim(c, ya elegidos)
    Con diversidad = 0 devuelve los k primeros. Los empates se resuelven por el orden de entrada.
    """
    n = len(similitudes)
    k = min(k, n)
    if k <= 0:
        return []
    if diversidad <= 0:
        return list(range(k))
    raiz = np.ones(Z.shape[1]) if w is None else np.sqrt(w)
    Zw = Z * raiz
    normas = np.linalg.norm(Zw, axis=1)
    normas[normas == 0] = 1.0
    U = Zw / normas[:, None]
    elegidos = [0]
    max_sim = U @ U[0]                       
    while len(elegidos) < k:
        valor = (1.0 - diversidad) * similitudes - diversidad * max_sim
        valor[elegidos] = -np.inf
        siguiente = int(np.argmax(valor))    
        elegidos.append(siguiente)
        max_sim = np.maximum(max_sim, U @ U[siguiente])
    return elegidos
