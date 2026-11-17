"""Aprendizaje a partir de las valoraciones del usuario.

El sistema aprende de dos maneras, las dos explicables:

1. A corto plazo, para una misma referencia (realimentación de Rocchio): los candidatos
   valorados para ese jugador desplazan la consulta hacia los que gustaron y la alejan de
   los que no; los valorados con 1-2 estrellas dejan de proponerse.

2. A largo plazo, para todo un grupo de posición (importancia de las variables): cada vez
   que el usuario valora de forma distinta a dos candidatos de una misma referencia nos
   dice cuál prefiere. Con todas esas comparaciones se ajustan unos multiplicadores m_i de
   los pesos de las variables para que la similitud coseno ponderada ordene a los
   candidatos como el usuario. Es un aprendizaje de ordenación por pares con el criterio
   BPR (Bayesian Personalized Ranking):

       minimizar  Σ_pares ω · log(1 + exp(−a · (sim_w(ref, preferido) − sim_w(ref, otro)))) + λ/2 · ‖ln m‖²

   con w = pesos de la búsqueda · m. La regularización hacia m = 1 hace que sin
   valoraciones no cambie nada y que con pocas (o contradictorias) el cambio sea prudente.
   Los multiplicadores aprendidos se aplican sobre los pesos del rol en cualquier búsqueda
   posterior del mismo grupo de posición.

Los parámetros (a, λ) se han elegido con usuarios simulados: ver evaluacion.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .similitud import EspacioGrupo, construir_espacio


@dataclass
class Personalizacion:
    """Lo que el sistema sabe del usuario para una búsqueda concreta."""

    multiplicadores: dict[str, float] = field(default_factory=dict)  # aprendidos para el grupo (media 1)
    aprendizaje: dict | None = None             # detalle del modelo ajustado (o None si no está activo)
    positivos: list[tuple[str, float]] = field(default_factory=list)   # (registro_id, peso) de esta referencia
    negativos: list[tuple[str, float]] = field(default_factory=list)
    descartados: set[str] = field(default_factory=set)   # persona_id con 1-2 estrellas para esta referencia
    valorados: dict[str, int] = field(default_factory=dict)   # persona_id → estrellas (esta referencia)
    n_grupo: int = 0                            # valoraciones vigentes del grupo de posición

    @property
    def hay_realimentacion(self) -> bool:
        return bool(self.positivos or self.negativos)

    def resumen(self) -> dict:
        return {
            "valoraciones_grupo": self.n_grupo, "valoraciones_referencia": len(self.valorados),
            "positivas_referencia": len(self.positivos), "negativas_referencia": len(self.negativos),
            "descartados": len(self.descartados), "rocchio_aplicado": self.hay_realimentacion,
            "pesos_aprendidos": bool(self.multiplicadores),
            "aprendizaje": self.aprendizaje,
        }


# --------------------------------------------------------------------------
# Aprendizaje de la importancia de las variables (ordenación por pares, criterio BPR)
# --------------------------------------------------------------------------
def _pesos_base(texto, variables: list[str]) -> np.ndarray:
    """Pesos de la búsqueda en la que se hizo la valoración (1 si no se guardaron)."""
    try:
        pesos = json.loads(texto) if isinstance(texto, str) and texto.strip() else {}
    except ValueError:
        pesos = {}
    w = np.array([float(pesos.get(v, 1.0)) for v in variables], dtype=float)
    return w if np.all(np.isfinite(w)) and (w >= 0).all() and w.sum() > 0 else np.ones(len(variables))


def pares_de_entrenamiento(valoraciones: pd.DataFrame, df: pd.DataFrame, espacio: EspacioGrupo) -> dict:
    """Convierte las valoraciones en comparaciones «para la referencia R prefiero A a B».

    Dos candidatos forman un par si se valoraron para el mismo registro de referencia con
    distinto número de estrellas. El peso del par crece con la diferencia de estrellas.
    """
    n = len(espacio.variables)
    por_registro = df[(df["grupo_posicion"] == espacio.grupo) & df["valido"]].set_index("registro_id")
    R, A, B, W, omega = [], [], [], [], []
    n_valoraciones = 0
    for referencia_id, grupo in valoraciones.groupby("objetivo_registro_id", sort=True):
        if referencia_id not in por_registro.index:
            continue
        z_ref = espacio.z_fila(por_registro.loc[referencia_id])
        items = []
        for v in grupo.itertuples():
            if v.candidato_registro_id in por_registro.index:
                items.append((int(v.estrellas), espacio.z_fila(por_registro.loc[v.candidato_registro_id]),
                              _pesos_base(getattr(v, "pesos_base_json", ""), espacio.variables)))
        n_valoraciones += len(items)
        for e_a, z_a, w_a in items:
            for e_b, z_b, w_b in items:
                if e_a > e_b:
                    base = (w_a + w_b) / 2.0
                    # Par no utilizable si algún vector queda sin norma con esos pesos
                    if min((base * z_ref ** 2).sum(), (base * z_a ** 2).sum(), (base * z_b ** 2).sum()) <= 1e-12:
                        continue
                    R.append(z_ref), A.append(z_a), B.append(z_b), W.append(base), omega.append((e_a - e_b) / 4.0)
    forma = (-1, n)
    return {"R": np.array(R, dtype=float).reshape(forma), "A": np.array(A, dtype=float).reshape(forma),
            "B": np.array(B, dtype=float).reshape(forma), "W": np.array(W, dtype=float).reshape(forma),
            "omega": np.array(omega, dtype=float), "n_valoraciones": n_valoraciones}


def _similitud_y_gradiente(R: np.ndarray, X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coseno ponderado fila a fila y su derivada respecto a cada peso."""
    a, b, d = (W * R * X).sum(axis=1), (W * R * R).sum(axis=1), (W * X * X).sum(axis=1)
    den = np.sqrt(b * d)
    s = a / den
    gradiente = (R * X) / den[:, None] - 0.5 * s[:, None] * ((R * R) / b[:, None] + (X * X) / d[:, None])
    return s, gradiente


def diferencias_de_similitud(pares: dict, m: np.ndarray) -> np.ndarray:
    """sim(ref, preferido) − sim(ref, otro) de cada par con los multiplicadores m."""
    W = pares["W"] * m
    return _similitud_y_gradiente(pares["R"], pares["A"], W)[0] - _similitud_y_gradiente(pares["R"], pares["B"], W)[0]


def ajustar_multiplicadores(pares: dict, nitidez: float = C.APRENDIZAJE_NITIDEZ, lam: float = C.APRENDIZAJE_LAMBDA,
                            limites: tuple[float, float] = C.APRENDIZAJE_LIMITES) -> np.ndarray:
    """Multiplicadores (media 1) que mejor ordenan los pares, regularizados hacia 1.

    Se optimiza θ = ln m con L-BFGS-B partiendo de θ = 0 (sin cambios).
    """
    from scipy.optimize import minimize

    R, A, B, Wb, omega = pares["R"], pares["A"], pares["B"], pares["W"], pares["omega"]
    n = R.shape[1]

    def perdida(theta: np.ndarray) -> tuple[float, np.ndarray]:
        W = Wb * np.exp(theta)
        s_a, g_a = _similitud_y_gradiente(R, A, W)
        s_b, g_b = _similitud_y_gradiente(R, B, W)
        d = nitidez * (s_a - s_b)
        valor = float(np.sum(omega * np.logaddexp(0.0, -d)) + lam / 2.0 * np.sum(theta ** 2))
        sigma = 1.0 / (1.0 + np.exp(np.clip(d, -50, 50)))          # σ(−d)
        # d(sim)/dθ_i = w_i · d(sim)/dw_i, porque w_i = base_i · e^θ_i
        g_theta = -nitidez * ((omega * sigma)[:, None] * ((g_a - g_b) * W)).sum(axis=0) + lam * theta
        return valor, g_theta

    cotas = [(float(np.log(limites[0])), float(np.log(limites[1])))] * n
    solucion = minimize(perdida, x0=np.zeros(n), jac=True, method="L-BFGS-B", bounds=cotas)
    m = np.exp(np.clip(solucion.x, cotas[0][0], cotas[0][1]))
    return m * n / m.sum()


def aprender_multiplicadores(valoraciones_grupo: pd.DataFrame, df: pd.DataFrame, grupo: str,
                             minutos_minimos: float = C.MINUTOS_MINIMOS_DEFECTO) -> dict | None:
    """Multiplicadores aprendidos para un grupo, o None si aún no hay comparaciones suficientes."""
    if valoraciones_grupo is None or len(valoraciones_grupo) == 0:
        return None
    espacio = construir_espacio(df, grupo, minutos_minimos)
    if not espacio.variables:
        return None
    pares = pares_de_entrenamiento(valoraciones_grupo, df, espacio)
    n_pares = len(pares["omega"])
    if n_pares < C.APRENDIZAJE_MIN_PARES:
        return None
    m = ajustar_multiplicadores(pares)
    antes = diferencias_de_similitud(pares, np.ones(len(m)))
    despues = diferencias_de_similitud(pares, m)
    return {
        "multiplicadores": {v: float(x) for v, x in zip(espacio.variables, m)},
        "n_valoraciones": int(pares["n_valoraciones"]), "n_pares": int(n_pares),
        # Proporción de comparaciones del usuario que la similitud ordena bien, antes y después de aprender
        "acierto_antes": float(np.mean(antes > 0)), "acierto_despues": float(np.mean(despues > 0)),
    }


# --------------------------------------------------------------------------
# Personalización de una búsqueda
# --------------------------------------------------------------------------
def construir_personalizacion(valoraciones_vigentes: pd.DataFrame | None, df: pd.DataFrame,
                              objetivo: pd.Series) -> Personalizacion:
    """Reúne lo aprendido que afecta a la búsqueda de sustitutos de `objetivo`."""
    p = Personalizacion()
    if valoraciones_vigentes is None or len(valoraciones_vigentes) == 0:
        return p
    grupo = objetivo["grupo_posicion"]
    del_grupo = valoraciones_vigentes[valoraciones_vigentes["grupo_posicion"] == grupo]
    p.n_grupo = int(len(del_grupo))

    aprendido = aprender_multiplicadores(del_grupo, df, grupo)
    if aprendido:
        p.multiplicadores = aprendido["multiplicadores"]
        p.aprendizaje = aprendido

    registros = set(df["registro_id"])
    de_referencia = del_grupo[del_grupo["objetivo_persona_id"] == objetivo["persona_id"]]
    for v in de_referencia.itertuples():
        estrellas = int(v.estrellas)
        p.valorados[v.candidato_persona_id] = estrellas
        if estrellas <= C.ESTRELLAS_NEGATIVA:
            p.descartados.add(v.candidato_persona_id)
        if v.candidato_registro_id not in registros:
            continue
        peso = abs(estrellas - 3) / 2.0
        if estrellas >= C.ESTRELLAS_POSITIVA:
            p.positivos.append((v.candidato_registro_id, peso))
        elif estrellas <= C.ESTRELLAS_NEGATIVA:
            p.negativos.append((v.candidato_registro_id, peso))
    return p
