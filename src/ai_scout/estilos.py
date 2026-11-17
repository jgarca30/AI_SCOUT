"""Estilos de juego: agrupamiento k-means y mapa PCA por grupo de posición.

Con las variables estandarizadas de cada grupo se entrena un k-means que reparte a los
jugadores en estilos y un PCA de dos componentes que permite dibujarlos en un mapa. 
El modelo se entrena con los jugadores que tienen los minutos mínimos;
 al resto se le asigna el estilo del centroide más cercano.

Los estilos se usan para describir a los jugadores, para filtrar candidatos del mismo
estilo que la referencia, para el mapa de la aplicación y para evaluar el recomendador.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .similitud import construir_espacio

MIN_JUGADORES_POR_ESTILO = 3  # con menos jugadores por grupo no se entrena el modelo


@dataclass
class EstilosGrupo:
    grupo: str
    variables: list[str]
    k: int
    centroides: np.ndarray            # k × n_variables, en unidades z
    nombres: list[str]                # nombre de cada estilo
    descripciones: list[str]
    tamanos: list[int]                # jugadores de la población de entrenamiento en cada estilo
    silueta: float | None
    componentes: np.ndarray           # 2 × n_variables (cargas del PCA)
    varianza_explicada: np.ndarray    # proporción de varianza de cada componente
    tabla: pd.DataFrame               # registro_id, estilo_id, estilo, pca_x, pca_y, en_entrenamiento

    def de_registro(self, registro_id: str) -> dict | None:
        fila = self.tabla[self.tabla["registro_id"] == registro_id]
        if fila.empty:
            return None
        f = fila.iloc[0]
        return {"id": int(f["estilo_id"]), "nombre": f["estilo"], "pca_x": float(f["pca_x"]),
                "pca_y": float(f["pca_y"])}


def _etiqueta(variable: str) -> str:
    return C.ETIQUETAS_CORTAS.get(variable, variable)


def _en_frase(variable: str) -> str:
    """Etiqueta para usar dentro de una frase: en minúsculas salvo las siglas (npxG, xAG)."""
    etiqueta = _etiqueta(variable).rstrip(".")
    return etiqueta if any(c.isupper() for c in etiqueta[1:]) else etiqueta.lower()


def nombrar_estilos(grupo: str, variables: list[str], centroides: np.ndarray) -> tuple[list[str], list[str]]:
    """Nombre y descripción de cada estilo a partir de su centroide (reglas deterministas).

    - Si alguna variable supera el umbral z, el estilo toma el arquetipo de la más destacada.
    - Si ninguna destaca: «Baja participación» cuando el centroide está claramente por debajo
      de la media y «Perfil equilibrado» en otro caso.
    """
    umbral = C.ESTILOS_UMBRAL_Z
    nombres, descripciones = [], []
    for centro in centroides:
        orden = np.argsort(-centro)
        altas = [variables[i] for i in orden if centro[i] >= umbral]
        bajas = [variables[i] for i in orden[::-1] if centro[i] <= -umbral]
        if altas:
            nombre = C.ARQUETIPOS.get(grupo, {}).get(altas[0], f"Destaca en {_en_frase(altas[0])}")
        elif centro.mean() <= -umbral / 2:
            nombre = "Baja participación con balón"
        else:
            nombre = "Perfil equilibrado"
        partes = []
        if altas:
            partes.append("Por encima de la media en " + ", ".join(_en_frase(v) for v in altas[:4]) + ".")
        if bajas:
            partes.append("Por debajo de la media en " + ", ".join(_en_frase(v) for v in bajas[:4]) + ".")
        if not partes:
            partes.append("Valores cercanos a la media del grupo en todas las variables.")
        nombres.append(nombre)
        descripciones.append(" ".join(partes))

    # Nombres repetidos: se distinguen con la segunda variable más destacada (o con un número)
    for nombre in set(nombres):
        repetidos = [i for i, n in enumerate(nombres) if n == nombre]
        if len(repetidos) < 2:
            continue
        for posicion, i in enumerate(repetidos, start=1):
            orden = np.argsort(-centroides[i])
            segunda = variables[orden[1]] if len(orden) > 1 and centroides[i][orden[1]] >= umbral else None
            nombres[i] = f"{nombre} ({_en_frase(segunda)})" if segunda else f"{nombre} {posicion}"
        if len({nombres[i] for i in repetidos}) < len(repetidos):
            for posicion, i in enumerate(repetidos, start=1):
                nombres[i] = f"{nombre} {posicion}"
    return nombres, descripciones


def calcular_estilos_grupo(df: pd.DataFrame, grupo: str, minutos_minimos: float = C.MINUTOS_MINIMOS_DEFECTO,
                           k: int | None = None) -> EstilosGrupo | None:
    """Entrena k-means y PCA para un grupo. Devuelve None si no hay jugadores suficientes."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    espacio = construir_espacio(df, grupo, minutos_minimos)
    poblacion = espacio.poblacion
    k = k or C.ESTILOS_K.get(grupo, 4)
    k = min(k, len(poblacion) // MIN_JUGADORES_POR_ESTILO)
    if k < 2 or len(espacio.variables) < 2:
        return None

    Z = espacio.z(poblacion)
    modelo = KMeans(n_clusters=k, n_init=10, random_state=C.ESTILOS_SEMILLA).fit(Z)
    pca = PCA(n_components=2, random_state=C.ESTILOS_SEMILLA).fit(Z)
    # Signo de cada componente fijado para que el resultado sea reproducible:
    # la carga de mayor valor absoluto queda siempre positiva.
    componentes = pca.components_.copy()
    for i in range(componentes.shape[0]):
        if componentes[i, np.argmax(np.abs(componentes[i]))] < 0:
            componentes[i] *= -1

    # Orden estable de los estilos: por la primera componente principal de su centroide
    orden = np.argsort(modelo.cluster_centers_ @ componentes[0])
    centroides = modelo.cluster_centers_[orden]
    reetiquetar = {int(antiguo): nuevo for nuevo, antiguo in enumerate(orden)}
    etiquetas_entrenamiento = np.array([reetiquetar[int(e)] for e in modelo.labels_])
    silueta = float(silhouette_score(Z, etiquetas_entrenamiento)) if len(set(etiquetas_entrenamiento)) > 1 else None
    nombres, descripciones = nombrar_estilos(grupo, espacio.variables, centroides)

    # Asignación de TODOS los registros válidos del grupo (también los de pocos minutos)
    todos = df[(df["grupo_posicion"] == grupo) & df["valido"]]
    Z_todos = espacio.z(todos)
    distancias = np.linalg.norm(Z_todos[:, None, :] - centroides[None, :, :], axis=2)
    estilo_id = distancias.argmin(axis=1)
    coordenadas = Z_todos @ componentes.T
    en_entrenamiento = set(poblacion["registro_id"])
    tabla = pd.DataFrame({
        "registro_id": todos["registro_id"].to_numpy(),
        "estilo_id": estilo_id,
        "estilo": [nombres[i] for i in estilo_id],
        "pca_x": coordenadas[:, 0],
        "pca_y": coordenadas[:, 1],
        "en_entrenamiento": [r in en_entrenamiento for r in todos["registro_id"]],
    })
    return EstilosGrupo(
        grupo=grupo, variables=espacio.variables, k=k, centroides=centroides, nombres=nombres,
        descripciones=descripciones, tamanos=[int((etiquetas_entrenamiento == i).sum()) for i in range(k)],
        silueta=silueta, componentes=componentes, varianza_explicada=pca.explained_variance_ratio_, tabla=tabla)


def calcular_estilos(df: pd.DataFrame, minutos_minimos: float = C.MINUTOS_MINIMOS_DEFECTO) -> dict[str, EstilosGrupo]:
    """Estilos de todos los grupos de posición (se omiten los grupos sin jugadores suficientes)."""
    estilos = {}
    for grupo in C.PERFILES:
        modelo = calcular_estilos_grupo(df, grupo, minutos_minimos)
        if modelo is not None:
            estilos[grupo] = modelo
    return estilos


def estilo_de(estilos: dict[str, EstilosGrupo] | None, grupo: str, registro_id: str) -> dict | None:
    if not estilos or grupo not in estilos:
        return None
    return estilos[grupo].de_registro(registro_id)


def tabla_estilos(estilos: dict[str, EstilosGrupo], df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de todos los jugadores con su estilo y su posición en el mapa (para la app y Power BI)."""
    partes = []
    for grupo, modelo in estilos.items():
        t = modelo.tabla.copy()
        t["grupo_posicion"] = grupo
        partes.append(t)
    if not partes:
        return pd.DataFrame(columns=["registro_id", "grupo_posicion", "estilo_id", "estilo", "pca_x", "pca_y",
                                     "en_entrenamiento"])
    tabla = pd.concat(partes, ignore_index=True)
    datos = df[["registro_id", "persona_id", "nombre", "equipo", "liga", "posicion_original", "edad_registrada",
                "minutos"]]
    return tabla.merge(datos, on="registro_id", how="left")


def eje_pca(modelo: EstilosGrupo, componente: int, n: int = 3) -> str:
    """Texto para el eje del mapa: variables con más carga en la componente y varianza explicada."""
    cargas = modelo.componentes[componente]
    orden = np.argsort(-np.abs(cargas))[:n]
    partes = [("+ " if cargas[i] > 0 else "− ") + _en_frase(modelo.variables[i]) for i in orden]
    return (f"Componente {componente + 1} ({modelo.varianza_explicada[componente] * 100:.0f} % de la varianza): "
            + ", ".join(partes))
