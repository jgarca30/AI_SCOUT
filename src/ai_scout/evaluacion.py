"""Evaluación del recomendador.

El sistema se evalúa desde varios ángulos complementarios:

1. Verdad-terreno aproximada. 
    (a) Autorrecuperación: un jugador que cambió de equipo a mitad de
    temporada aparece dos veces en el dataset; al buscar con uno de sus registros el otro debe
    salir arriba. 
    (b) Posición detallada: una etiqueta que la
    similitud no usa y con la que deberían coincidir los candidatos.
2. Estilos de juego: silueta del agrupamiento para distintos k y coherencia entre la similitud
   y los estilos.
3. Cobertura y diversidad: qué parte del catálogo llega a recomendarse, concentración de las
   recomendaciones y efecto del parámetro de diversidad .
4. Sensibilidad: cuánto cambian las recomendaciones al cambiar de rol o de minutos mínimos.
5. Aprendizaje con usuarios simulados: un usuario con preferencias ocultas valora candidatos
   y se mide si el sistema se acerca a su orden ideal.
6. Registro de uso real: validez y tiempos de los agentes LLM, diferencia entre el ranking
   numérico y el híbrido y valoraciones del usuario.
7. Agentes LLM en vivo: estabilidad entre repeticiones.

"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aprendizaje, estilos
from . import config as C
from .similitud import (construir_espacio, consulta_rocchio, cosenos, indice, pesos_de_rol, seleccionar_mmr,
                        vector_pesos)

K = 10  # tamaño de la lista evaluada (el Top 10 del Ojeador de datos)


# Utilidades
def poblacion_unica(df: pd.DataFrame, grupo: str, minutos: float = C.MINUTOS_MINIMOS_DEFECTO):
    """Espacio del grupo, población con un registro por persona (el de más minutos) y su matriz z."""
    espacio = construir_espacio(df, grupo, minutos)
    poblacion = espacio.poblacion.sort_values(["persona_id", "minutos", "registro_id"],
                                              ascending=[True, False, True]) \
        .drop_duplicates("persona_id").reset_index(drop=True)
    return espacio, poblacion, espacio.z(poblacion)


def _unitarios(Z: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    Zw = Z if w is None else Z * np.sqrt(w)
    normas = np.linalg.norm(Zw, axis=1)
    normas[normas == 0] = 1.0
    return Zw / normas[:, None]


def matriz_cosenos(Z: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Similitud coseno ponderada de todos con todos; la diagonal se pone a −inf (no se recomienda a uno mismo)."""
    U = _unitarios(Z, w)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    return S


def _top(S: np.ndarray, k: int = K) -> np.ndarray:
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


def gini(conteos: np.ndarray) -> float:
    """Índice de Gini: 0 = todos se recomiendan por igual; 1 = todo se concentra en uno."""
    x = np.sort(np.asarray(conteos, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1.0) / n)


# 1. Autorrecuperación (verdad-terreno: el mismo jugador en dos equipos)
def evaluar_autorrecuperacion(df: pd.DataFrame, minutos_minimos: int = 450) -> pd.DataFrame:
    """MRR y acierto@k al buscar el otro registro de un jugador que jugó en dos equipos.

    Se usa un umbral de minutos más bajo que el habitual para tener más casos; aun así son
    pocos, por lo que el resultado es orientativo. Se compara el coseno del sistema con la
    distancia euclídea y con el azar.
    """
    filas = []
    for grupo in C.PERFILES:
        espacio = construir_espacio(df, grupo, minutos_minimos)
        poblacion = espacio.poblacion.reset_index(drop=True)     # sin deduplicar: hacen falta los dos registros
        if len(poblacion) < 3 or not espacio.variables:
            continue
        Z = espacio.z(poblacion)
        personas = poblacion["persona_id"].to_numpy()
        veces = poblacion["persona_id"].value_counts()
        consultas = np.flatnonzero(poblacion["persona_id"].isin(set(veces[veces >= 2].index)).to_numpy())
        if len(consultas) == 0:
            continue
        distancias = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
        np.fill_diagonal(distancias, np.inf)
        medidas = {"Similitud coseno (la del sistema)": matriz_cosenos(Z), "Distancia euclídea": -distancias}
        n_candidatos = len(poblacion) - 1
        for medida, S in medidas.items():
            rangos = []
            for i in consultas:
                orden = np.argsort(-S[i], kind="stable")
                mismos = np.flatnonzero((personas[orden] == personas[i]) & (orden != i))
                rangos.append(int(mismos[0]) + 1)                 # posición del otro registro (1 = primero)
            rangos = np.array(rangos)
            filas.append({"grupo": grupo, "medida": medida, "consultas": len(rangos), "candidatos": n_candidatos,
                          "mrr": float(np.mean(1.0 / rangos)), "acierto_1": float(np.mean(rangos <= 1)),
                          "acierto_5": float(np.mean(rangos <= 5)), "acierto_10": float(np.mean(rangos <= 10)),
                          "rango_mediano": float(np.median(rangos))})
        armonico = float(np.sum(1.0 / np.arange(1, n_candidatos + 1)))
        filas.append({"grupo": grupo, "medida": "Azar (valor esperado)", "consultas": len(consultas),
                      "candidatos": n_candidatos, "mrr": armonico / n_candidatos, "acierto_1": 1 / n_candidatos,
                      "acierto_5": 5 / n_candidatos, "acierto_10": 10 / n_candidatos,
                      "rango_mediano": (n_candidatos + 1) / 2})
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    totales = []
    for medida, t in tabla.groupby("medida", sort=False):
        pesos = t["consultas"]
        totales.append({"grupo": "Todos", "medida": medida, "consultas": int(pesos.sum()),
                        "candidatos": int(t["candidatos"].sum()),
                        **{c: float(np.average(t[c], weights=pesos))
                           for c in ("mrr", "acierto_1", "acierto_5", "acierto_10", "rango_mediano")}})
    return pd.concat([tabla, pd.DataFrame(totales)], ignore_index=True)


def evaluar_posicion_detallada(df: pd.DataFrame) -> pd.DataFrame:
    """Coherencia con una etiqueta externa que la similitud no usa: la posición detallada.

    El grupo de posición solo mira la primera posición del jugador («MF» en «MF,FW»). Si la
    similitud capta el perfil real, los candidatos del Top 10 deberían compartir la posición
    completa de la referencia más a menudo que si se eligieran al azar dentro del grupo.
    """
    filas = []
    for grupo in C.PERFILES:
        _, poblacion, Z = poblacion_unica(df, grupo)
        if len(poblacion) <= K:
            continue
        posicion = poblacion["posicion_original"].to_numpy()
        vecinos = _top(matriz_cosenos(Z))
        misma = float((posicion[vecinos] == posicion[:, None]).mean())
        proporciones = poblacion["posicion_original"].value_counts(normalize=True)
        azar = float((proporciones ** 2).sum())
        filas.append({"grupo": grupo, "jugadores": len(poblacion), "posiciones_distintas": len(proporciones),
                      "misma_posicion_top10": misma, "esperado_por_azar": azar, "mejora_sobre_azar": misma / azar})
    return pd.DataFrame(filas)

# 2. Estilos de juego
def evaluar_estilos(df: pd.DataFrame, ks=range(2, 8)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(silueta e inercia para cada k, coherencia entre similitud y estilos)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    siluetas, coherencia = [], []
    for grupo in C.PERFILES:
        espacio = construir_espacio(df, grupo, C.MINUTOS_MINIMOS_DEFECTO)
        if len(espacio.poblacion) < 2 * max(ks):
            continue
        Z = espacio.z(espacio.poblacion)
        for k in ks:
            modelo = KMeans(n_clusters=k, n_init=10, random_state=C.ESTILOS_SEMILLA).fit(Z)
            siluetas.append({"grupo": grupo, "k": k, "silueta": float(silhouette_score(Z, modelo.labels_)),
                             "inercia": float(modelo.inertia_), "elegido": k == C.ESTILOS_K.get(grupo)})

        modelo = estilos.calcular_estilos_grupo(df, grupo)
        if modelo is None:
            continue
        _, poblacion, Zu = poblacion_unica(df, grupo)
        estilo = poblacion["registro_id"].map(dict(zip(modelo.tabla["registro_id"], modelo.tabla["estilo_id"]))).to_numpy()
        vecinos = _top(matriz_cosenos(Zu))
        mismo = float((estilo[vecinos] == estilo[:, None]).mean())
        proporciones = np.bincount(estilo.astype(int), minlength=modelo.k) / len(estilo)
        azar = float(np.sum(proporciones ** 2))
        coherencia.append({"grupo": grupo, "estilos": modelo.k, "silueta": modelo.silueta,
                           "varianza_mapa_2d": float(modelo.varianza_explicada.sum()),
                           "mismo_estilo_top10": mismo, "esperado_por_azar": azar, "mejora_sobre_azar": mismo / azar})
    return pd.DataFrame(siluetas), pd.DataFrame(coherencia)


# 3. Cobertura y diversidad
def evaluar_diversidad(df: pd.DataFrame, diversidades=(0.0, 0.3, 0.6)) -> pd.DataFrame:
    """Para cada nivel de diversidad: similitud media, diversidad dentro de la lista, cobertura y Gini.

    Cada jugador elegible hace de referencia una vez (sin filtros de liga ni de edad).
    """
    filas = []
    for grupo in C.PERFILES:
        _, poblacion, Z = poblacion_unica(df, grupo)
        if len(poblacion) <= K:
            continue
        S, U = matriz_cosenos(Z), _unitarios(Z)
        bolsa = np.argsort(-S, axis=1, kind="stable")[:, :max(C.MMR_BOLSA, K)]
        for d in diversidades:
            conteo = np.zeros(len(poblacion))
            similitud, intra = [], []
            for i in range(len(poblacion)):
                elegidos = bolsa[i][seleccionar_mmr(S[i, bolsa[i]], Z[bolsa[i]], K, d)]
                conteo[elegidos] += 1
                similitud.append(S[i, elegidos].mean())
                entre = U[elegidos] @ U[elegidos].T
                intra.append(1.0 - (entre.sum() - np.trace(entre)) / (K * (K - 1)))
            filas.append({"grupo": grupo, "diversidad": d, "jugadores": len(poblacion),
                          "indice_similitud_medio": indice(float(np.mean(similitud))),
                          "diversidad_intralista": float(np.mean(intra)),
                          "cobertura_catalogo": float((conteo > 0).mean()), "gini": gini(conteo)})
    return pd.DataFrame(filas)


# 4. Sensibilidad a los roles y a los minutos mínimos
def evaluar_sensibilidad(df: pd.DataFrame, minutos_alternativos: int = 900) -> pd.DataFrame:
    """Solapamiento medio del Top 10 al cambiar de rol y al cambiar el umbral de minutos."""
    filas = []
    for grupo in C.PERFILES:
        espacio, poblacion, Z = poblacion_unica(df, grupo)
        if len(poblacion) <= K:
            continue
        base = _top(matriz_cosenos(Z))
        for rol, definicion in C.ROLES[grupo].items():
            if rol == C.ROL_DEFECTO:
                continue
            con_rol = _top(matriz_cosenos(Z, vector_pesos(espacio.variables, pesos_de_rol(grupo, rol))))
            solape = np.mean([len(set(a) & set(b)) / K for a, b in zip(base, con_rol)])
            filas.append({"prueba": "Rol frente a equilibrado", "grupo": grupo, "detalle": definicion["nombre"],
                          "solapamiento_top10": float(solape), "candidatos_que_cambian": float(K * (1 - solape))})

        # Mismos candidatos (los que superan el umbral alto), escalados con cada población
        espacio_alto, poblacion_alta, Z_alto = poblacion_unica(df, grupo, minutos_alternativos)
        if len(poblacion_alta) <= K:
            continue
        comunes = poblacion.set_index("registro_id").reindex(poblacion_alta["registro_id"])
        if comunes["persona_id"].isna().any():
            continue
        top_bajo = _top(matriz_cosenos(espacio.z(comunes)))
        top_alto = _top(matriz_cosenos(Z_alto))
        solape = np.mean([len(set(a) & set(b)) / K for a, b in zip(top_bajo, top_alto)])
        filas.append({"prueba": "Escalado con otro umbral de minutos", "grupo": grupo,
                      "detalle": f"{C.MINUTOS_MINIMOS_DEFECTO} frente a {minutos_alternativos} minutos",
                      "solapamiento_top10": float(solape), "candidatos_que_cambian": float(K * (1 - solape))})
    return pd.DataFrame(filas)


# 5. Aprendizaje con usuarios simulados
def estrellas_simuladas(posicion_ideal: int) -> int:
    """Estrellas que da el usuario simulado según el puesto del candidato en SU orden ideal."""
    return 5 if posicion_ideal < 10 else 4 if posicion_ideal < 25 else 3 if posicion_ideal < 50 else \
        2 if posicion_ideal < 100 else 1


def _posiciones(similitudes: np.ndarray) -> np.ndarray:
    orden = np.argsort(-similitudes, kind="stable")
    posiciones = np.empty(len(orden), dtype=int)
    posiciones[orden] = np.arange(len(orden))
    return posiciones


def _ndcg(lista: np.ndarray, posicion_ideal: np.ndarray) -> float:
    ganancia = np.select([posicion_ideal[lista] < 10, posicion_ideal[lista] < 25, posicion_ideal[lista] < 50],
                         [3.0, 2.0, 1.0], 0.0)
    descuento = 1.0 / np.log2(np.arange(2, len(lista) + 2))
    return float((ganancia * descuento).sum() / (3.0 * descuento).sum())


ESCENARIOS_APRENDIZAJE = {
    "Valora los 10 candidatos": {"valorados": 10, "ruido": 0.0},
    "Valora solo el Top 3": {"valorados": 3, "ruido": 0.0},
    "Valora los 10, con un 20 % de valoraciones al azar": {"valorados": 10, "ruido": 0.2},
}


def simular_aprendizaje(df: pd.DataFrame, referencias_valoradas: int = 20, referencias_prueba: int = 40,
                        semillas=(0, 1), puntos=(0, 1, 3, 5, 10, 20), escenarios: dict | None = None) -> pd.DataFrame:
    """Curvas de aprendizaje de los pesos con usuarios simulados.

    El usuario simulado prefiere a los candidatos según la similitud ponderada con los pesos
    de un rol que el sistema desconoce (el sistema empieza con pesos iguales). En cada paso
    el sistema le enseña su Top 10 para una referencia nueva, el usuario lo valora y el
    sistema vuelve a aprender. La calidad se mide en referencias que el usuario no ha valorado.
    """
    escenarios = escenarios or ESCENARIOS_APRENDIZAJE
    filas = []
    for grupo in C.PERFILES:
        espacio, poblacion, Z = poblacion_unica(df, grupo)
        n, nv = Z.shape
        if n < referencias_valoradas + referencias_prueba + 100:
            continue
        for rol in C.ROLES[grupo]:
            if rol == C.ROL_DEFECTO:
                continue
            w_usuario = vector_pesos(espacio.variables, pesos_de_rol(grupo, rol))
            for semilla in semillas:
                rng_reparto = np.random.default_rng(semilla)
                orden = rng_reparto.permutation(n)
                entrenamiento = orden[:referencias_valoradas]
                prueba = orden[referencias_valoradas:referencias_valoradas + referencias_prueba]
                ideales = {}
                for i in prueba:
                    s = cosenos(Z[i], Z, w_usuario)
                    s[i] = -np.inf
                    ideales[i] = _posiciones(s)

                def calidad(w: np.ndarray) -> tuple[float, float]:
                    ndcg, precision = [], []
                    for i in prueba:
                        s = cosenos(Z[i], Z, w)
                        s[i] = -np.inf
                        lista = np.argsort(-s, kind="stable")[:K]
                        ndcg.append(_ndcg(lista, ideales[i]))
                        precision.append(float(np.mean(ideales[i][lista] < K)))
                    return float(np.mean(ndcg)), float(np.mean(precision))

                for escenario, opciones in escenarios.items():
                    rng = np.random.default_rng(1000 + semilla)
                    w_sistema = np.ones(nv)
                    R, A, B, omega = [], [], [], []
                    registrar = lambda paso, w: filas.append({                       # noqa: E731
                        "escenario": escenario, "grupo": grupo, "rol_del_usuario": C.ROLES[grupo][rol]["nombre"],
                        "semilla": semilla, "referencias_valoradas": paso,
                        **dict(zip(("ndcg_10", "precision_10"), calidad(w)))})
                    if 0 in puntos:
                        registrar(0, w_sistema)
                    for paso, i in enumerate(entrenamiento, start=1):
                        s_usuario = cosenos(Z[i], Z, w_usuario)
                        s_usuario[i] = -np.inf
                        ideal = _posiciones(s_usuario)
                        s_sistema = cosenos(Z[i], Z, w_sistema)
                        s_sistema[i] = -np.inf
                        mostrados = np.argsort(-s_sistema, kind="stable")[:K][:opciones["valorados"]]
                        estrellas = {}
                        for j in mostrados:
                            e = estrellas_simuladas(int(ideal[j]))
                            if opciones["ruido"] and rng.random() < opciones["ruido"]:
                                e = int(rng.integers(1, 6))
                            estrellas[int(j)] = e
                        for a, e_a in estrellas.items():
                            for b, e_b in estrellas.items():
                                if e_a > e_b:
                                    R.append(Z[i]), A.append(Z[a]), B.append(Z[b]), omega.append((e_a - e_b) / 4.0)
                        if omega:
                            pares = {"R": np.array(R), "A": np.array(A), "B": np.array(B),
                                     "W": np.ones((len(omega), nv)), "omega": np.array(omega)}
                            w_sistema = aprendizaje.ajustar_multiplicadores(pares)
                        if paso in puntos:
                            registrar(paso, w_sistema)
    return pd.DataFrame(filas)


def simular_rocchio(df: pd.DataFrame, referencias: int = 80, semilla: int = 0) -> pd.DataFrame:
    """Efecto de la realimentación de Rocchio en los 10 candidatos siguientes (colección residual).

    Dos tipos de usuario simulado:
    - otra importancia: quiere a alguien como la referencia, pero da otra importancia a las variables;
    - otro objetivo: quiere a alguien como la referencia pero con más de dos variables concretas.
    Tras valorar el Top 10 (o solo el Top 3) se compara qué proporción de los 10 candidatos
    siguientes está entre sus 30 preferidos, con y sin adaptar la consulta.
    """
    filas = []
    for tipo in ("Misma referencia, otra importancia de las variables",
                 "Como la referencia, pero con más de dos variables"):
        for valorados in (10, 3):
            sin, con = [], []
            for grupo in C.PERFILES:
                espacio, poblacion, Z = poblacion_unica(df, grupo)
                n, nv = Z.shape
                if n < referencias + 100:
                    continue
                rng = np.random.default_rng(semilla)
                elegidas = rng.permutation(n)[:referencias]
                for rol in C.ROLES[grupo]:
                    if rol == C.ROL_DEFECTO:
                        continue
                    otra_importancia = tipo.startswith("Misma")
                    w_usuario = vector_pesos(espacio.variables, pesos_de_rol(grupo, rol)) if otra_importancia \
                        else np.ones(nv)
                    for i in elegidas:
                        objetivo = Z[i].copy()
                        if not otra_importancia:
                            objetivo[rng.choice(nv, 2, replace=False)] += 1.0
                        s_usuario = cosenos(objetivo, Z, w_usuario)
                        s_usuario[i] = -np.inf
                        ideal = _posiciones(s_usuario)
                        s0 = cosenos(Z[i], Z)
                        s0[i] = -np.inf
                        mostrados = np.argsort(-s0, kind="stable")[:K]
                        estrellas = {int(j): estrellas_simuladas(int(ideal[j])) for j in mostrados[:valorados]}
                        positivos = [j for j, e in estrellas.items() if e >= C.ESTRELLAS_POSITIVA]
                        negativos = [j for j, e in estrellas.items() if e <= C.ESTRELLAS_NEGATIVA]
                        q = consulta_rocchio(
                            Z[i], Z[positivos] if positivos else None,
                            np.array([(estrellas[j] - 3) / 2 for j in positivos]) if positivos else None,
                            Z[negativos] if negativos else None,
                            np.array([(3 - estrellas[j]) / 2 for j in negativos]) if negativos else None)
                        s1 = cosenos(q, Z)
                        s1[i] = -np.inf
                        s0[mostrados] = -np.inf                      # se evalúa solo lo que aún no ha visto
                        s1[mostrados] = -np.inf
                        sin.append(np.mean(ideal[np.argsort(-s0, kind="stable")[:K]] < 30))
                        con.append(np.mean(ideal[np.argsort(-s1, kind="stable")[:K]] < 30))
            if sin:
                filas.append({"tipo_de_usuario": tipo, "candidatos_valorados": valorados, "busquedas": len(sin),
                              "relevantes_sin_realimentacion": float(np.mean(sin)),
                              "relevantes_con_rocchio": float(np.mean(con)),
                              "diferencia": float(np.mean(con) - np.mean(sin))})
    return pd.DataFrame(filas)

# 6. Registro de uso real (salidas/*.csv)
def _leer(directorio: Path, nombre: str) -> pd.DataFrame | None:
    ruta = directorio / f"{nombre}.csv"
    if not ruta.exists():
        return None
    try:
        tabla = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    except (OSError, ValueError):
        return None
    return tabla if len(tabla) else None


def resumen_del_uso(dir_salidas: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Indicadores calculados con las consultas reales guardadas en salidas/ (puede devolver tablas vacías)."""
    from scipy.stats import kendalltau

    directorio = Path(dir_salidas) if dir_salidas else C.DIR_SALIDAS
    tablas: dict[str, pd.DataFrame] = {}

    ejecuciones = _leer(directorio, "ejecuciones_agentes")
    if ejecuciones is not None:
        e = ejecuciones.copy()
        for col in ("duracion_ms", "tokens_entrada", "tokens_salida", "intento"):
            e[col] = pd.to_numeric(e[col], errors="coerce")
        reales = e[e["estado"] != "cache"]
        filas = []
        for agente, t in e.groupby("agente", sort=False):
            r = reales[reales["agente"] == agente]
            tareas = t.groupby("consulta_id")
            filas.append({
                "agente": agente, "tareas": int(tareas.ngroups), "llamadas_al_modelo": int(len(r)),
                "respuestas_de_cache": int((t["estado"] == "cache").sum()),
                "validas_al_primer_intento": float((r[r["intento"] == 1]["estado"] == "ok").mean()) if len(r) else np.nan,
                "tareas_resueltas": float(tareas["estado"].apply(lambda s: s.isin(["ok", "cache"]).any()).mean()),
                "duracion_media_s": float(r["duracion_ms"].mean() / 1000) if len(r) else np.nan,
                "tokens_medios": float((r["tokens_entrada"].fillna(0) + r["tokens_salida"].fillna(0)).mean())
                if len(r) else np.nan,
            })
        tablas["uso_agentes"] = pd.DataFrame(filas)
        errores = e[~e["estado"].isin(["ok", "cache"])]
        if len(errores):
            tablas["uso_errores"] = errores.groupby(["agente", "estado", "codigo_error"]).size() \
                .reset_index(name="veces").sort_values("veces", ascending=False)

    consultas = _leer(directorio, "consultas")
    if consultas is not None:
        c = consultas.copy()
        c["duracion_s"] = pd.to_numeric(c["duracion_total_ms"], errors="coerce") / 1000
        tablas["uso_consultas"] = c.groupby(["estado_consulta", "modo_ranking"]).agg(
            consultas=("consulta_id", "count"), duracion_media_s=("duracion_s", "mean")).reset_index()

    resultados = _leer(directorio, "resultados")
    if resultados is not None and consultas is not None:
        hibridas = set(consultas.loc[consultas["modo_ranking"] == "hibrido", "consulta_id"])
        r = resultados[resultados["consulta_id"].isin(hibridas)].copy()
        for col in ("ranking_preliminar", "ranking_final"):
            r[col] = pd.to_numeric(r[col], errors="coerce")
        filas = []
        for consulta_id, t in r.groupby("consulta_id"):
            if len(t) < 3:
                continue
            tau = kendalltau(t["ranking_preliminar"], t["ranking_final"]).statistic
            top_numerico = set(t.sort_values("ranking_preliminar").head(C.TOP_FINAL)["candidato_registro_id"])
            top_hibrido = set(t.sort_values("ranking_final").head(C.TOP_FINAL)["candidato_registro_id"])
            filas.append({"tau": tau, "cambio_medio": float((t["ranking_final"] - t["ranking_preliminar"]).abs().mean()),
                          "top3_cambia": top_numerico != top_hibrido,
                          "top3_comun": len(top_numerico & top_hibrido) / C.TOP_FINAL})
        if filas:
            t = pd.DataFrame(filas)
            tablas["uso_numerico_vs_hibrido"] = pd.DataFrame([{
                "consultas_hibridas": len(t), "tau_de_kendall_medio": float(t["tau"].mean()),
                "puestos_que_se_mueve_un_candidato": float(t["cambio_medio"].mean()),
                "consultas_con_otro_top3": float(t["top3_cambia"].mean()),
                "top3_en_comun": float(t["top3_comun"].mean())}])

    valoraciones = _leer(directorio, "valoraciones")
    if valoraciones is not None:
        v = valoraciones[valoraciones["vigente"] == "true"].copy()
        if len(v):
            v["estrellas"] = pd.to_numeric(v["estrellas"], errors="coerce")
            v["ranking_final"] = pd.to_numeric(v["ranking_final"], errors="coerce")
            v["tramo"] = np.where(v["ranking_final"] <= C.TOP_FINAL, f"Top {C.TOP_FINAL}", "Resto de la preselección")
            tablas["uso_valoraciones"] = v.groupby("tramo").agg(
                valoraciones=("estrellas", "count"), estrellas_medias=("estrellas", "mean"),
                positivas=("estrellas", lambda s: float((s >= C.ESTRELLAS_POSITIVA).mean())),
                negativas=("estrellas", lambda s: float((s <= C.ESTRELLAS_NEGATIVA).mean()))).reset_index()
    return tablas


# 7. Agentes LLM en vivo (opcional: consume cupo del proveedor)
def evaluar_llm(df: pd.DataFrame, dataset_version: str, proveedores: dict, n_consultas: int = 3,
                repeticiones: int = 2, modelos_estilos: dict | None = None, semilla: int = 0,
                avisar=None) -> dict[str, pd.DataFrame]:
    """Repite consultas reales sin caché para medir validez, tiempos y estabilidad de los agentes LLM.

    `proveedores` es un diccionario {rol: proveedor} con las claves analista, director y critico.
    No se guarda nada en salidas/ ni se usan las valoraciones del usuario.
    """
    from .modelos import Consulta
    from .orquestador import ejecutar_consulta

    elegibles = df[df["valido"] & ~df["es_portero"] & (df["minutos"] >= 1500)].sort_values("registro_id")
    por_grupo = max(1, int(np.ceil(n_consultas / len(C.PERFILES))))
    rng = np.random.default_rng(semilla)
    referencias = []
    for grupo in C.PERFILES:
        del_grupo = elegibles[elegibles["grupo_posicion"] == grupo]
        referencias += list(del_grupo.iloc[rng.permutation(len(del_grupo))[:por_grupo]]["registro_id"])
    referencias = referencias[:n_consultas]

    ejecuciones, corridas = [], []
    for registro_id in referencias:
        nombre = df.loc[df["registro_id"] == registro_id, "nombre"].iloc[0]
        for repeticion in range(1, repeticiones + 1):
            if avisar:
                avisar(f"{nombre}: repetición {repeticion} de {repeticiones}")
            inicio = time.perf_counter()
            r = ejecutar_consulta(Consulta(registro_id, usar_aprendizaje=False), df, dataset_version,
                                  proveedores.get("analista"), proveedores.get("director"), cache=None,
                                  exportar=False, proveedor_critico=proveedores.get("critico"),
                                  estilos=modelos_estilos)
            for e in r["ejecuciones"]:
                ejecuciones.append({**e, "referencia": nombre, "repeticion": repeticion})
            corridas.append({
                "referencia": nombre, "repeticion": repeticion, "modo_ranking": r["modo_ranking"],
                "duracion_s": time.perf_counter() - inicio,
                "revision": (r.get("revision") or {}).get("estado"),
                "top3": tuple(c["registro_id"] for c in r["top_final"]),
                "cualitativo": {c["registro_id"]: c["score_cualitativo"] for c in r["ranking"]},
                "movimiento": float(np.mean([abs(c["ranking_final"] - c["ranking_preliminar"]) for c in r["ranking"]])),
            })

    tablas: dict[str, pd.DataFrame] = {}
    if ejecuciones:
        e = pd.DataFrame(ejecuciones)
        tablas["llm_agentes"] = e.groupby("agente", sort=False).apply(lambda t: pd.Series({
            "llamadas": len(t), "validas_al_primer_intento": float((t[t["intento"] == 1]["estado"] == "ok").mean()),
            "reintentos": int((t["intento"] > 1).sum()),
            "errores_de_api": int((t["estado"] == "error_api").sum()),
            "duracion_media_s": float(t["duracion_ms"].mean() / 1000),
            "tokens_medios": float((t["tokens_entrada"].fillna(0) + t["tokens_salida"].fillna(0)).mean())}),
            include_groups=False).reset_index()
    c = pd.DataFrame(corridas)
    estabilidad = []
    for nombre, t in c.groupby("referencia", sort=False):
        tops = list(t["top3"])
        pares = [(a, b) for n, a in enumerate(tops) for b in tops[n + 1:]]
        comunes = [len(set(a) & set(b)) / max(len(a), 1) for a, b in pares]
        puntuaciones = pd.DataFrame(list(t["cualitativo"])).apply(pd.to_numeric, errors="coerce")
        estabilidad.append({
            "referencia": nombre, "repeticiones": len(t),
            "consultas_hibridas": float((t["modo_ranking"] == "hibrido").mean()),
            "top3_en_comun_entre_repeticiones": float(np.mean(comunes)) if comunes else np.nan,
            "mismo_top3": float(np.mean([a == b for a, b in pares])) if pares else np.nan,
            "desviacion_valoracion_ia": float(puntuaciones.std(ddof=0).mean()) if len(puntuaciones) > 1 else np.nan,
            "puestos_que_mueve_la_ia": float(t["movimiento"].mean()), "duracion_media_s": float(t["duracion_s"].mean()),
        })
    tablas["llm_estabilidad"] = pd.DataFrame(estabilidad)
    return tablas


# Ejecución completa e informe
def evaluar_todo(df: pd.DataFrame, dir_salidas: Path | str | None = None, rapido: bool = False,
                 avisar=None) -> dict[str, pd.DataFrame]:
    """Ejecuta todas las evaluaciones que no necesitan un modelo de lenguaje."""
    def paso(texto: str) -> None:
        if avisar:
            avisar(texto)

    tablas: dict[str, pd.DataFrame] = {}
    paso("Autorrecuperación (el mismo jugador en dos equipos)")
    tablas["autorrecuperacion"] = evaluar_autorrecuperacion(df)
    tablas["posicion_detallada"] = evaluar_posicion_detallada(df)
    paso("Estilos de juego")
    tablas["estilos_siluetas"], tablas["estilos_coherencia"] = evaluar_estilos(df)
    paso("Cobertura y diversidad")
    tablas["diversidad"] = evaluar_diversidad(df)
    paso("Sensibilidad a roles y minutos")
    tablas["sensibilidad"] = evaluar_sensibilidad(df)
    paso("Aprendizaje con usuarios simulados")
    opciones = dict(referencias_valoradas=10, referencias_prueba=20, semillas=(0,), puntos=(0, 1, 3, 5, 10)) \
        if rapido else {}
    tablas["aprendizaje_detalle"] = simular_aprendizaje(df, **opciones)
    if len(tablas["aprendizaje_detalle"]):
        tablas["aprendizaje"] = tablas["aprendizaje_detalle"].groupby(
            ["escenario", "referencias_valoradas"], sort=False)[["ndcg_10", "precision_10"]].mean().reset_index()
    paso("Realimentación de Rocchio con usuarios simulados")
    tablas["rocchio"] = simular_rocchio(df, referencias=30 if rapido else 80)
    paso("Registro de uso real")
    tablas.update(resumen_del_uso(dir_salidas))
    return tablas


TITULOS = {
    "autorrecuperacion": ("1. Autorrecuperación: el mismo jugador en dos equipos",
                          "Se busca con un registro de un jugador que cambió de equipo y se mira en qué puesto "
                          "aparece su otro registro. MRR y aciertos altos indican que la similitud reconoce al "
                          "mismo jugador. Hay pocos casos (umbral de 450 minutos), así que es orientativo."),
    "posicion_detallada": ("1b. Coherencia con la posición detallada (etiqueta que la similitud no usa)",
                           "Proporción de candidatos del Top 10 con la misma posición completa que la referencia "
                           "(por ejemplo «MF,FW»), frente a la esperada al azar dentro del grupo."),
    "estilos_siluetas": ("2a. Estilos de juego: silueta e inercia según el número de grupos",
                         "La silueta es baja en todos los casos: los estilos son una partición orientativa de un "
                         "continuo. Se eligió el k marcado por su interpretabilidad."),
    "estilos_coherencia": ("2b. Coherencia entre la similitud y los estilos",
                           "Proporción de candidatos del Top 10 que comparten estilo con la referencia, frente a "
                           "la esperada si se eligieran al azar."),
    "diversidad": ("3. Cobertura del catálogo y diversidad",
                   "Cada jugador hace de referencia una vez. Cobertura = jugadores recomendados al menos una "
                   "vez; Gini = concentración de las recomendaciones; diversidad intralista = 1 − similitud "
                   "media entre los 10 candidatos."),
    "sensibilidad": ("4. Sensibilidad a los roles y a los minutos mínimos",
                     "Solapamiento medio del Top 10 respecto a la configuración por defecto."),
    "aprendizaje": ("5a. Aprendizaje de los pesos con usuarios simulados",
                    "nDCG@10 y precisión@10 respecto al orden ideal de un usuario cuyas preferencias (los pesos "
                    "de un rol) el sistema desconoce, medido en referencias que no ha valorado. Media de todos "
                    "los roles."),
    "rocchio": ("5b. Realimentación de Rocchio con usuarios simulados",
                "Proporción de los 10 candidatos siguientes (aún no vistos) que están entre los 30 preferidos "
                "del usuario, con y sin adaptar la consulta."),
    "uso_agentes": ("6a. Uso real: agentes LLM", "Calculado con las consultas guardadas en salidas/."),
    "uso_errores": ("6b. Uso real: fallos de los agentes LLM", ""),
    "uso_consultas": ("6c. Uso real: consultas por estado y modo de ranking", ""),
    "uso_numerico_vs_hibrido": ("6d. Uso real: ranking numérico frente a híbrido",
                                "Tau de Kendall = 1 significa que el análisis del LLM no cambia el orden."),
    "uso_valoraciones": ("6e. Uso real: valoraciones del usuario",
                         "Si los candidatos del Top 3 reciben más estrellas que el resto, el ranking coincide "
                         "con el criterio del usuario."),
    "llm_agentes": ("7a. Agentes LLM en vivo: validez y tiempos", "Consultas repetidas sin caché."),
    "llm_estabilidad": ("7b. Agentes LLM en vivo: estabilidad entre repeticiones", ""),
}
ORDEN_INFORME = list(TITULOS)


def _celda(valor) -> str:
    if isinstance(valor, (bool, np.bool_)):
        return "sí" if valor else "no"
    if isinstance(valor, (float, np.floating)):
        return "—" if np.isnan(valor) else f"{valor:.3f}".replace(".", ",")
    return str(valor)


def tabla_markdown(tabla: pd.DataFrame) -> str:
    cabecera = "| " + " | ".join(str(c).replace("_", " ") for c in tabla.columns) + " |"
    separador = "|" + "|".join(" --- " for _ in tabla.columns) + "|"
    filas = ["| " + " | ".join(_celda(v) for v in fila) + " |" for fila in tabla.itertuples(index=False)]
    return "\n".join([cabecera, separador] + filas)


def escribir_resultados(tablas: dict[str, pd.DataFrame], dir_salidas: Path | str | None = None,
                        dataset_version: str = "") -> Path:
    """Guarda cada tabla en CSV y un informe en Markdown. Devuelve la carpeta."""
    carpeta = (Path(dir_salidas) if dir_salidas else C.DIR_SALIDAS) / C.SUBDIR_EVALUACION
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items():
        if tabla is not None and len(tabla):
            tabla.to_csv(carpeta / f"{nombre}.csv", index=False, encoding="utf-8")
    fecha = time.strftime("%Y-%m-%d %H:%M")
    lineas = ["# Evaluación de AI Scout", "",
              f"Generado el {fecha} · dataset {dataset_version or 'sin indicar'} · perfiles {C.PERFIL_VERSION} · "
              f"prompts {C.PROMPT_VERSION}", ""]
    for nombre in ORDEN_INFORME:
        tabla = tablas.get(nombre)
        if tabla is None or len(tabla) == 0:
            continue
        titulo, explicacion = TITULOS[nombre]
        lineas += [f"## {titulo}", ""]
        if explicacion:
            lineas += [explicacion, ""]
        lineas += [tabla_markdown(tabla), ""]
    (carpeta / "informe.md").write_text("\n".join(lineas), encoding="utf-8")
    return carpeta


def leer_resultados(dir_salidas: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Tablas de la última evaluación guardada (diccionario vacío si no hay ninguna)."""
    carpeta = (Path(dir_salidas) if dir_salidas else C.DIR_SALIDAS) / C.SUBDIR_EVALUACION
    tablas = {}
    for ruta in sorted(carpeta.glob("*.csv")) if carpeta.exists() else []:
        try:
            tablas[ruta.stem] = pd.read_csv(ruta)
        except (OSError, ValueError):
            continue
    return tablas
