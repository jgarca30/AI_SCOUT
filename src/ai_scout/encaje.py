"""Encaje en la plantilla de destino («química») y agente Analista de plantilla.

El dataset no tiene datos directos de química (minutos compartidos, redes de pase…), así
que el encaje se aproxima con cuatro indicadores calculados por código (0-100):

- estilo_equipo: parecido entre el juego del equipo actual del candidato y el del club de
  destino (perfil del equipo = estadísticas por 90 de toda su plantilla; coseno entre clubes).
- complemento: si el candidato cubre el hueco de la referencia (mismo estilo de juego) o, si
  no, cuánto escasea su estilo entre los compañeros de línea que quedan.
- compatriotas: jugadores de la plantilla de destino con su misma nacionalidad (compañeros
  de selección potenciales, idioma y adaptación).
- liga: si ya juega en la liga del club de destino.

El Analista de plantilla (LLM) recibe esos indicadores y la descripción de la plantilla,
puntúa el encaje de cada candidato de 0 a 4 y lo justifica. La nota de plantilla es la media
entre los indicadores y esa valoración; si el LLM falla, se usan solo los indicadores.
La nota pondera la puntuación final del Top 10.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .analista import _textos_con_terminos_prohibidos
from .director import cifras_no_respaldadas
from .redaccion import pulir

VARIABLES_EQUIPO = [f"{c}_p90" for c in C.COLUMNAS_P90] + list(C.COLUMNAS_PORCENTAJE.values())
PUNTOS_COMPATRIOTAS = {0: 0.0, 1: 60.0, 2: 85.0}   # 3 o más: 100
MINUTOS_COMPANERO = 300                            # minutos para contar como miembro habitual de la plantilla


def clubes(df: pd.DataFrame) -> list[str]:
    return sorted(df.loc[df["valido"] & ~df["es_portero"], "equipo"].unique())


def perfiles_de_equipo(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas por 90 minutos de cada club (suma de su plantilla de campo), estandarizadas entre clubes."""
    campo = df[df["valido"] & ~df["es_portero"]]
    filas = {}
    for equipo, t in campo.groupby("equipo"):
        minutos = t["minutos"].sum()
        fila = {f"{c}_p90": t[f"{c}_total"].sum() * 90.0 / minutos for c in C.COLUMNAS_P90}
        for col in C.COLUMNAS_PORCENTAJE.values():
            validos = t[col].notna()
            fila[col] = float(np.average(t.loc[validos, col], weights=t.loc[validos, "minutos"])) if validos.any() \
                else np.nan
        filas[equipo] = fila
    perfiles = pd.DataFrame.from_dict(filas, orient="index")[VARIABLES_EQUIPO]
    desviaciones = perfiles.std(ddof=0).replace(0, np.nan)
    return ((perfiles - perfiles.mean()) / desviaciones).fillna(0.0)


def _parecido_de_equipos(perfiles: pd.DataFrame, a: str, b: str) -> float | None:
    if a not in perfiles.index or b not in perfiles.index:
        return None
    if a == b:
        return 100.0
    u, v = perfiles.loc[a].to_numpy(), perfiles.loc[b].to_numpy()
    normas = float(np.linalg.norm(u) * np.linalg.norm(v))
    return 50.0 if normas == 0 else float(100.0 * (np.clip(u @ v / normas, -1, 1) + 1) / 2)


def evaluar(recuperacion: dict, df: pd.DataFrame, equipo_destino: str, modelos_estilos: dict | None = None) -> dict:
    """Indicadores de encaje de cada candidato en la plantilla del club de destino."""
    objetivo, grupo = recuperacion["objetivo"], recuperacion["grupo"]
    campo = df[df["valido"] & ~df["es_portero"]]
    plantilla = campo[(campo["equipo"] == equipo_destino) & (campo["persona_id"] != objetivo["persona_id"])]
    habituales = plantilla[plantilla["minutos"] >= MINUTOS_COMPANERO]
    liga_destino = campo.loc[campo["equipo"] == equipo_destino, "liga"].mode()
    liga_destino = liga_destino.iloc[0] if len(liga_destino) else None
    perfiles = perfiles_de_equipo(df)

    # Reparto de estilos entre los compañeros de línea que quedan (ponderado por minutos)
    reparto: dict[str, float] = {}
    modelo = (modelos_estilos or {}).get(grupo)
    if modelo is not None:
        estilo_de = dict(zip(modelo.tabla["registro_id"], modelo.tabla["estilo"]))
        linea = plantilla[plantilla["grupo_posicion"] == grupo]
        minutos_linea = linea["minutos"].sum()
        for estilo, t in linea.assign(estilo=linea["registro_id"].map(estilo_de)).groupby("estilo"):
            reparto[estilo] = float(t["minutos"].sum() / minutos_linea) if minutos_linea else 0.0

    candidatos = {}
    for c in recuperacion["candidatos"]:
        indicadores: dict[str, float | None] = {}
        indicadores["estilo_equipo"] = _parecido_de_equipos(perfiles, c["equipo"], equipo_destino)
        if modelo is not None and c.get("estilo"):
            indicadores["complemento"] = 100.0 if c["estilo"] == objetivo.get("estilo") \
                else 100.0 * (1.0 - reparto.get(c["estilo"], 0.0))
        else:
            indicadores["complemento"] = None
        paisanos = habituales[(habituales["nacionalidad"] == c["nacionalidad"])
                              & (habituales["persona_id"] != c["persona_id"])] if c.get("nacionalidad") else habituales.iloc[:0]
        n = int(paisanos["persona_id"].nunique())
        indicadores["compatriotas"] = PUNTOS_COMPATRIOTAS.get(n, 100.0)
        indicadores["liga"] = None if liga_destino is None else (100.0 if c["liga"] == liga_destino else 0.0)
        disponibles = {k: v for k, v in indicadores.items() if v is not None}
        pesos = {k: C.PESOS_ENCAJE[k] for k in disponibles}
        nota = sum(disponibles[k] * pesos[k] for k in disponibles) / sum(pesos.values()) if disponibles else 50.0
        candidatos[c["registro_id"]] = {
            "indicadores": {k: (None if v is None else round(v, 1)) for k, v in indicadores.items()},
            "score_indicadores": float(nota), "n_compatriotas": n,
            "compatriotas": sorted(paisanos.sort_values("minutos", ascending=False)["nombre"].unique()[:5]),
            "ya_en_el_club": c["equipo"] == equipo_destino,
        }

    return {
        "equipo": equipo_destino, "liga": liga_destino, "jugadores": int(plantilla["persona_id"].nunique()),
        "edad_media": None if plantilla["edad_registrada"].isna().all() else round(float(plantilla["edad_registrada"].mean()), 1),
        "nacionalidades": {k: int(v) for k, v in habituales.drop_duplicates("persona_id")["nacionalidad"]
                           .value_counts().head(6).items()},
        "estilos_de_la_linea": {k: round(v, 2) for k, v in sorted(reparto.items(), key=lambda kv: -kv[1])},
        "estilo_que_deja_la_referencia": objetivo.get("estilo"),
        "referencia_en_el_club": objetivo["equipo"] == equipo_destino,
        "candidatos": candidatos,
    }


# --------------------------------------------------------------------------
# Agente Analista de plantilla (LLM)
# --------------------------------------------------------------------------
SISTEMA = f"""Eres el Analista de plantilla de AI Scout, un recomendador académico de futbolistas.
Un club busca un sustituto para un jugador de referencia. Recibes la descripción de la PLANTILLA de destino y,
para cada candidato, unos INDICADORES de encaje (de 0 a 100) calculados con los datos:
- estilo_equipo: parecido entre el juego de su equipo actual y el del club de destino.
- complemento: 100 si tiene el mismo estilo de juego que la referencia (cubre su hueco); si no, es más alto
  cuanto más escasea su estilo entre los compañeros de línea que quedan.
- compatriotas: jugadores de la plantilla de destino con su misma nacionalidad (0, 60, 85 o 100).
- liga: 100 si ya juega en la liga del club de destino.
Tu tarea es valorar cuánto encajaría cada candidato en esa plantilla.

Reglas obligatorias:
1. Usa EXCLUSIVAMENTE los datos recibidos. No uses conocimiento previo sobre jugadores, clubes ni entrenadores.
2. "encaje": entero de 0 a 4 (0 = no encaja, 1 = encaje bajo, 2 = parcial, 3 = alto, 4 = muy alto), coherente
   con los indicadores.
3. "comentario": una o dos frases que expliquen el encaje a partir de los indicadores, con palabras
   corrientes («el juego de su equipo se parece al del club», «cubre el hueco de la referencia», «tiene
   compatriotas en la plantilla», «ya conoce la liga»). No escribas los nombres internos de los indicadores
   (estilo_equipo, score_indicadores…). Si citas una cifra, cópiala tal como aparece en los datos.
   Formula las conclusiones como hipótesis, no como hechos.
4. No hables de vestuario, carácter, idioma, salarios, precio, lesiones ni del entrenador: no están en los datos.
5. Incluye exactamente a todos los candidatos recibidos, identificados por "candidato_registro_id".
6. "diagnostico_plantilla": dos o tres frases sobre qué necesita la plantilla al perder a la referencia.
7. Responde ÚNICAMENTE con un objeto JSON con este formato:
{{"diagnostico_plantilla": "…", "candidatos": [{{"candidato_registro_id": "R0000", "encaje": 0-4,
  "comentario": "…"}}]}}
Escribe en español correcto, con todas sus tildes y la letra ñ."""

ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "diagnostico_plantilla": {"type": "string"},
        "candidatos": {"type": "array", "items": {
            "type": "object",
            "properties": {"candidato_registro_id": {"type": "string"},
                           "encaje": {"type": "integer", "minimum": 0, "maximum": 4},
                           "comentario": {"type": "string"}},
            "required": ["candidato_registro_id", "encaje", "comentario"]}},
    },
    "required": ["diagnostico_plantilla", "candidatos"],
}


def construir_payload(recuperacion: dict, evaluacion: dict) -> dict:
    objetivo = recuperacion["objetivo"]
    return {
        "plantilla_destino": {k: evaluacion[k] for k in ("equipo", "liga", "jugadores", "edad_media", "nacionalidades",
                                                         "estilos_de_la_linea", "estilo_que_deja_la_referencia")},
        "nota": "estilos_de_la_linea: reparto de minutos por estilo de juego entre los compañeros de la misma "
                "línea, sin contar a la referencia (de 0 a 1).",
        "referencia": {"posicion": objetivo["posicion_original"], "estilo": objetivo.get("estilo"),
                       "pertenece_al_club_de_destino": evaluacion["referencia_en_el_club"]},
        "candidatos": [{
            "candidato_registro_id": c["registro_id"], "estilo": c.get("estilo"), "liga_actual": c["liga"],
            "edad_registrada": c["edad_registrada"],
            "indicadores": evaluacion["candidatos"][c["registro_id"]]["indicadores"],
            "n_compatriotas": evaluacion["candidatos"][c["registro_id"]]["n_compatriotas"],
            "score_indicadores": round(evaluacion["candidatos"][c["registro_id"]]["score_indicadores"], 1),
        } for c in recuperacion["candidatos"]],
    }


def construir_mensaje(payload: dict) -> str:
    return ("Valora el encaje de estos candidatos en la plantilla de destino.\n\nDATOS:\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=1) + "\n```")


_CLAVES = {"estilo_equipo": "parecido con el juego del equipo", "score_indicadores": "nota de los indicadores",
           "n_compatriotas": "número de compatriotas", "estilos_de_la_linea": "estilos de la línea"}


def _sin_claves(texto: str) -> str:
    """Sustituye los nombres internos de los indicadores por palabras corrientes."""
    for clave, nombre in _CLAVES.items():
        texto = texto.replace(clave, nombre)
    return texto


def validar_respuesta(datos: dict, payload: dict) -> tuple[dict | None, list[str]]:
    errores: list[str] = []
    lista = datos.get("candidatos")
    if not isinstance(lista, list):
        return None, ["Falta la lista 'candidatos'."]
    esperados = [c["candidato_registro_id"] for c in payload["candidatos"]]
    valoraciones: dict[str, dict] = {}
    for item in lista:
        if not isinstance(item, dict):
            errores.append("Un elemento de 'candidatos' no es un objeto.")
            continue
        cid = item.get("candidato_registro_id")
        if cid not in esperados:
            errores.append(f"Candidato no solicitado: {cid!r}.")
            continue
        if cid in valoraciones:
            errores.append(f"Candidato repetido: {cid}.")
            continue
        encaje, comentario = item.get("encaje"), item.get("comentario")
        if not isinstance(encaje, int) or isinstance(encaje, bool) or not 0 <= encaje <= 4:
            errores.append(f"{cid}: 'encaje' debe ser un entero de 0 a 4.")
            continue
        if not isinstance(comentario, str) or not comentario.strip():
            errores.append(f"{cid}: falta 'comentario'.")
            continue
        if _textos_con_terminos_prohibidos([comentario]):
            errores.append(f"{cid}: el comentario afirma aspectos que no están en los datos.")
        malas = cifras_no_respaldadas([comentario], payload)
        if malas:
            errores.append(f"{cid}: cifras que no aparecen en los datos: {', '.join(str(m) for m in malas[:5])}.")
        valoraciones[cid] = {"encaje": encaje, "comentario": pulir(_sin_claves(comentario.strip()))}
    faltan = [c for c in esperados if c not in valoraciones]
    if faltan:
        errores.append(f"Faltan candidatos: {', '.join(faltan)}.")
    diagnostico = datos.get("diagnostico_plantilla", "")
    if not isinstance(diagnostico, str):
        errores.append("'diagnostico_plantilla' debe ser un texto.")
        diagnostico = ""
    if cifras_no_respaldadas([diagnostico], payload):
        errores.append("diagnostico_plantilla: cita cifras que no aparecen en los datos.")
    if errores:
        return None, errores
    return {"diagnostico": pulir(_sin_claves(diagnostico)), "candidatos": valoraciones}, []


# --------------------------------------------------------------------------
# Nota de plantilla y ponderación del ranking
# --------------------------------------------------------------------------
def comentario_por_indicadores(datos: dict, equipo: str) -> str:
    """Comentario determinista cuando no hay valoración del LLM."""
    ind = datos["indicadores"]
    partes = []
    if ind.get("estilo_equipo") is not None:
        partes.append(f"su equipo actual juega de forma {'parecida' if ind['estilo_equipo'] >= 60 else 'distinta'} "
                      f"a {equipo}")
    if ind.get("complemento") is not None:
        partes.append("cubre el estilo que deja la referencia" if ind["complemento"] >= 100
                      else "aporta un estilo poco presente en la línea" if ind["complemento"] >= 70
                      else "su estilo ya abunda en la línea")
    n = datos["n_compatriotas"]
    partes.append("no tiene compatriotas en la plantilla" if n == 0 else
                  f"tiene {n} compatriota{'s' if n > 1 else ''} en la plantilla")
    if ind.get("liga") is not None:
        partes.append("ya juega en la misma liga" if ind["liga"] else "viene de otra liga")
    return ("Ya pertenece al club. " if datos["ya_en_el_club"] else "") + (lambda t: t[:1].upper() + t[1:])("; ".join(partes)) + "."


def ponderar(ranking: list[dict], evaluacion: dict, valoracion_llm: dict | None, peso: float = None) -> list[dict]:
    """Añade la nota de plantilla a cada candidato, recalcula la puntuación final y reordena."""
    peso = C.PESO_PLANTILLA if peso is None else peso
    for c in ranking:
        datos = evaluacion["candidatos"][c["registro_id"]]
        llm = (valoracion_llm or {}).get("candidatos", {}).get(c["registro_id"])
        nota = datos["score_indicadores"] if llm is None else (datos["score_indicadores"] + 25.0 * llm["encaje"]) / 2.0
        c["score_sin_plantilla"] = c["score_final"]
        c["score_plantilla"] = float(nota)
        c["score_final"] = (1.0 - peso) * c["score_final"] + peso * nota
        c["plantilla"] = {**datos, "encaje_llm": None if llm is None else llm["encaje"],
                          "comentario": (("Ya pertenece al club. " if datos["ya_en_el_club"] else "") + llm["comentario"])
                          if llm else comentario_por_indicadores(datos, evaluacion["equipo"]),
                          "origen": "llm" if llm else "indicadores"}
    ranking.sort(key=lambda c: (-round(c["score_final"], 9), -round(c["score_numerico"], 9), -c["minutos"],
                                c["registro_id"]))
    for i, c in enumerate(ranking, start=1):
        c["ranking_final"] = i
    return ranking
