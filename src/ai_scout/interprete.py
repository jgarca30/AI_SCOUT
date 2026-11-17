"""Secretario técnico de consultas en lenguaje natural.

Convierte una petición como «busco un sustituto joven de Pedri en la Bundesliga, más creador»
en los parámetros de una consulta (jugador de referencia, ligas, edad máxima, minutos, rol…).

- Con modelo de lenguaje: el LLM extrae los parámetros en JSON y el código los valida.
  El LLM nunca elige el registro del jugador: solo copia el nombre que escribió el usuario
  y el código lo busca en el dataset.
- Sin modelo (o si falla): un intérprete por reglas hace el mismo trabajo con expresiones
  regulares. Es el mismo criterio de degradación que en el resto del sistema.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import pandas as pd

from . import config as C
from .datos import etiqueta_registro, normalizar_texto

EDAD_JOVEN = 23        # «joven» se interpreta como esta edad máxima
MINUTOS_TITULAR = 1800  # «titular» se interpreta como estos minutos mínimos

# Formas habituales de nombrar cada liga (sin tildes y en minúsculas)
SINONIMOS_LIGA = {
    "Premier League": ["premier league", "premier", "liga inglesa", "inglaterra"],
    "La Liga": ["la liga", "laliga", "liga espanola", "liga de espana", "espana"],
    "Serie A": ["serie a", "liga italiana", "calcio", "italia"],
    "Bundesliga": ["bundesliga", "liga alemana", "alemania"],
    "Ligue 1": ["ligue 1", "ligue1", "liga francesa", "francia"],
}
# Palabras que describen cada rol (sin tildes y en minúsculas)
PALABRAS_ROL = {
    "DF": {
        "central_marcador": ["central marcador", "marcador", "stopper", "contundente", "defensivo"],
        "salida_balon": ["salida de balon", "saque de balon", "buen pie", "con salida"],
        "lateral_ofensivo": ["lateral ofensivo", "carrilero", "ofensivo", "profundo"],
    },
    "MF": {
        "organizador": ["organizador", "regista", "distribuidor", "de posesion", "director de juego"],
        "recuperador": ["recuperador", "pivote", "destructor", "defensivo", "de contencion", "ancla"],
        "creador": ["creador", "mediapunta", "media punta", "creativo", "ultimo pase", "enganche"],
        "todocampista": ["todocampista", "box to box", "box-to-box", "llegador", "ida y vuelta"],
    },
    "FW": {
        "finalizador": ["finalizador", "goleador", "rematador", "delantero centro", "nueve", "de area"],
        "extremo_regateador": ["regateador", "desborde", "encarador", "regate", "desequilibrante"],
        "creador": ["creador", "asistente", "asistencias", "creativo", "ultimo pase"],
    },
}
CLAVES_ROL = sorted({rol for roles in C.ROLES.values() for rol in roles})
# Palabras de la petición que no pueden ser el nombre de un jugador
VOCABULARIO = {
    "busco", "buscar", "busca", "quiero", "necesito", "dame", "sustituto", "sustitutos", "sustituir", "relevo",
    "reemplazo", "recambio", "alternativa", "alternativas", "parecido", "parecidos", "similar", "similares",
    "como", "tipo", "estilo", "perfil", "jugador", "jugadores", "joven", "jovenes", "promesa", "liga", "ligas",
    "premier", "league", "serie", "bundesliga", "ligue", "anos", "menor", "menos", "maximo", "minimo", "minutos",
    "edad", "hasta", "para", "pero", "mas", "que", "con", "sin", "del", "los", "las", "una", "uno", "sea", "este",
    "extremo", "delantero", "defensa", "central", "lateral", "centrocampista", "mediocentro", "pivote", "titular",
    "espana", "italia", "francia", "alemania", "inglaterra", "mismo", "misma", "equipo", "club", "sub",
}


@dataclass
class Interpretacion:
    """Parámetros extraídos de una petición en lenguaje natural."""

    texto: str
    origen: str = "reglas"                    # "llm" o "reglas"
    nombre_detectado: str | None = None       # nombre tal como aparece en la petición
    objetivo_registro_id: str | None = None   # registro elegido como referencia
    alternativas: list[dict] = field(default_factory=list)   # otros registros que encajan con el nombre
    ligas: list[str] | None = None
    edad_maxima: int | None = None
    minutos_minimos: int | None = None
    rol: str | None = None
    solo_mismo_estilo: bool = False
    notas: list[str] = field(default_factory=list)   # avisos: interpretaciones y cosas que no se pueden aplicar

    def parametros(self) -> dict:
        """Argumentos para Consulta (solo los que se han detectado)."""
        p = {"texto_original": self.texto}
        if self.ligas:
            p["ligas_permitidas"] = list(self.ligas)
        if self.edad_maxima is not None:
            p["edad_maxima"] = self.edad_maxima
        if self.minutos_minimos is not None:
            p["minutos_minimos"] = self.minutos_minimos
        if self.rol:
            p["rol"] = self.rol
        if self.solo_mismo_estilo:
            p["solo_mismo_estilo"] = True
        return p

    def nombre_rol(self, df: pd.DataFrame | None = None) -> str:
        """Nombre del rol para el grupo de posición del jugador de referencia."""
        grupos = list(C.ROLES)
        if self.objetivo_registro_id and df is not None:
            fila = df[df["registro_id"] == self.objetivo_registro_id]
            if len(fila):
                grupos = [fila.iloc[0]["grupo_posicion"]] + grupos
        for grupo in grupos:
            if self.rol in C.ROLES.get(grupo, {}):
                return C.ROLES[grupo][self.rol]["nombre"]
        return self.rol or ""

    def resumen(self, df: pd.DataFrame | None = None) -> str:
        partes = []
        if self.objetivo_registro_id and df is not None:
            fila = df[df["registro_id"] == self.objetivo_registro_id]
            if len(fila):
                partes.append(f"referencia: {fila.iloc[0]['nombre']} ({fila.iloc[0]['equipo']})")
        if self.ligas:
            partes.append("ligas: " + ", ".join(self.ligas))
        if self.edad_maxima is not None:
            partes.append(f"edad ≤ {self.edad_maxima}")
        if self.minutos_minimos is not None:
            partes.append(f"minutos ≥ {self.minutos_minimos}")
        if self.rol:
            partes.append(f"rol: {self.nombre_rol(df)}")
        if self.solo_mismo_estilo:
            partes.append("solo el mismo estilo de juego")
        return " · ".join(partes) if partes else "no se ha reconocido ningún parámetro"


# --------------------------------------------------------------------------
# Búsqueda del jugador en el dataset
# --------------------------------------------------------------------------
_LETRAS = str.maketrans({"ø": "o", "đ": "d", "ł": "l", "ı": "i", "ß": "ss", "æ": "ae", "œ": "oe", "þ": "th", "ð": "d"})


def _plano(texto: object) -> str:
    """Texto sin tildes, en minúsculas y sin letras especiales (Ødegaard → odegaard)."""
    return normalizar_texto(texto).translate(_LETRAS)


def _fichas_nombre(texto: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", _plano(texto)) if t]


_INDICES: dict = {}


def _indice_nombres(df: pd.DataFrame) -> list[tuple]:
    """(registro_id, nombre plano, palabras, minutos, equipo plano) de los jugadores de campo válidos."""
    clave = (id(df), len(df))
    if clave not in _INDICES:
        _INDICES.clear()
        _INDICES[clave] = [(f.registro_id, _plano(f.nombre), _fichas_nombre(f.nombre), float(f.minutos),
                            _plano(f.equipo)) for f in df[~df["es_portero"] & df["valido"]].itertuples()]
    return _INDICES[clave]


def buscar_jugador(nombre: str, df: pd.DataFrame, equipo: str | None = None, limite: int = 5) -> list[dict]:
    """Registros de jugadores de campo cuyo nombre encaja con `nombre`, del más al menos probable.

    Combina coincidencia exacta, por palabras (p. ej. «Yamal» → «Lamine Yamal») y aproximada
    (erratas). A igual puntuación gana el registro con más minutos.
    """
    q_tokens = _fichas_nombre(nombre)
    q = " ".join(q_tokens)
    if not q_tokens:
        return []
    pista_equipo = _plano(equipo) if equipo else ""
    conjunto_q = set(q_tokens)
    comparador = difflib.SequenceMatcher(None)
    comparador.set_seq2(q)
    resultados = []
    for registro_id, n, n_tokens, minutos, equipo_plano in _indice_nombres(df):
        if not n_tokens:
            continue
        if n == q:
            puntos = 100.0
        elif conjunto_q <= set(n_tokens):
            puntos = 80.0 + 10.0 * len(q_tokens) / len(n_tokens)
        elif set(n_tokens) <= conjunto_q and len(n_tokens) >= 2:
            puntos = 75.0
        else:   # coincidencia aproximada (erratas); quick_ratio descarta barato la mayoría
            comparador.set_seq1(n)
            ratio = comparador.ratio() if comparador.quick_ratio() >= 0.82 else 0.0
            if ratio >= 0.82:
                puntos = 70.0 * ratio
            elif len(q_tokens) == 1 and len(q) >= 4:
                mejor = 0.0
                for b in n_tokens:
                    if len(b) >= 4 and abs(len(b) - len(q)) <= 2:
                        m = difflib.SequenceMatcher(None, q, b)
                        if m.quick_ratio() >= 0.85:
                            mejor = max(mejor, m.ratio())
                if mejor < 0.85:
                    continue
                puntos = 55.0 * mejor
            else:
                continue
        if pista_equipo and pista_equipo in equipo_plano:
            puntos += 15.0
        resultados.append((puntos, minutos, registro_id))
    resultados.sort(key=lambda r: (-r[0], -r[1], r[2]))
    por_registro = df.set_index("registro_id")
    return [{"registro_id": rid, "puntuacion": round(p, 1),
             "persona_id": por_registro.loc[rid, "persona_id"],
             "etiqueta": etiqueta_registro(por_registro.loc[rid])} for p, _, rid in resultados[:limite]]


def _resolver_jugador(interpretacion: Interpretacion, nombre: str | None, equipo: str | None,
                      df: pd.DataFrame) -> None:
    """Rellena objetivo_registro_id y alternativas a partir del nombre detectado."""
    if not nombre or not nombre.strip():
        interpretacion.notas.append("No se ha identificado un jugador de referencia en la petición.")
        return
    interpretacion.nombre_detectado = nombre.strip()
    encontrados = buscar_jugador(nombre, df, equipo)
    if not equipo and len(encontrados) > 1:
        # Si la petición nombra el equipo de alguno de los registros que encajan, se usa como pista
        texto_plano = _plano(interpretacion.texto)
        por_registro = df.set_index("registro_id")
        for e in encontrados:
            equipo_e = por_registro.loc[e["registro_id"], "equipo"]
            if re.search(rf"\b{re.escape(_plano(equipo_e))}\b", texto_plano):
                equipo = equipo_e
                encontrados = buscar_jugador(nombre, df, equipo)
                break
    if not encontrados:
        interpretacion.notas.append(f"No hay ningún jugador de campo llamado «{nombre.strip()}» en el dataset.")
        return
    mejor = encontrados[0]
    interpretacion.objetivo_registro_id = mejor["registro_id"]
    interpretacion.alternativas = [e for e in encontrados[1:] if e["puntuacion"] >= mejor["puntuacion"] - 15]
    otras_personas = {e["persona_id"] for e in interpretacion.alternativas} - {mejor["persona_id"]}
    if otras_personas and not equipo:
        interpretacion.notas.append(f"Hay varios jugadores que encajan con «{nombre.strip()}»; se ha elegido "
                                    f"{mejor['etiqueta']}.")


def _ajustar_rol_al_grupo(interpretacion: Interpretacion, df: pd.DataFrame) -> None:
    """Descarta el rol si no existe para el grupo de posición del jugador de referencia."""
    if not interpretacion.rol or not interpretacion.objetivo_registro_id:
        return
    grupo = df.loc[df["registro_id"] == interpretacion.objetivo_registro_id, "grupo_posicion"].iloc[0]
    if interpretacion.rol not in C.ROLES.get(grupo, {}):
        interpretacion.notas.append(f"El rol «{interpretacion.rol}» no existe para {C.GRUPOS[grupo].lower()}; "
                                    "se usa el rol equilibrado.")
        interpretacion.rol = None


# --------------------------------------------------------------------------
# Intérprete por reglas (sin modelo de lenguaje)
# --------------------------------------------------------------------------
_NEGACION = re.compile(r"(?:\bno\b|\bsin\b|\bexcepto\b|\bsalvo\b|\bfuera de\b|\bmenos\b|\bexcluye\w*\b)[\w\s]{0,18}$")
_FRASES_REFERENCIA = re.compile(
    r"(?:parecid[oa]s? a|similar(?:es)? a|sustitut[oa]s? (?:de|para|a)|sustituir a|relev[oa]s? (?:de|para)|"
    r"reemplaz[oa]r? (?:de|para|a)|recambio (?:de|para)|alternativas? a|del? estilo de|como|tipo)\s+(.+)")
_CORTE = re.compile(r"\s+(?:pero|que|con|sin|en|de la|del|de las|de los|para|menor|menos|sub|joven|maximo|"
                    r"hasta|y|o|mas)\b|[,.;:()]")


def _detectar_ligas(t: str) -> tuple[list[str] | None, list[str]]:
    incluidas, excluidas = [], []
    for liga, sinonimos in SINONIMOS_LIGA.items():
        for s in sinonimos:
            m = re.search(rf"\b{re.escape(s)}\b", t)
            if m:
                (excluidas if _NEGACION.search(t[:m.start()]) else incluidas).append(liga)
                break
    if incluidas:
        return [l for l in C.LIGAS_NORMALIZADAS if l in incluidas], []
    if excluidas:
        return [l for l in C.LIGAS_NORMALIZADAS if l not in excluidas], excluidas
    return None, []


def _detectar_edad(t: str) -> tuple[int | None, str | None]:
    m = re.search(r"\bsub[\s-]?(\d{2})\b", t)
    if m:
        return int(m.group(1)), None
    m = re.search(r"\b(?:menor(?:es)? de|menos de|por debajo de)\s+(\d{2})\b(?!\s*minutos)", t)
    if m:
        return int(m.group(1)) - 1, None
    m = (re.search(r"\b(?:hasta|maximo|como mucho|no mas de|edad maxima(?: de)?)\s+(\d{2})\b(?!\s*minutos)", t)
         or re.search(r"\b(\d{2})\s+anos\s+(?:o menos|como maximo|como mucho)", t)
         or re.search(r"\bde\s+(\d{2})\s+anos\b", t))
    if m:
        return int(m.group(1)), None
    if re.search(r"\b(?:joven|jovenes|promesa|promesas)\b", t):
        return EDAD_JOVEN, f"«joven» se ha interpretado como {EDAD_JOVEN} años o menos."
    return None, None


def _detectar_minutos(t: str) -> tuple[int | None, str | None]:
    m = (re.search(r"\b(?:minimo|al menos|mas de|con)\s+(\d{3,4})\s+minutos\b", t)
         or re.search(r"\b(\d{3,4})\s+minutos\b", t))
    if m:
        return int(m.group(1)), None
    if re.search(r"\btitular(?:es)?\b", t):
        return MINUTOS_TITULAR, f"«titular» se ha interpretado como al menos {MINUTOS_TITULAR} minutos."
    return None, None


def _detectar_nombre(texto: str, df: pd.DataFrame) -> str | None:
    """Busca en la petición un nombre de jugador del dataset."""
    t = _plano(texto)
    # 1) Tras una expresión como «parecido a», «sustituto de», «como»…
    m = _FRASES_REFERENCIA.search(t)
    if m:
        trozo = _CORTE.split(m.group(1).strip())[0].strip()
        tokens = trozo.split()
        for n in range(min(4, len(tokens)), 0, -1):
            candidato = " ".join(tokens[:n])
            if n == 1 and (candidato in VOCABULARIO or len(candidato) < 3):
                continue
            if buscar_jugador(candidato, df, limite=1):
                return candidato
    # 2) Cualquier secuencia de palabras que coincida con un jugador (de más larga a más corta)
    tokens = [x for x in re.split(r"[^\w]+", t) if x]
    nombres = set()
    for _, _, fichas, _, _ in _indice_nombres(df):
        nombres.add(" ".join(fichas))
        nombres.update(f for f in fichas if len(f) >= 5)
    for n in range(4, 0, -1):
        for i in range(len(tokens) - n + 1):
            trozo = tokens[i:i + n]
            if n == 1 and (trozo[0] in VOCABULARIO or len(trozo[0]) < 5):
                continue
            if any(x in VOCABULARIO for x in trozo) and n > 1:
                continue
            if " ".join(trozo) in nombres:
                return " ".join(trozo)
    return None


def interpretar_por_reglas(texto: str, df: pd.DataFrame) -> Interpretacion:
    """Intérprete determinista: expresiones regulares y búsqueda de nombres en el dataset."""
    t = _plano(texto)
    r = Interpretacion(texto=texto, origen="reglas")
    r.ligas, _ = _detectar_ligas(t)
    r.edad_maxima, nota = _detectar_edad(t)
    if nota:
        r.notas.append(nota)
    r.minutos_minimos, nota = _detectar_minutos(t)
    if nota:
        r.notas.append(nota)
    r.solo_mismo_estilo = bool(re.search(r"\bmismo (?:estilo|perfil) de juego\b|\bdel mismo estilo\b|\bmismo estilo\b", t))
    _resolver_jugador(r, _detectar_nombre(texto, df), None, df)
    if r.objetivo_registro_id:
        grupo = df.loc[df["registro_id"] == r.objetivo_registro_id, "grupo_posicion"].iloc[0]
        nombre_ref = _plano(r.nombre_detectado or "")
        resto = t.replace(nombre_ref, " ") if nombre_ref else t
        for rol, palabras in PALABRAS_ROL.get(grupo, {}).items():
            if any(re.search(rf"\b{re.escape(p)}\b", resto) for p in palabras):
                r.rol = rol
                break
    return r


# --------------------------------------------------------------------------
# Intérprete con modelo de lenguaje
# --------------------------------------------------------------------------
def _catalogo_roles() -> str:
    lineas = []
    for grupo, roles in C.ROLES.items():
        lineas.append(f"  {C.GRUPOS[grupo]}:")
        for clave, rol in roles.items():
            if clave != C.ROL_DEFECTO:
                lineas.append(f'    - "{clave}": {rol["nombre"]}. {rol["descripcion"]}')
    return "\n".join(lineas)


SISTEMA = f"""Eres el Secretario técnico (intérprete de las peticiones) de AI Scout, un recomendador académico de futbolistas.
El usuario describe en lenguaje natural qué sustituto busca. Tu tarea es extraer los parámetros de búsqueda.
No conoces el dataset: no completes ni corrijas nombres de jugadores.

Reglas obligatorias:
1. "jugador_referencia": el nombre del jugador al que se quiere sustituir o al que debe parecerse el
   candidato, copiado tal como lo escribe el usuario. null si no menciona ninguno.
   "equipo_referencia": el equipo de ese jugador solo si el usuario lo dice; si no, null.
2. "ligas": solo valores de esta lista: {", ".join(C.LIGAS_NORMALIZADAS)}. Lista vacía si no menciona ligas
   (significa todas). Si el usuario excluye ligas, devuelve las restantes.
3. "edad_maxima": entero o null. «menor de 23» o «menos de 23 años» → 22; «sub-23», «hasta 23» o «máximo 23»
   → 23; «joven» → {EDAD_JOVEN}.
4. "minutos_minimos": entero o null. «titular» → {MINUTOS_TITULAR}.
5. "rol": una de las claves de este catálogo, o null si el usuario no describe el tipo de jugador que busca:
{_catalogo_roles()}
6. "solo_mismo_estilo": true ÚNICAMENTE si el usuario escribe expresamente «mismo estilo» o «mismo perfil de
   juego». Expresiones como «parecido a», «similar a», «como» o «tipo» NO lo activan: en esos casos, false.
7. "no_aplicable": lista de frases breves con lo que pide el usuario y no se puede expresar con estos
   parámetros (por ejemplo pierna hábil, altura, precio, nacionalidad o un equipo de destino). Lista vacía si
   todo se ha podido recoger.
8. Responde ÚNICAMENTE con un objeto JSON con este formato:
{{"jugador_referencia": "nombre" | null, "equipo_referencia": "equipo" | null, "ligas": ["..."],
  "edad_maxima": 0 | null, "minutos_minimos": 0 | null, "rol": "clave" | null, "solo_mismo_estilo": false,
  "no_aplicable": ["..."]}}"""


def _opcional(tipo: dict) -> dict:
    return {"anyOf": [tipo, {"type": "null"}]}


ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "jugador_referencia": _opcional({"type": "string"}),
        "equipo_referencia": _opcional({"type": "string"}),
        "ligas": {"type": "array", "items": {"type": "string", "enum": list(C.LIGAS_NORMALIZADAS)}},
        "edad_maxima": _opcional({"type": "integer"}),
        "minutos_minimos": _opcional({"type": "integer"}),
        "rol": _opcional({"type": "string", "enum": CLAVES_ROL}),
        "solo_mismo_estilo": {"type": "boolean"},
        "no_aplicable": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["jugador_referencia", "equipo_referencia", "ligas", "edad_maxima", "minutos_minimos", "rol",
                 "solo_mismo_estilo", "no_aplicable"],
}


def construir_mensaje(texto: str) -> str:
    return f"PETICIÓN DEL USUARIO:\n«{texto.strip()}»"


def _entero_en_rango(valor, minimo: int, maximo: int) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool) and minimo <= valor <= maximo


def validar_respuesta(datos: dict, texto: str, df: pd.DataFrame) -> tuple[Interpretacion | None, list[str]]:
    """Valida la salida del Secretario técnico y resuelve el jugador en el dataset."""
    errores = []
    nombre, equipo = datos.get("jugador_referencia"), datos.get("equipo_referencia")
    if nombre is not None and not isinstance(nombre, str):
        errores.append("'jugador_referencia' debe ser un texto o null.")
    if equipo is not None and not isinstance(equipo, str):
        errores.append("'equipo_referencia' debe ser un texto o null.")
    ligas = datos.get("ligas", [])
    if not isinstance(ligas, list) or any(l not in C.LIGAS_NORMALIZADAS for l in ligas):
        errores.append(f"'ligas' solo admite: {', '.join(C.LIGAS_NORMALIZADAS)}.")
    edad, minutos = datos.get("edad_maxima"), datos.get("minutos_minimos")
    if edad is not None and not _entero_en_rango(edad, 15, 45):
        errores.append("'edad_maxima' debe ser un entero entre 15 y 45, o null.")
    if minutos is not None and not _entero_en_rango(minutos, 90, 3420):
        errores.append("'minutos_minimos' debe ser un entero entre 90 y 3420, o null.")
    rol = datos.get("rol")
    if rol is not None and rol not in CLAVES_ROL:
        errores.append(f"'rol' debe ser una de estas claves o null: {', '.join(CLAVES_ROL)}.")
    notas = datos.get("no_aplicable", [])
    if not isinstance(notas, list) or not all(isinstance(x, str) for x in notas):
        errores.append("'no_aplicable' debe ser una lista de textos.")
    if not isinstance(datos.get("solo_mismo_estilo", False), bool):
        errores.append("'solo_mismo_estilo' debe ser true o false.")
    if errores:
        return None, errores

    r = Interpretacion(texto=texto, origen="llm",
                       ligas=[l for l in C.LIGAS_NORMALIZADAS if l in ligas] or None,
                       edad_maxima=edad, minutos_minimos=minutos,
                       rol=None if rol == C.ROL_DEFECTO else rol,
                       solo_mismo_estilo=bool(datos.get("solo_mismo_estilo", False)))
    r.notas.extend(f"No se puede aplicar: {x.strip()}" for x in notas if x.strip())
    _resolver_jugador(r, nombre, equipo, df)
    if r.objetivo_registro_id is None:   # el LLM no vio el nombre: segunda oportunidad con las reglas
        detectado = _detectar_nombre(texto, df)
        if detectado:
            r.notas = [n for n in r.notas if "jugador" not in n]
            _resolver_jugador(r, detectado, equipo, df)
    _ajustar_rol_al_grupo(r, df)
    return r, []


def interpretar(texto: str, df: pd.DataFrame, proveedor=None, cache=None,
                consulta_id: str = "INTERPRETE") -> tuple[Interpretacion, list[dict]]:
    """Interpreta la petición con el LLM si está disponible y, si no (o si falla), con reglas.

    Devuelve (interpretación, ejecuciones registradas del agente).
    """
    from .ejecutor_llm import ejecutar_agente

    ejecuciones: list[dict] = []
    if proveedor is not None and texto.strip():
        resultado, ejecuciones = ejecutar_agente(
            "interprete", proveedor, SISTEMA, construir_mensaje(texto),
            lambda datos: validar_respuesta(datos, texto, df), consulta_id, max_tokens=2000, cache=cache)
        if resultado is not None:
            return resultado, ejecuciones
    r = interpretar_por_reglas(texto, df)
    if proveedor is not None and ejecuciones:
        r.notas.append("El intérprete con IA no ha dado una respuesta válida; se han usado reglas.")
    return r, ejecuciones
