"""Configuración central de AI Scout.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
RUTA_CSV = RAIZ / "dataset" / "players_data-2024_2025.csv"
RUTA_CORRESPONDENCIAS = RAIZ / "config" / "correspondencias_identidad.csv"
DIR_PROCESADO = RAIZ / "datos" / "procesado"
DIR_CACHE_LLM = RAIZ / "datos" / "cache_llm"
DIR_SALIDAS = RAIZ / "salidas"

# --------------------------------------------------------------------------
# Versiones (cualquier cambio en perfiles, prompts o esquemas sube su versión)
# --------------------------------------------------------------------------
VERSION_PREPARACION = "1.0"
PERFIL_VERSION = "perfiles-v1"
PROMPT_VERSION = "prompts-v2"  # v2: el director nombra a los jugadores por su nombre
ESQUEMA_VERSION = "esquema-v2"  # v2: esquemas JSON de salida (salida estructurada en Gemini)
TEMPORADA = "2024-2025"

# --------------------------------------------------------------------------
# Parámetros del recomendador
# --------------------------------------------------------------------------
MINUTOS_MINIMOS_DEFECTO = 600
TOP_PRESELECCION = 10
TOP_FINAL = 3
PESO_NUMERICO = 0.8
PESO_CUALITATIVO = 0.2
MIN_DIMENSIONES_EVALUABLES = 2

# Competiciones del CSV
LIGAS = {
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "it Serie A": "Serie A",
    "de Bundesliga": "Bundesliga",
    "fr Ligue 1": "Ligue 1",
}
LIGAS_NORMALIZADAS = list(LIGAS.values())

GRUPOS = {"DF": "Defensas", "MF": "Centrocampistas", "FW": "Delanteros"}

# Columnas acumuladas del CSV que se transforman a valores por 90 minutos
COLUMNAS_P90 = ["Tkl", "Int", "Clr", "Recov", "PrgP", "PrgC", "KP", "PPA",
                "Touches", "npxG", "xAG", "Succ"]
# Columnas que ya son porcentajes (no se dividen por minutos)
COLUMNAS_PORCENTAJE = {"Cmp%": "Cmp_pct"}

# Perfiles de variables por grupo de posición (versión inicial, pesos iguales)
PERFILES = {
    "DF": ["Tkl_p90", "Int_p90", "Clr_p90", "Recov_p90", "PrgP_p90", "PrgC_p90", "Cmp_pct"],
    "MF": ["PrgP_p90", "KP_p90", "PPA_p90", "Recov_p90", "Int_p90", "Touches_p90", "PrgC_p90", "Cmp_pct"],
    "FW": ["npxG_p90", "xAG_p90", "KP_p90", "PrgC_p90", "PPA_p90", "Touches_p90", "Succ_p90"],
}

# Descripción y unidad de cada variable procesada (se muestra en la app y se
# envía a los agentes LLM para que sepan qué significa cada cifra)
VARIABLES = {
    "Tkl_p90": ("Entradas por 90 minutos", "por 90"),
    "Int_p90": ("Intercepciones por 90 minutos", "por 90"),
    "Clr_p90": ("Despejes por 90 minutos", "por 90"),
    "Recov_p90": ("Recuperaciones de balón por 90 minutos", "por 90"),
    "PrgP_p90": ("Pases progresivos por 90 minutos", "por 90"),
    "PrgC_p90": ("Conducciones progresivas por 90 minutos", "por 90"),
    "KP_p90": ("Pases clave por 90 minutos", "por 90"),
    "PPA_p90": ("Pases completados al área por 90 minutos", "por 90"),
    "Touches_p90": ("Toques de balón por 90 minutos", "por 90"),
    "npxG_p90": ("Goles esperados sin penaltis (npxG) por 90 minutos", "por 90"),
    "xAG_p90": ("Goles esperados de los tiros asistidos (xAG) por 90 minutos", "por 90"),
    "Succ_p90": ("Regates completados por 90 minutos", "por 90"),
    "Cmp_pct": ("Precisión de pase", "%"),
}
ETIQUETAS_CORTAS = {
    "Tkl_p90": "Entradas", "Int_p90": "Intercepciones", "Clr_p90": "Despejes",
    "Recov_p90": "Recuperaciones", "PrgP_p90": "Pases progresivos",
    "PrgC_p90": "Conducciones progr.", "KP_p90": "Pases clave", "PPA_p90": "Pases al área",
    "Touches_p90": "Toques", "npxG_p90": "npxG", "xAG_p90": "xAG",
    "Succ_p90": "Regates", "Cmp_pct": "Precisión de pase",
}

# Dimensiones que puntúa el Agente 2 (escala 0-4)
DIMENSIONES = {
    "progresion_creacion": "Semejanza en la forma de hacer avanzar el balón y crear ocasiones "
                           "(pases progresivos, conducciones, pases clave, pases al área, xG/xAG).",
    "equilibrio_ofensivo_defensivo": "Semejanza en el equilibrio entre acciones ofensivas y defensivas "
                                     "disponibles (entradas, intercepciones, recuperaciones, despejes frente a "
                                     "acciones de creación).",
    "coherencia_perfil": "Coherencia global del conjunto de indicadores del candidato con el perfil de referencia.",
}
ESCALA_PUNTUACION = {
    0: "divergencia clara", 1: "coincidencia baja", 2: "coincidencia parcial",
    3: "coincidencia alta", 4: "coincidencia muy alta",
}
VEREDICTOS = {
    "candidato_para_revisar": "Candidato para revisar",
    "candidato_con_reservas": "Candidato con reservas",
    "evidencia_insuficiente": "Evidencia insuficiente",
}

# Términos que el LLM no puede usar como afirmación (no hay datos que los
# respalden). Se comparan sin tildes y en minúsculas.
TERMINOS_SIN_RESPALDO = [
    "lesion", "salario", "sueldo", "contrato", "precio", "traspaso", "valor de mercado",
    "velocidad", "fisico", "personalidad", "mentalidad", "liderazgo", "bajo presion",
    "mapa de calor", "estado de forma", "lectura de juego",
]

LIMITACIONES_GENERALES = [
    "Los índices comparan perfiles estadísticos de una sola temporada.",
    "El contexto táctico del equipo, el nivel de cada liga, el estado físico y la situación contractual no están en los datos.",
    "La edad es la registrada en el dataset no necesariamente la actual.",
]

# --------------------------------------------------------------------------
# Modelos de lenguaje (se leen del entorno / archivo .env)
# --------------------------------------------------------------------------
try:  # python-dotenv es opcional
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
except ImportError:  # pragma: no cover
    pass


def entorno(nombre: str, defecto: str | None = None) -> str | None:
    valor = os.getenv(nombre)
    return valor if valor not in (None, "") else defecto


LLM_PROVEEDOR = (entorno("LLM_PROVEEDOR", "anthropic") or "ninguno").lower()
MODELOS_DEFECTO = {
    "anthropic": {"analista": "claude-sonnet-5-5", "director": "claude-haiku-5-5"},
    "gemini": {"analista": "gemini-3.5-flash-lite", "director": "gemini-3.5-flash-lite"},
}
# Modelos que admiten el reenvío automático a otro modelo si la petición se rechaza
MODELOS_CON_FALLBACK = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
LLM_TIMEOUT_S = float(entorno("LLM_TIMEOUT_S", "60"))
LLM_MAX_INTENTOS = 2
LLM_TOPE_TOKENS_CONSULTA = int(entorno("LLM_TOPE_TOKENS_CONSULTA", "80000"))
LLM_USAR_CACHE = entorno("LLM_CACHE", "1") != "0"

PRECIOS_USD_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-3.5-flash-lite": (0.0, 0.0),
    "gemini-3.1-flash-lite": (0.0, 0.0),
    "gemini-3.6-flash": (0.0, 0.0),
    "gemini-3.8-flash": (0.0, 0.0),
}
