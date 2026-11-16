"""Configuración central de AI Scout.
"""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

try:  
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
except ImportError: 
    pass


def entorno(nombre: str, defecto: str | None = None) -> str | None:
    valor = os.getenv(nombre)
    return valor if valor not in (None, "") else defecto


RUTA_CSV = RAIZ / "dataset" / "players_data-2024_2025.csv"
RUTA_CORRESPONDENCIAS = RAIZ / "config" / "correspondencias_identidad.csv"
DIR_PROCESADO = RAIZ / "datos" / "procesado"
DIR_CACHE_LLM = RAIZ / "datos" / "cache_llm"

DIR_SALIDAS = Path(entorno("AI_SCOUT_SALIDAS") or RAIZ / "salidas")
# Subcarpetas de salidas/
SUBDIR_HISTORIAL = "historial"      # resultado completo de cada consulta (JSON) para reabrirla sin recalcular
SUBDIR_EVALUACION = "evaluacion"    # tablas e informe de evaluar_sistema.py

# Versiones 
VERSION_PREPARACION = "1.0"
PERFIL_VERSION = "perfiles-v2"  # v2: roles de juego con pesos por variable
PROMPT_VERSION = "prompts-v4"  # v3: pesos del rol y estilos en los datos; agentes intérprete y revisor
ESQUEMA_VERSION = "esquema-v3"  # v3: esquemas del Secretario técnico y del Jefe de scouting
TEMPORADA = "2024-2025"

# Parámetros del recomendador
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

# Perfiles de variables por grupo de posición
PERFILES = {
    "DF": ["Tkl_p90", "Int_p90", "Clr_p90", "Recov_p90", "PrgP_p90", "PrgC_p90", "Cmp_pct"],
    "MF": ["PrgP_p90", "KP_p90", "PPA_p90", "Recov_p90", "Int_p90", "Touches_p90", "PrgC_p90", "Cmp_pct"],
    "FW": ["npxG_p90", "xAG_p90", "KP_p90", "PrgC_p90", "PPA_p90", "Touches_p90", "Succ_p90"],
}

# Roles de juego: qué variables pesan más al buscar un sustituto
# Cada rol da un peso relativo a las variables del perfil (1 = importancia normal; las no citadas valen 1). 
# #Los pesos se normalizan a media 1 antes de calcular la similitud coseno ponderada.
ROL_DEFECTO = "equilibrado"
PESO_MAXIMO_VARIABLE = 3.0  # tope de los deslizadores de la app
ROLES = {
    "DF": {
        "equilibrado": {"nombre": "Equilibrado", "pesos": {},
                        "descripcion": "Todas las variables del perfil pesan lo mismo."},
        "central_marcador": {"nombre": "Central marcador",
                             "pesos": {"Clr_p90": 2.5, "Int_p90": 2.0, "Tkl_p90": 2.0, "PrgC_p90": 0.5},
                             "descripcion": "Prioriza despejes, intercepciones y entradas."},
        "salida_balon": {"nombre": "Defensa con salida de balón",
                         "pesos": {"PrgP_p90": 2.5, "Cmp_pct": 2.0, "Recov_p90": 1.5},
                         "descripcion": "Prioriza pases progresivos, precisión de pase y recuperaciones."},
        "lateral_ofensivo": {"nombre": "Lateral ofensivo",
                             "pesos": {"PrgC_p90": 2.5, "PrgP_p90": 1.5, "Tkl_p90": 1.5, "Clr_p90": 0.5},
                             "descripcion": "Prioriza conducciones y pases progresivos; los despejes pesan menos."},
    },
    "MF": {
        "equilibrado": {"nombre": "Equilibrado", "pesos": {},
                        "descripcion": "Todas las variables del perfil pesan lo mismo."},
        "organizador": {"nombre": "Mediocentro organizador",
                        "pesos": {"PrgP_p90": 2.5, "Touches_p90": 2.0, "Cmp_pct": 2.0},
                        "descripcion": "Prioriza pases progresivos, participación y precisión de pase."},
        "recuperador": {"nombre": "Pivote recuperador",
                        "pesos": {"Recov_p90": 2.5, "Int_p90": 2.5, "KP_p90": 0.5, "PPA_p90": 0.5},
                        "descripcion": "Prioriza recuperaciones e intercepciones; la creación pesa menos."},
        "creador": {"nombre": "Creador / mediapunta",
                    "pesos": {"KP_p90": 2.5, "PPA_p90": 2.5, "PrgC_p90": 1.5},
                    "descripcion": "Prioriza pases clave, pases al área y conducciones."},
        "todocampista": {"nombre": "Todocampista",
                         "pesos": {"PrgC_p90": 2.0, "Recov_p90": 2.0, "PrgP_p90": 1.5},
                         "descripcion": "Prioriza conducciones, recuperaciones y pases progresivos."},
    },
    "FW": {
        "equilibrado": {"nombre": "Equilibrado", "pesos": {},
                        "descripcion": "Todas las variables del perfil pesan lo mismo."},
        "finalizador": {"nombre": "Delantero finalizador",
                        "pesos": {"npxG_p90": 3.0, "Touches_p90": 0.5},
                        "descripcion": "Prioriza los goles esperados; la participación pesa menos."},
        "extremo_regateador": {"nombre": "Extremo regateador",
                               "pesos": {"Succ_p90": 2.5, "PrgC_p90": 2.5},
                               "descripcion": "Prioriza regates completados y conducciones progresivas."},
        "creador": {"nombre": "Atacante creador",
                    "pesos": {"xAG_p90": 2.5, "KP_p90": 2.5, "PPA_p90": 2.0},
                    "descripcion": "Prioriza asistencias esperadas, pases clave y pases al área."},
    },
}

# Estilos de juego (agrupamiento k-means sobre las variables estandarizadas + mapa PCA)
# k = 4 por grupo: la silueta es casi igual que con k = 3 y los grupos son más interpretables
# (la tabla de siluetas por k la genera evaluar_sistema.py)
#Los estilos son una partición  orientativa de un continuo, no categorías cerradas.
ESTILOS_K = {"DF": 4, "MF": 4, "FW": 4}
ESTILOS_SEMILLA = 0
ESTILOS_UMBRAL_Z = 0.5  # un estilo «destaca» en una variable si su centroide supera este valor z
# Nombre del estilo según la variable más destacada de su centroide
ARQUETIPOS = {
    "DF": {"Clr_p90": "Central de área", "Int_p90": "Defensa de anticipación", "Tkl_p90": "Defensa de duelo",
           "Recov_p90": "Defensa recuperador", "PrgP_p90": "Defensa de salida de balón",
           "PrgC_p90": "Defensa de conducción", "Cmp_pct": "Defensa seguro en el pase"},
    "MF": {"Int_p90": "Pivote defensivo", "Recov_p90": "Recuperador", "KP_p90": "Creador", "PPA_p90": "Creador",
           "PrgC_p90": "Conductor", "PrgP_p90": "Organizador", "Touches_p90": "Organizador",
           "Cmp_pct": "Mediocentro de posesión"},
    "FW": {"npxG_p90": "Finalizador", "xAG_p90": "Asistente", "KP_p90": "Creador", "PPA_p90": "Creador",
           "Succ_p90": "Regateador", "PrgC_p90": "Conductor", "Touches_p90": "Atacante participativo"},
}

# Valoraciones del usuario y aprendizaje
# El usuario puntúa a los candidatos de 1 a 5 estrellas. La relevancia es (estrellas - 3) / 2:
# 4-5 = positiva, 3 = neutra (no influye), 1-2 = negativa.
ESTRELLAS_POSITIVA = 4
ESTRELLAS_NEGATIVA = 2
# Realimentación de relevancia de Rocchio sobre el vector de la referencia:
#   q' = ALFA·referencia + BETA·media(valorados en positivo) − GAMMA·media(valorados en negativo)
# BETA y GAMMA elegidos con usuarios simulados (evaluar_sistema.py): ayudan mucho cuando el usuario
# busca «alguien como la referencia pero con más de algo» y apenas cambian nada cuando no.
ROCCHIO_ALFA = 1.0
ROCCHIO_BETA = 0.5
ROCCHIO_GAMMA = 0.5
# Aprendizaje de la importancia de cada variable por grupo de posición: ordenación por pares
# (criterio BPR) sobre la similitud coseno ponderada. Parámetros elegidos con usuarios simulados
# (evaluar_sistema.py): aprenden rápido con valoraciones coherentes y no empeoran con ruido.
APRENDIZAJE_MIN_PARES = 1            # comparaciones necesarias para activar el aprendizaje
APRENDIZAJE_NITIDEZ = 25.0           # «a»: cuánto cuenta una diferencia de similitud entre dos candidatos
APRENDIZAJE_LAMBDA = 5.0             # fuerza de la regularización (más alto = aprende más despacio)
APRENDIZAJE_LIMITES = (0.25, 4.0)    # rango permitido de cada multiplicador

# Encaje en la plantilla del club de destino: peso de cada indicador en la nota de plantilla
# y peso de esa nota en la puntuación final (final = (1 − peso) × puntuación + peso × plantilla)
PESOS_ENCAJE = {"estilo_equipo": 0.4, "complemento": 0.3, "compatriotas": 0.2, "liga": 0.1}
PESO_PLANTILLA = 0.15

# Diversificación de la preselección (MMR): tamaño de la bolsa de candidatos que se reordena
MMR_BOLSA = 50

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

# Dimensiones que puntúa el Analista de rendimiento (escala 0-4)
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

# Modelos de lenguaje (se leen del entorno / archivo .env)
LLM_PROVEEDOR = (entorno("LLM_PROVEEDOR", "anthropic") or "ninguno").lower()
# Roles de los agentes LLM: Secretario técnico, Analista de rendimiento, Analista de plantilla, Director deportivo y Jefe de scouting
ROLES_LLM = ("interprete", "analista", "plantilla", "director", "critico")
# Nombre con el que se presenta cada agente en la aplicación y en los informes
NOMBRES_AGENTES = {
    "interprete": "Secretario técnico", "ojeador": "Ojeador de datos", "analista": "Analista de rendimiento",
    "plantilla": "Analista de plantilla", "director": "Director deportivo", "critico": "Jefe de scouting",
    "director_correccion": "Director deportivo (corrección)",
}
MODELOS_DEFECTO = {
    "anthropic": {"interprete": "claude-haiku-5-5", "analista": "claude-sonnet-5-5",
                  "director": "claude-haiku-5-5", "critico": "claude-haiku-5-5", "plantilla": "claude-haiku-5-5"},
    "gemini": {rol: "gemini-3.5-flash-lite" for rol in ROLES_LLM},
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
