"""Respuestas simuladas de los agentes LLM para las pruebas (sin llamadas externas)."""
import json
import re

from ai_scout.llm import ErrorLLM


def payload(mensaje: str) -> dict:
    return json.loads(re.search(r"```json\n(.*)\n```", mensaje, re.S).group(1))


def analista_valido(sistema: str, usuario: str, puntuaciones=(3, 2)) -> str:
    p = payload(usuario)
    ref = p["referencia"]["valores"]
    salida = []
    for c in p["candidatos"]:
        variables = list(c["valores"])
        dims = []
        for nombre, punt, var in zip(("progresion_creacion", "coherencia_perfil"), puntuaciones, variables):
            dims.append({"nombre": nombre, "puntuacion": punt,
                         "evidencias": [{"variable": var, "valor_objetivo": ref[var],
                                         "valor_candidato": c["valores"][var]}],
                         "justificacion": "Valores cercanos en el indicador citado."})
        salida.append({"candidato_registro_id": c["candidato_registro_id"], "estado": "evaluado",
                       "dimensiones": dims, "semejanzas": ["Volumen de progresión parecido."],
                       "diferencias": ["Menos acciones defensivas."],
                       "limitaciones": ["No hay datos de lesiones ni de contexto táctico."],
                       "resumen": "Perfil próximo al de referencia."})
    return json.dumps({"candidatos": salida}, ensure_ascii=False)


def analista_valor_inventado(sistema, usuario):
    datos = json.loads(analista_valido(sistema, usuario))
    datos["candidatos"][0]["dimensiones"][0]["evidencias"][0]["valor_candidato"] += 5
    return json.dumps(datos)


def analista_candidato_extra(sistema, usuario):
    datos = json.loads(analista_valido(sistema, usuario))
    extra = dict(datos["candidatos"][0], candidato_registro_id="R9999")
    datos["candidatos"].append(extra)
    return json.dumps(datos)


def analista_termino_prohibido(sistema, usuario):
    datos = json.loads(analista_valido(sistema, usuario))
    datos["candidatos"][0]["semejanzas"] = ["Gran velocidad y buena lectura bajo presión."]
    return json.dumps(datos)


def analista_no_evaluable(sistema, usuario):
    return analista_valido(sistema, usuario, puntuaciones=(3,))


def director_valido(sistema: str, usuario: str) -> str:
    p = payload(usuario)
    fichas = [{"candidato_registro_id": c["candidato_registro_id"],
               "resumen": f"Índice de similitud de {c['indice_similitud']} con {c['minutos']} minutos.",
               "fortalezas": ["Progresión parecida a la referencia."],
               "diferencias_relevantes": ["Menor participación con balón."],
               "reservas": ["No se conoce su situación contractual."],
               "veredicto": "candidato_para_revisar"} for c in p["candidatos"]]
    return json.dumps({"resumen_consulta": "Perfiles estadísticamente parecidos al de referencia.",
                       "limitaciones_generales": ["El contexto táctico no está en los datos."],
                       "candidatos": fichas}, ensure_ascii=False)


def director_cifra_inventada(sistema, usuario):
    datos = json.loads(director_valido(sistema, usuario))
    datos["candidatos"][0]["resumen"] = "Tiene un 73,5 % de probabilidad de rendir igual."
    return json.dumps(datos, ensure_ascii=False)


def texto_no_json(sistema, usuario):
    return "Lo siento, aquí tienes mi análisis en prosa."


class Secuencia:
    """Devuelve respuestas distintas en cada llamada (para probar reintentos)."""

    def __init__(self, *funciones):
        self.funciones = list(funciones)
        self.n = 0

    def __call__(self, sistema, usuario):
        f = self.funciones[min(self.n, len(self.funciones) - 1)]
        self.n += 1
        return f(sistema, usuario)


def error_api(transitorio: bool):
    def f(sistema, usuario):
        raise ErrorLLM("fallo simulado", "http_503" if transitorio else "http_401", transitorio=transitorio)
    return f


# ---- Jefe de scouting (revisor) ----
def critico_aprueba(sistema: str, usuario: str) -> str:
    p = payload(usuario)
    return json.dumps({"revisiones": [{"candidato_registro_id": c["candidato_registro_id"], "aprobada": True,
                                       "problemas": []} for c in p["fichas"]["candidatos"]],
                       "valoracion_global": "Fichas coherentes con los datos."}, ensure_ascii=False)


def critico_rechaza_primera(sistema: str, usuario: str) -> str:
    datos = json.loads(critico_aprueba(sistema, usuario))
    datos["revisiones"][0].update(aprobada=False, problemas=["La fortaleza citada no concuerda con los valores."])
    return json.dumps(datos, ensure_ascii=False)


def critico_cifra_inventada(sistema: str, usuario: str) -> str:
    datos = json.loads(critico_aprueba(sistema, usuario))
    datos["revisiones"][0].update(aprobada=False, problemas=["Dice 12,34 pases cuando el dato es 987,65."])
    return json.dumps(datos, ensure_ascii=False)


def critico_incoherente(sistema: str, usuario: str) -> str:
    datos = json.loads(critico_aprueba(sistema, usuario))
    datos["revisiones"][0]["problemas"] = ["Aprobada pero con un problema."]
    return json.dumps(datos, ensure_ascii=False)


def director_corregido(sistema: str, usuario: str) -> str:
    """Segunda versión del director tras las observaciones del revisor."""
    datos = json.loads(director_valido(sistema, usuario))
    for ficha in datos["candidatos"]:
        ficha["fortalezas"] = ["Versión corregida tras la revisión."]
    return json.dumps(datos, ensure_ascii=False)


# ---- Secretario técnico (intérprete) ----
def interprete(**campos):
    base = {"jugador_referencia": None, "equipo_referencia": None, "ligas": [], "edad_maxima": None,
            "minutos_minimos": None, "rol": None, "solo_mismo_estilo": False, "no_aplicable": []}
    base.update(campos)
    return lambda sistema, usuario: json.dumps(base, ensure_ascii=False)


# ---- Analista de plantilla ----
def plantilla_valida(sistema: str, usuario: str, encaje: int = 3) -> str:
    p = payload(usuario)
    return json.dumps({"diagnostico_plantilla": "La línea pierde a su jugador de referencia.",
                       "candidatos": [{"candidato_registro_id": c["candidato_registro_id"], "encaje": encaje,
                                       "comentario": "Encaje razonable según los indicadores."}
                                      for c in p["candidatos"]]}, ensure_ascii=False)


def plantilla_invalida(sistema: str, usuario: str) -> str:
    datos = json.loads(plantilla_valida(sistema, usuario))
    datos["candidatos"][0]["encaje"] = 9
    return json.dumps(datos, ensure_ascii=False)
