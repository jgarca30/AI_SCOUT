"""Pulido determinista de los textos que escriben los modelos de lenguaje.
"""
from __future__ import annotations

import re

from . import config as C

# Palabras que sin tilde (o sin eñe) no existen en español o no tienen sentido en un informe.
# Clave: forma sin tilde; valor: forma correcta. Se aplican también en plural y femenino si se indica.
_PALABRAS = {
    "busqueda": "búsqueda", "busquedas": "búsquedas", "hibrido": "híbrido", "hibrida": "híbrida",
    "minimo": "mínimo", "minima": "mínima", "minimos": "mínimos", "minimas": "mínimas",
    "maximo": "máximo", "maxima": "máxima", "maximos": "máximos", "maximas": "máximas",
    "indice": "índice", "indices": "índices", "numero": "número", "numeros": "números",
    "analisis": "análisis", "ultimo": "último", "ultima": "última", "ultimos": "últimos", "ultimas": "últimas",
    "tambien": "también", "ademas": "además", "segun": "según", "balon": "balón", "linea": "línea",
    "lineas": "líneas", "metrica": "métrica", "metricas": "métricas", "exito": "éxito", "util": "útil",
    "utiles": "útiles", "dificil": "difícil", "facil": "fácil", "debil": "débil", "debiles": "débiles",
    "agil": "ágil", "movil": "móvil", "jovenes": "jóvenes", "aereo": "aéreo", "aereos": "aéreos",
    "rapido": "rápido", "rapida": "rápida", "solido": "sólido", "solida": "sólida", "unico": "único",
    "unica": "única", "pequeno": "pequeño", "pequena": "pequeña", "pequenos": "pequeños", "pequenas": "pequeñas",
    "companero": "compañero", "companeros": "compañeros", "desempeno": "desempeño", "tamano": "tamaño",
    "senala": "señala", "senalar": "señalar", "ensena": "enseña", "anade": "añade", "anadir": "añadir",
    "volumenes": "volúmenes", "caracter": "carácter", "tecnicamente": "técnicamente",
    "estadisticamente": "estadísticamente", "practicamente": "prácticamente", "proximo": "próximo",
    "proxima": "próxima", "proximos": "próximos", "proximas": "próximas", "optimo": "óptimo", "optima": "óptima",
    "deficit": "déficit", "superavit": "superávit", "regimen": "régimen", "nucleo": "núcleo",
    "area": "área", "areas": "áreas", "mas": "más", "anos": "años", "ano": "año", "asi": "así",
}
# Estas solo se corrigen en minúscula: con mayúscula podrían ser un nombre propio
_SOLO_MINUSCULA = {"area", "areas", "mas", "anos", "ano", "asi", "linea", "lineas"}
# Raíces de adjetivos esdrújulos: estadistic-o/a/os/as → estadístic-…
_RAICES = {
    "estadistic": "estadístic", "tecnic": "técnic", "tactic": "táctic", "fisic": "físic", "numeric": "numéric",
    "especific": "específic", "caracteristic": "característic", "basic": "básic", "practic": "práctic",
    "atletic": "atlétic", "dinamic": "dinámic", "estetic": "estétic", "tipic": "típic", "logic": "lógic",
}

_ES_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_CODIGO = re.compile(r"\b([A-Za-z]+_(?:p90|pct))\b")


def _como(original: str, corregida: str) -> str:
    """Devuelve la palabra corregida conservando la mayúscula inicial de la original."""
    return corregida.capitalize() if original[0].isupper() else corregida


def _corregir(palabra: str) -> str:
    if palabra.isupper() and len(palabra) > 1:        # siglas: no se tocan
        return palabra
    minuscula = palabra.lower()
    es_propia = palabra[0].isupper()
    if minuscula in _PALABRAS and not (es_propia and minuscula in _SOLO_MINUSCULA):
        return _como(palabra, _PALABRAS[minuscula])
    if len(minuscula) > 5 and minuscula.endswith(("cion", "sion")):          # -ción y -sión siempre llevan tilde
        return _como(palabra, minuscula[:-2] + "ón")
    for raiz, con_tilde in _RAICES.items():
        if minuscula.startswith(raiz) and minuscula[len(raiz):] in ("o", "a", "os", "as"):
            return _como(palabra, con_tilde + minuscula[len(raiz):])
    return palabra


def restaurar_tildes(texto: str) -> str:
    """Restaura tildes y eñes inequívocas. No modifica un texto que ya está bien escrito."""
    return _ES_PALABRA.sub(lambda m: _corregir(m.group(0)), texto)


def nombrar_variables(texto: str) -> str:
    """Sustituye los códigos de variable («Succ_p90») por su nombre («regates»)."""
    def nombre(m: re.Match) -> str:
        etiqueta = C.ETIQUETAS_CORTAS.get(m.group(1))
        if etiqueta is None:
            return m.group(1)
        etiqueta = etiqueta.rstrip(".")
        return etiqueta if any(c.isupper() for c in etiqueta[1:]) else etiqueta.lower()    # npxG y xAG se respetan
    return _CODIGO.sub(nombre, texto)


def pulir(texto):
    """Aplica todas las correcciones a un texto (cualquier otro valor se devuelve sin cambios)."""
    return restaurar_tildes(nombrar_variables(texto)) if isinstance(texto, str) else texto


def pulir_lista(textos) -> list:
    return [pulir(t) for t in textos] if isinstance(textos, list) else textos
