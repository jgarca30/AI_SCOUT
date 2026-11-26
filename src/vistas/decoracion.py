"""Decoración de la interfaz: estilos CSS, cabecera con el campo, tarjetas de cifras, cadena de agentes y cromos.

Todo se dibuja con HTML/SVG propio (sin imágenes externas) y funciona con el tema claro y el oscuro.
"""
from __future__ import annotations

from html import escape
from urllib.parse import quote
from pathlib import Path

import streamlit as st

from ai_scout import config as C
from ai_scout.director import fmt

RAIZ = Path(__file__).resolve().parents[2]
LOGO = RAIZ / "assets" / "logo.svg"
ICONO = RAIZ / "assets" / "icono.svg"

POSICION = {"DF": "Defensa", "MF": "Centrocampista", "FW": "Delantero"}
AGENTES = [
    ("chat", "Secretario técnico", "Convierte tu petición en filtros"),
    ("travel_explore", "Ojeador de datos", "Similitud coseno → 10 candidatos"),
    ("analytics", "Analista de rendimiento", "Valora a los 10 con IA"),
    ("groups", "Analista de plantilla", "Encaje en el club de destino"),
    ("edit_note", "Director deportivo", "Redacta las fichas del Top 3"),
    ("verified", "Jefe de scouting", "Revisa y pide correcciones"),
]

CSS = """
<style>
.ais-icono{font-family:'Material Symbols Rounded';font-weight:normal;font-style:normal;line-height:1;
  letter-spacing:normal;text-transform:none;white-space:nowrap;direction:ltr;-webkit-font-smoothing:antialiased;
  font-feature-settings:'liga';user-select:none}
.ais-hero{position:relative;overflow:hidden;border-radius:18px;background-color:#1d5a2b;color:#f2f8ef;
  padding:30px 34px;margin-bottom:.5rem;min-height:190px}
.ais-hero::after{content:'';position:absolute;right:4%;top:12%;bottom:12%;width:24%}
.ais-hero .ais-contenido{position:relative;max-width:62%}
.ais-hero .ais-sobre{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:#bfe6b3}
.ais-hero h1{color:#fff;font-size:2.6rem;font-weight:800;margin:.1rem 0 .2rem;padding:0;line-height:1.1}
.ais-hero p{color:#dcefd5;font-size:1.05rem;margin:0 0 .9rem}
.ais-chip{display:inline-block;font-size:.78rem;padding:3px 11px;margin:0 6px 6px 0;border-radius:99px;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.28);color:#f2f8ef}
.ais-cifras{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:.4rem 0 1rem}
.ais-cifra{border-radius:14px;padding:14px 16px;border:1px solid rgba(63,163,77,.35);background:rgba(63,163,77,.08);
  display:flex;align-items:center;gap:14px}
.ais-cifra .ais-icono{font-size:28px;color:#3fa34d;background:rgba(63,163,77,.15);border-radius:12px;padding:8px}
.ais-cifra b{display:block;font-size:1.6rem;font-weight:700;line-height:1.1}
.ais-cifra span.ais-txt{font-size:.85rem;opacity:.75}
.ais-agentes{display:flex;align-items:stretch;gap:4px;margin:.4rem 0 .8rem}
.ais-agente{flex:1 1 0;min-width:0;border-radius:14px;padding:12px 8px;border:1px solid rgba(128,128,128,.28);
  background:rgba(128,128,128,.06);text-align:center}
.ais-agente .ais-icono{font-size:30px;color:#3fa34d}
.ais-agente .ais-paso{font-size:.7rem;opacity:.6;text-transform:uppercase;letter-spacing:.06em}
.ais-agente b{display:block;font-size:.92rem;margin:2px 0}
.ais-agente span.ais-txt{font-size:.8rem;opacity:.75;line-height:1.3;display:block}
.ais-flecha{align-self:center;color:#3fa34d;font-size:22px}
[data-testid="stColumn"]:has(> div > [class*="st-key-tarjeta_cromo_"]) > div{height:100%}
[class*="st-key-tarjeta_cromo_"]{height:100%;justify-content:space-between;
  border-radius:16px;padding:14px 14px 10px;border:1.5px solid #c9a447;
  background:linear-gradient(160deg,rgba(201,164,71,.16),rgba(201,164,71,.03) 60%);transition:transform .15s}
[class*="st-key-tarjeta_cromo_"]:hover{transform:translateY(-3px)}
.ais-cromo{text-align:center}
.ais-avatar{width:58px;height:58px;margin:2px auto 8px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:1.2rem;background:#c9a447;color:#2b2410;
  border:3px solid rgba(255,255,255,.35)}
.ais-cromo b{display:block;font-size:1rem}
.ais-cromo .ais-club{font-size:.8rem;opacity:.75}
.ais-cromo .ais-estilo{display:inline-block;margin-top:6px;font-size:.75rem;padding:2px 10px;border-radius:99px;
  background:rgba(201,164,71,.22);color:inherit}
@media (max-width:760px){.ais-hero .ais-contenido{max-width:100%}.ais-hero::after{display:none}
  .ais-agentes{flex-wrap:wrap}.ais-agente{flex-basis:40%}.ais-flecha{display:none}}
</style>
"""

# Campo de fútbol visto desde arriba (franjas de césped y líneas) y pizarra con un 4-3-3. Van como imágenes de
# fondo en el CSS porque st.html elimina las etiquetas <svg> al sanear el HTML.
_CAMPO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 220" preserveAspectRatio="xMidYMid slice">
<g fill="#ffffff" fill-opacity=".045">{franjas}</g>
<g fill="none" stroke="#ffffff" stroke-opacity=".22" stroke-width="2">
<rect x="14" y="14" width="972" height="192" rx="2"/>
<line x1="500" y1="14" x2="500" y2="206"/><circle cx="500" cy="110" r="42"/>
<rect x="14" y="58" width="92" height="104"/><rect x="14" y="86" width="34" height="48"/>
<rect x="894" y="58" width="92" height="104"/><rect x="952" y="86" width="34" height="48"/>
<path d="M106 84 A30 30 0 0 1 106 136"/><path d="M894 84 A30 30 0 0 0 894 136"/></g>
<circle cx="500" cy="110" r="3" fill="#ffffff" fill-opacity=".35"/></svg>"""
_PIZARRA = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160">
<g fill="none" stroke="#f2d27a" stroke-opacity=".55" stroke-width="2" stroke-dasharray="5 5">
<path d="M70 80 Q110 40 150 30"/><path d="M100 130 Q145 120 168 85"/></g>
<g fill="#f2d27a" fill-opacity=".9">{jugadores}</g></svg>"""
_FORMACION = [(12, 80), (55, 15), (55, 58), (55, 102), (55, 145), (110, 38), (100, 80), (110, 122),
              (170, 22), (185, 80), (170, 138)]


def _url_svg(svg: str) -> str:
    return 'url("data:image/svg+xml,' + quote(" ".join(svg.split())) + '")'


@st.cache_data(show_spinner=False)
def _fondo_cabecera() -> str:
    franjas = "".join(f'<rect x="{x}" y="14" width="54" height="192"/>' for x in range(14, 986, 108))
    jugadores = "".join(f'<circle cx="{x}" cy="{y}" r="7"/>' for x, y in _FORMACION)
    campo, pizarra = _url_svg(_CAMPO.format(franjas=franjas)), _url_svg(_PIZARRA.format(jugadores=jugadores))
    return (f".ais-hero{{background-image:{campo};background-size:cover;background-position:center}}"
            f".ais-hero::after{{background:{pizarra} center/contain no-repeat}}")


def estilos() -> None:
    """Hoja de estilos común a todas las páginas (no ocupa espacio en la página)."""
    st.html(CSS)


def logo() -> None:
    if LOGO.exists():
        st.logo(str(LOGO), icon_image=str(ICONO) if ICONO.exists() else None, size="large")


def cabecera() -> None:
    """Banner de la portada. No depende de los datos, así que se ve mientras se cargan."""
    ligas = "".join(f'<span class="ais-chip">{escape(l)}</span>' for l in C.LIGAS_NORMALIZADAS)
    st.html(f"""
    <style>{_fondo_cabecera()}</style>
    <div class="ais-hero">
      <div class="ais-contenido">
        <div class="ais-sobre">Temporada {escape(C.TEMPORADA)} · 5 grandes ligas europeas</div>
        <h1>AI Scout</h1>
        <p>Encuentra el sustituto de cualquier jugador con estadística y un equipo de agentes de IA.</p>
        <div>{ligas}</div>
      </div>
    </div>""")


def tarjetas_cifras(cifras: list[tuple[str, str, str]]) -> None:
    """Fila de tarjetas (icono de Material Symbols, valor, texto)."""
    st.html('<div class="ais-cifras">' + "".join(
        f'<div class="ais-cifra"><span class="ais-icono">{icono}</span>'
        f'<div><b>{escape(valor)}</b><span class="ais-txt">{escape(texto)}</span></div></div>'
        for icono, valor, texto in cifras) + "</div>")


def cadena_agentes() -> None:
    piezas = []
    for i, (icono, nombre, texto) in enumerate(AGENTES, start=1):
        if i > 1:
            piezas.append('<span class="ais-icono ais-flecha">arrow_forward</span>')
        piezas.append(f'<div class="ais-agente"><span class="ais-icono">{icono}</span>'
                      f'<div class="ais-paso">Paso {i}</div><b>{escape(nombre)}</b>'
                      f'<span class="ais-txt">{escape(texto)}</span></div>')
    st.html('<div class="ais-agentes">' + "".join(piezas) + "</div>")


def _iniciales(nombre: str) -> str:
    partes = [p for p in nombre.split() if p[:1].isalpha()]
    return "".join(p[0] for p in (partes[:1] + partes[-1:] if len(partes) > 1 else partes)).upper()


def cromo(jugador: dict) -> None:
    """Parte de arriba del cromo de un jugador (el botón lo añade quien lo llama, dentro del mismo contenedor)."""
    estilo = f'<span class="ais-estilo">{escape(jugador["estilo"])}</span>' if jugador.get("estilo") else ""
    st.html(f"""
    <div class="ais-cromo">
      <div class="ais-avatar">{escape(_iniciales(jugador["nombre"]))}</div>
      <b>{escape(jugador["nombre"])}</b>
      <div class="ais-club">{escape(jugador["equipo"])}</div>
      <div class="ais-club">{escape(jugador["liga"])}</div>
      <div class="ais-club">{POSICION[jugador["grupo_posicion"]]} ·{fmt(jugador["minutos"], 0)} min</div>
      {estilo}
    </div>""")
