"""Informe de ojeador en PDF, y tabla de la preselección en CSV a partir del resultado de una consulta.


"""
from __future__ import annotations

import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as C
from .redaccion import restaurar_tildes

AZUL = (47, 84, 150)        # referencia
NARANJA = (197, 90, 17)     # candidato
GRIS = (110, 110, 110)
GRIS_CLARO = (238, 240, 244)
COLOR_VEREDICTO = {"candidato_para_revisar": (46, 125, 50), "candidato_con_reservas": (178, 120, 0),
                   "evidencia_insuficiente": (110, 110, 110)}
TEXTO_REVISION = {"aprobada": "Ficha revisada y aprobada por el Jefe de scouting.",
                  "corregida": "Ficha corregida por el director tras la revisión del Jefe de scouting.",
                  "con_observaciones": "El Jefe de scouting dejó observaciones que no se pudieron corregir:"}

# (normal, negrita, cursiva) de fuentes con buen soporte de caracteres latinos extendidos
FUENTES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeuii.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
     "/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf"),
]


def _num(x, dec: int = 1) -> str:
    """Número con coma decimal y punto de millar; «—» si no hay valor."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _corto(nombre: str, largo: int = 18) -> str:
    return nombre if len(nombre) <= largo else nombre[:largo - 1].rstrip() + "…"


def _edad(edad) -> str:
    return "edad no disponible" if edad is None or pd.isna(edad) else f"{int(edad)} años"


def _ficha_jugador(j: dict) -> str:
    partes = [j["equipo"], j["liga"], j["posicion_original"], _edad(j.get("edad_registrada")),
              f"{_num(j['minutos'], 0)} min"]
    if j.get("estilo"):
        partes.append(f"estilo: {j['estilo']}")
    return " · ".join(partes)


def tabla_preseleccion(resultado: dict) -> pd.DataFrame:
    """Tabla del Top 10 tal como se muestra en la app (sirve para la descarga en CSV)."""
    return pd.DataFrame([{
        "Puesto": c["ranking_final"], "Jugador": c["nombre"], "Equipo": c["equipo"], "Liga": c["liga"],
        "Posición": c["posicion_original"], "Edad": c["edad_registrada"], "Minutos": c["minutos"],
        "Estilo": c.get("estilo"), "Índice de similitud": round(c["score_numerico"], 1),
        "Similitud base": None if c.get("score_base") is None else round(c["score_base"], 1),
        "Valoración IA": None if c.get("score_cualitativo") is None else round(c["score_cualitativo"], 1),
        "Plantilla": None if c.get("score_plantilla") is None else round(c["score_plantilla"], 1),
        "Puntuación final": round(c["score_final"], 1), "Puesto numérico": c["ranking_preliminar"],
        "Veredicto": C.VEREDICTOS.get((c.get("ficha") or {}).get("veredicto"), ""),
        "Tu valoración": c.get("valoracion_usuario"),
    } for c in resultado["ranking"]])


def csv_preseleccion(resultado: dict) -> bytes:
    """CSV con punto y coma y BOM, para que Excel lo abra bien en español."""
    return tabla_preseleccion(resultado).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def generar_pdf(resultado: dict, valoraciones: dict[str, int] | None = None) -> bytes:
    """PDF del informe de ojeador. `valoraciones`: estrellas actuales del usuario por registro_id (opcional)."""
    from fpdf import FPDF

    valoraciones = valoraciones or {}
    consulta, rec = resultado["consulta"], resultado["recuperacion"]
    objetivo, variables = rec["objetivo"], rec["variables"]

    class Informe(FPDF):
        unicode = False

        def footer(self):
            self.set_y(-12)
            self.set_font(self.fuente, "", 8)
            self.set_text_color(*GRIS)
            self.cell(0, 5, self.t(f"AI Scout · consulta {consulta['consulta_id']} · página {self.page_no()}"),
                      align="C")

        def t(self, texto) -> str:
            """Texto apto para la fuente activa."""
            texto = "" if texto is None else str(texto)
            if self.unicode:
                return texto
            texto = texto.replace("≤", "<=").replace("≥", ">=").replace("−", "-").replace("…", "...")
            plano = "".join(c if ord(c) < 256 else unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode()
                            or "?" for c in texto)
            return plano.encode("latin-1", "replace").decode("latin-1")

    pdf = Informe(format="A4")
    pdf.fuente = "Helvetica"
    for normal, negrita, cursiva in FUENTES:
        if all(Path(f).exists() for f in (normal, negrita, cursiva)):
            try:
                pdf.add_font("Informe", "", normal)
                pdf.add_font("Informe", "B", negrita)
                pdf.add_font("Informe", "I", cursiva)
                pdf.fuente, pdf.unicode = "Informe", True
                break
            except Exception:   # fuente ilegible: se prueba la siguiente  # noqa: BLE001
                continue
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_title(pdf.t(f"AI Scout · sustitutos para {objetivo['nombre']}"))
    pdf.add_page()
    ancho = pdf.epw

    def texto(contenido, tam=10, estilo="", color=(0, 0, 0), alto=5.0, sangria=0.0):
        pdf.set_font(pdf.fuente, estilo, tam)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin + sangria)
        pdf.multi_cell(ancho - sangria, alto, pdf.t(contenido), new_x="LMARGIN", new_y="NEXT")

    def titulo_seccion(contenido):
        if pdf.get_y() > pdf.h - 45:
            pdf.add_page()
        pdf.ln(3)
        texto(contenido, 12, "B", AZUL, 6)
        pdf.set_draw_color(*AZUL)
        pdf.line(pdf.l_margin, pdf.get_y() + 0.5, pdf.l_margin + ancho, pdf.get_y() + 0.5)
        pdf.ln(2.5)

    def lista(titulo, elementos):
        if not elementos:
            return
        texto(titulo, 9.5, "B", alto=5)
        for e in elementos:
            texto(f"•  {e}", 9.5, alto=4.6, sangria=3)

    # ---- Cabecera ----
    pdf.set_fill_color(*AZUL)
    pdf.rect(0, 0, pdf.w, 26, style="F")
    pdf.set_xy(pdf.l_margin, 7)
    pdf.set_font(pdf.fuente, "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(ancho, 8, pdf.t("AI Scout · Informe de ojeador"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(pdf.fuente, "", 9)
    fecha = consulta.get("fecha_utc", "")[:10] or datetime.now().strftime("%Y-%m-%d")
    pdf.cell(ancho, 5, pdf.t(f"Recomendador de sustitutos por similitud estadística · temporada {C.TEMPORADA} · "
                             f"consulta del {fecha}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(31)
    texto(f"Sustitutos para {objetivo['nombre']}", 15, "B", alto=7)
    texto(_ficha_jugador(objetivo), 9.5, color=GRIS)

    # ---- Parámetros ----
    titulo_seccion("Búsqueda")
    filtros = [f"Ligas: {', '.join(consulta['ligas_permitidas'])}", f"Minutos mínimos: {_num(consulta['minutos_minimos'], 0)}"]
    if consulta.get("edad_maxima") is not None:
        filtros.append(f"Edad máxima: {consulta['edad_maxima']} años")
    if consulta.get("solo_mismo_estilo"):
        filtros.append("Solo jugadores del mismo estilo de juego")
    if consulta.get("diversidad"):
        filtros.append(f"Diversidad de la preselección: {_num(consulta['diversidad'], 1)}")
    texto(" · ".join(filtros), 9.5)
    modo = "híbrido (similitud + análisis con IA)" if resultado["modo_ranking"] == "hibrido" else "numérico"
    texto(f"Rol buscado: {rec.get('rol_nombre', 'Equilibrado')} · Ranking {modo} · "
          f"{rec['n_candidatos_validos']} jugadores comparados del grupo {C.GRUPOS[rec['grupo']].lower()}", 9.5)
    p = resultado.get("personalizacion") or {}
    if p.get("valoraciones_grupo"):
        partes = [f"{p['valoraciones_grupo']} valoraciones tuyas en este grupo de posición"]
        if p.get("pesos_aprendidos"):
            partes.append("pesos de las variables ajustados con lo aprendido")
        if p.get("rocchio_aplicado"):
            partes.append(f"búsqueda adaptada con {p['valoraciones_referencia']} valoraciones de esta referencia")
        if p.get("descartados"):
            partes.append(f"{p['descartados']} candidato(s) descartado(s) por ti")
        texto("Personalización: " + " · ".join(partes) + ".", 9.5)
    if consulta.get("texto_original"):
        texto(f"Petición original: «{consulta['texto_original']}»", 9.5, "I", GRIS)
    destacadas = sorted(rec.get("pesos", {}).items(), key=lambda kv: -kv[1])
    if destacadas and destacadas[0][1] > 1.05:
        texto("Variables con más peso: " + ", ".join(
            f"{C.ETIQUETAS_CORTAS.get(v, v)} (×{_num(w, 2)})" for v, w in destacadas[:4] if w > 1.05), 9.5)
    if resultado.get("resumen_consulta"):
        pdf.ln(1.5)
        texto(resultado["resumen_consulta"], 10, "I")
    for aviso in resultado.get("avisos", []):
        texto(f"Aviso: {aviso}", 9, color=NARANJA)

    # ---- Fichas ----
    titulo_seccion("Fichas de ojeador")
    for c in resultado["top_final"]:
        ficha = c["ficha"]
        if pdf.get_y() > pdf.h - 95:
            pdf.add_page()
        y0 = pdf.get_y()
        pdf.set_fill_color(*GRIS_CLARO)
        pdf.rect(pdf.l_margin, y0, ancho, 13, style="F")
        pdf.set_xy(pdf.l_margin + 2, y0 + 1.2)
        pdf.set_font(pdf.fuente, "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(ancho - 60, 6, pdf.t(f"{c['ranking_final']}. {c['nombre']}"))
        pdf.set_font(pdf.fuente, "B", 10)
        pdf.set_text_color(*COLOR_VEREDICTO.get(ficha.get("veredicto"), GRIS))
        pdf.cell(56, 6, pdf.t(C.VEREDICTOS.get(ficha.get("veredicto"), "")), align="R")
        pdf.set_xy(pdf.l_margin + 2, y0 + 7)
        pdf.set_font(pdf.fuente, "", 8.5)
        pdf.set_text_color(*GRIS)
        pdf.cell(ancho - 4, 5, pdf.t(_ficha_jugador(c)))
        pdf.set_y(y0 + 15)

        puntuaciones = [f"Índice de similitud: {_num(c['score_numerico'])}"]
        if resultado["modo_ranking"] == "hibrido":
            puntuaciones.append(f"Analista de rendimiento: {_num(c.get('score_cualitativo'))}")
        if c.get("plantilla"):
            puntuaciones.append(f"Encaje en la plantilla: {_num(c['score_plantilla'])}")
        if resultado["modo_ranking"] == "hibrido" or c.get("plantilla"):
            puntuaciones.append(f"Puntuación final: {_num(c['score_final'])}")
        estrellas = valoraciones.get(c["registro_id"], c.get("valoracion_usuario"))
        if estrellas:
            puntuaciones.append(f"Tu valoración: {int(estrellas)} de 5 estrellas")
        texto(" · ".join(puntuaciones), 9.5, "B", AZUL)
        texto(ficha.get("resumen", ""), 10)
        lista("Fortalezas", ficha.get("fortalezas"))
        lista("Diferencias", ficha.get("diferencias_relevantes"))
        lista("Reservas", ficha.get("reservas"))
        analisis = c.get("analisis")
        if analisis:
            notas = [f"{restaurar_tildes(d['nombre'].replace('_', ' '))}: {d['puntuacion']}/4"
                     for d in analisis.get("dimensiones", [])
                     if d.get("puntuacion") is not None]
            lista("Analista de rendimiento", [x for x in (analisis.get("resumen"), " · ".join(notas)) if x])
        if c.get("plantilla"):
            lista("Analista de plantilla", [c["plantilla"]["comentario"]])

        # Tabla de métricas con barras de percentil (referencia en azul, candidato en naranja)
        pdf.ln(1.5)
        columnas = [(40, "Variable", "L", GRIS), (30, _corto(objetivo["nombre"]), "R", AZUL),
                    (30, _corto(c["nombre"]), "R", NARANJA), (16, "Peso", "R", GRIS),
                    (ancho - 116, "Percentil en su grupo (0-100)", "L", GRIS)]
        pdf.set_font(pdf.fuente, "B", 8)
        for w, cab, alin, color in columnas:      # el color de cada nombre es el de su barra
            pdf.set_text_color(*color)
            pdf.cell(w, 5, pdf.t(cab), align=alin)
        pdf.ln(5)
        pdf.set_font(pdf.fuente, "", 8.5)
        pdf.set_text_color(0, 0, 0)
        pesos = rec.get("pesos", {})
        for d in c["diferencias"]:
            v = d["variable"]
            y = pdf.get_y()
            pdf.cell(columnas[0][0], 4.8, pdf.t(C.ETIQUETAS_CORTAS.get(v, v)))
            pdf.cell(columnas[1][0], 4.8, _num(d["valor_objetivo"], 2), align="R")
            pdf.cell(columnas[2][0], 4.8, _num(d["valor_candidato"], 2), align="R")
            pdf.cell(columnas[3][0], 4.8, "×" + _num(pesos.get(v, 1.0), 2) if pdf.unicode
                     else "x" + _num(pesos.get(v, 1.0), 2), align="R")
            x_barra, w_barra = pdf.get_x() + 3, columnas[4][0] - 6
            pdf.set_fill_color(225, 228, 234)
            pdf.rect(x_barra, y + 0.7, w_barra, 3.4, style="F")
            for desplazamiento, jugador, color in ((0.7, objetivo, AZUL), (2.4, c, NARANJA)):
                percentil = jugador.get("percentiles", {}).get(v)
                if percentil is not None and not pd.isna(percentil):
                    pdf.set_fill_color(*color)
                    pdf.rect(x_barra, y + desplazamiento, w_barra * max(0.0, min(100.0, percentil)) / 100.0, 1.7,
                             style="F")
            pdf.ln(4.8)

        revision = ficha.get("revision") or {}
        if revision.get("estado") in TEXTO_REVISION:
            pdf.ln(1)
            texto(TEXTO_REVISION[revision["estado"]], 8.5, "I", GRIS, alto=4.2)
            if revision["estado"] == "con_observaciones":
                for problema in revision.get("problemas", []):
                    texto(f"•  {problema}", 8.5, "I", GRIS, alto=4.2, sangria=3)
        origen = "Ficha redactada por el Director deportivo (modelo de lenguaje)." if ficha.get("origen") == "llm" \
            else "Ficha generada automáticamente, sin modelo de lenguaje."
        texto(origen, 8, "I", GRIS, alto=4)
        pdf.ln(3)

    # ---- Encaje en la plantilla ----
    plantilla = resultado.get("plantilla")
    if plantilla:
        titulo_seccion(f"Encaje en la plantilla de {plantilla['equipo']}")
        if plantilla.get("diagnostico"):
            texto(plantilla["diagnostico"], 10, "I")
        estilos_linea = ", ".join(f"{e} ({_num(p * 100, 0)} %)" for e, p in (plantilla.get("estilos_de_la_linea") or {}).items())
        if estilos_linea:
            texto(f"Estilos entre los compañeros de línea, sin la referencia: {estilos_linea}.", 9.5)
        for c in resultado["ranking"]:
            texto(f"{c['ranking_final']}. {c['nombre']} · nota de plantilla {_num(c['score_plantilla'])} · "
                  f"{c['plantilla']['comentario']}", 9, alto=4.6)
        origen = "el Analista de plantilla (modelo de lenguaje) y los indicadores" if plantilla["origen"] == "llm" \
            else "los indicadores, sin modelo de lenguaje"
        texto(f"Nota calculada con {origen}; pesa {_num(plantilla['peso'], 2)} en la puntuación final. Es una "
              "aproximación: el dataset no tiene datos directos de química entre jugadores.", 8, "I", GRIS, alto=4)

    # ---- Preselección ----
    if pdf.get_y() + 20 + 5 * len(resultado["ranking"]) > pdf.h - 16:   # la tabla no se parte entre páginas
        pdf.add_page()
    titulo_seccion(f"Preselección (Top {len(resultado['ranking'])})")
    hibrido = resultado["modo_ranking"] == "hibrido"
    columnas = [(8, "#", "R"), (46, "Jugador", "L"), (34, "Equipo", "L"), (25, "Liga", "L"), (10, "Edad", "R"),
                (14, "Min.", "R"), (17, "Similitud", "R")] + ([(12, "IA", "R")] if hibrido else []) \
        + ([(16, "Plantilla", "R")] if plantilla else []) + ([(14, "Final", "R")] if hibrido or plantilla else [])
    if plantilla and hibrido:      # con todas las columnas, se estrechan las de texto para caber en la página
        columnas[1], columnas[2], columnas[3] = (40, "Jugador", "L"), (26, "Equipo", "L"), (23, "Liga", "L")
    pdf.set_font(pdf.fuente, "B", 8.5)
    pdf.set_text_color(*GRIS)
    for w, cab, alin in columnas:
        pdf.cell(w, 5.5, pdf.t(cab), align=alin)
    pdf.ln(5.5)
    pdf.set_font(pdf.fuente, "", 8.5)
    pdf.set_text_color(0, 0, 0)
    for n, c in enumerate(resultado["ranking"]):
        if n % 2 == 0:
            pdf.set_fill_color(*GRIS_CLARO)
            pdf.rect(pdf.l_margin, pdf.get_y(), sum(w for w, _, _ in columnas), 5, style="F")
        celdas = [str(c["ranking_final"]), _corto(c["nombre"], 24 if plantilla and hibrido else 28), _corto(c["equipo"], 21), c["liga"],
                  "—" if c["edad_registrada"] is None else str(c["edad_registrada"]), _num(c["minutos"], 0),
                  _num(c["score_numerico"])]
        celdas[2] = _corto(c["equipo"], 16 if plantilla and hibrido else 21)
        if hibrido:
            celdas.append(_num(c.get("score_cualitativo")))
        if plantilla:
            celdas.append(_num(c.get("score_plantilla")))
        if hibrido or plantilla:
            celdas.append(_num(c["score_final"]))
        for (w, _, alin), valor in zip(columnas, celdas):
            pdf.cell(w, 5, pdf.t(valor), align=alin)
        pdf.ln(5)

    # ---- Trabajo de los agentes ----
    if resultado.get("ejecuciones"):
        titulo_seccion("Trabajo de los agentes")
        estados = {"ok": "respuesta válida", "cache": "respuesta recuperada de la caché",
                   "validacion_fallida": "respuesta rechazada por la validación", "json_invalido": "respuesta ilegible",
                   "error_api": "error del proveedor"}
        for e in resultado["ejecuciones"]:
            duracion = f" · {_num((e.get('duracion_ms') or 0) / 1000, 1)} s" if e.get("duracion_ms") else ""
            texto(f"•  {C.NOMBRES_AGENTES.get(e['agente'], e['agente'])}: {estados.get(e['estado'], e['estado'])}"
                  f"{duracion}", 9, alto=4.5)

    # ---- Limitaciones y trazabilidad ----
    titulo_seccion("Limitaciones y trazabilidad")
    for limitacion in resultado.get("limitaciones_generales", []):
        texto(f"•  {limitacion}", 9, alto=4.5)
    texto("•  Los índices miden parecido estadístico, no la probabilidad de éxito de un fichaje.", 9, alto=4.5)
    modelos = sorted({f"{e['proveedor']}/{e['modelo']}" for e in resultado.get("ejecuciones", [])})
    pdf.ln(1.5)
    texto(f"Dataset {resultado['dataset_version']} · {resultado['perfil_version']} · {C.PROMPT_VERSION} · "
          + (f"modelos: {', '.join(modelos)}" if modelos else "sin modelos de lenguaje") +
          f" · consulta {consulta['consulta_id']}", 8, color=GRIS, alto=4)
    return bytes(pdf.output())
