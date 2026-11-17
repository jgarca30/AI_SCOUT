"""Aplicación web de AI Scout (Streamlit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vistas import aprendizaje, buscar, estilos_de_juego, evaluacion, historial  # noqa: E402

st.set_page_config(page_title="AI Scout", page_icon="⚽", layout="wide")

paginas = {
    "buscar": st.Page(buscar.pagina, title="Buscar sustitutos", icon=":material/search:", url_path="buscar",
                      default=True),
    "estilos": st.Page(estilos_de_juego.pagina, title="Estilos de juego", icon=":material/scatter_plot:",
                       url_path="estilos"),
    "aprendizaje": st.Page(aprendizaje.pagina, title="Lo aprendido de ti", icon=":material/psychology:",
                           url_path="aprendizaje"),
    "historial": st.Page(historial.pagina, title="Historial", icon=":material/history:", url_path="historial"),
    "evaluacion": st.Page(evaluacion.pagina, title="Evaluación", icon=":material/monitoring:",
                          url_path="evaluacion"),
}
st.session_state["paginas"] = paginas   # para cambiar de página desde el código (st.switch_page)

pagina = st.navigation(list(paginas.values()))
with st.sidebar:
    st.title("⚽ AI Scout")
pagina.run()
