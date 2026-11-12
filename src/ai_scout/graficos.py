"""Gráficos de la aplicación (Plotly)."""
from __future__ import annotations

from . import config as C

COLOR_REFERENCIA = "#2F5496"
COLOR_CANDIDATO = "#C55A11"


def radar_percentiles(objetivo: dict, candidato: dict, variables: list[str]):
    """Radar de percentiles (0-100) dentro del grupo de posición: referencia frente a candidato."""
    import plotly.graph_objects as go

    etiquetas = [C.ETIQUETAS_CORTAS.get(v, v) for v in variables]
    cerrar = lambda xs: xs + xs[:1]  # noqa: E731
    fig = go.Figure()
    for nombre, datos, color in ((objetivo["nombre"], objetivo, COLOR_REFERENCIA),
                                 (candidato["nombre"], candidato, COLOR_CANDIDATO)):
        valores = [round(datos["percentiles"].get(v, 0), 1) for v in variables]
        reales = [datos["valores"][v] for v in variables]
        fig.add_trace(go.Scatterpolar(
            r=cerrar(valores), theta=cerrar(etiquetas), name=nombre, fill="toself", opacity=0.55,
            line=dict(color=color, width=2),
            customdata=cerrar(reales),
            hovertemplate="%{theta}<br>Percentil %{r:.0f}<br>Valor %{customdata:.2f}<extra>" + nombre + "</extra>",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], tickfont=dict(size=10))),
        legend=dict(orientation="h", y=-0.12), margin=dict(l=40, r=40, t=30, b=30), height=430,
    )
    return fig
