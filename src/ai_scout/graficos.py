"""Gráficos de la aplicación (Plotly)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

COLOR_REFERENCIA = "#2F5496"
COLOR_CANDIDATO = "#C55A11"
# Colores de los candidatos cuando se comparan varios a la vez y de los estilos de juego
COLORES_CANDIDATOS = ["#C55A11", "#2E8B57", "#8E44AD"]
COLORES_ESTILOS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
COLOR_SUBE, COLOR_BAJA, COLOR_NEUTRO = "#2E8B57", "#C0392B", "#8A8F98"
MARGEN = dict(l=40, r=30, t=40, b=40)


def _etiqueta(variable: str) -> str:
    return C.ETIQUETAS_CORTAS.get(variable, variable)


def radar_percentiles(objetivo: dict, candidatos: dict | list[dict], variables: list[str]):
    """Radar de percentiles (0-100) dentro del grupo de posición: referencia frente a uno o varios candidatos."""
    import plotly.graph_objects as go

    if isinstance(candidatos, dict):
        candidatos = [candidatos]
    etiquetas = [_etiqueta(v) for v in variables]
    cerrar = lambda xs: xs + xs[:1]  # noqa: E731
    series = [(objetivo, COLOR_REFERENCIA)] + [(c, COLORES_CANDIDATOS[i % len(COLORES_CANDIDATOS)])
                                              for i, c in enumerate(candidatos)]
    fig = go.Figure()
    for datos, color in series:
        nombre = datos["nombre"]
        valores = [round(datos["percentiles"].get(v, 0), 1) for v in variables]
        reales = [datos["valores"][v] for v in variables]
        fig.add_trace(go.Scatterpolar(
            r=cerrar(valores), theta=cerrar(etiquetas), name=nombre, fill="toself",
            opacity=0.55 if len(series) <= 2 else 0.4, line=dict(color=color, width=2),
            customdata=cerrar(reales),
            hovertemplate="%{theta}<br>Percentil %{r:.0f}<br>Valor %{customdata:.2f}<extra>" + nombre + "</extra>",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], tickfont=dict(size=10))),
        legend=dict(orientation="h", y=-0.12), margin=dict(l=40, r=40, t=30, b=30), height=430,
    )
    return fig


def mapa_estilos(tabla: pd.DataFrame, nombres_estilos: list[str], eje_x: str, eje_y: str,
                 objetivo_id: str | None = None, candidatos: list[dict] | None = None, altura: int = 560):
    """Mapa de jugadores (dos primeras componentes principales) coloreado por estilo de juego.

    `tabla`: jugadores de un grupo con registro_id, nombre, equipo, minutos, estilo, pca_x y pca_y.
    Si se indican, la referencia se dibuja como una estrella y los candidatos con su puesto.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, estilo in enumerate(nombres_estilos):
        t = tabla[tabla["estilo"] == estilo]
        fig.add_trace(go.Scatter(
            x=t["pca_x"], y=t["pca_y"], mode="markers", name=estilo,
            marker=dict(size=6, color=COLORES_ESTILOS[i % len(COLORES_ESTILOS)], opacity=0.55),
            customdata=np.stack([t["nombre"], t["equipo"], t["minutos"]], axis=-1) if len(t) else None,
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]:,.0f} min<extra>" + estilo
                          + "</extra>",
        ))
    por_registro = tabla.set_index("registro_id")
    if candidatos:
        presentes = [c for c in candidatos if c["registro_id"] in por_registro.index]
        if presentes:
            puntos = por_registro.loc[[c["registro_id"] for c in presentes]]
            fig.add_trace(go.Scatter(
                x=puntos["pca_x"], y=puntos["pca_y"], mode="markers+text", name="Candidatos",
                text=[str(c["ranking_final"]) for c in presentes], textposition="middle center",
                textfont=dict(color="white", size=11),
                marker=dict(size=20, color=COLOR_CANDIDATO, line=dict(color="white", width=1.5)),
                customdata=np.stack([[c["nombre"] for c in presentes], [c["equipo"] for c in presentes],
                                     [round(c["score_final"], 1) for c in presentes]], axis=-1),
                hovertemplate="%{text}. %{customdata[0]} · %{customdata[1]}<br>Puntuación %{customdata[2]}"
                              "<extra>Candidato</extra>",
            ))
    if objetivo_id is not None and objetivo_id in por_registro.index:
        ref = por_registro.loc[objetivo_id]
        fig.add_trace(go.Scatter(
            x=[ref["pca_x"]], y=[ref["pca_y"]], mode="markers+text", name="Referencia", text=[ref["nombre"]],
            textposition="top center", textfont=dict(size=13),
            marker=dict(size=24, symbol="star", color=COLOR_REFERENCIA, line=dict(color="white", width=1.5)),
            hovertemplate=f"{ref['nombre']} · {ref['equipo']}<extra>Referencia</extra>",
        ))
    fig.update_layout(xaxis_title=eje_x, yaxis_title=eje_y, height=altura, margin=MARGEN,
                      legend=dict(orientation="h", y=-0.18), hovermode="closest")
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def barras_pesos(pesos_rol: dict[str, float], pesos: dict[str, float]):
    """Importancia de cada variable: lo que marca el rol y lo que se ha aplicado tras lo aprendido."""
    import plotly.graph_objects as go

    variables = list(pesos)
    etiquetas = [_etiqueta(v) for v in variables][::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=etiquetas, x=[pesos_rol.get(v, 1.0) for v in variables][::-1], orientation="h",
                         name="Rol buscado", marker_color=COLOR_NEUTRO,
                         hovertemplate="%{y}: ×%{x:.2f}<extra>Rol buscado</extra>"))
    fig.add_trace(go.Bar(y=etiquetas, x=[pesos[v] for v in variables][::-1], orientation="h",
                         name="Aplicado en esta búsqueda", marker_color=COLOR_REFERENCIA,
                         hovertemplate="%{y}: ×%{x:.2f}<extra>Aplicado</extra>"))
    fig.add_vline(x=1.0, line_dash="dot", line_color="rgba(128,128,128,0.7)")
    fig.update_layout(barmode="group", height=60 + 42 * len(variables), margin=MARGEN,
                      xaxis_title="Peso (1 = importancia normal)", legend=dict(orientation="h", y=-0.25))
    return fig


def barras_multiplicadores(multiplicadores: dict[str, float], titulo_x: str = "Multiplicador aprendido "
                                                                              "(1 = sin cambios)"):
    """Lo aprendido de las valoraciones: variables que ganan peso (verde) y que lo pierden (rojo)."""
    import plotly.graph_objects as go

    orden = sorted(multiplicadores, key=multiplicadores.get)
    valores = [multiplicadores[v] for v in orden]
    fig = go.Figure(go.Bar(
        y=[_etiqueta(v) for v in orden], x=[x - 1.0 for x in valores], base=1.0, orientation="h",
        marker_color=[COLOR_SUBE if x >= 1 else COLOR_BAJA for x in valores],
        text=[f"×{x:.2f}".replace(".", ",") for x in valores], textposition="outside",
        hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.add_vline(x=1.0, line_color="rgba(128,128,128,0.8)")
    margen = max(0.25, max(abs(x - 1.0) for x in valores) * 1.35) if valores else 0.25
    fig.update_layout(height=60 + 36 * len(orden), margin=MARGEN, xaxis_title=titulo_x,
                      xaxis_range=[1 - margen, 1 + margen], showlegend=False)
    return fig


def barras_desplazamiento(desplazamiento: dict[str, float]):
    """Cómo cambia el perfil buscado al aplicar las valoraciones de esta referencia (Rocchio)."""
    import plotly.graph_objects as go

    orden = sorted(desplazamiento, key=desplazamiento.get)
    valores = [desplazamiento[v] for v in orden]
    fig = go.Figure(go.Bar(
        y=[_etiqueta(v) for v in orden], x=valores, orientation="h",
        marker_color=[COLOR_SUBE if x >= 0 else COLOR_BAJA for x in valores],
        hovertemplate="%{y}: %{x:+.2f} desviaciones<extra></extra>"))
    fig.add_vline(x=0.0, line_color="rgba(128,128,128,0.8)")
    fig.update_layout(height=60 + 34 * len(orden), margin=MARGEN, showlegend=False,
                      xaxis_title="Cambio del perfil buscado (desviaciones típicas)")
    return fig


def mapa_centroides(variables: list[str], centroides: np.ndarray, nombres: list[str]):
    """Perfil medio de cada estilo de juego (valores z: 0 = media del grupo)."""
    import plotly.graph_objects as go

    limite = float(max(1.5, np.abs(centroides).max()))
    fig = go.Figure(go.Heatmap(
        z=centroides, x=[_etiqueta(v) for v in variables], y=nombres, zmin=-limite, zmax=limite,
        colorscale="RdBu", text=np.round(centroides, 1), texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: %{z:+.2f} desviaciones<extra></extra>",
        colorbar=dict(title="z")))
    fig.update_layout(height=110 + 48 * len(nombres), margin=dict(l=40, r=30, t=20, b=40))
    fig.update_yaxes(autorange="reversed")
    return fig


def curvas_aprendizaje(tabla: pd.DataFrame, medida: str = "precision_10"):
    """Evolución de la calidad del ranking según cuántas referencias ha valorado el usuario simulado."""
    import plotly.graph_objects as go

    nombres = {"precision_10": "Precisión en el Top 10", "ndcg_10": "nDCG en el Top 10"}
    fig = go.Figure()
    for i, (escenario, t) in enumerate(tabla.groupby("escenario", sort=False)):
        fig.add_trace(go.Scatter(x=t["referencias_valoradas"], y=t[medida], mode="lines+markers", name=escenario,
                                 line=dict(color=COLORES_ESTILOS[i % len(COLORES_ESTILOS)], width=2.5),
                                 hovertemplate="%{x} referencias: %{y:.3f}<extra>" + escenario + "</extra>"))
    fig.update_layout(height=380, margin=MARGEN, xaxis_title="Referencias valoradas por el usuario",
                      yaxis_title=nombres.get(medida, medida), legend=dict(orientation="h", y=-0.3))
    return fig


def curvas_silueta(tabla: pd.DataFrame):
    """Silueta del agrupamiento según el número de estilos; el k elegido se marca con un rombo."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (grupo, t) in enumerate(tabla.groupby("grupo", sort=False)):
        color = COLORES_ESTILOS[i % len(COLORES_ESTILOS)]
        fig.add_trace(go.Scatter(x=t["k"], y=t["silueta"], mode="lines+markers", name=C.GRUPOS.get(grupo, grupo),
                                 line=dict(color=color, width=2)))
        elegido = t[t["elegido"].astype(bool)]
        fig.add_trace(go.Scatter(x=elegido["k"], y=elegido["silueta"], mode="markers", showlegend=False,
                                 marker=dict(size=14, symbol="diamond", color=color, line=dict(color="white", width=1)),
                                 hovertemplate="k elegido = %{x}<extra></extra>"))
    fig.update_layout(height=340, margin=MARGEN, xaxis_title="Número de estilos (k)", yaxis_title="Silueta",
                      legend=dict(orientation="h", y=-0.3))
    return fig


def barras_comparacion(tabla: pd.DataFrame, categoria: str, medidas: dict[str, str], titulo_y: str = ""):
    """Barras agrupadas genéricas: una barra por medida (columna → nombre) y categoría."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (columna, nombre) in enumerate(medidas.items()):
        fig.add_trace(go.Bar(x=tabla[categoria], y=tabla[columna], name=nombre,
                             marker_color=[COLOR_NEUTRO, COLOR_REFERENCIA, COLOR_CANDIDATO][i % 3],
                             hovertemplate="%{x}: %{y:.3f}<extra>" + nombre + "</extra>"))
    fig.update_layout(barmode="group", height=340, margin=MARGEN, yaxis_title=titulo_y,
                      legend=dict(orientation="h", y=-0.3))
    return fig
