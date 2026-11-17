"""Página «Estilos de juego»: mapa de jugadores y perfil de cada estilo."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_scout import config as C
from ai_scout import estilos as E
from ai_scout import graficos

from . import comun
from .comun import fmt


def pagina() -> None:
    df, manifiesto, modelos, tabla, _ = comun.contexto()
    st.header("Estilos de juego")
    st.markdown(
        "Los jugadores de cada grupo de posición se reparten en **estilos** con un agrupamiento *k-means* sobre "
        "sus estadísticas estandarizadas, y se dibujan en un mapa con las dos primeras **componentes "
        "principales** (PCA). Los estilos sirven para describir a los jugadores, para filtrar candidatos del "
        "mismo estilo que la referencia y para evaluar el recomendador.")
    grupos = [g for g in C.PERFILES if g in modelos]
    if not grupos:
        st.info("No hay jugadores suficientes para calcular estilos de juego.")
        return
    grupo = st.segmented_control("Grupo de posición", grupos, format_func=C.GRUPOS.get, default=grupos[0],
                                 key="estilos_grupo") or grupos[0]
    modelo = modelos[grupo]
    del_grupo = tabla[tabla["grupo_posicion"] == grupo]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estilos", modelo.k)
    c2.metric("Jugadores que entrenan el modelo", fmt(sum(modelo.tamanos), 0),
              help=f"Jugadores con al menos {C.MINUTOS_MINIMOS_DEFECTO} minutos. Al resto se le asigna el estilo "
                   "del centroide más cercano.")
    c3.metric("Silueta", fmt(modelo.silueta, 2),
              help="Entre −1 y 1. Los valores bajos son normales aquí: los estilos son una partición orientativa "
                   "de un continuo, no categorías cerradas.")
    c4.metric("Varianza explicada por el mapa", f"{fmt(modelo.varianza_explicada.sum() * 100, 0)} %",
              help="Parte de la variabilidad de las estadísticas que recogen los dos ejes del mapa.")

    a, b = st.columns([2, 1], vertical_alignment="bottom")
    nombres = del_grupo.sort_values(["nombre", "minutos"], ascending=[True, False])
    etiquetas = {r.registro_id: f"{r.nombre} · {r.equipo}" for r in nombres.itertuples()}
    resaltado = a.selectbox("Resaltar a un jugador", list(etiquetas), format_func=etiquetas.get, index=None,
                            placeholder="Escribe un nombre…", key=f"estilos_resaltar_{grupo}")
    todos = b.toggle(f"Incluir a los de menos de {C.MINUTOS_MINIMOS_DEFECTO} minutos", key="estilos_todos")
    visibles = del_grupo if todos else del_grupo[del_grupo["en_entrenamiento"] | (del_grupo["registro_id"] == resaltado)]
    st.plotly_chart(graficos.mapa_estilos(visibles, modelo.nombres, E.eje_pca(modelo, 0), E.eje_pca(modelo, 1),
                                          resaltado))
    if resaltado:
        fila = del_grupo[del_grupo["registro_id"] == resaltado].iloc[0]
        st.markdown(f"**{fila['nombre']}** ({fila['equipo']}) · estilo: **{fila['estilo']}**. "
                    + modelo.descripciones[modelo.nombres.index(fila["estilo"])])

    st.subheader("Perfil medio de cada estilo")
    st.plotly_chart(graficos.mapa_centroides(modelo.variables, modelo.centroides, modelo.nombres))
    st.caption("Valores en desviaciones típicas respecto a la media del grupo: azul = por encima, rojo = por debajo.")
    entrenamiento = del_grupo[del_grupo["en_entrenamiento"]]
    resumen = []
    for i, nombre in enumerate(modelo.nombres):
        ejemplos = entrenamiento[entrenamiento["estilo"] == nombre].sort_values("minutos", ascending=False).head(5)
        resumen.append({"Estilo": nombre, "Jugadores": modelo.tamanos[i], "Descripción": modelo.descripciones[i],
                        "Ejemplos (más minutos)": ", ".join(ejemplos["nombre"])})
    st.dataframe(pd.DataFrame(resumen), hide_index=True)

    st.subheader("Jugadores de un estilo")
    estilo = st.selectbox("Estilo", modelo.nombres, key=f"estilos_lista_{grupo}")
    lista = del_grupo[del_grupo["estilo"] == estilo].sort_values("minutos", ascending=False)
    st.dataframe(lista[["nombre", "equipo", "liga", "posicion_original", "edad_registrada", "minutos"]].rename(columns={
        "nombre": "Jugador", "equipo": "Equipo", "liga": "Liga", "posicion_original": "Pos.",
        "edad_registrada": "Edad", "minutos": "Minutos"}), hide_index=True, height=320)
    st.caption(f"Dataset {manifiesto['dataset_version']} · la tabla completa se exporta a "
               "`salidas/estilos_jugadores.csv` para Power BI.")
