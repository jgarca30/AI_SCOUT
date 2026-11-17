"""Página «Lo que el sistema ha aprendido de ti»: valoraciones y pesos aprendidos."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_scout import aprendizaje as A
from ai_scout import config as C
from ai_scout import graficos
from ai_scout import valoraciones as V

from . import comun
from .comun import fmt


def _al_editar(clave: str, vigentes: pd.DataFrame) -> None:
    """Cambia (o retira, con 0) las valoraciones editadas en la tabla."""
    for fila, estrellas in comun.estrellas_editadas(st.session_state.get(clave), "Estrellas"):
        original = vigentes.iloc[fila].to_dict()
        original.update(estrellas=estrellas, valoracion_id=None, fecha_utc=None)
        V.registrar_fila(original)
    st.session_state["version_valoraciones"] = comun.version_valoraciones() + 1
    st.session_state["valoraciones_pendientes"] = True


def _olvidar() -> None:
    V.borrar_todas()
    st.session_state["version_valoraciones"] = comun.version_valoraciones() + 1
    st.session_state["confirmar_olvido"] = False


def pagina() -> None:
    df, manifiesto, _, _, _ = comun.contexto()
    st.header("Lo que el sistema ha aprendido de ti")
    st.markdown(
        "Cada vez que puntúas a un candidato con estrellas, el sistema aprende de dos maneras:\n"
        "- **Para esa misma referencia** (realimentación de Rocchio): la próxima búsqueda se acerca a los "
        "candidatos que te gustaron, se aleja de los que no y deja de proponer a los de 1-2 estrellas.\n"
        "- **Para todo el grupo de posición** (aprendizaje de ordenación por pares): si prefieres a un candidato "
        "frente a otro de la misma búsqueda, el sistema ajusta la importancia de cada variable para ordenar a "
        "los candidatos como tú. Esos pesos se aplican después a cualquier referencia del mismo grupo.")
    try:
        vigentes = V.vigentes(dataset_version=manifiesto["dataset_version"])
    except (OSError, ValueError, KeyError):
        st.error("No se han podido leer las valoraciones guardadas en `salidas/valoraciones.csv`.")
        return
    if vigentes.empty:
        st.info("Todavía no has valorado a ningún candidato. Haz una búsqueda y puntúa con estrellas en las "
                "fichas o en la pestaña «Preselección».", icon=":material/star:")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valoraciones vigentes", len(vigentes))
    c2.metric("Referencias valoradas", vigentes["objetivo_persona_id"].nunique())
    c3.metric("Estrellas medias", fmt(vigentes["estrellas"].mean(), 1))
    c4.metric("Candidatos descartados", int((vigentes["estrellas"] <= C.ESTRELLAS_NEGATIVA).sum()),
              help="Valorados con 1 o 2 estrellas: no vuelven a proponerse para su referencia.")

    st.subheader("Importancia aprendida de cada variable")
    grupos = [g for g in C.PERFILES if (vigentes["grupo_posicion"] == g).any()]
    for pestana, grupo in zip(st.tabs([C.GRUPOS[g] for g in grupos]), grupos):
        with pestana:
            del_grupo = vigentes[vigentes["grupo_posicion"] == grupo]
            aprendido = A.aprender_multiplicadores(del_grupo, df, grupo)
            if not aprendido:
                st.info(f"Hay {len(del_grupo)} valoraciones de {C.GRUPOS[grupo].lower()}, pero todavía ninguna "
                        "comparación: puntúa con distinta nota a dos candidatos de una misma búsqueda.")
                continue
            izq, der = st.columns([3, 2])
            izq.plotly_chart(graficos.barras_multiplicadores(aprendido["multiplicadores"]),
                             key=f"multiplicadores_{grupo}")
            der.metric("Comparaciones aprendidas", aprendido["n_pares"],
                       help="Parejas de candidatos de una misma búsqueda a las que diste distinta puntuación.")
            der.metric("Tus comparaciones bien ordenadas", f"{fmt(aprendido['acierto_despues'] * 100, 0)} %",
                       delta=f"{fmt((aprendido['acierto_despues'] - aprendido['acierto_antes']) * 100, 0)} puntos",
                       help="Proporción de tus comparaciones que la similitud ordena como tú, con los pesos "
                            "aprendidos. La variación es respecto a los pesos sin aprender.")
            der.caption("En verde, las variables que han ganado importancia; en rojo, las que la han perdido. "
                        "Se aplican multiplicando los pesos del rol que elijas.")

    st.subheader("Tus valoraciones")
    orden = vigentes.sort_values("fecha_utc", ascending=False).reset_index(drop=True)
    tabla = pd.DataFrame({
        "Fecha": pd.to_datetime(orden["fecha_utc"], errors="coerce", utc=True).dt.strftime("%d/%m/%Y %H:%M"),
        "Referencia": orden["objetivo_nombre"], "Candidato": orden["candidato_nombre"],
        "Grupo": orden["grupo_posicion"].map(C.GRUPOS), "Puesto en la lista": orden["ranking_final"],
        "Estrellas": orden["estrellas"],
    })
    clave = f"tabla_valoraciones_{comun.version_valoraciones()}"
    izq, der = st.columns([3, 1])
    with izq:
        st.data_editor(tabla, hide_index=True, key=clave, disabled=[c for c in tabla.columns if c != "Estrellas"],
                       on_change=_al_editar, args=(clave, orden), height=330,
                       column_config={"Estrellas": st.column_config.NumberColumn(
                           "Estrellas ★", min_value=0, max_value=5, step=1, format="%d ★",
                           help="Cambia la nota de 1 a 5, o pon 0 para retirar la valoración.")})
        st.caption("Puedes cambiar una nota escribiendo de 1 a 5, o retirarla con un 0.")
    with der:
        reparto = vigentes["estrellas"].value_counts().reindex(range(1, 6), fill_value=0)
        st.bar_chart(pd.DataFrame({"Valoraciones": reparto.to_numpy()}, index=[f"{e} ★" for e in reparto.index]),
                     height=300)

    st.subheader("Olvidar lo aprendido")
    st.checkbox("Quiero borrar todas mis valoraciones", key="confirmar_olvido")
    st.button("Borrar valoraciones", on_click=_olvidar, disabled=not st.session_state.get("confirmar_olvido"),
              icon=":material/delete:")
    st.caption("Se borra `salidas/valoraciones.csv`. Las búsquedas volverán a usar los pesos del rol sin "
               "personalizar.")
