# AI SCOUT

PROYECTO FINAL SISTEMAS DE INFORMACIÓN Y GESTIÓN Y BUSINESS INTELLIGENCE     CURSO 2026/2027

Autor: Jorge García Andrés
Escuela de Ingenierías Industrial, Informática y Aeroespacial, Universidad de León

---

AI Scout es un sistema recomendador multiagente de futbolistas. Eliges un jugador de referencia y el sistema
devuelve los perfiles estadísticos más parecidos de las cinco grandes ligas, extraídos de un dataset de la temporada 2024-2025, con una ficha de ojeador que explica en qué se parecen y en qué no.

- **Secretario técnico** (LLM o reglas): convierte una petición en lenguaje natural («sustituto joven de Pedri
  en la Bundesliga, más creador») en los filtros de la búsqueda.
- **Ojeador de datos**: filtra la población, estandariza las métricas por 90 minutos y calcula la
  similitud coseno **ponderada por el rol buscado** → Top 10. Aplica lo aprendido de las valoraciones del usuario.
- **Analista de rendimiento** (LLM): valora a los 10 candidatos usando solo los datos recibidos
  (anonimizados). Su respuesta se valida antes de usarse.
- **Analista de plantilla** (LLM + indicadores): valora cómo encajaría cada candidato en la plantilla del club de
  destino (juego del equipo, hueco que cubre, compatriotas y liga). Su nota pondera la puntuación final.
- **Director deportivo** (LLM): redacta la ficha de ojeador del Top 3 sin cambiar el ranking.
- **Jefe de scouting** (LLM): comprueba las fichas contra los datos y, si hay errores, pide una corrección al
  director (patrón evaluador-optimizador).
- **Orquestador**: valida la consulta, coordina a los agentes, reintenta una vez si una respuesta no es válida y,
  si falla el LLM, termina la consulta en **modo numérico** con fichas por plantilla.

## Qué hace el recomendador

- **Roles de juego**: cada rol (organizador, creador, finalizador…) da más peso a unas variables; también se
  pueden ajustar los pesos a mano.
- **Estilos de juego**: agrupamiento *k-means* por grupo de posición y mapa PCA. Sirven para describir a los
  jugadores, filtrar por el mismo estilo y evaluar el sistema.
- **Aprende de tus valoraciones** (1-5 estrellas):
  - para la misma referencia, adapta la búsqueda hacia lo que te gustó (realimentación de Rocchio) y deja de
    proponer a los que puntuaste con 1-2 estrellas;
  - para todo el grupo de posición, ajusta la importancia de cada variable con un aprendizaje de ordenación por
    pares (criterio BPR) sobre tus comparaciones entre candidatos.
- **Diversidad** opcional de la preselección (MMR).
- **Evaluación**: verdad-terreno aproximada, coherencia con los estilos, cobertura, sensibilidad, usuarios
  simulados y registro de uso real (`python src\evaluar_sistema.py`).

## Estructura

```
dataset/            CSV original (Kaggle · FBref), nunca se modifica
config/             tabla manual de identidades de jugadores
diagramas/          diagramas UML y de dominio (Draw.io)
memoria/            memoria del proyecto y presentación
src/
  app.py              aplicación web (Streamlit): define las páginas
  vistas/             páginas: buscar, estilos de juego, aprendizaje, historial y evaluación
  preparar_datos.py   preparación del dataset
  evaluar_sistema.py  evaluación del recomendador
  ai_scout/
    config.py         parámetros: perfiles, roles, umbrales, pesos, modelos
    datos.py          carga, validación, métricas por 90 e identidades
    similitud.py      espacio estandarizado, coseno ponderado, Rocchio y MMR
    estilos.py        estilos de juego (k-means) y mapa PCA
    valoraciones.py   registro de las estrellas del usuario
    aprendizaje.py    lo que se aprende de las valoraciones
    interprete.py     Secretario técnico (lenguaje natural, con LLM o por reglas)
    ojeador.py        Ojeador de datos (similitud coseno ponderada)
    analista.py       Analista de rendimiento (prompt + validación)
    director.py       Director deportivo (prompt + validación + ficha por plantilla)
    critico.py        Jefe de scouting (revisión de las fichas)
    encaje.py         Analista de plantilla (encaje en el club de destino)
    redaccion.py      pulido de los textos de los LLM (tildes, nombres de variables)
    ejecutor_llm.py   reintentos, validación y trazas de cada llamada
    llm.py            clientes Claude y Gemini, caché
    ranking.py        ranking híbrido o numérico
    orquestador.py    flujo completo de una consulta
    exportador.py     tablas CSV para Power BI e historial
    informe.py        informe de ojeador en PDF y Top 10 en CSV
    evaluacion.py     métricas y simulaciones de la evaluación
    graficos.py       radar, mapa de estilos, pesos y curvas
tests/              pruebas automáticas (pytest)
salidas/            se crea al usar la app: tablas para Power BI
datos/              se crea al preparar los datos (ignorada por git)
```

## Instalación (Windows)

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Abre `.env` y pega tu clave. Por defecto se usa **Gemini** (`LLM_PROVEEDOR=gemini`).

 Para usar Claude, pon `LLM_PROVEEDOR=anthropic` y tu clave en `ANTHROPIC_API_KEY`.

Sin clave, la aplicación funciona igualmente en modo numérico.

## Uso

```bat
python src\preparar_datos.py      :: valida el CSV y genera los datos procesados (opcional: la app lo hace sola)
streamlit run src\app.py          :: abre la aplicación en http://localhost:8501
python src\evaluar_sistema.py     :: evalúa el recomendador (añade --llm 3 para medir también los agentes LLM)
python -m pytest                  :: ejecuta las pruebas
```

Páginas de la app:

- **Buscar sustitutos**: describe lo que buscas o usa el formulario (jugador, rol, ligas, edad, minutos). El
  resultado tiene pestañas con las fichas, un comparador, el mapa de estilos, el Top 10, cómo se ha calculado y
  las trazas. Puntúa a los candidatos con estrellas y pulsa **Recalcular**. Se puede descargar el informe en PDF.
- **Estilos de juego**: mapa de jugadores y perfil de cada estilo.
- **Lo aprendido de ti**: tus valoraciones y los pesos que el sistema ha aprendido.
- **Historial**: abre una consulta anterior sin recalcular, o repítela.
- **Evaluación**: resultados de la evaluación con gráficos.

Si cambias `.env`, reinicia la app. Para hacer pruebas sin mezclarlas con tus consultas reales, define la
variable `AI_SCOUT_SALIDAS` con otra carpeta.

## Salidas para Power BI

Cada consulta se añade (sin sobrescribir las anteriores) a las tablas de `salidas/`:

| Tabla                       | Clave             | Contenido                                                             |
| --------------------------- | ----------------- | --------------------------------------------------------------------- |
| `consultas.csv`           | `consulta_id`   | jugador de referencia, filtros, modo del ranking, estado, tiempos     |
| `resultados.csv`          | `resultado_id`  | una fila por candidato: similitud, scores, rankings, ficha, veredicto |
| `evidencias.csv`          | `evidencia_id`  | valores comparados por variable                                       |
| `registros.csv`           | `registro_id`   | datos descriptivos de los jugadores que aparecen                      |
| `ejecuciones_agentes.csv` | `ejecucion_id`  | cada llamada a un agente: modelo, intento, estado, tokens, coste      |
| `incidencias.csv`         | `incidencia_id` | problemas de datos y degradaciones                                    |
| `valoraciones.csv`        | `valoracion_id` | estrellas del usuario a cada candidato (con su vigencia)              |
| `estilos_jugadores.csv`   | `registro_id`   | estilo de juego y posición en el mapa PCA de todos los jugadores     |

Además, `salidas/historial/` guarda el resultado completo de cada consulta y `salidas/evaluacion/` las tablas y
el informe de la evaluación.

## Dataset

*Football Players Stats (2024-2025)*, Hubert Sidorowicz, Kaggle (datos de FBref). 2.854 registros y 267 columnas;
con jugadores de campo y al menos 600 minutos quedan 1.701 registros (712 DF, 563 MF, 426 FW).
