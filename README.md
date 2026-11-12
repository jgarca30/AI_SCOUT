# AI SCOUT

PROYECTO FINAL SISTEMAS DE INFORMACIÓN Y GESTIÓN Y BUSINESS INTELLIGENCE     CURSO 2026/2027

Autor: Jorge García Andrés
Escuela de Ingenierías Industrial, Informática y Aeroespacial, Universidad de León

---

AI Scout es un sistema recomendador multiagente de futbolistas. Eliges un jugador de referencia y el sistema
devuelve los perfiles estadísticos más parecidos de las cinco grandes ligas, extraídos de un dataset de la temporada 2024-2025, con una ficha de ojeador que explica en qué se parecen y en qué no.

- **Agente 1 · Ojeador estadístico** : filtra la población, estandariza las métricas por 90 minutos
  y calcula la similitud coseno → Top 10.
- **Agente 2 · Analista de perfiles** (LLM): valora a los 10 candidatos usando solo los datos recibidos
  (anonimizados). Su respuesta se valida antes de usarse.
- **Agente 3 · Director deportivo** (LLM): redacta la ficha de ojeador del Top 3 sin cambiar el ranking.
- **Orquestador**: valida la consulta, coordina a los agentes, reintenta una vez si una respuesta no es válida y,
  si falla el LLM, termina la consulta en **modo numérico** con fichas por plantilla.

## Estructura

```
dataset/            CSV original (Kaggle · FBref), nunca se modifica
config/             tabla manual de identidades de jugadores
diagramas/          diagramas UML y de dominio (Draw.io)
memoria/            memoria del proyecto y presentación
src/
  app.py            aplicación web (Streamlit)
  preparar_datos.py preparación del dataset
  ai_scout/
    config.py       parámetros: perfiles, umbrales, pesos, modelos
    datos.py        carga, validación, métricas por 90 e identidades
    ojeador.py      Agente 1 (similitud coseno)
    analista.py     Agente 2 (prompt + validación)
    director.py     Agente 3 (prompt + validación + ficha por plantilla)
    ejecutor_llm.py reintentos, validación y trazas de cada llamada
    llm.py          clientes Claude y Gemini, caché
    ranking.py      ranking híbrido o numérico
    orquestador.py  flujo completo de una consulta
    exportador.py   tablas CSV para Power BI
    graficos.py     radar de percentiles
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
python -m pytest                  :: ejecuta las pruebas
```

En la app: elige el jugador de referencia, las ligas, la edad máxima (opcional) y los minutos mínimos, y pulsa
**Buscar sustitutos**. El interruptor *Usar análisis con IA* permite comparar con el ranking puramente numérico.

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

## Dataset

*Football Players Stats (2024-2025)*, Hubert Sidorowicz, Kaggle (datos de FBref). 2.854 registros y 267 columnas;
con jugadores de campo y al menos 600 minutos quedan 1.701 registros (712 DF, 563 MF, 426 FW).
