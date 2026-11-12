"""Ranking final: híbrido (numérico + cualitativo) o solo numérico."""
from __future__ import annotations


def calcular_ranking(candidatos: list[dict], analisis: dict | None, peso_numerico: float,
                     peso_cualitativo: float) -> tuple[list[dict], str, str | None]:
    """Devuelve (candidatos ordenados con score_final y ranking_final, modo, motivo de degradación).

    Regla de la memoria: si el análisis LLM falta o algún candidato no es
    evaluable, TODA la consulta usa el ranking numérico. Nunca se mezclan
    fórmulas distintas en una misma consulta.
    """
    motivo = None
    modo = "hibrido"
    if analisis is None:
        modo, motivo = "numerico", "sin análisis cualitativo válido"
    else:
        no_evaluables = [c["registro_id"] for c in candidatos
                         if analisis.get(c["registro_id"], {}).get("score_cualitativo") is None]
        if no_evaluables:
            modo = "numerico"
            motivo = f"análisis no evaluable para {len(no_evaluables)} candidato(s)"

    resultado = []
    for c in candidatos:
        c = dict(c)
        a = (analisis or {}).get(c["registro_id"])
        c["analisis"] = a
        c["score_cualitativo"] = a["score_cualitativo"] if a else None
        c["estado_analisis"] = a["estado_analisis"] if a else "no_disponible"
        if modo == "hibrido":
            c["score_final"] = peso_numerico * c["score_numerico"] + peso_cualitativo * c["score_cualitativo"]
        else:
            c["score_final"] = c["score_numerico"]
        resultado.append(c)

    # Desempates: score_final, score_numerico, minutos y registro_id
    resultado.sort(key=lambda c: (-round(c["score_final"], 9), -round(c["score_numerico"], 9),
                                  -c["minutos"], c["registro_id"]))
    for i, c in enumerate(resultado, start=1):
        c["ranking_final"] = i
    return resultado, modo, motivo
