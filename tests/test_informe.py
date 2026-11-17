"""Informe de ojeador en PDF y tabla de la preselección en CSV."""
import io

import pandas as pd
import pytest

import tests.test_simulador as S
from ai_scout import estilos, informe, orquestador
from ai_scout.llm import ProveedorSimulado
from ai_scout.modelos import Consulta
from conftest import id_de

pytest.importorskip("fpdf")


@pytest.fixture()
def resultado_hibrido(df_sintetico):
    return orquestador.ejecutar_consulta(
        Consulta(id_de(df_sintetico, "MF Jugador 06"), rol="creador", edad_maxima=30,
                 texto_original="un creador como MF Jugador 06, máximo 30 años"),
        df_sintetico, "test", ProveedorSimulado(S.analista_valido),
        ProveedorSimulado(S.Secuencia(S.director_valido, S.director_cifra_inventada)), exportar=False,
        proveedor_critico=ProveedorSimulado(S.critico_rechaza_primera), estilos=estilos.calcular_estilos(df_sintetico))


def test_pdf_de_una_consulta_hibrida(resultado_hibrido):
    pdf = informe.generar_pdf(resultado_hibrido, {resultado_hibrido["top_final"][0]["registro_id"]: 4})
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000


def test_pdf_en_modo_numerico_y_sin_edad(df_sintetico):
    r = orquestador.ejecutar_consulta(Consulta(id_de(df_sintetico, "MF Jugador 02"), top_preseleccion=30),
                                      df_sintetico, "test", exportar=False)
    assert any(c["edad_registrada"] is None for c in r["ranking"])        # «Sin Edad» está en la lista
    assert informe.generar_pdf(r).startswith(b"%PDF")


def test_pdf_sin_fuente_unicode_translitera_los_nombres(resultado_hibrido, monkeypatch):
    monkeypatch.setattr(informe, "FUENTES", [])                            # obliga a usar Helvetica
    resultado_hibrido["top_final"][0]["nombre"] = "Martin Ødegaard Šeško Çalhanoğlu"
    resultado_hibrido["recuperacion"]["objetivo"]["nombre"] = "Łukasz «el 10» ≤ ≥ …"
    assert informe.generar_pdf(resultado_hibrido).startswith(b"%PDF")


def test_csv_de_la_preseleccion(resultado_hibrido):
    tabla = informe.tabla_preseleccion(resultado_hibrido)
    assert list(tabla["Puesto"]) == list(range(1, len(resultado_hibrido["ranking"]) + 1))
    assert tabla["Estilo"].notna().all() and tabla["Veredicto"].iloc[0] == "Candidato para revisar"
    datos = informe.csv_preseleccion(resultado_hibrido)
    assert datos.startswith(b"\xef\xbb\xbf")                               # BOM: Excel reconoce la codificación
    leida = pd.read_csv(io.BytesIO(datos), sep=";", decimal=",", encoding="utf-8-sig")
    assert list(leida.columns) == list(tabla.columns) and len(leida) == len(tabla)
    assert leida["Puntuación final"].iloc[0] == pytest.approx(tabla["Puntuación final"].iloc[0])
