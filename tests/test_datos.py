import numpy as np
import pandas as pd
import pytest

from ai_scout import config as C
from ai_scout import datos
from conftest import id_de


def test_calculo_por_90_usa_minutos_exactos(raw_sintetico, df_sintetico):
    raw = raw_sintetico.iloc[0]
    fila = df_sintetico.iloc[0]
    assert fila["PrgP_p90"] == pytest.approx(raw["PrgP"] * 90 / raw["Min"])
    assert fila["npxG_p90"] == pytest.approx(raw["npxG"] * 90 / raw["Min"])


def test_porcentajes_no_se_transforman(raw_sintetico, df_sintetico):
    assert df_sintetico.iloc[3]["Cmp_pct"] == raw_sintetico.iloc[3]["Cmp%"]


def test_ligas_normalizadas(df_sintetico):
    assert set(df_sintetico["liga"]) <= set(C.LIGAS_NORMALIZADAS)


def test_porteros_excluidos(df_sintetico):
    portero = df_sintetico[df_sintetico["nombre"] == "Portero Uno"].iloc[0]
    assert portero["es_portero"] and not portero["valido"]


def test_edad_ausente_no_se_imputa(df_sintetico):
    fila = df_sintetico[df_sintetico["nombre"] == "Sin Edad"].iloc[0]
    assert pd.isna(fila["edad_registrada"]) and fila["valido"]


def test_metrica_obligatoria_ausente_excluye_sin_imputar(df_sintetico):
    fila = df_sintetico[df_sintetico["nombre"] == "Sin Precision"].iloc[0]
    assert not fila["valido"] and "Cmp_pct" in fila["motivo_invalido"]
    assert pd.isna(fila["Cmp_pct"])


def test_misma_persona_en_dos_equipos(df_sintetico):
    t = df_sintetico[df_sintetico["nombre"] == "Traspasado"]
    assert len(t) == 2 and t["persona_id"].nunique() == 1
    assert t["registro_id"].nunique() == 2


def test_homonimos_no_se_fusionan(df_sintetico):
    h = df_sintetico[df_sintetico["nombre"] == "Homonimo"]
    assert h["persona_id"].nunique() == 2


def test_correspondencia_manual_tiene_prioridad(raw_sintetico):
    df0, _ = datos.preparar(raw_sintetico)
    rid = id_de(df0, "Homonimo", "Equipo D")
    otro = df0[(df0["nombre"] == "Homonimo") & (df0["equipo"] == "Equipo C")].iloc[0]["persona_id"]
    manual = pd.DataFrame({"registro_id": [rid], "persona_id": [otro]})
    df1, _ = datos.preparar(raw_sintetico, manual)
    assert df1[df1["nombre"] == "Homonimo"]["persona_id"].nunique() == 1


def test_columnas_obligatorias(raw_sintetico):
    with pytest.raises(datos.ErrorDatos):
        datos.preparar(raw_sintetico.drop(columns=["Recov"]))


def test_minutos_cero_invalido(raw_sintetico):
    raw = raw_sintetico.copy()
    raw.loc[0, "Min"] = 0
    df, _ = datos.preparar(raw)
    assert not df.iloc[0]["valido"] and np.isnan(df.iloc[0]["PrgP_p90"])


def test_guardar_y_recargar(tmp_path, raw_sintetico):
    ruta = tmp_path / "mini.csv"
    raw_sintetico.to_csv(ruta, index=False)
    df1, _, man = datos.preparar_desde_csv(ruta, tmp_path / "proc", tmp_path / "no_existe.csv")
    df2, man2 = datos.cargar_o_preparar(ruta, tmp_path / "proc")
    assert man2["huella_sha256"] == man["huella_sha256"]
    assert list(df2["registro_id"]) == list(df1["registro_id"])
    assert df2["valido"].dtype == bool
    assert df2["PrgP_p90"].to_numpy() == pytest.approx(df1["PrgP_p90"].to_numpy(), nan_ok=True)


# ---- Dataset real (se omiten si no está el CSV) ----
def test_real_dimensiones_y_poblacion(datos_reales):
    df, _, man = datos_reales
    assert (man["filas"], man["columnas"]) == (2854, 267)
    assert man["huella_sha256"].startswith("18fe8dbf3864")
    elegibles = df[df["valido"] & (df["minutos"] >= 600)]
    assert len(elegibles) == 1701
    assert elegibles["grupo_posicion"].value_counts().to_dict() == {"DF": 712, "MF": 563, "FW": 426}


def test_real_dos_rodri_distintos(datos_reales):
    df = datos_reales[0]
    rodri = df[df["nombre"] == "Rodri"]
    assert len(rodri) == 2 and rodri["persona_id"].nunique() == 2
