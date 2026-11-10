import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_scout import config as C  # noqa: E402
from ai_scout import datos  # noqa: E402

LIGAS_CSV = list(C.LIGAS)


def fila(rk, nombre, pos, equipo, liga, edad, nacido, minutos, rng, nacion="es ESP", **fijos):
    f = {"Rk": rk, "Player": nombre, "Nation": nacion, "Pos": pos, "Squad": equipo, "Comp": liga,
         "Age": edad, "Born": nacido, "Min": minutos, "90s": round(minutos / 90, 1)}
    for col in C.COLUMNAS_P90:
        f[col] = float(rng.integers(0, 120)) if col not in ("npxG", "xAG") else round(float(rng.uniform(0, 12)), 1)
    f["Cmp%"] = round(float(rng.uniform(60, 92)), 1)
    f.update(fijos)
    return f


@pytest.fixture(scope="session")
def raw_sintetico() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    filas, rk = [], 0
    for grupo in ("DF", "MF", "FW"):
        for i in range(18):
            rk += 1
            filas.append(fila(rk, f"{grupo} Jugador {i:02d}", grupo, f"Equipo {i % 6}", LIGAS_CSV[i % 5],
                              19 + i % 14, 2005 - i % 14, 700 + 100 * i, rng))
    especiales = [
        ("Portero Uno", "GK", "Equipo 1", LIGAS_CSV[0], 30, 1994, 3000, {}),
        ("Sin Edad", "MF", "Equipo 2", LIGAS_CSV[1], np.nan, np.nan, 1500, {}),
        ("Poco Minutos", "MF", "Equipo 3", LIGAS_CSV[2], 28, 1996, 73, {}),
        ("Traspasado", "MF", "Equipo A", LIGAS_CSV[0], 24, 2000, 900, {}),
        ("Traspasado", "MF", "Equipo B", LIGAS_CSV[3], 24, 2000, 1200, {}),
        ("Homonimo", "DF", "Equipo C", LIGAS_CSV[1], 29, 1995, 1500, {}),
        ("Homonimo", "FW", "Equipo D", LIGAS_CSV[2], 24, 2000, 1500, {}),
        ("Sin Precision", "DF", "Equipo E", LIGAS_CSV[4], 25, 1999, 1500, {"Cmp%": np.nan}),
    ]
    for nombre, pos, equipo, liga, edad, nacido, minutos, fijos in especiales:
        rk += 1
        filas.append(fila(rk, nombre, pos, equipo, liga, edad, nacido, minutos, rng, **fijos))
    return pd.DataFrame(filas)


@pytest.fixture(scope="session")
def df_sintetico(raw_sintetico) -> pd.DataFrame:
    df, _ = datos.preparar(raw_sintetico)
    return df


def id_de(df, nombre, equipo=None) -> str:
    sel = df[df["nombre"] == nombre]
    if equipo:
        sel = sel[sel["equipo"] == equipo]
    return sel.iloc[0]["registro_id"]


@pytest.fixture(scope="session")
def datos_reales(tmp_path_factory):
    if not C.RUTA_CSV.exists():
        pytest.skip("No está el CSV real en dataset/")
    destino = tmp_path_factory.mktemp("procesado")
    df, incidencias, manifiesto = datos.preparar_desde_csv(C.RUTA_CSV, destino)
    return df, incidencias, manifiesto
