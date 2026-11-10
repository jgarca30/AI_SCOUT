"""Carga, validación y transformación del dataset (CU-01 Preparar dataset).
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

COLUMNAS_OBLIGATORIAS = ["Rk", "Player", "Nation", "Pos", "Squad", "Comp", "Age", "Born", "Min"] + \
    C.COLUMNAS_P90 + list(C.COLUMNAS_PORCENTAJE)

COLUMNAS_TEXTO = ["registro_id", "persona_id", "nombre", "nacionalidad", "posicion_original",
                  "grupo_posicion", "equipo", "liga_original", "liga", "temporada", "motivo_invalido"]


class ErrorDatos(ValueError):
    """Error de validación o carga del dataset."""


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def huella_sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def normalizar_texto(texto: object) -> str:
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def _incidencia(codigo: str, descripcion: str, severidad: str = "aviso",
                registro_id: str | None = None, fase: str = "preparacion") -> dict:
    return {"fase": fase, "severidad": severidad, "codigo": codigo,
            "registro_id": registro_id, "descripcion": descripcion}


# --------------------------------------------------------------------------
# Preparación
# --------------------------------------------------------------------------
def preparar(df_raw: pd.DataFrame, correspondencias: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Devuelve (tabla procesada, lista de incidencias)."""
    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df_raw.columns]
    if faltan:
        raise ErrorDatos(f"Faltan columnas obligatorias en el CSV: {', '.join(faltan)}")

    incidencias: list[dict] = []
    raw = df_raw.copy()

    # Identificador estable del registro dentro de esta versión del dataset
    rk = pd.to_numeric(raw["Rk"], errors="coerce")
    if rk.notna().all() and rk.is_unique:
        registro_id = rk.astype(int).map(lambda x: f"R{x:04d}")
    else:
        incidencias.append(_incidencia("rk_no_unico", "La columna Rk no es única; se usa el orden de fila.",
                                       "aviso"))
        registro_id = pd.Series([f"R{i + 1:04d}" for i in range(len(raw))], index=raw.index)

    df = pd.DataFrame({
        "registro_id": registro_id.astype(str),
        "nombre": raw["Player"].astype(str).str.strip(),
        "nacionalidad": raw["Nation"].fillna("").astype(str).str.strip(),
        "ano_nacimiento": pd.to_numeric(raw["Born"], errors="coerce"),
        "posicion_original": raw["Pos"].fillna("").astype(str).str.strip(),
        "equipo": raw["Squad"].astype(str).str.strip(),
        "liga_original": raw["Comp"].astype(str).str.strip(),
        "temporada": C.TEMPORADA,
        "edad_registrada": pd.to_numeric(raw["Age"], errors="coerce"),
        "minutos": pd.to_numeric(raw["Min"], errors="coerce"),
    })
    df["grupo_posicion"] = df["posicion_original"].str.split(",").str[0].str.strip()
    df["es_portero"] = df["grupo_posicion"].eq("GK")

    # Ligas
    df["liga"] = df["liga_original"].map(C.LIGAS)
    for _, fila in df[df["liga"].isna()].iterrows():
        incidencias.append(_incidencia("liga_desconocida", f"Competición no reconocida: {fila['liga_original']}",
                                       "aviso", fila["registro_id"]))
    df["liga"] = df["liga"].fillna(df["liga_original"])

    # Grupos de posición
    for _, fila in df[~df["grupo_posicion"].isin(list(C.GRUPOS) + ["GK"])].iterrows():
        incidencias.append(_incidencia("posicion_desconocida", f"Posición no reconocida: '{fila['posicion_original']}'",
                                       "error", fila["registro_id"]))

    # Edad ausente
    for _, fila in df[df["edad_registrada"].isna()].iterrows():
        incidencias.append(_incidencia("edad_ausente", f"{fila['nombre']} ({fila['equipo']}) no tiene edad registrada.",
                                       "info", fila["registro_id"]))

    # Validación de minutos
    minutos_ok = df["minutos"].notna() & (df["minutos"] > 0)

    # Estadísticas por 90 minutos
    for col in C.COLUMNAS_P90:
        valores = pd.to_numeric(raw[col], errors="coerce")
        negativos = valores < 0
        for rid in df.loc[negativos, "registro_id"]:
            incidencias.append(_incidencia("valor_negativo", f"{col} negativo", "error", rid))
        valores = valores.where(~negativos)
        df[f"{col}_p90"] = np.where(minutos_ok, valores * 90.0 / df["minutos"], np.nan)
        df[f"{col}_total"] = valores

    for col, nuevo in C.COLUMNAS_PORCENTAJE.items():
        valores = pd.to_numeric(raw[col], errors="coerce")
        fuera = (valores < 0) | (valores > 100)
        for rid in df.loc[fuera, "registro_id"]:
            incidencias.append(_incidencia("porcentaje_fuera_de_rango", f"{col} fuera de 0-100", "error", rid))
        df[nuevo] = valores.where(~fuera)

    # Validez de cada registro para su grupo
    motivos = []
    for i, fila in df.iterrows():
        if fila["es_portero"]:
            motivos.append("portero")
            continue
        if fila["grupo_posicion"] not in C.PERFILES:
            motivos.append("posicion_desconocida")
            continue
        if not minutos_ok.loc[i]:
            motivos.append("minutos_invalidos")
            continue
        ausentes = [v for v in C.PERFILES[fila["grupo_posicion"]] if pd.isna(fila[v])]
        motivos.append("metricas_ausentes:" + ",".join(ausentes) if ausentes else "")
    df["motivo_invalido"] = motivos
    df["valido"] = df["motivo_invalido"].eq("")
    for _, fila in df[df["motivo_invalido"].str.startswith(("metricas", "minutos"))].iterrows():
        incidencias.append(_incidencia("registro_excluido", f"Registro excluido: {fila['motivo_invalido']}",
                                       "aviso", fila["registro_id"]))

    df["persona_id"], inc_identidad = asignar_personas(df, correspondencias)
    incidencias.extend(inc_identidad)
    return df, incidencias


def asignar_personas(df: pd.DataFrame, correspondencias: pd.DataFrame | None = None) -> tuple[pd.Series, list[dict]]:
    """Agrupa registros de una misma persona.

    Regla automática: mismo nombre + año de nacimiento + nacionalidad.
    Si falta el año de nacimiento no se fusiona nada. La tabla manual de
    correspondencias (config/correspondencias_identidad.csv) tiene prioridad.
    """
    incidencias = []
    clave = (df["nombre"].map(normalizar_texto) + "|" +
             df["ano_nacimiento"].map(lambda x: "" if pd.isna(x) else str(int(x))) + "|" +
             df["nacionalidad"].map(normalizar_texto))
    sin_nacimiento = df["ano_nacimiento"].isna()
    clave = clave.where(~sin_nacimiento, "registro|" + df["registro_id"])

    codigos = {}
    persona = []
    for k in clave:
        if k not in codigos:
            codigos[k] = f"P{len(codigos) + 1:04d}"
        persona.append(codigos[k])
    persona = pd.Series(persona, index=df.index)

    if correspondencias is not None and len(correspondencias):
        mapa = dict(zip(correspondencias["registro_id"].astype(str), correspondencias["persona_id"].astype(str)))
        manual = df["registro_id"].map(mapa)
        persona = manual.fillna(persona)
        for rid in df.loc[manual.notna(), "registro_id"]:
            incidencias.append(_incidencia("identidad_manual", "persona_id asignado por la tabla manual", "info", rid))

    # Avisos útiles para la revisión manual
    tmp = df.assign(persona_id=persona)
    for nombre, grupo in tmp.groupby(tmp["nombre"].map(normalizar_texto)):
        if len(grupo) < 2:
            continue
        ids = ", ".join(grupo["registro_id"])
        if grupo["persona_id"].nunique() == 1:
            incidencias.append(_incidencia("persona_varios_registros",
                                           f"{grupo['nombre'].iloc[0]}: misma persona en varios equipos ({ids}).",
                                           "info"))
        else:
            incidencias.append(_incidencia("posible_homonimo",
                                           f"{grupo['nombre'].iloc[0]}: registros con el mismo nombre tratados como "
                                           f"personas distintas ({ids}). Revisar si procede.", "aviso"))
    return persona, incidencias


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------
def leer_correspondencias(ruta: Path = C.RUTA_CORRESPONDENCIAS) -> pd.DataFrame | None:
    if not Path(ruta).exists():
        return None
    tabla = pd.read_csv(ruta, dtype=str, comment="#").dropna(subset=["registro_id", "persona_id"])
    return tabla if len(tabla) else None


def preparar_desde_csv(ruta_csv: Path = C.RUTA_CSV, dir_salida: Path = C.DIR_PROCESADO,
                       ruta_correspondencias: Path = C.RUTA_CORRESPONDENCIAS) -> tuple[pd.DataFrame, list[dict], dict]:
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise ErrorDatos(f"No se encuentra el dataset en {ruta_csv}")
    huella = huella_sha256(ruta_csv)
    raw = pd.read_csv(ruta_csv)
    df, incidencias = preparar(raw, leer_correspondencias(ruta_correspondencias))
    version = huella[:12]
    for n, inc in enumerate(incidencias, start=1):
        inc["incidencia_id"] = f"D{version[:6]}-{n:05d}"
        inc["dataset_version"] = version
        inc["consulta_id"] = None
    manifiesto = {
        "dataset_version": version,
        "fichero": ruta_csv.name,
        "huella_sha256": huella,
        "filas": int(raw.shape[0]),
        "columnas": int(raw.shape[1]),
        "temporada": C.TEMPORADA,
        "version_preparacion": C.VERSION_PREPARACION,
        "perfil_version": C.PERFIL_VERSION,
        "fecha_preparacion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registros_validos_campo": int((df["valido"]).sum()),
        "porteros": int(df["es_portero"].sum()),
        "incidencias": len(incidencias),
    }
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    df.to_csv(dir_salida / "registros_procesados.csv", index=False, encoding="utf-8")
    pd.DataFrame(incidencias).to_csv(dir_salida / "incidencias_datos.csv", index=False, encoding="utf-8")
    (dir_salida / "manifiesto.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, incidencias, manifiesto


def _leer_procesado(dir_procesado: Path) -> pd.DataFrame:
    df = pd.read_csv(dir_procesado / "registros_procesados.csv",
                     dtype={c: str for c in COLUMNAS_TEXTO}, keep_default_na=False,
                     na_values={"edad_registrada": [""], "ano_nacimiento": [""], "minutos": [""]})
    for col in df.columns:
        if col.endswith(("_p90", "_total", "_pct")) or col in ("edad_registrada", "ano_nacimiento", "minutos"):
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
    for col in ("valido", "es_portero"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def cargar_o_preparar(ruta_csv: Path = C.RUTA_CSV, dir_procesado: Path = C.DIR_PROCESADO) -> tuple[pd.DataFrame, dict]:
    """Carga los datos procesados; si no existen o el CSV ha cambiado, los regenera."""
    dir_procesado = Path(dir_procesado)
    ruta_manifiesto = dir_procesado / "manifiesto.json"
    if ruta_manifiesto.exists() and (dir_procesado / "registros_procesados.csv").exists() and Path(ruta_csv).exists():
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        if (manifiesto.get("huella_sha256") == huella_sha256(Path(ruta_csv))
                and manifiesto.get("version_preparacion") == C.VERSION_PREPARACION):
            return _leer_procesado(dir_procesado), manifiesto
    df, _, manifiesto = preparar_desde_csv(ruta_csv, dir_procesado)
    return df, manifiesto


def etiqueta_registro(fila: pd.Series | dict) -> str:
    """Texto para distinguir registros en la app: nombre, equipo, liga, posición y minutos."""
    minutos = fila["minutos"]
    minutos_txt = "—" if pd.isna(minutos) else f"{int(minutos):,}".replace(",", ".")
    return f"{fila['nombre']} · {fila['equipo']} ({fila['liga']}) · {fila['posicion_original']} · {minutos_txt} min"
