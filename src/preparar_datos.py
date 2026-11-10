"""
Prepara el dataset (CU-01): valida el CSV, calcula las métricas por 90 minutos,
asigna identidades y guarda los datos procesados y el informe de incidencias.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_scout import config as C  # noqa: E402
from ai_scout.datos import ErrorDatos, preparar_desde_csv  # noqa: E402
from ai_scout.exportador import exportar_incidencias_datos  # noqa: E402


def main() -> int:
    try:
        df, incidencias, manifiesto = preparar_desde_csv()
    except ErrorDatos as e:
        print(f"ERROR: {e}")
        return 1
    exportar_incidencias_datos(incidencias)

    elegibles = df[df["valido"] & (df["minutos"] >= C.MINUTOS_MINIMOS_DEFECTO)]
    print("Dataset preparado")
    print(f"  Archivo:            {manifiesto['fichero']} (versión {manifiesto['dataset_version']})")
    print(f"  Filas x columnas:   {manifiesto['filas']} x {manifiesto['columnas']}")
    print(f"  Porteros excluidos: {manifiesto['porteros']}")
    print(f"  Elegibles (>= {C.MINUTOS_MINIMOS_DEFECTO} min): {len(elegibles)} "
          f"{dict(Counter(elegibles['grupo_posicion']))}")
    print(f"  Incidencias:        {len(incidencias)} {dict(Counter(i['codigo'] for i in incidencias))}")
    print(f"  Datos procesados en {C.DIR_PROCESADO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
