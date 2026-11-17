"""
Evalúa el recomendador y escribe las tablas y un informe en salidas/evaluacion/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_scout import datos, estilos, evaluacion, llm  # noqa: E402
from ai_scout.datos import ErrorDatos  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de AI Scout")
    parser.add_argument("--rapido", action="store_true", help="simulaciones más cortas")
    parser.add_argument("--llm", type=int, default=0, metavar="N",
                        help="número de consultas reales que se repiten con los agentes LLM (0 = ninguna)")
    parser.add_argument("--repeticiones", type=int, default=2, help="repeticiones de cada consulta con --llm")
    args = parser.parse_args()

    try:
        df, manifiesto = datos.cargar_o_preparar()
    except ErrorDatos as e:
        print(f"ERROR: {e}")
        return 1

    inicio = time.perf_counter()
    tablas = evaluacion.evaluar_todo(df, rapido=args.rapido, avisar=lambda t: print(f"· {t}…", flush=True))

    if args.llm > 0:
        proveedores = {rol: llm.crear_proveedor(rol) for rol in ("analista", "director", "critico")}
        if proveedores["analista"] is None:
            print("· No hay modelo de lenguaje configurado en .env: se omite la evaluación en vivo.")
        else:
            print(f"· Agentes LLM en vivo: {args.llm} consultas × {args.repeticiones} repeticiones…", flush=True)
            tablas.update(evaluacion.evaluar_llm(
                df, manifiesto["dataset_version"], proveedores, args.llm, args.repeticiones,
                estilos.calcular_estilos(df), avisar=lambda t: print(f"    {t}", flush=True)))

    carpeta = evaluacion.escribir_resultados(tablas, dataset_version=manifiesto["dataset_version"])
    print(f"\nEvaluación terminada en {time.perf_counter() - inicio:.0f} s. Informe: {carpeta / 'informe.md'}\n")
    for nombre in ("autorrecuperacion", "estilos_coherencia", "aprendizaje", "rocchio"):
        tabla = tablas.get(nombre)
        if tabla is not None and len(tabla):
            print(evaluacion.TITULOS[nombre][0])
            print(tabla.round(3).to_string(index=False), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
