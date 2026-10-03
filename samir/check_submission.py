#!/usr/bin/env python3
"""
check_submission.py
-------------------
Validador estricto de entregables para la Hackathon BCP.
Asegura que el archivo de predicciones cumpla al 100% con los requerimientos
oficiales de la competencia antes de subirlo a la plataforma.

Reglas de validación:
1. Columnas exactas: ['id_cliente', 'prediccion'] (sin columnas extras ni nombres distintos).
2. Cantidad exacta de filas: 9,900 registros (excluyendo cabecera).
3. Integridad y orden de IDs: Coincidencia 1 a 1 y en el mismo orden que test.csv.
4. Ausencia total de nulos o infinitos en las predicciones.
5. Rango continuo válido: Probabilidades estrictamente en el intervalo [0.0, 1.0].
6. Diagnóstico de distribución: Detección de colapso de varianza (predicciones constantes).
"""

from pathlib import Path
from typing import Union
import sys
import pandas as pd
import numpy as np

Union_Path = Union[str, Path]


EXPECTED_COLUMNS = ["id_cliente", "prediccion"]
EXPECTED_ROWS = 9900


def find_test_file() -> Path:
    """Localiza el archivo test.csv en data/ o en la raíz del proyecto."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "data" / "test.csv",
        base_dir / "test.csv"
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se pudo encontrar el archivo test.csv de referencia en 'data/' ni en la raíz."
    )


def validate_submission(
    submission_path: Union_Path,
    test_path: Union_Path = None,
    verbose: bool = True
) -> bool:
    """
    Ejecuta el protocolo de validación estricta sobre el archivo de entrega.

    Args:
        submission_path: Ruta al archivo CSV generado.
        test_path: Ruta opcional al archivo test.csv de referencia.
        verbose: Si True, imprime diagnósticos detallados en consola.

    Returns:
        bool: True si pasa todas las pruebas con éxito.
    """
    sub_file = Path(submission_path)
    if not sub_file.exists():
        raise FileNotFoundError(f"❌ El archivo de entrega no existe: {sub_file}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if test_path is None:
        test_file = find_test_file()
    else:
        test_file = Path(test_path)

    if verbose:
        print("=" * 65)
        print(f"[*] VALIDANDO ENTREGA: {sub_file.name}")
        print(f"[*] Referencia de test: {test_file.name}")
        print("=" * 65)

    # 1. Carga de datos
    df_sub = pd.read_csv(sub_file)
    df_test = pd.read_csv(test_file)

    # 2. Validación de columnas
    if list(df_sub.columns) != EXPECTED_COLUMNS:
        raise AssertionError(
            f"[ERROR] Discrepancia en columnas. "
            f"Esperado: {EXPECTED_COLUMNS} | Encontrado: {list(df_sub.columns)}"
        )
    if verbose:
        print("  [OK] Estructura de columnas correcta: ['id_cliente', 'prediccion']")

    # 3. Validación de cantidad de filas
    if len(df_sub) != EXPECTED_ROWS:
        raise AssertionError(
            f"[ERROR] Cantidad de filas incorrecta. "
            f"Esperado exactamente {EXPECTED_ROWS} filas | Encontrado: {len(df_sub)}"
        )
    if len(df_sub) != len(df_test):
        raise AssertionError(
            f"[ERROR] El numero de filas de la entrega ({len(df_sub)}) no coincide "
            f"con el tamano de test.csv ({len(df_test)})."
        )
    if verbose:
        print(f"  [OK] Cantidad exacta de filas verificada: {EXPECTED_ROWS:,}")

    # 4. Validación de orden e integridad de id_cliente
    test_ids = df_test["id_cliente"].values
    sub_ids = df_sub["id_cliente"].values
    if not np.array_equal(sub_ids, test_ids):
        diff_idx = np.where(sub_ids != test_ids)[0]
        first_diff = diff_idx[0]
        raise AssertionError(
            f"[ERROR] Desalineacion de IDs en la fila {first_diff}. "
            f"Esperado id_cliente={test_ids[first_diff]}, Encontrado={sub_ids[first_diff]}. "
            f"El orden y conjunto de clientes debe ser estrictamente identico al test."
        )
    if verbose:
        print("  [OK] Orden e integridad de id_cliente 100% sincronizado con test.csv")

    # 5. Validación de ausencia de nulos e infinitos
    null_count = df_sub["prediccion"].isna().sum()
    if null_count > 0:
        raise AssertionError(f"[ERROR] Se encontraron {null_count} valores nulos (NaN) en 'prediccion'.")

    preds = df_sub["prediccion"].values.astype(float)
    if not np.isfinite(preds).all():
        raise AssertionError("[ERROR] Se detectaron valores no finitos (inf / -inf) en las predicciones.")
    if verbose:
        print("  [OK] Cero valores nulos (NaN) o infinitos detectados")

    # 6. Validación de rango continuo [0.0, 1.0]
    min_val = float(np.min(preds))
    max_val = float(np.max(preds))
    if min_val < 0.0 or max_val > 1.0:
        raise AssertionError(
            f"[ERROR] Predicciones fuera del rango de probabilidad [0.0, 1.0]. "
            f"Min: {min_val:.6f}, Max: {max_val:.6f}"
        )
    if verbose:
        print(f"  [OK] Rango de probabilidades valido: [{min_val:.6f}, {max_val:.6f}]")

    # 7. Auditoría estadística de calidad
    variance = float(np.var(preds))
    std = float(np.std(preds))
    mean = float(np.mean(preds))
    p25, p50, p75 = np.percentile(preds, [25, 50, 75])

    if std < 1e-5:
        print("  [!] ADVERTENCIA CRITICA: La desviacion estandar de las predicciones es casi 0.")
        print("      El modelo esta prediciendo valores constantes (colapso de Gini).")

    if verbose:
        print("\n[*] Perfil Estadistico de la Entrega:")
        print(f"      - Media:      {mean:.5f}")
        print(f"      - Desv. Est:  {std:.5f}")
        print(f"      - P25:        {p25:.5f}")
        print(f"      - Mediana:    {p50:.5f}")
        print(f"      - P75:        {p75:.5f}")
        print(f"      - Minimo:     {min_val:.5f}")
        print(f"      - Maximo:     {max_val:.5f}")
        print("=" * 65)
        print(" [OK] VERIFICACION EXITOSA: Archivo 100% apto para submission oficial.")
        print("=" * 65)

    return True


if __name__ == "__main__":
    # Ruta por defecto al primer baseline
    default_sub = Path(__file__).resolve().parent / "submissions" / "sub_v1_baseline.csv"
    
    target_path = sys.argv[1] if len(sys.argv) > 1 else default_sub
    try:
        validate_submission(target_path)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FALLO DE VALIDACIÓN]: {exc}", file=sys.stderr)
        sys.exit(1)
