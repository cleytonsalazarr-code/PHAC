"""
src/data_audit.py
-----------------
Auditoría exhaustiva y empírica del dataset de la Hackathon BCP.

REGLA: No se asume ningún valor. Todo resultado proviene de los datos reales.

Genera:
    outputs/data_audit_report.csv  — métricas clave en formato tabular.
    Impresión detallada en consola.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOL_COLS = [
    "tiene_tarjeta_credito", "activo_movil",
    "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"
]


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga train.csv y test.csv desde data/ o raíz."""
    def _find(name: str) -> Path:
        for candidate in [ROOT_DIR / "data" / name, ROOT_DIR / name]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No se encontró: {name}")

    df_train = pd.read_csv(_find("train.csv"))
    df_test = pd.read_csv(_find("test.csv"))
    return df_train, df_test


def _section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def audit_shape_and_schema(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    _section("1. SHAPE Y ESQUEMA")

    report = {}

    print(f"\nTrain shape:  {df_train.shape[0]:,} filas x {df_train.shape[1]} columnas")
    print(f"Test  shape:  {df_test.shape[0]:,} filas x {df_test.shape[1]} columnas")
    report["train_rows"] = df_train.shape[0]
    report["train_cols"] = df_train.shape[1]
    report["test_rows"] = df_test.shape[0]
    report["test_cols"] = df_test.shape[1]

    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols

    print(f"\nColumnas solo en Train: {only_in_train}")
    print(f"Columnas solo en Test:  {only_in_test}")
    report["only_in_train"] = str(only_in_train)
    report["only_in_test"] = str(only_in_test)

    print("\nTipos de datos (Train):")
    for col, dtype in df_train.dtypes.items():
        print(f"  {col:<40} {str(dtype)}")

    mem_train_mb = df_train.memory_usage(deep=True).sum() / 1e6
    mem_test_mb = df_test.memory_usage(deep=True).sum() / 1e6
    print(f"\nMemoria Train: {mem_train_mb:.2f} MB")
    print(f"Memoria Test:  {mem_test_mb:.2f} MB")
    report["memory_train_mb"] = round(mem_train_mb, 2)
    report["memory_test_mb"] = round(mem_test_mb, 2)

    return report


def audit_quality(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    _section("2. CALIDAD — NULOS, INFINITOS, DUPLICADOS")

    report = {}

    # Nulos por columna
    null_train = df_train.isna().sum()
    null_test = df_test.isna().sum()
    total_null_train = int(null_train.sum())
    total_null_test = int(null_test.sum())

    print(f"\nNulos totales Train: {total_null_train}")
    print(f"Nulos totales Test:  {total_null_test}")
    if total_null_train > 0:
        print("  Columnas con nulos (Train):")
        print(null_train[null_train > 0].to_string())
    report["null_train_total"] = total_null_train
    report["null_test_total"] = total_null_test

    # Infinitos en columnas numéricas
    num_cols = df_train.select_dtypes(include=[np.number]).columns
    inf_train = (df_train[num_cols]
                 .apply(lambda s: np.isinf(s).sum())
                 .sum())
    print(f"\nInfinitos en Train (numéricas): {int(inf_train)}")
    report["inf_train_total"] = int(inf_train)

    # Duplicados completos
    dup_full_train = int(df_train.duplicated().sum())
    dup_full_test = int(df_test.duplicated().sum())
    print(f"\nDuplicados completos Train: {dup_full_train}")
    print(f"Duplicados completos Test:  {dup_full_test}")
    report["dup_full_train"] = dup_full_train
    report["dup_full_test"] = dup_full_test

    # Duplicados (id_cliente, mes)
    dup_key_train = int(df_train.duplicated(subset=["id_cliente", "mes"]).sum())
    dup_key_test = int(df_test.duplicated(subset=["id_cliente", "mes"]).sum())
    print(f"\nDuplicados (id_cliente, mes) Train: {dup_key_train}")
    print(f"Duplicados (id_cliente, mes) Test:  {dup_key_test}")
    report["dup_id_mes_train"] = dup_key_train
    report["dup_id_mes_test"] = dup_key_test

    # Valores sospechosos en numéricas
    print("\nValores sospechosos (mín / máx) en numéricas (Train):")
    for col in num_cols:
        mn, mx = df_train[col].min(), df_train[col].max()
        if mn < 0 and col not in ["delta_saldo", "delta_visitas"]:  # solo reporte
            print(f"  ATENCION: {col}  min={mn:.4f}  max={mx:.4f}")

    return report


def audit_time(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    _section("3. DIMENSIÓN TEMPORAL")

    report = {}

    mes_min_train = int(df_train["mes"].min())
    mes_max_train = int(df_train["mes"].max())
    mes_test = int(df_test["mes"].unique()[0]) if df_test["mes"].nunique() == 1 else -1

    print(f"\nTrain: mes mínimo = {mes_min_train}  |  mes máximo = {mes_max_train}")
    print(f"Test:  mes = {df_test['mes'].unique()}")
    report["mes_min_train"] = mes_min_train
    report["mes_max_train"] = mes_max_train
    report["mes_test"] = mes_test

    # Observaciones y tasa de conversión por mes
    print("\nObservaciones y tasa conversión por mes (Train):")
    monthly = df_train.groupby("mes").agg(
        n=("objetivo", "count"),
        positivos=("objetivo", "sum"),
    )
    monthly["tasa_conv_pct"] = (monthly["positivos"] / monthly["n"] * 100).round(2)
    print(monthly.to_string())
    report["monthly_stats"] = monthly.to_dict()

    return report


def audit_clients(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    _section("4. DIMENSIÓN DE CLIENTES")

    report = {}

    n_clients_train = df_train["id_cliente"].nunique()
    n_clients_test = df_test["id_cliente"].nunique()
    print(f"\nClientes únicos Train: {n_clients_train:,}")
    print(f"Clientes únicos Test:  {n_clients_test:,}")
    report["n_clients_train"] = n_clients_train
    report["n_clients_test"] = n_clients_test

    # Meses por cliente
    months_per_client = df_train.groupby("id_cliente")["mes"].count()
    print("\nDistribución de meses por cliente (Train):")
    dist = months_per_client.value_counts().sort_index()
    for n_months, count in dist.items():
        pct = count / n_clients_train * 100
        print(f"  {n_months} mes(es): {count:,} clientes ({pct:.1f}%)")
    report["avg_months_per_client"] = round(float(months_per_client.mean()), 2)
    report["max_months_per_client"] = int(months_per_client.max())
    report["min_months_per_client"] = int(months_per_client.min())

    # Clientes presentes en train Y test
    clients_in_both = set(df_train["id_cliente"]) & set(df_test["id_cliente"])
    pct_overlap = len(clients_in_both) / n_clients_test * 100
    print(f"\nClientes en Train y Test (overlap): {len(clients_in_both):,} ({pct_overlap:.1f}% del test)")
    report["overlap_clients"] = len(clients_in_both)
    report["overlap_pct_of_test"] = round(pct_overlap, 2)

    return report


def audit_target(df_train: pd.DataFrame) -> dict:
    _section("5. DISTRIBUCIÓN DEL TARGET")

    report = {}

    total = len(df_train)
    positivos = int(df_train["objetivo"].sum())
    negativos = total - positivos
    tasa_global = positivos / total

    print(f"\nTotal observaciones:     {total:,}")
    print(f"Objetivo = 1 (conv.):    {positivos:,}")
    print(f"Objetivo = 0 (no conv.): {negativos:,}")
    print(f"Tasa positiva global:    {tasa_global:.4f}  ({tasa_global*100:.2f}%)")
    print(f"Imbalance Ratio:         {negativos/positivos:.2f} : 1")

    report["total_obs"] = total
    report["positivos"] = positivos
    report["negativos"] = negativos
    report["positive_rate"] = round(tasa_global, 4)
    report["imbalance_ratio"] = round(negativos / positivos, 2)

    print("\nTasa de conversión por mes:")
    for mes, grp in df_train.groupby("mes"):
        tasa = grp["objetivo"].mean()
        print(f"  {mes}: {tasa:.4f} ({tasa*100:.2f}%)")

    return report


def audit_censorship(df_train: pd.DataFrame) -> dict:
    """
    VERIFICACIÓN EMPÍRICA DE CENSURA DESPUÉS DE PRIMERA CONVERSIÓN.

    No se asume que la hipótesis es verdadera. Se comprueba con los datos.
    """
    _section("6. CENSURA / ESTRUCTURA DE PRIMERA CONVERSIÓN (EMPÍRICO)")

    report = {}

    df_sorted = df_train.sort_values(["id_cliente", "mes"]).copy()

    # Clientes con alguna conversión
    clientes_con_conv = df_sorted[df_sorted["objetivo"] == 1]["id_cliente"].unique()
    report["clientes_con_conversion"] = int(len(clientes_con_conv))
    print(f"\nClientes con al menos 1 conversion: {len(clientes_con_conv):,}")

    # ¿Cuántos clientes tienen más de una conversión?
    conv_por_cliente = (df_sorted.groupby("id_cliente")["objetivo"].sum())
    multi_conv = (conv_por_cliente > 1).sum()
    print(f"Clientes con >1 conversion (viola hipotesis): {int(multi_conv)}")
    report["clientes_multi_conversion"] = int(multi_conv)

    if int(multi_conv) == 0:
        print("  [OK] Hipotesis confirmada: ningún cliente convierte más de una vez.")
    else:
        print("  [ATENCION] Existen clientes con conversiones repetidas.")

    # ¿Aparecen registros DESPUÉS de objetivo=1?
    print("\nVerificando si aparecen registros DESPUES de la primera conversion...")
    post_conversion_rows = []

    for cliente, grp in df_sorted.groupby("id_cliente"):
        grp = grp.reset_index(drop=True)
        conv_idx = grp[grp["objetivo"] == 1].index.tolist()
        if conv_idx:
            first_conv = conv_idx[0]
            # ¿Hay filas DESPUÉS de la primera conversión?
            rows_after = grp.loc[first_conv + 1:]
            if len(rows_after) > 0:
                post_conversion_rows.append({
                    "id_cliente": cliente,
                    "mes_conversion": grp.loc[first_conv, "mes"],
                    "filas_post_conversion": len(rows_after),
                    "objetivo_post": int(rows_after["objetivo"].sum())
                })

    n_post_conv = len(post_conversion_rows)
    print(f"Clientes con filas después de la conversión: {n_post_conv}")
    report["clientes_con_filas_post_conversion"] = n_post_conv

    if n_post_conv == 0:
        print("  [OK] CENSURA CONFIRMADA: Todos los clientes desaparecen después de su primera conversión.")
        report["censorship_confirmed"] = True
    else:
        print(f"  [ATENCION] {n_post_conv} clientes tienen registros POSTERIORES a su conversión.")
        report["censorship_confirmed"] = False
        # Mostrar muestra de violaciones
        df_post = pd.DataFrame(post_conversion_rows[:10])
        print("  Muestra de violaciones:")
        print(df_post.to_string(index=False))

    return report


def run_full_audit() -> pd.DataFrame:
    """Ejecuta la auditoría completa y guarda el reporte."""
    print("=" * 70)
    print("  AUDITORIA INTEGRAL DE DATOS — HACKATHON BCP")
    print("=" * 70)

    df_train, df_test = _load_data()

    all_reports = {}
    all_reports.update(audit_shape_and_schema(df_train, df_test))
    all_reports.update(audit_quality(df_train, df_test))
    all_reports.update(audit_time(df_train, df_test))
    all_reports.update(audit_clients(df_train, df_test))
    all_reports.update(audit_target(df_train))
    all_reports.update(audit_censorship(df_train))

    # Guardar reporte clave como CSV
    scalar_report = {
        k: v for k, v in all_reports.items()
        if not isinstance(v, dict)
    }
    report_df = pd.DataFrame([scalar_report])
    report_path = OUTPUT_DIR / "data_audit_report.csv"
    report_df.to_csv(report_path, index=False)

    _section("AUDITORIA COMPLETADA")
    print(f"Reporte guardado en: {report_path}")
    print(f"\nResumen ejecutivo:")
    print(f"  Train:               {all_reports['train_rows']:,} filas")
    print(f"  Test:                {all_reports['test_rows']:,} filas")
    print(f"  Tasa de conversion:  {all_reports['positive_rate']*100:.2f}%")
    print(f"  Imbalance Ratio:     {all_reports['imbalance_ratio']:.2f}:1")
    print(f"  Clientes train:      {all_reports['n_clients_train']:,}")
    print(f"  Overlap train/test:  {all_reports['overlap_clients']:,} ({all_reports['overlap_pct_of_test']:.1f}%)")
    print(f"  Censura confirmada:  {all_reports.get('censorship_confirmed', 'N/A')}")
    print(f"  Multi-conversion:    {all_reports['clientes_multi_conversion']} clientes")

    return report_df


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_full_audit()
