#!/usr/bin/env python3
"""
setup_project.py
----------------
Script de inicialización de la arquitectura modular para la Hackathon BCP.
Crea la estructura de directorios, paquetes y traslada los conjuntos de datos
a la carpeta 'data/' correspondiente.
"""

from pathlib import Path
import shutil
import sys

def setup_directory_structure():
    base_dir = Path(__file__).resolve().parent

    # 1. Definición de directorios requeridos
    directories = [
        base_dir / "data",
        base_dir / "src",
        base_dir / "notebooks",
        base_dir / "submissions"
    ]

    print("[*] Creando estructura modular de directorios...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [+] Directorio verificado/creado: {directory.relative_to(base_dir)}/")

    # 2. Creación de archivos __init__.py en paquetes Python
    init_file = base_dir / "src" / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            '"""BCP Hackathon - Predictive Pipeline Package."""\n__version__ = "0.1.0"\n',
            encoding="utf-8"
        )
        print("  [+] Archivo inicializado: src/__init__.py")
    else:
        print("  [=] Archivo ya presente: src/__init__.py")

    # 3. Traslado seguro de datos si están en la raíz del proyecto
    dataset_files = [
        "train.csv",
        "test.csv",
        "metaData.csv",
        "sample_submission.csv"
    ]

    print("\n[*] Verificando ubicacion de datasets...")
    for filename in dataset_files:
        root_path = base_dir / filename
        data_path = base_dir / "data" / filename

        if root_path.exists() and not data_path.exists():
            print(f"  [->] Moviendo {filename} -> data/{filename}")
            shutil.move(str(root_path), str(data_path))
        elif data_path.exists():
            print(f"  [v] Dataset presente en: data/{filename}")
        else:
            print(f"  [!] Advertencia: {filename} no encontrado en raiz ni en data/")

    # 4. Creación de .gitkeep en carpetas vacías para control de versiones
    for empty_dir in ["notebooks", "submissions"]:
        gitkeep = base_dir / empty_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print("\n[OK] Setup completado exitosamente con arquitectura modular lista.")

if __name__ == "__main__":
    setup_directory_structure()
