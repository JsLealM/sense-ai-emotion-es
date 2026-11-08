"""
Script de Descarga y Verificación Previa de emotion2vec+ Large.
Ideal para compañeros o nuevos entornos (ej. WSL / Linux / Windows) que clonan
el repositorio por primera vez y requieren precargar los pesos (~1.2 GB).

Uso:
    python scripts/download_model.py [--device cuda:0|cpu]
"""

import sys
import os
import argparse
import time

# Agregar directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.emotion2vec_engine import find_local_emotion2vec_snapshot, load_emotion2vec_model


def setup_and_download(device: str = "cuda:0"):
    print("=" * 70)
    print("   AURAVOICE ENGINE — DESCARGA Y PREPARACIÓN DE PESOS DEL MODELO")
    print("=" * 70)

    # 1. Comprobación de Hardware y CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n[1] Entorno de Ejecución:")
    print(f"    - Python:          {sys.version.split()[0]}")
    print(f"    - PyTorch:         {torch.__version__}")
    print(f"    - CUDA Disponible: {cuda_available}")

    if "cuda" in device and not cuda_available:
        print("    [AVISO] Se solicitó CUDA pero no hay GPU compatible disponible.")
        print("            Cambiando automáticamente a modo CPU.")
        device = "cpu"

    if cuda_available and "cuda" in device:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2)
        print(f"    - Dispositivo:     {gpu_name} ({vram_mb} MB VRAM)")
    else:
        print(f"    - Dispositivo:     Procesador Central (CPU)")

    # 2. Verificar estado de la caché local
    print(f"\n[2] Comprobación de Caché Local:")
    existing_snapshot = find_local_emotion2vec_snapshot()
    if existing_snapshot:
        print(f"    [OK] Los pesos ya se encuentran descargados en este equipo:")
        print(f"         {existing_snapshot}")
        print("    No es necesario volver a descargar.")
    else:
        print("    [!] No se encontró copia local de los pesos.")
        print("    Descargando emotion2vec/emotion2vec_plus_large desde Hugging Face...")
        print("    (Tamaño aprox: ~1.2 GB. Por favor espera a que finalice la descarga)...")

    # 3. Descarga / Inicialización
    t0 = time.perf_counter()
    try:
        model = load_emotion2vec_model(device=device)
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"\n[3] Modelo cargado con éxito en {elapsed} segundos.")
    except Exception as exc:
        print(f"\n[ERROR] Ocurrió una falla durante la descarga o carga del modelo:")
        print(f"        {exc}")
        sys.exit(1)

    # 4. Verificación de Inferencia Rápida (con sample_es.wav)
    sample_wav = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples", "sample_es.wav"))
    if os.path.exists(sample_wav):
        print(f"\n[4] Ejecutando pase de verificación con audio de muestra ({os.path.basename(sample_wav)})...")
        try:
            from src.emotion2vec_engine import run_emotion2vec_inference
            inf_res = run_emotion2vec_inference(model, sample_wav, device=device)
            if inf_res["status"] == "EXITOSA":
                pred = inf_res["parsed_result"]["predominant_emotion"]
                pct = inf_res["parsed_result"]["predominant_percent"]
                lat = inf_res["metrics"]["latency_ms"]
                print(f"    [OK] Inferencia completada con éxito en {lat} ms.")
                print(f"    [OK] Emoción detectada: {pred} ({pct}%)")
            else:
                print(f"    [AVISO] Inferencia retornó: {inf_res.get('error_message')}")
        except Exception as exc:
            print(f"    [AVISO] Verificación con audio omitida: {exc}")

    # 5. Resumen Final
    print("\n" + "=" * 70)
    print("[OK] PREPARACIÓN COMPLETADA CON ÉXITO")
    print("=" * 70)
    print("El modelo está listo para funcionar de manera 100% offline.")
    print("Ya puedes ejecutar la aplicación con:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y verificación de emotion2vec+ Large")
    parser.add_argument("--device", default="cuda:0", help="Dispositivo objetivo (cuda:0 o cpu)")
    args = parser.parse_args()

    setup_and_download(device=args.device)
