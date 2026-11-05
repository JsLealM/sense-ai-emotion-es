"""
Script de verificación técnica de NVIDIA CUDA y aceleración GPU con PyTorch.
Ejecutar con: python tests/test_cuda.py
"""

import sys
import os
import time

# Agregar directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.cuda_monitor import get_cuda_status, CUDAMemoryTracker


def run_cuda_verification():
    print("=" * 70)
    print("   VERIFICACIÓN TÉCNICA DE NVIDIA CUDA & PYTORCH (TEST SUITE)")
    print("=" * 70)

    # 1. Inspección básica
    print(f"\n[1] Versión de Python:    {sys.version.split()[0]}")
    print(f"[2] Versión de PyTorch:   {torch.__version__}")
    print(f"[3] PyTorch CUDA Version: {torch.version.cuda}")
    print(f"[4] CUDA Disponible:      {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[ERROR] torch.cuda.is_available() devolvió False.")
        print("Verifica que los controladores NVIDIA y la rueda PyTorch con CUDA estén correctamente instalados.")
        sys.exit(1)

    # 2. Detalles de la GPU
    device_idx = 0
    gpu_name = torch.cuda.get_device_name(device_idx)
    capability = torch.cuda.get_device_capability(device_idx)
    print(f"\n[5] GPU Detectada [ID {device_idx}]: {gpu_name}")
    print(f"[6] Compute Capability:       {capability[0]}.{capability[1]}")
    
    status = get_cuda_status()
    print(f"[7] VRAM Total:               {status['vram_total_mb']} MB (~{round(status['vram_total_mb']/1024, 2)} GB)")
    print(f"[8] VRAM Libre Inicial:       {status['vram_free_mb']} MB")

    # 3. Test de Asignación y Operaciones Tensoriales en GPU
    print("\n" + "-" * 70)
    print("   EJECUCIÓN DE PRUEBA DE CÁLCULO TENSORIAL EN GPU")
    print("-" * 70)

    tracker = CUDAMemoryTracker(device="cuda:0")
    tracker.start()

    # Creación explícita de tensores en cuda:0
    t0 = time.perf_counter()
    size = 4000
    a = torch.randn(size, size, device="cuda:0", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda:0", dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    mem_stats = tracker.stop()

    print(f"[OK] Multiplicación de matrices ({size}x{size}) ejecutada con éxito.")
    print(f"  - Dispositivo del tensor resultado: {c.device}")
    print(f"  - Es tensor CUDA:                   {c.is_cuda}")
    print(f"  - Tiempo de ejecución en GPU:       {round((t1 - t0) * 1000, 2)} ms")
    print(f"  - VRAM Inicial:                     {mem_stats['start_mb']} MB")
    print(f"  - VRAM Pico durante cómputo:        {mem_stats['peak_mb']} MB")
    print(f"  - VRAM Final:                       {mem_stats['end_mb']} MB")

    assert c.is_cuda, "El tensor resultante no reside en CUDA"
    assert "cuda" in str(c.device), f"Dispositivo inesperado: {c.device}"

    # Limpieza
    del a, b, c
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("[OK] RESULTADO: El entorno CUDA está plenamente operativo para inferencia en GPU.")
    print("=" * 70)


if __name__ == "__main__":
    run_cuda_verification()
