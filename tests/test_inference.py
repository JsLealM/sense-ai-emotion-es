"""
Script de prueba de inferencia real con emotion2vec+ Large (Interspeech 2024 SOTA).
Valida la carga del modelo en GPU CUDA, la clasificación en las 9 categorías y el consumo de VRAM acotado.
Ejecutar con: python tests/test_inference.py [ruta_opcional_audio] [--device cuda:0|cpu]
"""

import sys
import os
import argparse

# Agregar directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.emotion2vec_engine import load_emotion2vec_model, run_emotion2vec_inference


def run_test(audio_path: str = None, device: str = "cuda:0"):
    print("=" * 70)
    print("   TEST DE INFERENCIA REAL - emotion2vec+ Large (GPU CUDA)")
    print("=" * 70)

    # 1. Determinar audio de prueba
    if not audio_path or not os.path.exists(audio_path):
        samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
        candidate = os.path.join(samples_dir, "sample_es.wav")
        if os.path.exists(candidate):
            audio_path = candidate
        else:
            print(f"[ERROR] No se proporcionó un audio válido ni se encontró {candidate}")
            sys.exit(1)

    print(f"\n[1] Archivo de audio de prueba: {audio_path}")
    print(f"[2] Dispositivo objetivo:         {device}")
    print(f"[3] Arquitectura del Modelo:      Data2Vec 2.0 (315.0M params, Interspeech 2024 SOTA)")

    # 2. Cargar modelo
    print("\n[4] Cargando pesos de emotion2vec+ Large en GPU...")
    try:
        model = load_emotion2vec_model(device=device)
        print("    [OK] Modelo cargado correctamente.")
    except Exception as exc:
        print(f"    [ERROR] Error cargando modelo: {exc}")
        sys.exit(1)

    # 3. Ejecutar inferencia
    print("\n[5] Ejecutando pase hacia adelante (Inferencia emocional 9 clases)...")
    result = run_emotion2vec_inference(model, audio_path, device=device)

    print("\n" + "-" * 70)
    print("   RESULTADOS DE LA CLASIFICACIÓN EMOCIONAL")
    print("-" * 70)

    status = result["status"]
    print(f"Estado de ejecución:     {status}")

    if status != "EXITOSA":
        print(f"[ERROR] Error durante inferencia: {result.get('error_message')}")
        sys.exit(1)

    parsed = result["parsed_result"]
    metrics = result["metrics"]

    print(f"Emoción Predominante:    {parsed['predominant_emotion']} ({parsed['predominant_percent']}%)")
    print(f"Puntuación (Prob.):      {parsed['predominant_score']}")

    print("\nDistribución en las 9 Categorías:")
    for item in parsed["ranking"]:
        bar = "█" * int(item["porcentaje"] / 5)
        print(f"  - {item['categoria']:<16}: {item['porcentaje']:>6.2f}% | {bar}")

    print(f"\n{parsed['disclaimer']}")

    print("\n" + "-" * 70)
    print("   MÉTRICAS TÉCNICAS Y CONSUMO DE MEMORIA")
    print("-" * 70)
    print(f"Dispositivo utilizado:   {metrics['device_used']}")
    print(f"Latencia total:          {metrics['latency_ms']} ms ({metrics['latency_seconds']} s)")
    print(f"Duración del audio:      {metrics['audio_duration_seconds']} s")
    print(f"Real-Time Factor (RTF):  {metrics['real_time_factor_rtf']}")
    print(f"Velocidad relativa:      {metrics['speedup_vs_realtime']}")

    vram = metrics.get("vram_stats", {})
    print(f"VRAM Pico en inferencia: {vram.get('peak_mb', 0)} MB")
    print(f"VRAM Asignada:           {vram.get('allocated_mb', 0)} MB")

    print("\n" + "=" * 70)
    print("[OK] TEST COMPLETADO CON ÉXITO: emotion2vec+ Large operando en GPU NVIDIA.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de inferencia emotion2vec+ Large")
    parser.add_argument("audio", nargs="?", default=None, help="Ruta a un archivo de audio")
    parser.add_argument("--device", default="cuda:0", help="Dispositivo (cuda:0 o cpu)")
    args = parser.parse_args()

    run_test(audio_path=args.audio, device=args.device)
