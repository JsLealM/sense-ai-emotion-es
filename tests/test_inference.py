"""
Script de prueba de inferencia real con W2V-BERT Emotion ES (UMUTeam).
Valida la carga del modelo en GPU CUDA, la clasificación en las 6 categorías y el consumo de VRAM.
Ejecutar con: python tests/test_inference.py [ruta_opcional_audio] [--device cuda:0|cpu]
"""

import sys
import os
import argparse

# Agregar directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_loader import load_emotion_model
from src.inference import run_emotion_inference
from src.cuda_monitor import verify_model_device


def run_test(audio_path: str = None, device: str = "cuda:0", use_fp16: bool = True):
    print("=" * 70)
    print("   TEST DE INFERENCIA REAL - W2V-BERT EMOTION ES (GPU CUDA)")
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
    print(f"[3] Optimización FP16:            {use_fp16 and 'cuda' in device}")

    # 2. Cargar modelo
    print("\n[4] Cargando pesos de W2V-BERT Emotion ES en GPU...")
    try:
        model, feat_extractor, meta = load_emotion_model(device_name=device, use_fp16=use_fp16)
        print("    [OK] Modelo cargado correctamente.")
    except Exception as exc:
        print(f"    [ERROR] Error cargando modelo: {exc}")
        sys.exit(1)

    # 3. Verificar que los tensores estén físicamente en GPU
    dev_info = verify_model_device(model)
    print(f"[5] Verificación de Tensores:    {dev_info['device']}")
    print(f"    - Reside en CUDA:             {dev_info['is_on_cuda']}")
    print(f"    - Parámetros contados:        {dev_info['param_count']:,}")
    print(f"    - Precisión (Dtype):          {meta['dtype']}")

    # 4. Ejecutar inferencia
    print("\n[6] Ejecutando pase hacia adelante (Inferencia emocional)...")
    result = run_emotion_inference(model, feat_extractor, audio_path, device=device)

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

    print("\nDistribución en las 6 Categorías:")
    for item in parsed["ranking"]:
        bar = "█" * int(item["porcentaje"] / 5)
        print(f"  - {item['categoria']:<10}: {item['porcentaje']:>6.2f}% | {bar}")

    print(f"\n{parsed['disclaimer']}")

    print("\n" + "-" * 70)
    print("   MÉTRICAS TÉCNICAS Y CONSUMO DE MEMORIA")
    print("-" * 70)
    print(f"Dispositivo utilizado:   {metrics['device_used']}")
    print(f"Latencia total:          {metrics['latency_ms']} ms ({metrics['latency_seconds']} s)")
    print(f"Duración del audio:      {metrics['audio_duration_seconds']} s")
    print(f"Real-Time Factor (RTF):  {metrics['real_time_factor_rtf']}")
    print(f"Velocidad relativa:      {metrics['speedup_vs_realtime']}")

    vram = metrics["vram_stats"]
    print(f"VRAM Inicial:            {vram['start_mb']} MB")
    print(f"VRAM Pico en inferencia: {vram['peak_mb']} MB")
    print(f"VRAM Final:              {vram['end_mb']} MB")

    print("\n" + "=" * 70)
    print("[OK] TEST COMPLETADO CON ÉXITO: W2V-BERT Emotion ES operando en GPU NVIDIA.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de inferencia W2V-BERT Emotion ES")
    parser.add_argument("audio", nargs="?", default=None, help="Ruta a un archivo de audio en español")
    parser.add_argument("--device", default="cuda:0", help="Dispositivo (cuda:0 o cpu)")
    parser.add_argument("--no-fp16", action="store_true", help="Desactivar precisión FP16")
    args = parser.parse_args()

    run_test(audio_path=args.audio, device=args.device, use_fp16=not args.no_fp16)
