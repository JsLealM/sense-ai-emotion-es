"""
Módulo Orquestador de Inferencia de Emociones en la Voz (W2V-BERT Emotion ES).
Ejecuta la inferencia en GPU CUDA o CPU, mide tiempos sincronizados y captura consumo de VRAM.
"""

import time
from typing import Dict, Any, Optional
import torch

from src.cuda_monitor import CUDAMemoryTracker, get_cuda_status
from src.result_parser import parse_emotion_probabilities
from src.audio_processor import load_audio_array_16k, get_audio_metadata


def run_emotion_inference(
    model: Any,
    feature_extractor: Any,
    audio_path: str,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Ejecuta la clasificación emocional en una grabación de voz.
    
    Parámetros:
      - model: instancia de AutoModelForAudioClassification
      - feature_extractor: SeamlessM4TFeatureExtractor
      - audio_path: ruta al archivo de audio
      - device: "cuda:0" o "cpu"
    """
    # 1. Metadatos y carga del audio a 16kHz
    meta_audio = get_audio_metadata(audio_path)
    audio_array, audio_duration = load_audio_array_16k(audio_path)

    # 2. Inicializar rastreador de memoria VRAM y sincronización CUDA
    tracker = CUDAMemoryTracker(device=device)
    tracker.start()

    start_time = time.perf_counter()
    execution_status = "EXITOSA"
    error_message = None
    logits_tensor = None

    try:
        # 3. Extracción de características (Fbank de 80 canales a 16kHz)
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        # Transferir tensores al dispositivo correspondiente (cuda:0 o cpu)
        cuda_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    cuda_inputs[k] = v.to(device=model_device, dtype=model_dtype)
                else:
                    cuda_inputs[k] = v.to(device=model_device)
            else:
                cuda_inputs[k] = v

        # 4. Inferencia (Pase hacia adelante / Forward pass)
        with torch.no_grad():
            outputs = model(**cuda_inputs)
            logits_tensor = outputs.logits

    except torch.cuda.OutOfMemoryError as oom_exc:
        execution_status = "ERROR_OOM_VRAM"
        error_message = (
            f"VRAM insuficiente en {device}: {oom_exc}. "
            "Para GPU de 4 GB: se recomienda activar FP16 o acortar la duración del audio."
        )
        torch.cuda.empty_cache()
    except Exception as exc:
        execution_status = "ERROR_INFERENCIA"
        error_message = str(exc)

    end_time = time.perf_counter()
    mem_stats = tracker.stop()

    latency_seconds = round(end_time - start_time, 4)
    rtf = round(latency_seconds / audio_duration, 4) if audio_duration > 0 else 0.0

    # 5. Parsear probabilidades en las 6 categorías en español
    parsed = {}
    if execution_status == "EXITOSA" and logits_tensor is not None:
        parsed = parse_emotion_probabilities(logits_tensor, model.config.id2label)

    # 6. Telemetría de hardware
    cuda_telemetry = get_cuda_status() if "cuda" in device else None

    return {
        "status": execution_status,
        "error_message": error_message,
        "parsed_result": parsed,
        "metrics": {
            "latency_seconds": latency_seconds,
            "latency_ms": round(latency_seconds * 1000, 2),
            "audio_duration_seconds": audio_duration,
            "real_time_factor_rtf": rtf,
            "speedup_vs_realtime": f"{round(1 / rtf, 1)}x más rápido que tiempo real" if rtf > 0 else "N/A",
            "vram_stats": mem_stats,
            "device_used": str(next(model.parameters()).device),
            "is_cuda": "cuda" in str(next(model.parameters()).device)
        },
        "audio_metadata": meta_audio,
        "cuda_telemetry": cuda_telemetry
    }
