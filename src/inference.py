"""
Módulo Orquestador de Inferencia de Emociones en la Voz (W2V-BERT Emotion ES).
Ejecuta la inferencia en GPU CUDA o CPU, mide tiempos sincronizados y captura consumo de VRAM.
"""

import time
from typing import Dict, Any, Optional
import torch

from src.cuda_monitor import CUDAMemoryTracker, get_cuda_status
from src.result_parser import parse_emotion_probabilities, EMOTION_TRANSLATION, EMOTION_COLORS
from src.audio_processor import load_audio_array_16k, get_audio_metadata
import numpy as np


def run_timeline_batch_inference(
    model: Any,
    feature_extractor: Any,
    audio_array: np.ndarray,
    audio_duration: float,
    device: str = "cuda:0",
    window_sec: float = 3.0,
    stride_sec: float = 2.0
) -> Dict[str, Any]:
    """
    Ejecuta el crono-análisis en paralelo mediante ventana deslizante y GPU batching.
    Divide la señal de audio en ventanas de longitud window_sec con paso stride_sec,
    empaca los segmentos en un lote (batch tensor) y evalúa todas las ventanas
    en un único pase hacia adelante (forward pass) en la GPU NVIDIA CUDA.
    """
    sr = 16000
    total_samples = len(audio_array)

    if total_samples == 0:
        return {"status": "VACIO", "segments": [], "total_duration_s": 0.0}

    # Si el audio es corto (menor o igual a la ventana), evaluar como un solo segmento
    if audio_duration <= window_sec:
        chunks = [audio_array]
        spans = [(0.0, audio_duration)]
    else:
        win_samples = int(window_sec * sr)
        stride_samples = int(stride_sec * sr)
        chunks = []
        spans = []
        curr = 0
        while curr < total_samples:
            end = min(curr + win_samples, total_samples)
            chunk = audio_array[curr:end]
            if len(chunk) < int(1.0 * sr) and len(chunks) > 0:
                break
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            chunks.append(chunk)
            spans.append((round(curr / sr, 2), round(end / sr, 2)))
            curr += stride_samples

    # Extracción de características en lote (Batch Feature Extraction)
    inputs = feature_extractor(
        chunks,
        sampling_rate=sr,
        padding=True,
        return_tensors="pt"
    )

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    cuda_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                cuda_inputs[k] = v.to(device=model_device, dtype=model_dtype)
            else:
                cuda_inputs[k] = v.to(device=model_device)
        else:
            cuda_inputs[k] = v

    # Ejecución sincronizada en CUDA
    if "cuda" in str(model_device):
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**cuda_inputs, return_dict=True)
        logits = outputs.logits

    if "cuda" in str(model_device):
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    batch_latency_ms = round((t_end - t_start) * 1000, 2)
    probs_tensor = torch.softmax(logits, dim=-1)

    segments = []
    for i, span in enumerate(spans):
        top_idx = torch.argmax(probs_tensor[i]).item()
        top_label_eng = model.config.id2label[top_idx]
        top_label_es = EMOTION_TRANSLATION.get(top_label_eng.lower(), top_label_eng)
        pct = round(probs_tensor[i][top_idx].item() * 100, 1)
        color = EMOTION_COLORS.get(top_label_es, "#574A43")

        seg_dur = round(span[1] - span[0], 2)
        pct_width = round((seg_dur / max(0.1, audio_duration)) * 100, 1)

        segments.append({
            "index": i + 1,
            "start_s": span[0],
            "end_s": span[1],
            "duration_s": seg_dur,
            "width_percent": pct_width,
            "emotion": top_label_es,
            "confidence_percent": pct,
            "color": color
        })

    return {
        "status": "EXITOSA",
        "total_duration_s": audio_duration,
        "num_windows": len(segments),
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "batch_latency_ms": batch_latency_ms,
        "ms_per_window": round(batch_latency_ms / max(1, len(segments)), 2),
        "segments": segments,
        "logits": logits.detach().cpu()
    }


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
            outputs = model(**cuda_inputs, return_dict=True)
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

    # 5. Crono-análisis en lote mediante GPU batching
    timeline_result = None
    if execution_status == "EXITOSA" and audio_array is not None:
        try:
            timeline_result = run_timeline_batch_inference(
                model=model,
                feature_extractor=feature_extractor,
                audio_array=audio_array,
                audio_duration=audio_duration,
                device=device,
                window_sec=3.0,
                stride_sec=2.0
            )
        except Exception as t_exc:
            timeline_result = {"status": "ERROR", "error": str(t_exc), "segments": []}

    # 6. Parsear probabilidades en las 6 categorías en español
    # Para audios de varias ventanas (> 3.5s), agregamos los logits de todas las ventanas
    # (Temporal Average Pooling) para que la emoción dominante del audio refleje fielmente
    # las transiciones afectivas activas (ej. Tristeza o Alegría) y evitar el sesgo de dilución
    # hacia 'Neutral' que sufren los modelos W2V-BERT al procesar señales largas no segmentadas.
    parsed = {}
    if execution_status == "EXITOSA":
        if (
            timeline_result
            and "logits" in timeline_result
            and timeline_result["logits"] is not None
            and timeline_result.get("num_windows", 0) > 1
        ):
            mean_logits = timeline_result["logits"].mean(dim=0, keepdim=True)
            parsed = parse_emotion_probabilities(mean_logits, model.config.id2label)
        elif logits_tensor is not None:
            parsed = parse_emotion_probabilities(logits_tensor, model.config.id2label)

    # 7. Telemetría de hardware
    cuda_telemetry = get_cuda_status() if "cuda" in device else None

    return {
        "status": execution_status,
        "error_message": error_message,
        "parsed_result": parsed,
        "timeline_result": timeline_result,
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
