"""
Módulo de Inferencia para emotion2vec+ Large (Alibaba / Interspeech 2024).
Modelo fundacional auto-supervisado para Reconocimiento de Emociones en el Habla (SER).
Procesa audio crudo a 16 kHz (100% acústico, cero multimodal) y clasifica en 9 estados emocionales.
Compatible con PyTorch NVIDIA CUDA y CPU.
"""

import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.audio_processor import load_audio_array_16k, convert_to_wav_16k_mono, get_audio_metadata
from src.result_parser import compute_circumplex_state, EMOTION_COLORS, ETHICAL_DISCLAIMER
from src.cuda_monitor import get_cuda_status

# Mapeo oficial de etiquetas chinas/inglesas de emotion2vec a español
E2V_LABEL_MAP = {
    "生气/angry": "Enojo",
    "厌恶/disgusted": "Disgusto",
    "恐惧/fearful": "Miedo",
    "开心/happy": "Alegría",
    "中立/neutral": "Neutral",
    "其他/other": "Otro",
    "难过/sad": "Tristeza",
    "吃惊/surprised": "Sorpresa",
    "<unk>": "Desconocido"
}

_E2V_CACHE = {}


def load_emotion2vec_model(device: str = "cuda:0") -> Any:
    """
    Carga el modelo emotion2vec/emotion2vec_plus_large usando FunASR.
    Reutiliza la instancia en caché si ya está cargada en el dispositivo especificado.
    """
    cache_key = f"emotion2vec_plus_large_{device}"
    if cache_key in _E2V_CACHE:
        return _E2V_CACHE[cache_key]

    from funasr import AutoModel

    # Si se pide CUDA pero no está disponible, caer a CPU
    actual_device = device if ("cuda" in device and torch.cuda.is_available()) else "cpu"

    model = AutoModel(
        model="emotion2vec/emotion2vec_plus_large",
        hub="hf",
        device=actual_device,
        disable_update=True
    )
    _E2V_CACHE[cache_key] = model
    return model


def run_emotion2vec_inference(
    model: Any,
    audio_path: str,
    device: str = "cuda:0",
    window_sec: float = 3.0,
    stride_sec: float = 2.0
) -> Dict[str, Any]:
    """
    Ejecuta inferencia completa con emotion2vec+ Large:
    1. Inferencia a nivel de proferencia completa (utterance-level)
    2. Crono-análisis por ventanas deslizantes (timeline)
    3. Mapeo a las 9 clases en español y modelo circunflejo de Russell 2D
    """
    t_start = time.perf_counter()

    # 1. Metadatos y conversión a WAV 16k mono
    meta_audio = get_audio_metadata(audio_path)
    wav_path = convert_to_wav_16k_mono(audio_path)
    audio_arr, duration = load_audio_array_16k(wav_path)

    # 2. Inferencia global de audio completo
    try:
        raw_res = model.generate(wav_path, output_dir=None, granularity="utterance")
    except Exception as exc:
        if os.path.exists(wav_path) and wav_path != audio_path:
            try: os.remove(wav_path)
            except: pass
        raise RuntimeError(f"Fallo en inferencia de emotion2vec: {exc}")

    # 3. Parsear distribución de probabilidades de las 9 clases
    first_item = raw_res[0] if raw_res and len(raw_res) > 0 else {}
    raw_labels = first_item.get("labels", [])
    raw_scores = first_item.get("scores", [])

    detailed_list = []
    emotion_scores = {}

    for raw_l, score in zip(raw_labels, raw_scores):
        es_label = E2V_LABEL_MAP.get(raw_l, raw_l)
        pct = round(float(score) * 100, 2)
        score_val = round(float(score), 4)

        emotion_scores[es_label] = score_val
        detailed_list.append({
            "id": len(detailed_list),
            "raw_label": raw_l,
            "categoria": es_label,
            "probabilidad": score_val,
            "porcentaje": pct,
            "color": EMOTION_COLORS.get(es_label, "#64748B")
        })

    detailed_list.sort(key=lambda x: x["probabilidad"], reverse=True)
    predominant = detailed_list[0] if detailed_list else {
        "categoria": "Neutral", "probabilidad": 1.0, "porcentaje": 100.0, "raw_label": "neutral"
    }

    # Vector Circunflejo de Russell 2D
    circumplex = compute_circumplex_state(emotion_scores)

    # Formatear lista de desglose para la UI
    formatted_breakdown = []
    for item in detailed_list:
        formatted_breakdown.append({
            "name": item["categoria"],
            "raw_cat": item["categoria"],
            "percent": item["porcentaje"],
            "prob": item["probabilidad"],
            "color": item["color"]
        })

    parsed_result = {
        "predominant_emotion": predominant["categoria"],
        "predominant_raw": predominant["raw_label"],
        "predominant_score": predominant["probabilidad"],
        "predominant_percent": predominant["porcentaje"],
        "circumplex": circumplex,
        "formatted_breakdown": formatted_breakdown,
        "scores_dict": emotion_scores,
        "ranking": detailed_list,
        "disclaimer": ETHICAL_DISCLAIMER
    }

    # 4. Crono-Análisis por Ventanas Deslizantes (Timeline)
    sr = 16000
    total_samples = len(audio_arr)
    segments = []

    if duration > 1.5 and total_samples > int(1.0 * sr):
        win_samples = int(window_sec * sr)
        stride_samples = int(stride_sec * sr)
        curr = 0
        w_idx = 1

        import soundfile as sf
        import tempfile

        while curr < total_samples:
            end = min(curr + win_samples, total_samples)
            chunk = audio_arr[curr:end]
            if len(chunk) < int(0.8 * sr) and len(segments) > 0:
                break
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))

            # Guardar chunk temporal para el extractor
            fd, tmp_chunk_path = tempfile.mkstemp(suffix="_chunk.wav")
            os.close(fd)
            sf.write(tmp_chunk_path, chunk, sr)

            try:
                chunk_res = model.generate(tmp_chunk_path, output_dir=None, granularity="utterance")
                c_item = chunk_res[0] if chunk_res else {}
                c_labels = c_item.get("labels", [])
                c_scores = c_item.get("scores", [])
                if c_scores:
                    top_idx = int(np.argmax(c_scores))
                    top_raw = c_labels[top_idx]
                    top_es = E2V_LABEL_MAP.get(top_raw, top_raw)
                    top_pct = round(float(c_scores[top_idx]) * 100, 1)
                else:
                    top_es = "Neutral"
                    top_pct = 50.0
            except Exception:
                top_es = "Neutral"
                top_pct = 50.0
            finally:
                if os.path.exists(tmp_chunk_path):
                    try: os.remove(tmp_chunk_path)
                    except: pass

            start_s = round(curr / sr, 2)
            end_s = round(min(end, total_samples) / sr, 2)
            seg_dur = round(end_s - start_s, 2)
            pct_w = round((seg_dur / max(0.1, duration)) * 100, 1)

            segments.append({
                "index": w_idx,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": seg_dur,
                "width_percent": pct_w,
                "emotion": top_es,
                "confidence_percent": top_pct,
                "color": EMOTION_COLORS.get(top_es, "#64748B")
            })

            curr += stride_samples
            w_idx += 1
    else:
        # Audio muy corto: un solo segmento
        segments.append({
            "index": 1,
            "start_s": 0.0,
            "end_s": duration,
            "duration_s": duration,
            "width_percent": 100.0,
            "emotion": predominant["categoria"],
            "confidence_percent": predominant["porcentaje"],
            "color": predominant.get("color", "#64748B")
        })

    timeline_result = {
        "status": "EXITOSA",
        "total_duration_s": duration,
        "num_windows": len(segments),
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "segments": segments
    }

    t_end = time.perf_counter()
    latency_sec = t_end - t_start

    # Limpieza de archivo temporal
    if os.path.exists(wav_path) and wav_path != audio_path:
        try: os.remove(wav_path)
        except: pass

    # Telemetría CUDA
    cuda_telemetry = get_cuda_status() if "cuda" in device else None

    return {
        "status": "EXITOSA",
        "error_message": None,
        "model_id": "emotion2vec/emotion2vec_plus_large",
        "backbone": "emotion2vec+ Large (Data2Vec 2.0)",
        "num_params_str": "315.0M params",
        "num_classes": 9,
        "classes_names": list(emotion_scores.keys()),
        "parsed_result": parsed_result,
        "timeline_result": timeline_result,
        "metrics": {
            "latency_seconds": round(latency_sec, 4),
            "latency_ms": round(latency_sec * 1000, 2),
            "audio_duration_seconds": duration,
            "real_time_factor_rtf": round(latency_sec / max(0.01, duration), 3),
            "speedup_vs_realtime": f"{round(duration / max(0.001, latency_sec), 1)}x más rápido que tiempo real",
            "vram_stats": {"status": "Activo"},
            "device_used": device,
            "is_cuda": "cuda" in device
        },
        "audio_metadata": meta_audio,
        "cuda_telemetry": cuda_telemetry
    }
