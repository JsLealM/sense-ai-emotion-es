"""
Módulo de Inferencia para emotion2vec+ Large (emotion2vec / FunASR, arXiv:2312.15185).
Modelo fundacional auto-supervisado para Reconocimiento de Emociones en el Habla (SER).
Procesa audio crudo a 16 kHz (100% acústico, cero multimodal) y clasifica en 9 estados emocionales.
Compatible con PyTorch NVIDIA CUDA y CPU.
"""

import os
import sys
import time
import contextlib
from typing import Dict, Any, List, Optional
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.audio_processor import load_audio_array_16k, convert_to_wav_16k_mono, get_audio_metadata
from src.result_parser import compute_circumplex_state, EMOTION_COLORS, ETHICAL_DISCLAIMER
from src.cuda_monitor import get_cuda_status

@contextlib.contextmanager
def silence_stdio():
    """Silencia completamente stdout y stderr para evitar contaminación del log de consola."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

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


def unload_emotion2vec_model():
    """
    Descarga de la GPU y libera la memoria VRAM ocupada por emotion2vec.
    Garantiza que la VRAM quede disponible para otros modelos o buffers.
    """
    global _E2V_CACHE
    for k, model in list(_E2V_CACHE.items()):
        try:
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to("cpu")
        except Exception:
            pass
    _E2V_CACHE.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Capas convolucionales del extractor de características de Data2Vec 2.0 (kernel, stride),
# tomadas de feature_encoder_spec en el config.yaml del modelo:
# [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]
_CNN_SPEC = [(10, 5)] + [(3, 2)] * 4 + [(2, 2)] * 2


def cnn_output_frames(num_samples: int) -> int:
    """Número de frames que produce la CNN para una señal de num_samples muestras (≈ 50 por segundo)."""
    length = int(num_samples)
    for kernel, stride in _CNN_SPEC:
        length = (length - kernel) // stride + 1
        if length <= 0:
            return 0
    return length


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Inspecciona el modelo cargado y devuelve datos medidos, no declarados:
    número real de parámetros, tipo de dato, dispositivo y bloques Transformer.
    El resultado se guarda en el propio objeto para no recalcularlo.
    """
    cached = getattr(model, "_auravoice_info", None)
    if cached is not None:
        return cached

    net = getattr(model, "model", model)
    params = list(net.parameters()) if hasattr(net, "parameters") else []
    total = sum(p.numel() for p in params)
    first = params[0] if params else None
    # El decoder de Data2Vec solo se usa en el preentrenamiento (predicción enmascarada): no viene en el
    # checkpoint de Hugging Face ni participa en extract_features(). Se excluye del conteo de inferencia.
    decoder = 0
    if hasattr(net, "named_parameters"):
        decoder = sum(p.numel() for n, p in net.named_parameters() if ".decoder." in f".{n}")
    used = total - decoder

    main_blocks, prenet_blocks = set(), set()
    if hasattr(net, "named_parameters"):
        for name, _ in net.named_parameters():
            parts = name.split(".")
            if "blocks" in parts:
                idx = parts[parts.index("blocks") + 1]
                if "context_encoder" in parts:
                    prenet_blocks.add(idx)
                else:
                    main_blocks.add(idx)

    dtype = str(first.dtype).replace("torch.", "") if first is not None else "desconocido"
    info = {
        "params": used,
        "params_str": f"{used / 1e6:.1f}M params",
        "params_in_memory": total,
        "params_decoder_unused": decoder,
        "dtype": dtype,
        "dtype_label": {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}.get(dtype, dtype),
        "device": str(first.device) if first is not None else "desconocido",
        "transformer_blocks": len(main_blocks) + len(prenet_blocks),
        "prenet_blocks": len(prenet_blocks),
        "main_blocks": len(main_blocks),
        "weights_mb": round(sum(p.numel() * p.element_size() for p in params) / (1024 ** 2), 1),
    }
    try:
        model._auravoice_info = info
    except Exception:
        pass
    return info


def get_loaded_model_info() -> Optional[Dict[str, Any]]:
    """Datos reales del primer modelo ya cargado en memoria, o None si aún no se ha cargado ninguno."""
    for model in _E2V_CACHE.values():
        return get_model_info(model)
    return None


def _sync(device: str):
    """Espera a que la GPU termine el trabajo encolado (CUDA es asíncrono) antes de leer el reloj."""
    if HAS_TORCH and "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def find_local_emotion2vec_snapshot() -> Optional[str]:
    """
    Busca si los pesos del modelo ya existen en la caché local de Hugging Face.
    Retorna la ruta absoluta al snapshot si contiene 'model.pt', de lo contrario None.
    """
    hub_cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--emotion2vec--emotion2vec_plus_large/snapshots"
    )
    if os.path.isdir(hub_cache_dir):
        snapshots = [
            os.path.join(hub_cache_dir, s)
            for s in os.listdir(hub_cache_dir)
            if os.path.isdir(os.path.join(hub_cache_dir, s))
        ]
        if snapshots:
            snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for snap in snapshots:
                if os.path.isfile(os.path.join(snap, "model.pt")):
                    return snap
    return None


def load_emotion2vec_model(device: str = "cuda:0") -> Any:
    """
    Carga el modelo emotion2vec/emotion2vec_plus_large usando FunASR.
    Si el modelo ya fue descargado previamente, lo carga 100% offline desde la caché
    local en disco, silenciando advertencias de inicialización.
    Si se ejecuta por primera vez en una máquina nueva (ej. WSL de un compañero),
    descarga automáticamente los pesos desde Hugging Face mostrando el progreso.
    """
    cache_key = f"emotion2vec_plus_large_{device}"
    if cache_key in _E2V_CACHE:
        return _E2V_CACHE[cache_key]

    from funasr import AutoModel

    # Si se pide CUDA pero no está disponible, caer a CPU
    actual_device = device if ("cuda" in device and torch.cuda.is_available()) else "cpu"

    local_snapshot = find_local_emotion2vec_snapshot()

    if local_snapshot:
        # Carga rápida y 100% offline desde el disco local
        with silence_stdio():
            model = AutoModel(
                model=local_snapshot,
                hub="hf",
                device=actual_device,
                disable_update=True
            )
    else:
        # Primera descarga en una máquina nueva (ej. WSL de un compañero)
        print("\n" + "=" * 70)
        print(" [AuraVoice Engine] Primera ejecución detectada en este equipo / WSL.")
        print(" Descargando pesos de emotion2vec/emotion2vec_plus_large desde Hugging Face (~1.2 GB)...")
        print(" Esta descarga se realiza UNA SOLA VEZ y queda cacheada permanentemente en disco.")
        print("=" * 70 + "\n")
        model = AutoModel(
            model="emotion2vec/emotion2vec_plus_large",
            hub="hf",
            device=actual_device,
            disable_update=False
        )
        print("\n[AuraVoice Engine] Descarga completada con éxito. Ejecución lista.\n")

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
    Ejecuta inferencia acústica optimizada con emotion2vec+ Large:
    - Previene OOM en audios de media y larga duración (> 8s) mediante agregación temporal
      por ventaneo: la memoria por ventana no depende de la duración del audio.
    - Para audios cortos (<= 8s), realiza inferencia global directa y crono-análisis.
    - Mapea a las 9 clases oficiales en español y modelo circunflejo de Russell 2D.
    """
    import soundfile as sf
    import tempfile

    is_cuda_run = HAS_TORCH and "cuda" in str(device) and torch.cuda.is_available()
    if is_cuda_run:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    # Contadores medidos de las pasadas del modelo (forward) y de las copias CPU <-> GPU
    model_stats = {"passes": 0, "model_s": 0.0, "h2d_bytes": 0, "d2h_bytes": 0}

    def timed_generate(path: str, num_samples: int):
        """Llama al modelo midiendo solo el tiempo del forward, con sincronización CUDA."""
        _sync(device)
        t0 = time.perf_counter()
        with silence_stdio():
            res = model.generate(path, output_dir=None, granularity="utterance", disable_pbar=True)
        _sync(device)
        model_stats["model_s"] += time.perf_counter() - t0
        model_stats["passes"] += 1
        # Ida: la forma de onda float32. Vuelta: 9 probabilidades + embeddings [T, 1024] float32
        model_stats["h2d_bytes"] += int(num_samples) * 4
        model_stats["d2h_bytes"] += 9 * 4 + cnn_output_frames(num_samples) * 1024 * 4
        return res

    # 1. Metadatos y conversión a WAV 16k mono
    meta_audio = get_audio_metadata(audio_path)
    wav_path = convert_to_wav_16k_mono(audio_path)
    audio_arr, duration = load_audio_array_16k(wav_path)
    sr = 16000
    total_samples = len(audio_arr)

    detailed_list = []
    emotion_scores = {}
    segments = []

    # 2. Inferencia Adaptativa con Control Riguroso de VRAM
    # Si duration > 8.0s, procesar en un único barrido de ventanas para evitar la matriz de atención O(T^2)
    # que causa CUDA OOM en secuencias largas (>3000 frames) en GPUs de 4 GB.
    if duration > 8.0 and total_samples > int(1.0 * sr):
        # Configurar longitud de ventana y paso (respetando la configuración del usuario)
        eff_win_sec = float(window_sec) if window_sec else 3.0
        eff_stride_sec = float(stride_sec) if stride_sec else max(0.5, round(eff_win_sec * 0.67, 1))

        win_samples = int(eff_win_sec * sr)
        stride_samples = int(eff_stride_sec * sr)
        curr = 0
        w_idx = 1
        all_chunk_scores = []
        labels_ref = None

        with torch.inference_mode():
            while curr < total_samples:
                end = min(curr + win_samples, total_samples)
                chunk = audio_arr[curr:end]
                if len(chunk) < int(0.8 * sr) and len(segments) > 0:
                    break
                if len(chunk) < win_samples:
                    chunk = np.pad(chunk, (0, win_samples - len(chunk)))

                fd, tmp_chunk_path = tempfile.mkstemp(suffix="_chunk.wav")
                os.close(fd)
                sf.write(tmp_chunk_path, chunk, sr)

                try:
                    c_res = timed_generate(tmp_chunk_path, len(chunk))
                    c_item = c_res[0] if c_res else {}
                    if labels_ref is None and "labels" in c_item:
                        labels_ref = c_item["labels"]

                    c_scores = c_item.get("scores", [])
                    if c_scores:
                        all_chunk_scores.append(c_scores)
                        top_idx = int(np.argmax(c_scores))
                        top_raw = c_item["labels"][top_idx]
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

        # Agregar probabilidades globales promediando tensores de cada fragmento
        if all_chunk_scores and labels_ref:
            mean_scores = np.mean(all_chunk_scores, axis=0)
            raw_labels = labels_ref
            raw_scores = mean_scores.tolist()
        else:
            raw_labels = list(E2V_LABEL_MAP.keys())[:9]
            raw_scores = [0.11] * 9

    else:
        # Audio corto (<= 8.0s): inferencia global directa
        with torch.inference_mode():
            try:
                raw_res = timed_generate(wav_path, total_samples)
            except Exception as exc:
                if os.path.exists(wav_path) and wav_path != audio_path:
                    try: os.remove(wav_path)
                    except: pass
                raise RuntimeError(f"Fallo en inferencia de emotion2vec: {exc}")

        first_item = raw_res[0] if raw_res and len(raw_res) > 0 else {}
        raw_labels = first_item.get("labels", [])
        raw_scores = first_item.get("scores", [])

        # Timeline y agregación temporal para audios cortos
        short_chunk_scores = []
        labels_ref_short = None
        if duration > 1.5 and total_samples > int(1.0 * sr):
            win_samples = int(window_sec * sr)
            stride_samples = int(stride_sec * sr)
            curr = 0
            w_idx = 1
            with torch.inference_mode():
                while curr < total_samples:
                    end = min(curr + win_samples, total_samples)
                    chunk = audio_arr[curr:end]
                    if len(chunk) < int(0.8 * sr) and len(segments) > 0:
                        break
                    if len(chunk) < win_samples:
                        chunk = np.pad(chunk, (0, win_samples - len(chunk)))

                    fd, tmp_chunk_path = tempfile.mkstemp(suffix="_chunk.wav")
                    os.close(fd)
                    sf.write(tmp_chunk_path, chunk, sr)

                    try:
                        chunk_res = timed_generate(tmp_chunk_path, len(chunk))
                        c_item = chunk_res[0] if chunk_res else {}
                        c_scores = c_item.get("scores", [])
                        if c_scores:
                            short_chunk_scores.append(c_scores)
                            if labels_ref_short is None and "labels" in c_item:
                                labels_ref_short = c_item["labels"]
                            top_idx = int(np.argmax(c_scores))
                            top_raw = c_item.get("labels", [])[top_idx]
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

            # Promediar ventanas si hubo segmentación para evitar el colapso de atención del audio completo
            if short_chunk_scores:
                raw_scores = np.mean(short_chunk_scores, axis=0).tolist()
                if labels_ref_short:
                    raw_labels = labels_ref_short
        else:
            segments.append({
                "index": 1,
                "start_s": 0.0,
                "end_s": duration,
                "duration_s": duration,
                "width_percent": 100.0,
                "emotion": "Neutral",
                "confidence_percent": 100.0,
                "color": EMOTION_COLORS.get("Neutral", "#64748B")
            })

    # 3. Parsear distribución de probabilidades de las 9 clases
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

    # Si audio es muy corto y no hubo segmentos, actualizar con la emoción predominante
    if len(segments) == 1 and segments[0]["emotion"] == "Neutral" and predominant["categoria"] != "Neutral":
        segments[0]["emotion"] = predominant["categoria"]
        segments[0]["confidence_percent"] = predominant["porcentaje"]
        segments[0]["color"] = predominant.get("color", "#64748B")

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

    timeline_result = {
        "status": "EXITOSA",
        "total_duration_s": duration,
        "num_windows": len(segments),
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "segments": segments
    }

    _sync(device)
    t_end = time.perf_counter()
    latency_sec = t_end - t_start

    peak_vram_mb = None
    allocated_vram_mb = None
    if is_cuda_run:
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        allocated_vram_mb = round(torch.cuda.memory_allocated() / (1024 ** 2), 1)

    # Limpieza del WAV temporal
    if os.path.exists(wav_path) and wav_path != audio_path:
        try: os.remove(wav_path)
        except: pass

    model_info = get_model_info(model)
    cuda_telemetry = get_cuda_status() if is_cuda_run else None

    # Consistencia temporal: fracción de ventanas que coinciden con la emoción global
    if segments:
        agree = sum(1 for seg in segments if seg["emotion"] == predominant["categoria"])
        window_agreement_pct = round(agree / len(segments) * 100, 1)
    else:
        window_agreement_pct = None

    model_ms = model_stats["model_s"] * 1000
    total_ms = latency_sec * 1000

    return {
        "status": "EXITOSA",
        "error_message": None,
        "model_id": "emotion2vec/emotion2vec_plus_large",
        "backbone": "emotion2vec+ Large (Data2Vec 2.0)",
        "model_info": model_info,
        "num_params": model_info["params"],
        "num_params_str": model_info["params_str"],
        "num_classes": len(emotion_scores),
        "classes_names": list(emotion_scores.keys()),
        "parsed_result": parsed_result,
        "timeline_result": timeline_result,
        "metrics": {
            "latency_seconds": round(latency_sec, 4),
            "latency_ms": round(total_ms, 2),
            "model_forward_ms": round(model_ms, 2),
            "host_overhead_ms": round(max(0.0, total_ms - model_ms), 2),
            "forward_passes": model_stats["passes"],
            "h2d_bytes": model_stats["h2d_bytes"],
            "d2h_bytes": model_stats["d2h_bytes"],
            "audio_duration_seconds": duration,
            "real_time_factor_rtf": round(latency_sec / max(0.01, duration), 3),
            "speedup_vs_realtime": f"{round(duration / max(0.001, latency_sec), 1)}x más rápido que tiempo real",
            "vram_stats": {"peak_mb": peak_vram_mb, "allocated_mb": allocated_vram_mb},
            "window_agreement_pct": window_agreement_pct,
            "device_used": model_info["device"],
            "precision": model_info["dtype_label"],
            "is_cuda": "cuda" in model_info["device"]
        },
        "audio_metadata": meta_audio,
        "cuda_telemetry": cuda_telemetry
    }
