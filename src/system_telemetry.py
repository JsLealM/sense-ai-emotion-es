"""
Módulo de Telemetría de Sistema y Generador de Logs para AuraTerminal.

Todo valor que aparece en la consola sale de una medición real:
  - psutil (CPU, RAM, memoria del proceso),
  - torch.cuda (dispositivo, VRAM asignada y pico),
  - nvidia-smi (utilización y temperatura de la GPU en el momento de la consulta),
  - las métricas que devuelve run_emotion2vec_inference (latencias, pasadas, bytes copiados),
  - las medidas acústicas de audio_telemetry (RMS, pico, F0 con pYIN, % de tramas con voz).
Cuando un dato no está disponible se muestra "N/D" en lugar de inventarlo.
"""

import os
import datetime
from typing import Dict, Any, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.cuda_monitor import get_cuda_status


ND = "N/D"


def _fmt(value: Any, suffix: str = "") -> str:
    """Formatea un valor medido o devuelve N/D si no existe."""
    if value is None or value == "---":
        return ND
    return f"{value}{suffix}"


def _fmt_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return ND
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.2f} MiB"
    return f"{num_bytes / 1024:.1f} KiB"


def get_system_cpu_usage() -> Dict[str, Any]:
    """Métricas reales de CPU y RAM. Sin psutil, los campos no medibles quedan en None."""
    info: Dict[str, Any] = {
        "cpu_percent": None,
        "cpu_count": os.cpu_count(),
        "cpu_physical": None,
        "ram_total_gb": None,
        "ram_used_gb": None,
        "ram_percent": None,
        "process_rss_gb": None,
        "source": "os",
    }
    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            info.update({
                "cpu_percent": round(float(psutil.cpu_percent(interval=0.1)), 1),
                "cpu_count": psutil.cpu_count(logical=True),
                "cpu_physical": psutil.cpu_count(logical=False),
                "ram_total_gb": round(mem.total / (1024 ** 3), 1),
                "ram_used_gb": round(mem.used / (1024 ** 3), 1),
                "ram_percent": round(mem.percent, 1),
                "process_rss_gb": round(psutil.Process().memory_info().rss / (1024 ** 3), 2),
                "source": "psutil",
            })
        except Exception:
            pass
    return info


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(tag: str, kind: str, text: str) -> Dict[str, str]:
    return {"time": _now(), "tag": tag, "type": kind, "text": text}


def _filter(logs: List[Dict[str, str]], filter_tag: str) -> List[Dict[str, str]]:
    if filter_tag == "ALL":
        return logs
    if filter_tag in ("INFO", "WARN"):
        return [l for l in logs if l["type"] in (filter_tag, "INFO")]
    if filter_tag == "CUDA":
        return [l for l in logs if "CUDA" in l["tag"]]
    if filter_tag == "CPU":
        return [l for l in logs if "CPU" in l["tag"] or "RAM" in l["tag"]]
    return logs


def generate_terminal_logs(
    filter_tag: str = "ALL",
    inference_meta: Optional[Dict[str, Any]] = None,
    current_device: str = "cuda:0",
    model_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Construye la consola de telemetría de hardware con datos medidos.
    inference_meta es el diccionario 'metrics' de la última inferencia (más 'predominant'/'confidence').
    """
    cpu = get_system_cpu_usage()
    is_cuda = "cuda" in current_device.lower() and HAS_TORCH and torch.cuda.is_available()
    gpu = get_cuda_status() if is_cuda else {}
    logs: List[Dict[str, str]] = []

    if is_cuda:
        cc = gpu.get("compute_capability")
        logs.append(_log("CUDA:INIT", "CUDA",
            f"{gpu.get('device_name')} [ID {gpu.get('current_device')}] | CUDA {gpu.get('cuda_version')} | "
            f"Compute capability {cc[0]}.{cc[1] if cc else ''} | VRAM total {gpu.get('vram_total_mb')} MiB"
            if cc else f"{gpu.get('device_name')} | CUDA {gpu.get('cuda_version')}"))
        logs.append(_log("CUDA:STREAM", "CUDA",
            f"VRAM asignada por PyTorch: {gpu.get('vram_allocated_mb')} MiB | reservada: {gpu.get('vram_reserved_mb')} MiB | "
            f"nvidia-smi ahora: {_fmt(gpu.get('gpu_utilization_percent'), '%')} uso, {_fmt(gpu.get('gpu_temperature_c'), ' °C')}"))
    else:
        logs.append(_log("CPU:INIT", "CPU",
            f"Ejecución en CPU: {_fmt(cpu['cpu_physical'])} núcleos físicos / {_fmt(cpu['cpu_count'])} hilos lógicos | "
            f"uso actual {_fmt(cpu['cpu_percent'], '%')}"))
        logs.append(_log("RAM:STATUS", "CPU",
            f"RAM del sistema: {_fmt(cpu['ram_used_gb'])} / {_fmt(cpu['ram_total_gb'])} GB | proceso: {_fmt(cpu['process_rss_gb'], ' GB')}"))

    if model_info:
        logs.append(_log("MODEL:INFO", "INFO",
            f"emotion2vec+ Large en {model_info['device']} | {model_info['params_str']} contados con sum(p.numel()) | "
            f"dtype {model_info['dtype']} | {model_info['transformer_blocks']} bloques Transformer "
            f"({model_info['prenet_blocks']} prenet + {model_info['main_blocks']}) | pesos {model_info['weights_mb']} MiB"))
    else:
        logs.append(_log("MODEL:INFO", "INFO", "Modelo aún no cargado: se carga en la primera inferencia (@st.cache_resource)."))

    if not inference_meta:
        logs.append(_log("PIPELINE", "INFO", "Ejecuta una inferencia para ver las mediciones de esa corrida."))
        return _filter(logs, filter_tag)

    m = inference_meta
    passes = m.get("forward_passes")
    vram = m.get("vram_stats") or {}
    logs.append(_log("CUDA:HtoD" if is_cuda else "CPU:INPUT", "CUDA" if is_cuda else "CPU",
        f"Entrada al modelo: {passes} pasada(s), {_fmt_bytes(m.get('h2d_bytes'))} de forma de onda float32"
        + (" copiados CPU → GPU (wav.to(device))" if is_cuda else "")))
    logs.append(_log("CUDA:KERNEL" if is_cuda else "CPU:FORWARD", "CUDA" if is_cuda else "CPU",
        f"Forward del modelo: {m.get('model_forward_ms')} ms en total"
        + (" (medido entre torch.cuda.synchronize())" if is_cuda else "")))
    logs.append(_log("CPU:HOST", "CPU",
        f"Trabajo en CPU fuera del modelo (ffmpeg, ventaneo, E/S de WAV temporales, post-proceso): {m.get('host_overhead_ms')} ms"))
    logs.append(_log("CUDA:DtoH" if is_cuda else "CPU:OUTPUT", "CUDA" if is_cuda else "CPU",
        f"Salida: {_fmt_bytes(m.get('d2h_bytes'))} (9 probabilidades + embeddings [T, 1024] por pasada)"
        + (" copiados GPU → CPU (.tolist() / .cpu())" if is_cuda else "")))
    if is_cuda:
        logs.append(_log("CUDA:PEAK", "CUDA",
            f"VRAM pico durante esta inferencia: {_fmt(vram.get('peak_mb'), ' MiB')} | asignada al terminar: {_fmt(vram.get('allocated_mb'), ' MiB')}"))
    logs.append(_log("MODEL:E2V", "INFO",
        f"Latencia total {m.get('latency_ms')} ms en {m.get('device_used')} ({m.get('precision')}) | "
        f"RTF {m.get('real_time_factor_rtf')} | Veredicto: {m.get('predominant')} ({m.get('confidence')})"))
    return _filter(logs, filter_tag)


def _energy_label(rms_db: Any) -> str:
    """Clasificación por umbrales fijos del nivel RMS medido (dBFS). Es una regla nuestra, no del modelo."""
    if not isinstance(rms_db, (int, float)):
        return ND
    if rms_db > -20:
        return "nivel alto"
    if rms_db > -35:
        return "nivel medio"
    return "nivel bajo"


def generate_decision_reasoning_logs(
    parsed_result: Optional[Dict[str, Any]] = None,
    inference_meta: Optional[Dict[str, Any]] = None,
    audio_telemetry_meta: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    timeline_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Trazabilidad de la decisión con datos reales: medidas de la señal, configuración de ventanas,
    salida softmax del modelo, margen entre clases, consistencia entre ventanas y proyección de Russell.
    Nota: describe entradas y salidas; no es interpretabilidad interna del Transformer.
    """
    if not parsed_result or not inference_meta:
        return [
            _log("XAI:INIT", "INFO", "Trazabilidad en espera de una inferencia."),
            _log("XAI:PIPELINE", "INFO",
                 "Cadena: audio 16 kHz mono → ventanas → emotion2vec+ (CNN + Transformer) → promedio → softmax 9 clases → Russell 2D."),
            _log("XAI:NOTE", "INFO",
                 "Las medidas acústicas (RMS, F0) se calculan aparte para describir la señal; el modelo recibe la forma de onda cruda."),
        ]

    a = audio_telemetry_meta or {}
    m = inference_meta
    pred = parsed_result.get("predominant_emotion", ND)
    conf = parsed_result.get("predominant_percent", ND)
    circ = parsed_result.get("circumplex", {})
    breakdown = parsed_result.get("formatted_breakdown", [])

    logs = [
        _log("XAI:INGEST", "INFO",
             f"Señal: {_fmt(a.get('num_samples'))} muestras a 16 kHz mono | RMS {_fmt(a.get('rms_db'), ' dBFS')} "
             f"({_energy_label(a.get('rms_db'))}) | pico {_fmt(a.get('peak_db'), ' dBFS')}"),
        _log("XAI:PROSODY", "INFO",
             f"pYIN: {_fmt(a.get('voiced_ratio_pct'), '%')} de tramas con voz | F0 mediano {_fmt(a.get('f0_hz'), ' Hz')} | "
             f"variación de F0 {_fmt(a.get('f0_cv_pct'), '%')}"),
    ]
    if timeline_result:
        logs.append(_log("XAI:CHUNK", "INFO",
             f"Ventaneo: {timeline_result.get('num_windows')} ventana(s) de {timeline_result.get('window_sec')} s, "
             f"paso {timeline_result.get('stride_sec')} s | {m.get('forward_passes')} pasada(s) del modelo"))
    if model_info:
        logs.append(_log("XAI:DATA2VEC", "INFO",
             f"Modelo: CNN 7 capas + {model_info['transformer_blocks']} bloques Transformer, {model_info['params_str']}, "
             f"{model_info['dtype']} en {model_info['device']} | forward {m.get('model_forward_ms')} ms"))
    logs.append(_log("XAI:SOFTMAX", "INFO", f"Softmax (promedio de ventanas): '{pred}' con {conf}%"))
    if len(breakdown) >= 2:
        t1, t2 = breakdown[0], breakdown[1]
        logs.append(_log("XAI:MARGIN", "INFO",
             f"Margen sobre la segunda clase: {round(t1['percent'] - t2['percent'], 1)} puntos ('{t2['name']}' con {t2['percent']}%)"
             + (" → predicción incierta (< 50%)" if t1["percent"] < 50 else "")))
    if m.get("window_agreement_pct") is not None:
        logs.append(_log("XAI:TEMPORAL", "INFO",
             f"Consistencia: {m.get('window_agreement_pct')}% de las ventanas coinciden con '{pred}'"))
    logs.append(_log("XAI:RUSSELL_2D", "INFO",
         f"Russell 2D (promedio ponderado de coordenadas fijas por clase): valencia {circ.get('valence_str', ND)}, "
         f"activación {circ.get('arousal', ND)} → '{circ.get('title', ND)}'"))
    logs.append(_log("XAI:VEREDICTO", "INFO",
         f"Resultado: '{pred}' ({conf}%). Es una clasificación estadística de la voz, no un diagnóstico."))
    return logs
