"""
Módulo de Telemetría Acústica y Extracción de Formas de Onda (Waveform & Spectrum).
Calcula la envolvente visual de amplitudes (barras de ondas sonoras),
nivel RMS en dB, pico máximo dB, F0 (pYIN) y porcentaje de tramas con voz para AuraVoice.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np


def extract_waveform_bars(
    audio_array: Optional[np.ndarray] = None,
    num_bars: int = 50,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calcula una serie de alturas normalizadas (entre 0.08 y 1.0) para las barras
    de la onda de audio visual, junto con métricas de RMS y pico máximo.
    """
    if audio_array is not None and len(audio_array) > 0:
        # Calcular RMS y Peak dB reales
        abs_audio = np.abs(audio_array)
        peak_val = float(np.max(abs_audio)) if len(abs_audio) > 0 else 0.001
        rms_val = float(np.sqrt(np.mean(audio_array**2))) if len(audio_array) > 0 else 0.001

        # Conversión a dBFS
        peak_db = round(20 * math.log10(max(peak_val, 1e-5)), 1)
        rms_db = round(20 * math.log10(max(rms_val, 1e-5)), 1)

        # Muestreo de ventanas para las barras
        chunk_size = max(1, len(audio_array) // num_bars)
        bars = []
        for i in range(num_bars):
            start = i * chunk_size
            end = min(len(audio_array), start + chunk_size)
            if start < end:
                chunk_peak = float(np.max(np.abs(audio_array[start:end])))
                # Escalar de forma logarítmica/perceptual
                bar_height = min(1.0, max(0.10, chunk_peak ** 0.65))
            else:
                bar_height = 0.15
            bars.append(round(bar_height, 3))

        # Medidas reales de la señal (no son entradas del modelo; el modelo recibe la forma de onda cruda)
        acoustic = measure_voice_features(audio_array)
        f0_hz = acoustic["f0_median_hz"] if acoustic["f0_median_hz"] is not None else "---"
        f0_var = acoustic["f0_cv_pct"]
        jitter_str = f"Variación F0: {f0_var}%" if f0_var is not None else "Sin voz detectada"
        vad_active = f"{acoustic['voiced_ratio_pct']}%"
    else:
        # En espera (sin audio cargado todavía)
        bars = [0.08] * num_bars
        rms_db = "---"
        peak_db = "---"
        f0_hz = "---"
        jitter_str = "En espera"
        vad_active = "---"
        acoustic = {"f0_median_hz": None, "f0_cv_pct": None, "voiced_ratio_pct": None, "num_samples": 0}

    return {
        "bars": bars,
        "rms_db": rms_db,
        "peak_db": peak_db,
        "f0_hz": f0_hz,
        "jitter": jitter_str,
        "vad_active_pct": vad_active,
        "f0_cv_pct": acoustic["f0_cv_pct"],
        "voiced_ratio_pct": acoustic["voiced_ratio_pct"],
        "num_samples": acoustic["num_samples"],
    }


_FEATURE_CACHE: Dict[str, Dict[str, Any]] = {}


def measure_voice_features(audio_array: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Mide sobre la señal real con pYIN (librosa), tramas de 20 ms:
      - voiced_ratio_pct: % de tramas que pYIN marca como voz (sonoras).
      - f0_median_hz: tono fundamental mediano en las tramas con voz.
      - f0_cv_pct: coeficiente de variación del F0 (desviación / media), indicador de entonación.
    El resultado se cachea por contenido porque Streamlit vuelve a ejecutar el script en cada clic.
    """
    import hashlib
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
    n = len(audio_array)
    key = hashlib.md5(audio_array.tobytes()).hexdigest()
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]

    result = {"f0_median_hz": None, "f0_cv_pct": None, "voiced_ratio_pct": 0.0, "num_samples": n}
    if n >= 2048:
        try:
            import librosa
            f0, voiced_flag, _ = librosa.pyin(
                audio_array, fmin=65.0, fmax=500.0, sr=sr, frame_length=1024, hop_length=int(0.02 * sr)
            )
            result["voiced_ratio_pct"] = round(float(np.mean(voiced_flag)) * 100, 1)
            f0_voiced = f0[voiced_flag & np.isfinite(f0)]
            if len(f0_voiced) >= 5:
                mean = float(np.mean(f0_voiced))
                result["f0_median_hz"] = int(round(float(np.median(f0_voiced))))
                result["f0_cv_pct"] = round(float(np.std(f0_voiced)) / max(mean, 1e-6) * 100, 1)
        except Exception:
            pass

    if len(_FEATURE_CACHE) > 16:
        _FEATURE_CACHE.clear()
    _FEATURE_CACHE[key] = result
    return result
