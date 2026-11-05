"""
Módulo de Procesamiento e Inspección de Archivos de Audio.
Soporta formatos WAV, MP3, M4A, OGG y FLAC, normalizando a 16kHz mono para W2V-BERT Emotion ES.
"""

import os
import subprocess
import tempfile
from typing import Dict, Any, Tuple
import soundfile as sf
import numpy as np


def get_audio_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extrae la tasa de muestreo, duración en segundos, número de canales y formato del archivo.
    """
    metadata: Dict[str, Any] = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "format": os.path.splitext(file_path)[1].lower().replace(".", ""),
        "duration_seconds": 0.0,
        "sample_rate": 0,
        "channels": 1,
        "size_kb": round(os.path.getsize(file_path) / 1024, 2) if os.path.exists(file_path) else 0.0
    }

    try:
        info = sf.info(file_path)
        metadata.update({
            "duration_seconds": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format.lower(),
        })
    except Exception:
        # Fallback usando ffprobe si soundfile no puede abrir directamente formatos como m4a
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=sample_rate,channels",
                "-of", "default=noprint_wrappers=1:nokey=1", file_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if len(lines) >= 3:
                metadata["sample_rate"] = int(lines[0])
                metadata["channels"] = int(lines[1])
                metadata["duration_seconds"] = round(float(lines[2]), 2)
        except Exception as exc:
            metadata["error"] = str(exc)

    return metadata


def convert_to_wav_16k_mono(input_path: str, output_path: str = None) -> str:
    """
    Convierte cualquier archivo de audio soportado a formato WAV 16kHz Mono mediante ffmpeg.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_16k_mono.wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",           # Forzar canal mono
        "-ar", "16000",       # Forzar 16000 Hz requerido por W2V-BERT
        "-c:a", "pcm_s16le",  # PCM 16-bit estándar
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_path


def load_audio_array_16k(audio_path: str) -> Tuple[np.ndarray, float]:
    """
    Carga un archivo de audio como un array NumPy 1D normalizado a 16000 Hz.
    Retorna (audio_array_float32, duracion_en_segundos).
    """
    # 1. Asegurar formato 16kHz mono
    wav_path = convert_to_wav_16k_mono(audio_path)
    
    # 2. Leer señal con soundfile
    audio_data, sr = sf.read(wav_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration = float(len(audio_data)) / float(sr)

    # Eliminar temporal
    if os.path.exists(wav_path) and wav_path != audio_path:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return audio_data, round(duration, 2)
