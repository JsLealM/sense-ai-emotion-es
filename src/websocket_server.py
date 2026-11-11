"""
Servidor WebSocket Asíncrono para Inferencia Emocional Continua en Tiempo Real.

Recibe fragmentos de audio PCM Float32 a 16 kHz mono desde el navegador, los acumula
en un búfer circular acotado y, cada STRIDE_SEC segundos de audio nuevo, ejecuta
emotion2vec+ Large sobre la última ventana de WINDOW_SEC segundos en la GPU.

Modelo de concurrencia:
  - El bucle de eventos asyncio atiende la red (recepción de chunks, envío de JSON).
  - La inferencia (bloqueante, GPU) se delega a un hilo con asyncio.to_thread para
    no congelar el bucle mientras la GPU trabaja.
  - Un asyncio.Lock serializa el acceso al modelo: una sola GPU y un solo modelo
    compartido entre todos los clientes conectados.
  - Si llega audio mientras una inferencia está en curso, no se encolan inferencias
    atrasadas: se procesa siempre la ventana más reciente (backpressure por descarte).
  - Un hilo aparte mantiene abierto `nvidia-smi -lms 100` y guarda una muestra de uso,
    temperatura, potencia y memoria de la GPU cada 100 ms. Para cada ventana se reporta
    el uso pico y medio de la GPU durante su inferencia.

Uso:
    python src/websocket_server.py              # ws://0.0.0.0:8765 en cuda:0
    python src/websocket_server.py --port 9000 --device cpu
"""

import argparse
import atexit
import asyncio
import json
import logging
import os
import signal
import sys
import subprocess
import tempfile
import threading
import time
from collections import deque

import numpy as np
import soundfile as sf
import torch
import websockets

# Asegurar importación de módulos del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.emotion2vec_engine import load_emotion2vec_model, run_emotion2vec_inference, get_model_info  # noqa: E402

# El registro raíz queda en WARNING para que FunASR no imprima cada capa al cargar el modelo;
# solo el logger propio del servidor muestra mensajes INFO.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("auravoice.ws")
log.setLevel(logging.INFO)
# Las comprobaciones de salud TCP desde Streamlit no completan el handshake WebSocket;
# se silencian para que no ensucien la consola.
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

SAMPLE_RATE = 16000
WINDOW_SEC = 3.0            # Ventana de análisis enviada al modelo
STRIDE_SEC = 1.0            # Audio nuevo necesario entre dos inferencias
MAX_BUFFER_SEC = 5.0        # Tope del búfer circular (evita crecimiento de RAM)
SILENCE_RMS = 0.004         # Umbral de energía por debajo del cual se considera silencio

WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
STRIDE_SAMPLES = int(STRIDE_SEC * SAMPLE_RATE)
MAX_BUFFER_SAMPLES = int(MAX_BUFFER_SEC * SAMPLE_RATE)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class GpuSampler:
    """
    Lee nvidia-smi en modo continuo (una muestra cada SAMPLE_MS) en un hilo daemon.
    utilization.gpu es el % de tiempo en que al menos un kernel CUDA se ejecutó durante
    el último periodo de muestreo del driver; no es el % de núcleos ocupados. En GPUs de
    portátil el driver muestrea con poca resolución (en la RTX 3050 Laptop reporta 0 % o
    100 %), así que puede no ver ráfagas de ~100 ms. Por eso el servidor envía además
    'duty_pct': tiempo de forward medido con torch.cuda.synchronize() / 1 s de audio.
    """
    SAMPLE_MS = 100
    QUERY = "utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total"

    def __init__(self):
        self.samples = deque(maxlen=600)   # ~60 s de historia
        self.lock = threading.Lock()
        self.available = False
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits",
                 "-lms", str(self.SAMPLE_MS), "-i", "0"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self.available = True
            threading.Thread(target=self._reader, daemon=True).start()
        except (FileNotFoundError, OSError):
            self.proc = None

    @staticmethod
    def _num(txt):
        try:
            return float(txt.strip())
        except ValueError:
            return None  # p. ej. "[N/A]" en potencia de algunas laptops

    def _reader(self):
        for line in self.proc.stdout:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            util, temp, power, mem, mem_tot = (self._num(p) for p in parts[:5])
            with self.lock:
                self.samples.append((time.monotonic(), util, temp, power, mem, mem_tot))
        self.available = False

    def latest(self):
        with self.lock:
            s = self.samples[-1] if self.samples else None
        if not s:
            return None
        return {"util": s[1], "temp_c": s[2], "power_w": s[3], "mem_used_mib": s[4], "mem_total_mib": s[5]}

    def stats_between(self, t0: float, t1: float):
        """Uso pico y medio entre t0 y t1. Se extiende 300 ms porque el driver promedia hacia atrás."""
        with self.lock:
            win = [s for s in self.samples if t0 <= s[0] <= t1 + 0.3 and s[1] is not None]
        if not win:
            return None
        utils = [s[1] for s in win]
        return {"util_peak": max(utils), "util_mean": round(sum(utils) / len(utils), 1), "n_samples": len(win)}

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


GPU_SAMPLER = None


def gpu_snapshot(t0=None, t1=None) -> dict:
    """Datos de nvidia-smi para enviar al navegador (None si no hay GPU NVIDIA)."""
    if GPU_SAMPLER is None or not GPU_SAMPLER.available:
        return None
    snap = GPU_SAMPLER.latest() or {}
    if t0 is not None:
        snap.update(GPU_SAMPLER.stats_between(t0, t1) or {})
    return snap
MODEL = None
MODEL_LOCK = asyncio.Lock()


def get_server_model():
    """Carga (una sola vez) el modelo emotion2vec+ Large en el dispositivo configurado."""
    global MODEL
    if MODEL is None:
        log.info(f"Cargando emotion2vec+ Large en {DEVICE}...")
        MODEL = load_emotion2vec_model(device=DEVICE)
        log.info(f"Modelo cargado en {DEVICE}.")
    return MODEL


def infer_window(model, window: np.ndarray) -> dict:
    """
    Ejecuta la inferencia sobre una ventana de audio (bloqueante; se llama desde un hilo).
    Reutiliza run_emotion2vec_inference, que espera una ruta de archivo, mediante un WAV temporal.
    """
    fd, tmp_path = tempfile.mkstemp(suffix="_stream.wav")
    os.close(fd)
    try:
        sf.write(tmp_path, window, SAMPLE_RATE, subtype="FLOAT")
        return run_emotion2vec_inference(model, tmp_path, device=DEVICE)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def gpu_memory_mb() -> float:
    if "cuda" in DEVICE and torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
    return 0.0


def build_payload(result: dict, window_index: int, rms: float, queue_ms: float, gpu: dict = None) -> dict:
    parsed = result["parsed_result"]
    metrics = result["metrics"]
    circ = parsed.get("circumplex", {})
    return {
        "status": "OK",
        "window": window_index,
        "predominant": parsed["predominant_emotion"],
        "confidence": parsed["predominant_percent"],
        "color": next((b["color"] for b in parsed["formatted_breakdown"]
                       if b["name"] == parsed["predominant_emotion"]), "#574A43"),
        "valence": circ.get("valence", 0.0),
        "arousal": circ.get("arousal", 0.0),
        "circumplex_title": circ.get("title", ""),
        "breakdown": [{"name": b["name"], "percent": b["percent"], "color": b["color"]}
                      for b in parsed["formatted_breakdown"]],
        "latency_ms": metrics["latency_ms"],
        "queue_ms": round(queue_ms, 1),
        "rms": round(rms, 4),
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if "cuda" in DEVICE else "CPU",
        "vram_mb": gpu_memory_mb(),
        "vram_peak_mb": metrics["vram_stats"].get("peak_mb"),
        "model_forward_ms": metrics["model_forward_ms"],
        "precision": metrics["precision"],
        # Fracción de cada segundo de audio que la GPU pasa ejecutando el modelo
        "duty_pct": round(metrics["model_forward_ms"] / (STRIDE_SEC * 1000) * 100, 1),
        "gpu": gpu,
        "timestamp": time.strftime("%H:%M:%S"),
    }


async def handler(websocket):
    client = websocket.remote_address
    log.info(f"Cliente conectado: {client}")

    model = get_server_model()
    buffer = np.zeros(0, dtype=np.float32)
    new_samples = 0          # Muestras recibidas desde la última inferencia
    window_index = 0

    info = get_model_info(model)
    await websocket.send(json.dumps({
        "status": "READY", "device": info["device"],
        "window_sec": WINDOW_SEC, "stride_sec": STRIDE_SEC,
        "params_str": info["params_str"], "dtype": info["dtype_label"],
    }))

    try:
        async for message in websocket:
            # ---------------- Comandos de control (texto) ----------------
            if isinstance(message, str):
                if message == "PING":
                    await websocket.send(json.dumps({"status": "PONG", "device": DEVICE}))
                elif message == "RESET":
                    buffer = np.zeros(0, dtype=np.float32)
                    new_samples = 0
                    await websocket.send(json.dumps({"status": "RESET_OK"}))
                continue

            # ---------------- Audio (binario, Float32 16 kHz mono) ----------------
            if len(message) % 4 != 0:
                continue  # Trama corrupta: no es múltiplo de 4 bytes (float32)
            chunk = np.frombuffer(message, dtype=np.float32)
            if chunk.size == 0:
                continue

            buffer = np.concatenate((buffer, chunk))[-MAX_BUFFER_SAMPLES:]
            new_samples += chunk.size

            if len(buffer) < WINDOW_SAMPLES or new_samples < STRIDE_SAMPLES:
                continue
            if MODEL_LOCK.locked():
                continue  # La GPU está ocupada: se analizará la ventana más reciente después

            new_samples = 0
            window_index += 1
            window = buffer[-WINDOW_SAMPLES:].copy()
            rms = float(np.sqrt(np.mean(np.square(window))))

            if not np.isfinite(rms) or rms < SILENCE_RMS:
                await websocket.send(json.dumps({
                    "status": "SILENCE", "window": window_index, "rms": round(rms, 5),
                    "timestamp": time.strftime("%H:%M:%S"), "gpu": gpu_snapshot(),
                }))
                continue

            t_queue = time.perf_counter()
            async with MODEL_LOCK:
                queue_ms = (time.perf_counter() - t_queue) * 1000
                try:
                    t_inf0 = time.monotonic()
                    result = await asyncio.to_thread(infer_window, model, window)
                    t_inf1 = time.monotonic()
                except Exception as exc:  # Nunca tumbar la conexión por un fallo de inferencia
                    log.error(f"Fallo de inferencia (ventana {window_index}): {exc}")
                    await websocket.send(json.dumps({"status": "ERROR", "message": str(exc)}))
                    continue

            if result.get("status") == "EXITOSA":
                if GPU_SAMPLER is not None and GPU_SAMPLER.available:
                    await asyncio.sleep(0.3)  # deja llegar las muestras de nvidia-smi de esta inferencia
                payload = build_payload(result, window_index, rms, queue_ms, gpu_snapshot(t_inf0, t_inf1))
                g = payload.get("gpu") or {}
                log.info(f"[{client[0]}] ventana {window_index}: {payload['predominant']} "
                         f"{payload['confidence']}% | {payload['latency_ms']} ms | {DEVICE} | "
                         f"GPU pico {g.get('util_peak', 'N/D')}% media {g.get('util_mean', 'N/D')}%")
                await websocket.send(json.dumps(payload))
            else:
                await websocket.send(json.dumps({
                    "status": "ERROR", "message": result.get("error_message", "Inferencia fallida"),
                }))
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as exc:
        log.error(f"Error no fatal con {client}: {exc}")
    finally:
        log.info(f"Cliente desconectado: {client} (búfer liberado)")


async def main(host: str, port: int):
    log.info(f"=== SERVIDOR WEBSOCKET AURAVOICE en ws://{host}:{port} | dispositivo {DEVICE} ===")
    global GPU_SAMPLER
    get_server_model()  # Precarga antes de aceptar conexiones
    async with websockets.serve(handler, host, port, max_size=2 * 1024 * 1024):
        if "cuda" in DEVICE:  # Solo tras abrir el puerto, para no dejar nvidia-smi huérfano si falla
            GPU_SAMPLER = GpuSampler()
            atexit.register(GPU_SAMPLER.stop)
            log.info(f"Muestreo de GPU con nvidia-smi cada {GpuSampler.SAMPLE_MS} ms: "
                     f"{'activo' if GPU_SAMPLER.available else 'no disponible'}")
        log.info("Servidor listo. Esperando clientes... (Ctrl+C para detener)")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor WebSocket de streaming emocional AuraVoice")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default=None, help="cuda:0 o cpu (por defecto: cuda:0 si está disponible)")
    args = parser.parse_args()
    if args.device:
        DEVICE = args.device if ("cuda" not in args.device or torch.cuda.is_available()) else "cpu"

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt  # mismo cierre ordenado que Ctrl+C (detiene nvidia-smi)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        log.info("Servidor detenido por el usuario.")
    finally:
        if GPU_SAMPLER is not None:
            GPU_SAMPLER.stop()
