"""
Módulo de Telemetría y Monitoreo de CUDA y GPU NVIDIA.
Proporciona utilidades para inspeccionar hardware, medir memoria VRAM y verificar
que los tensores y modelos residan realmente en la GPU.
"""

from typing import Dict, Any, Optional
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_cuda_status() -> Dict[str, Any]:
    """
    Obtiene el estado completo del subsistema CUDA y de la GPU disponible.
    
    Retorna un diccionario estructurado con:
      - is_available: bool
      - device_name: str
      - device_count: int
      - current_device: int
      - compute_capability: tuple
      - pytorch_version: str
      - cuda_version: Optional[str]
      - vram_total_mb: float
      - vram_allocated_mb: float
      - vram_reserved_mb: float
      - vram_free_mb: float
    """
    if not HAS_TORCH:
        return {
            "is_available": False,
            "pytorch_version": "No instalado (Modo UI Demo)",
            "cuda_version": None,
            "device_count": 0,
            "device_name": "No detectada (CPU)",
            "compute_capability": None,
            "current_device": None,
            "vram_total_mb": 0.0,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
            "vram_free_mb": 0.0,
        }

    cuda_available = torch.cuda.is_available()
    status: Dict[str, Any] = {
        "is_available": cuda_available,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": "No detectada (CPU)",
        "compute_capability": None,
        "current_device": None,
        "vram_total_mb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "vram_free_mb": 0.0,
    }

    if cuda_available and torch.cuda.device_count() > 0:
        device_idx = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device_idx)
        
        total_bytes = device_props.total_memory
        allocated_bytes = torch.cuda.memory_allocated(device_idx)
        reserved_bytes = torch.cuda.memory_reserved(device_idx)
        free_bytes = total_bytes - reserved_bytes

        status.update({
            "current_device": device_idx,
            "device_name": device_props.name,
            "compute_capability": (device_props.major, device_props.minor),
            "vram_total_mb": round(total_bytes / (1024 ** 2), 2),
            "vram_allocated_mb": round(allocated_bytes / (1024 ** 2), 2),
            "vram_reserved_mb": round(reserved_bytes / (1024 ** 2), 2),
            "vram_free_mb": round(free_bytes / (1024 ** 2), 2),
            "gpu_utilization_percent": 0.0,
            "gpu_temperature_c": 0.0
        })

        # Consulta en tiempo real a nivel de silicio mediante nvidia-smi
        try:
            import subprocess
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits']
            out_smi = subprocess.check_output(cmd, text=True, timeout=1.0).strip().split(',')
            if len(out_smi) >= 2:
                status["gpu_utilization_percent"] = float(out_smi[0].strip())
                status["gpu_temperature_c"] = float(out_smi[1].strip())
        except Exception:
            pass

    return status


def verify_model_device(model_obj: Any) -> Dict[str, Any]:
    """
    Inspecciona los parámetros reales de un modelo PyTorch o instancia de FunASR
    para verificar en qué dispositivo físico residen sus tensores.
    """
    device_found = "desconocido"
    is_on_cuda = False
    param_count = 0

    try:
        # FunASR AutoModel suele almacenar la red subyacente en model_obj.model
        target_model = getattr(model_obj, "model", model_obj)
        if hasattr(target_model, "parameters"):
            params = list(target_model.parameters())
            if params:
                first_param = params[0]
                device_found = str(first_param.device)
                is_on_cuda = first_param.is_cuda
                param_count = sum(p.numel() for p in params)
        elif hasattr(model_obj, "device"):
            device_found = str(model_obj.device)
            is_on_cuda = "cuda" in device_found
    except Exception as exc:
        device_found = f"Error al inspeccionar: {exc}"

    return {
        "device": device_found,
        "is_on_cuda": is_on_cuda,
        "param_count": param_count,
        "verified": is_on_cuda
    }


class CUDAMemoryTracker:
    """
    Context manager y rastreador para medir el consumo de VRAM y tiempo de sincronización
    durante una operación de inferencia en GPU.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.is_cuda = "cuda" in device and HAS_TORCH and torch.cuda.is_available()
        self.start_allocated_mb = 0.0
        self.peak_allocated_mb = 0.0
        self.end_allocated_mb = 0.0
        self.delta_allocated_mb = 0.0
        self.start_event = None
        self.end_event = None
        self.elapsed_ms = 0.0

    def start(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    def stop(self) -> Dict[str, float]:
        if self.is_cuda:
            torch.cuda.synchronize()
            self.end_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            self.peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.delta_allocated_mb = max(0.0, self.end_allocated_mb - self.start_allocated_mb)
        
        return {
            "start_mb": round(self.start_allocated_mb, 2),
            "peak_mb": round(self.peak_allocated_mb, 2),
            "end_mb": round(self.end_allocated_mb, 2),
            "delta_mb": round(self.delta_allocated_mb, 2),
        }
