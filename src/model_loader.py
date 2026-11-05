"""
Módulo de Carga y Definición del Modelo W2V-BERT Emotion ES (UMUTeam).
Implementa la arquitectura oficial CustomAudioClassification de UMUTeam,
garantizando la carga completa y fidedigna de los 580M parámetros entrenados
en GPU CUDA (con optimización FP16 para 4GB VRAM) o CPU.
"""

from typing import Tuple, Any, Dict, Optional, Union
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
    Wav2Vec2BertPreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from src.cuda_monitor import verify_model_device, get_cuda_status

_MODEL_CACHE: Dict[str, Any] = {}
DEFAULT_MODEL_ID = "UMUTeam/w2v-bert-emotion-es"


class ClassificationHead(nn.Module):
    """Cabezal de clasificación oficial de UMUTeam para reconocimiento de emociones."""

    def __init__(self, config, num_extra_dims: int = 0):
        super().__init__()
        total_dims = getattr(config, "classifier_proj_size", 768)
        self.dense = nn.Linear(total_dims, total_dims)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(total_dims, config.num_labels)
        self.layernorm = nn.LayerNorm(total_dims)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomAudioClassification(Wav2Vec2BertPreTrainedModel):
    """
    Arquitectura oficial de UMUTeam para W2V-BERT Emotion ES.
    Integra el backbone acústico Wav2Vec2BertModel, proyector lineal y el cabezal ClassificationHead.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.classifier = ClassificationHead(config)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


def load_emotion_model(
    device_name: str = "cuda:0",
    use_fp16: bool = True,
    model_id: str = DEFAULT_MODEL_ID
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Carga el modelo UMUTeam/w2v-bert-emotion-es y su extractor de características.
    Utiliza CustomAudioClassification para garantizar coincidencia del 100% de los pesos de UMUTeam.
    """
    is_cuda_requested = "cuda" in device_name.lower()
    if is_cuda_requested and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó inferencia en CUDA, pero torch.cuda.is_available() es False.")

    cache_key = f"{model_id}_{device_name}_{use_fp16}"
    if cache_key in _MODEL_CACHE:
        model, feature_extractor = _MODEL_CACHE[cache_key]
        verification = verify_model_device(model)
        return model, feature_extractor, {
            "cached": True,
            "device": device_name,
            "dtype": str(next(model.parameters()).dtype),
            "verification": verification,
            "model_id": model_id,
        }

    # 1. Cargar extractor de características (80-channel log-mel, 16kHz)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # 2. Cargar configuración oficial
    config = AutoConfig.from_pretrained(model_id)

    # 3. Instanciar la arquitectura CustomAudioClassification
    model = CustomAudioClassification(config)

    # 4. Cargar los pesos exactos desde model.safetensors (compatible con Linux, WSL y Windows)
    weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    state_dict = load_file(weights_path)
    load_result = model.load_state_dict(state_dict, strict=True)
    print(f"[ModelLoader] Pesos de UMUTeam cargados con éxito (strict=True): {load_result}")

    # 5. Configurar precisión y dispositivo
    device_target = torch.device(device_name if is_cuda_requested else "cpu")
    if is_cuda_requested and use_fp16:
        model = model.half()  # Convertir a float16 para optimizar VRAM en 4GB

    model.to(device_target)
    model.eval()

    # Guardar en caché
    _MODEL_CACHE[cache_key] = (model, feature_extractor)

    verification = verify_model_device(model)

    metadata = {
        "cached": False,
        "device": device_name,
        "dtype": str(next(model.parameters()).dtype),
        "verification": verification,
        "model_id": model_id,
        "labels": config.id2label,
        "num_labels": config.num_labels,
        "cuda_status": get_cuda_status() if is_cuda_requested else None
    }

    return model, feature_extractor, metadata
