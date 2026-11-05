"""
Módulo de Parseo e Interpretación de Resultados Emocionales.
Modela y traduce las 6 categorías de UMUTeam/w2v-bert-emotion-es:
Alegría, Tristeza, Enojo, Miedo, Disgusto y Neutral.
"""

from typing import Dict, Any, List
import torch

EMOTION_TRANSLATION = {
    "anger": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "joy": "Alegría",
    "neutral": "Neutral",
    "sadness": "Tristeza",
}

EMOTION_COLORS = {
    "Alegría": "#EAB308",    # Amarillo / Dorado
    "Tristeza": "#3B82F6",   # Azul
    "Enojo": "#EF4444",      # Rojo
    "Miedo": "#8B5CF6",      # Púrpura
    "Disgusto": "#10B981",   # Verde Esmeralda
    "Neutral": "#6B7280",    # Gris
}

EMOTION_EMOJIS = {
    "Alegría": "",
    "Tristeza": "",
    "Enojo": "",
    "Miedo": "",
    "Disgusto": "",
    "Neutral": "",
}

ETHICAL_DISCLAIMER = (
    "Nota ética importante: El modelo analiza características acústicas y prosódicas de la voz "
    "(frecuencia fundamental, energía, formantes). Esto constituye una clasificación estadística de patrones sonoros, "
    "no un diagnóstico psicológico del estado mental de una persona."
)


def parse_emotion_probabilities(
    logits: torch.Tensor,
    id2label: Dict[int, str]
) -> Dict[str, Any]:
    """
    Convierte los logits brutos del modelo en una distribución de probabilidad softmax
    y estructura los resultados en las 6 categorías en español.
    """
    if logits.ndim > 1:
        logits = logits[0]

    # Aplicar Softmax para obtener probabilidades que sumen 1.0 (100%)
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()

    emotion_scores: Dict[str, float] = {}
    detailed_list: List[Dict[str, Any]] = []

    for idx, prob in enumerate(probs):
        raw_label = id2label.get(idx, f"label_{idx}")
        es_label = EMOTION_TRANSLATION.get(raw_label.lower(), raw_label.capitalize())
        prob_percent = round(float(prob) * 100, 2)
        score_val = round(float(prob), 4)

        emotion_scores[es_label] = score_val
        detailed_list.append({
            "id": idx,
            "raw_label": raw_label,
            "categoria": es_label,
            "probabilidad": score_val,
            "porcentaje": prob_percent,
            "color": EMOTION_COLORS.get(es_label, "#3B82F6")
        })

    # Ordenar de mayor a menor probabilidad
    detailed_list.sort(key=lambda x: x["probabilidad"], reverse=True)
    predominant = detailed_list[0]

    return {
        "predominant_emotion": predominant["categoria"],
        "predominant_raw": predominant["raw_label"],
        "predominant_score": predominant["probabilidad"],
        "predominant_percent": predominant["porcentaje"],
        "predominant_emoji": "",
        "scores_dict": emotion_scores,
        "ranking": detailed_list,
        "disclaimer": ETHICAL_DISCLAIMER
    }
