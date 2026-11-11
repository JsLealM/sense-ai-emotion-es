"""
Módulo de Parseo e Interpretación de Resultados Emocionales.
Modela y traduce las 9 categorías oficiales de emotion2vec/emotion2vec_plus_large:
Alegría, Tristeza, Enojo, Miedo, Disgusto, Neutral, Sorpresa, Otro y Neutro / Desc.
Incorpora el Modelo Circunflejo de Russell (Valence & Arousal) y descriptores latentes.
"""

from typing import Dict, Any, List, Tuple
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

EMOTION_TRANSLATION = {
    # Clases oficiales de emotion2vec+ (9 clases)
    "angry": "Enojo",
    "disgusted": "Disgusto",
    "fearful": "Miedo",
    "happy": "Alegría",
    "neutral": "Neutral",
    "other": "Otro",
    "sad": "Tristeza",
    "surprised": "Sorpresa",
    "unknown": "Neutro / Desc.",
    # Variantes estándar
    "anger": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "joy": "Alegría",
    "sadness": "Tristeza",
    "alegría": "Alegría",
    "tristeza": "Tristeza",
    "enojo": "Enojo",
    "sorpresa": "Sorpresa",
}

# Paleta AuraVoice armonizada
EMOTION_COLORS = {
    "Alegría": "#EA580C",       # Naranja cálido / Terracota
    "Tristeza": "#0D9488",      # Teal profundo / Calma
    "Enojo": "#F43F5E",         # Carmesí
    "Miedo": "#E11D48",         # Rosa oscuro
    "Disgusto": "#D97706",      # Ámbar
    "Neutral": "#574A43",       # Carbón espresso / Neutro cálido
    "Sorpresa": "#8B5CF6",      # Púrpura eléctrico / Asombro
    "Otro": "#64748B",          # Pizarra
    "Neutro / Desc.": "#475569" # Gris azulado
}

# Coordenadas circunflejas canónicas de Russell (Valence: -1.0 a +1.0, Arousal: -1.0 a +1.0)
CIRCUMPLEX_COORDS = {
    "Alegría": (0.82, 0.65),
    "Neutral": (0.15, 0.10),
    "Tristeza": (-0.75, -0.45),
    "Enojo": (-0.70, 0.82),
    "Miedo": (-0.65, 0.70),
    "Disgusto": (-0.60, 0.35),
    "Sorpresa": (0.65, 0.85),
    "Otro": (0.00, 0.20),
    "Neutro / Desc.": (0.00, 0.00)
}

ETHICAL_DISCLAIMER = (
    "Nota ética importante: El modelo analiza características acústicas y prosódicas de la voz "
    "(frecuencia fundamental, energía, formantes). Esto constituye una clasificación estadística de patrones sonoros, "
    "no un diagnóstico psicológico del estado mental de una persona."
)


def compute_circumplex_state(scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Calcula el vector de Valencia (Valence) y Activación (Arousal) mediante
    la combinación lineal ponderada de las probabilidades predichas por el modelo.
    Retorna el descriptor compuesto del estado emocional latente.
    """
    total_w = sum(scores.values()) or 1.0
    val_sum = 0.0
    aro_sum = 0.0

    for emo, p in scores.items():
        v, a = CIRCUMPLEX_COORDS.get(emo, (0.0, 0.0))
        val_sum += (p / total_w) * v
        aro_sum += (p / total_w) * a

    valence = round(val_sum, 2)
    arousal = round(max(0.05, (aro_sum + 1.0) / 2.0), 2)  # Escala 0.0 a 1.0

    # Determinar título latente compuesto y etiqueta semántica
    max_emo = max(scores.items(), key=lambda x: x[1])[0] if scores else "Neutral"
    max_prob = scores.get(max_emo, 0.0)

    # Si hay una emoción dominante con alta confianza (>40%), respetar su identidad primaria
    if max_emo == "Neutral" and max_prob >= 0.4:
        title = "Neutral / Reposo Acústico"
        tag = "● Neutral"
        tag_color = "#574A43"
    elif max_emo == "Sorpresa" and max_prob >= 0.4:
        title = "Sorpresa / Asombro Intenso"
        tag = "● Asombrado"
        tag_color = "#8B5CF6"
    elif max_emo == "Tristeza" and max_prob >= 0.4:
        title = "Tristeza / Introspección"
        tag = "● Decaído"
        tag_color = "#0D9488"
    elif max_emo == "Enojo" and max_prob >= 0.4:
        title = "Enojo / Tensión Elevada"
        tag = "● Agitado"
        tag_color = "#F43F5E"
    elif max_emo == "Alegría" and max_prob >= 0.4:
        title = "Alegría / Entusiasmo"
        tag = "● Expresivo"
        tag_color = "#EA580C"
    elif max_emo == "Miedo" and max_prob >= 0.4:
        title = "Miedo / Alerta Tensa"
        tag = "● Temeroso"
        tag_color = "#E11D48"
    elif max_emo == "Disgusto" and max_prob >= 0.4:
        title = "Disgusto / Rechazo Acústico"
        tag = "● Disgustado"
        tag_color = "#D97706"
    elif valence >= 0.3:
        if arousal >= 0.5:
            title = "Alegría / Entusiasmo"
            tag = "● Expresivo"
            tag_color = "#EA580C"
        else:
            title = "Alegría / Calma Positiva"
            tag = "● Positivo"
            tag_color = "#EA580C"
    elif valence <= -0.3:
        if arousal >= 0.5:
            title = "Enojo / Tensión Elevada"
            tag = "● Agitado"
            tag_color = "#F43F5E"
        else:
            title = "Tristeza / Introspección"
            tag = "● Decaído"
            tag_color = "#0D9488"
    else:
        if arousal >= 0.65:
            title = "Alerta / Atención Activa"
            tag = "● Atento"
            tag_color = "#EA580C"
        else:
            title = "Neutral / Reposo Acústico"
            tag = "● Neutral"
            tag_color = "#574A43"

    val_sign = f"+{valence}" if valence >= 0 else f"{valence}"

    return {
        "valence": valence,
        "arousal": arousal,
        "valence_str": val_sign,
        "title": title,
        "tag": tag,
        "tag_color": tag_color
    }


def parse_emotion_probabilities(
    logits: Any,
    id2label: Dict[int, str]
) -> Dict[str, Any]:
    """
    Convierte los logits brutos del modelo en una distribución de probabilidad softmax
    y estructura los resultados en las categorías en español, vector circunflejo
    y métricas de incidencia de AuraVoice.
    """
    if HAS_TORCH and isinstance(logits, torch.Tensor):
        if logits.ndim > 1:
            logits = logits[0]
        probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    else:
        probs = np.array(logits, dtype=np.float32)
        if probs.ndim > 1:
            probs = probs[0]
        exp_p = np.exp(probs - np.max(probs))
        probs = exp_p / np.sum(exp_p)

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
            "color": EMOTION_COLORS.get(es_label, "#EA580C")
        })

    # Ordenar de mayor a menor probabilidad
    detailed_list.sort(key=lambda x: x["probabilidad"], reverse=True)
    predominant = detailed_list[0]

    # Calcular vector circunflejo Valence/Arousal
    circumplex = compute_circumplex_state(emotion_scores)

    # Desglose canónico de las 9 clases de emotion2vec
    class_mapping = {
        "Alegría": "Alegría",
        "Neutral": "Neutral",
        "Tristeza": "Tristeza",
        "Enojo": "Enojo",
        "Miedo": "Miedo",
        "Disgusto": "Disgusto",
        "Sorpresa": "Sorpresa",
        "Otro": "Otro",
        "Neutro / Desc.": "Neutro / Desc."
    }

    formatted_breakdown = []
    for i, item in enumerate(detailed_list):
        formatted_name = class_mapping.get(item["categoria"], item["categoria"])
        # Usar el color canónico de la categoría emocional
        item_color = EMOTION_COLORS.get(item["categoria"], "#574A43")
        formatted_breakdown.append({
            "name": formatted_name,
            "raw_cat": item["categoria"],
            "percent": item["porcentaje"],
            "prob": item["probabilidad"],
            "color": item_color
        })

    return {
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
