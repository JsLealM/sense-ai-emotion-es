"""
Módulo de Parseo e Interpretación de Resultados Emocionales.
Modela y traduce las 6 categorías de UMUTeam/w2v-bert-emotion-es:
Alegría, Tristeza, Enojo, Miedo, Disgusto y Neutral.
Incorpora el Modelo Circunflejo de Russell (Valence & Arousal) y descriptores latentes.
"""

from typing import Dict, Any, List, Tuple
import torch

EMOTION_TRANSLATION = {
    "anger": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "joy": "Alegría",
    "neutral": "Neutral",
    "sadness": "Tristeza",
}

# Paleta AuraVoice armonizada
EMOTION_COLORS = {
    "Alegría": "#EA580C",    # Naranja cálido / Terracota
    "Tristeza": "#0D9488",   # Teal profundo / Calma
    "Enojo": "#F43F5E",      # Carmesí
    "Miedo": "#E11D48",      # Rosa oscuro
    "Disgusto": "#D97706",   # Ámbar
    "Neutral": "#574A43",    # Carbón espresso / Neutro cálido
}

# Coordenadas circunflejas canónicas de Russell (Valence: -1.0 a +1.0, Arousal: -1.0 a +1.0)
CIRCUMPLEX_COORDS = {
    "Alegría": (0.82, 0.65),
    "Neutral": (0.15, 0.10),
    "Tristeza": (-0.75, -0.45),
    "Enojo": (-0.70, 0.82),
    "Miedo": (-0.65, 0.70),
    "Disgusto": (-0.60, 0.35)
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
    if valence >= 0.3:
        if arousal >= 0.5:
            title = "Entusiasmo / Expresivo"
            tag = "● Dinámico"
            tag_color = "#EA580C"
        else:
            title = "Calma Positiva / Receptivo"
            tag = "● Receptivo"
            tag_color = "#EA580C"
    elif valence <= -0.3:
        if arousal >= 0.5:
            title = "Tensión / Alerta Elevada"
            tag = "● Agitado"
            tag_color = "#F43F5E"
        else:
            title = "Atenuación / Introspección"
            tag = "● Reservado"
            tag_color = "#0D9488"
    else:
        if arousal >= 0.5:
            title = "Atención Activa / Vigilante"
            tag = "● Atento"
            tag_color = "#EA580C"
        else:
            title = "Equilibrio Neutral / Neutro"
            tag = "● Receptivo"
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
    logits: torch.Tensor,
    id2label: Dict[int, str]
) -> Dict[str, Any]:
    """
    Convierte los logits brutos del modelo en una distribución de probabilidad softmax
    y estructura los resultados en las 6 categorías en español, vector circunflejo
    y métricas de incidencia de AuraVoice.
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
            "color": EMOTION_COLORS.get(es_label, "#EA580C")
        })

    # Ordenar de mayor a menor probabilidad
    detailed_list.sort(key=lambda x: x["probabilidad"], reverse=True)
    predominant = detailed_list[0]

    # Calcular vector circunflejo Valence/Arousal
    circumplex = compute_circumplex_state(emotion_scores)

    # Desglose de Top clases con formato refinado
    class_mapping = {
        "Alegría": "Entusiasmo / Alegría",
        "Neutral": "Neutral / Pausado",
        "Tristeza": "Calma / Serenidad",
        "Enojo": "Tensión / Ansiedad",
        "Miedo": "Incertidumbre / Alerta",
        "Disgusto": "Frustración / Tensión"
    }

    formatted_breakdown = []
    palette_shades = ["#EA580C", "#F43F5E", "#E15A75", "#D97706", "#8F7D74", "#574A43"]
    for i, item in enumerate(detailed_list):
        formatted_name = class_mapping.get(item["categoria"], item["categoria"])
        formatted_breakdown.append({
            "name": formatted_name,
            "raw_cat": item["categoria"],
            "percent": item["porcentaje"],
            "prob": item["probabilidad"],
            "color": palette_shades[min(i, len(palette_shades)-1)]
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
