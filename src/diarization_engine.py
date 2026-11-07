"""
Módulo de Diarización y Seguimiento de Transición Emocional Multi-Hablante.
Gestiona el estado temporal de los hablantes (ej. Alex y Dra. Silva),
su cuota de tiempo de habla y la línea de tiempo de transiciones emocionales.
Diseñado para la integración con modelos futuros de VAD en tiempo real y diarización.
"""

from typing import Dict, Any, List


# Paleta de colores semánticos AuraVoice para emociones en la línea temporal
DIARIZATION_COLORS = {
    "alegria": "#EA580C",      # Terracota / Naranja cálido
    "calma": "#0D9488",        # Verde Azulado / Teal
    "tension": "#F43F5E",      # Carmesí / Rosa
    "neutral": "#EBE3DC",      # Arena / Marfil neutro
    "silencio": "#F5EFEB"      # Fondo suave
}


def get_default_diarization_state() -> Dict[str, Any]:
    """
    Retorna el estado de diarización y transición emocional de los últimos 120s,
    preparado tanto para visualización estática como para actualización en streaming.
    """
    return {
        "window_duration_seconds": 120,
        "speakers": [
            {
                "id": "spk_1",
                "name": "Hablante 1: Alex",
                "active": True,
                "spoken_time_s": 68,
                "spoken_percent": 57,
                "predominant_tone": "Entusiasta",
                "segments": [
                    {"label": "Alegría / Entusiasmo", "duration_pct": 28, "color": DIARIZATION_COLORS["alegria"]},
                    {"label": "Calma / Serenidad", "duration_pct": 18, "color": DIARIZATION_COLORS["calma"]},
                    {"label": "Neutral / Pausa", "duration_pct": 12, "color": DIARIZATION_COLORS["neutral"]},
                    {"label": "Tensión", "duration_pct": 8, "color": DIARIZATION_COLORS["tension"]},
                    {"label": "Alegría / Entusiasmo", "duration_pct": 34, "color": DIARIZATION_COLORS["alegria"]},
                ]
            },
            {
                "id": "spk_2",
                "name": "Hablante 2: Dra. Silva",
                "active": False,
                "spoken_time_s": 44,
                "spoken_percent": 36,
                "predominant_tone": "Calma & Empatía",
                "segments": [
                    {"label": "Silencio / Escucha", "duration_pct": 22, "color": DIARIZATION_COLORS["neutral"]},
                    {"label": "Calma / Serenidad", "duration_pct": 30, "color": DIARIZATION_COLORS["calma"]},
                    {"label": "Pausa", "duration_pct": 16, "color": DIARIZATION_COLORS["neutral"]},
                    {"label": "Calma / Serenidad", "duration_pct": 32, "color": DIARIZATION_COLORS["calma"]},
                ]
            }
        ],
        "legend": [
            {"label": "Alegría / Entusiasmo", "color": DIARIZATION_COLORS["alegria"]},
            {"label": "Calma / Serenidad", "color": DIARIZATION_COLORS["calma"]},
            {"label": "Tensión", "color": DIARIZATION_COLORS["tension"]},
            {"label": "Neutral / Silencio", "color": DIARIZATION_COLORS["neutral"]}
        ],
        "time_ticks": ["-02:00", "-01:30", "-01:00", "-00:30", "AHORA (00:00)"]
    }


def update_diarization_from_inference(
    state: Dict[str, Any],
    speaker_id: str,
    detected_emotion: str,
    duration_s: float
) -> Dict[str, Any]:
    """
    Hook de extensión preparado para las futuras actualizaciones en tiempo real:
    agrega un segmento clasificado por la ventana deslizante al historial de diarización.
    """
    color_map = {
        "Alegría": DIARIZATION_COLORS["alegria"],
        "Calma": DIARIZATION_COLORS["calma"],
        "Enojo": DIARIZATION_COLORS["tension"],
        "Tensión": DIARIZATION_COLORS["tension"],
        "Miedo": DIARIZATION_COLORS["tension"],
        "Tristeza": DIARIZATION_COLORS["calma"],
        "Neutral": DIARIZATION_COLORS["neutral"]
    }
    target_color = color_map.get(detected_emotion, DIARIZATION_COLORS["alegria"])

    for spk in state["speakers"]:
        if spk["id"] == speaker_id:
            spk["segments"].append({
                "label": detected_emotion,
                "duration_pct": max(5, int(duration_s)),
                "color": target_color
            })
            # Mantener la suma de segmentos dentro de la ventana de 120s
            if len(spk["segments"]) > 8:
                spk["segments"].pop(0)
    return state

