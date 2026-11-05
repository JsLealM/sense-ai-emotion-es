"""
Módulo de Evaluación Experimental y Métricas para Reconocimiento de Emociones.
Calcula Matriz de Confusión, Accuracy, Precision, Recall y F1-score comparando
las predicciones del modelo contra etiquetas reales (Ground Truth) de referencia.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ALL_EMOTIONS = ["Alegría", "Tristeza", "Enojo", "Miedo", "Disgusto", "Neutral"]


def compute_evaluation_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calcula métricas estándar de clasificación supervisada.
    
    IMPORTANTE: y_true DEBEN ser etiquetas reales anotadas previamente por humanos o
    provenientes de un dataset etiquetado independiente, NUNCA las predicciones del modelo.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"error": "No hay datos para evaluar."}

    if len(y_true) != len(y_pred):
        raise ValueError(f"Longitudes dispares: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    if labels is None:
        labels = ALL_EMOTIONS

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc * 100, 2),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_mac, 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "classification_report": report
    }
