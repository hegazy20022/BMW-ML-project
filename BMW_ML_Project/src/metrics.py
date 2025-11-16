from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
    }

def format_metrics(prefix: str, m: Dict[str, float]) -> str:
    return (f"{prefix} -> acc={m['accuracy']:.4f} "
            f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")
