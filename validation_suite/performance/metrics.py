import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_pred = (proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred))
    }
