import numpy as np

def brier_score(y_true, proba) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    return float(np.mean((proba - y_true) ** 2))

def calibration_summary(y_true, proba) -> dict:
    return {"brier_score": brier_score(y_true, proba)}
