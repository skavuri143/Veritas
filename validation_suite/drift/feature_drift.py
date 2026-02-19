import numpy as np
import pandas as pd

def psi(expected: np.ndarray, actual: np.ndarray, bins=10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)
    e = e_hist / max(e_hist.sum(), 1)
    a = a_hist / max(a_hist.sum(), 1)
    eps = 1e-6
    e = np.clip(e, eps, 1)
    a = np.clip(a, eps, 1)
    return float(np.sum((a - e) * np.log(a / e)))

def feature_drift_report(X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> dict:
    num_cols = [c for c in X_ref.columns if pd.api.types.is_numeric_dtype(X_ref[c])]
    psi_scores = {}
    for c in num_cols:
        psi_scores[c] = psi(X_ref[c].to_numpy(float), X_cur[c].to_numpy(float), bins=10)
    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    return {"psi_by_feature": psi_scores, "max_psi": float(max_psi)}
