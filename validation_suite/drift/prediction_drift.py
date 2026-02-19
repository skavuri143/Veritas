import numpy as np

def prediction_drift(proba_ref: np.ndarray, proba_cur: np.ndarray, bins=10) -> dict:
    ref = np.asarray(proba_ref)
    cur = np.asarray(proba_cur)
    hist_ref, edges = np.histogram(ref, bins=bins, range=(0,1), density=True)
    hist_cur, _ = np.histogram(cur, bins=edges, density=True)
    diff = float(np.mean(np.abs(hist_ref - hist_cur)))
    return {"hist_l1_diff": diff}
