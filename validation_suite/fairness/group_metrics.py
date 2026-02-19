import numpy as np
import pandas as pd

def group_confusion_metrics(df: pd.DataFrame, group_col: str) -> dict:
    """
    df must contain columns: _y, _pred and group_col
    """
    out = {}
    for g, sub in df.groupby(df[group_col].astype(str)):
        y = sub["_y"].to_numpy()
        p = sub["_pred"].to_numpy()

        tp = ((p == 1) & (y == 1)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        tn = ((p == 0) & (y == 0)).sum()

        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        sr = float(p.mean()) if len(p) else 0.0

        out[g] = {"tpr": tpr, "fpr": fpr, "selection_rate": sr, "count": int(len(sub))}
    return out

def gap(metrics_by_group: dict, key: str) -> float:
    vals = [v[key] for v in metrics_by_group.values()]
    if not vals:
        return 0.0
    return float(max(vals) - min(vals))
