import os
import json
import pandas as pd
from typing import Optional, Dict, Any


def _default_profile_path(dataset_name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "data_profile", f"{dataset_name}_schema.json")


def _load_missingness_baseline(profile_path: str) -> Dict[str, float]:
    """
    If your schema profile contains missing_ratio per column, use it as baseline reference.
    Returns dict: {col_name: missing_ratio}
    """
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = json.load(f)

    baseline = {}
    for c in prof.get("columns", []):
        if isinstance(c, dict) and c.get("name") is not None:
            mr = c.get("missing_ratio", None)
            if mr is not None:
                baseline[str(c["name"])] = float(mr)
    return baseline


def missingness_check(
    X: pd.DataFrame,
    max_missing_ratio: float = 0.30,
    topk: int = 10,
    dataset_name: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Missingness check:
    - flags columns whose missing ratio exceeds max_missing_ratio
    - (optional) also reports delta vs stored baseline profile missingness

    Works without profile; profile is only used to compute deltas for reporting.
    """
    ratios = X.isna().mean().astype(float).to_dict()

    # top-k worst missingness
    worst = sorted(ratios.items(), key=lambda x: -x[1])[:topk]
    fail_cols = {k: float(v) for k, v in ratios.items() if v > max_missing_ratio}

    baseline = None
    deltas = None
    used_profile = None

    if profile_path is None and dataset_name is not None:
        profile_path = _default_profile_path(dataset_name)

    if profile_path is not None and os.path.exists(profile_path):
        try:
            baseline = _load_missingness_baseline(profile_path)
            deltas = {}
            for col, cur in ratios.items():
                if col in baseline:
                    deltas[col] = float(cur - baseline[col])
            used_profile = profile_path
        except Exception:
            baseline = None
            deltas = None
            used_profile = profile_path

    overall_missing = float(X.isna().sum().sum() / max(X.size, 1))

    return {
        "ok": len(fail_cols) == 0,
        "max_missing_ratio_allowed": float(max_missing_ratio),
        "overall_missing_ratio": overall_missing,
        "top_missingness": [(k, float(v)) for k, v in worst],
        "fail_columns": fail_cols,
        "baseline_profile": used_profile,
        "missingness_delta_vs_baseline": deltas, 
    }
