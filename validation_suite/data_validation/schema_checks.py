import os
import json
import pandas as pd
from typing import Optional, List, Dict, Any


def _default_profile_path(dataset_name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "data_profile", f"{dataset_name}_schema.json")


def _load_expected_columns_from_profile(profile_path: str) -> List[str]:
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = json.load(f)

    cols = prof.get("columns", [])
    names = [c.get("name") for c in cols if isinstance(c, dict) and c.get("name")]
    return names


def schema_check(
    X: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Schema validation:
    - If expected_columns is given: use it.
    - Else if profile_path is given: load expected columns from that JSON.
    - Else if dataset_name is given: load from data/data_profile/{dataset}_schema.json.

    Returns an auditable dict with missing/unexpected columns.
    """

    used_source = None

    if expected_columns is not None:
        exp = list(expected_columns)
        used_source = "expected_columns_arg"

    else:
        if profile_path is None and dataset_name is not None:
            profile_path = _default_profile_path(dataset_name)

        if profile_path is None:
            return {
                "ok": True,
                "skipped": True,
                "reason": "No expected_columns, profile_path, or dataset_name provided",
                "missing_columns": [],
                "unexpected_columns": [],
                "source": None,
            }

        if not os.path.exists(profile_path):
            return {
                "ok": False,
                "skipped": False,
                "reason": f"Profile file not found: {profile_path}",
                "missing_columns": [],
                "unexpected_columns": [],
                "source": profile_path,
            }

        exp = _load_expected_columns_from_profile(profile_path)
        used_source = profile_path

    missing = [c for c in exp if c not in X.columns]
    unexpected = [c for c in X.columns if c not in exp]

    ok = (len(missing) == 0)

    return {
        "ok": ok,
        "skipped": False,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "expected_count": len(exp),
        "observed_count": int(X.shape[1]),
        "source": used_source,
    }
