import os
import json
import pandas as pd
from typing import Optional, List, Dict, Any


def _default_profile_path(dataset_name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "data_profile", f"{dataset_name}_schema.json")


def _load_forbidden_from_profile(profile_path: str) -> List[str]:
    """
    Optional: if you add a 'forbidden_columns' list to your schema JSON later,
    this will auto-load it.
    """
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    fc = prof.get("forbidden_columns", [])
    return [str(x) for x in fc] if isinstance(fc, list) else []


def leakage_check(
    X: pd.DataFrame,
    forbidden_columns: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Leakage check:
    - flags if any forbidden columns exist in the feature matrix X.

    Sources for forbidden columns:
    1) forbidden_columns argument (highest priority)
    2) profile file (if contains forbidden_columns)
    3) otherwise: check is skipped (returns ok=True, skipped=True)
    """
    used_source = None

    if forbidden_columns is not None:
        forbidden = list(forbidden_columns)
        used_source = "forbidden_columns_arg"
    else:
        # try load from profile
        if profile_path is None and dataset_name is not None:
            profile_path = _default_profile_path(dataset_name)

        if profile_path is not None and os.path.exists(profile_path):
            forbidden = _load_forbidden_from_profile(profile_path)
            used_source = profile_path
        else:
            return {
                "ok": True,
                "skipped": True,
                "reason": "No forbidden_columns provided and no profile forbidden_columns available",
                "forbidden_present": [],
                "source": None,
            }

        if not forbidden:
            return {
                "ok": True,
                "skipped": True,
                "reason": "Profile exists but contains no forbidden_columns list",
                "forbidden_present": [],
                "source": used_source,
            }

    present = [c for c in forbidden if c in X.columns]

    return {
        "ok": len(present) == 0,
        "skipped": False,
        "forbidden_present": present,
        "forbidden_expected": forbidden,
        "source": used_source,
    }
