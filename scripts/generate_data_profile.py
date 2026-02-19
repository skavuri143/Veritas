import os
import json
import argparse
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(ROOT, "data", "data_profile")
os.makedirs(PROFILE_DIR, exist_ok=True)


def _safe_json(obj):
    """Convert pandas/numpy types to JSON-safe python types."""
    if obj is None:
        return None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    try:
        # numpy scalars
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    return obj


def profile_dataframe(df: pd.DataFrame, dataset_name: str) -> dict:
    n_rows, n_cols = df.shape

    col_profiles = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        missing_ratio = float(missing / max(n_rows, 1))

        entry = {
            "name": col,
            "dtype": dtype,
            "missing_count": missing,
            "missing_ratio": missing_ratio,
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            entry.update({
                "min": _safe_json(s_num.min(skipna=True)),
                "max": _safe_json(s_num.max(skipna=True)),
                "mean": _safe_json(s_num.mean(skipna=True)),
                "std": _safe_json(s_num.std(skipna=True)),
                "p50": _safe_json(s_num.quantile(0.50)),
                "p95": _safe_json(s_num.quantile(0.95)),
            })

        # Datetime sample
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry.update({
                "min": _safe_json(s.min(skipna=True)),
                "max": _safe_json(s.max(skipna=True)),
            })

        # Categorical/text stats
        else:
            # limit unique count computation if huge
            nunique = int(s.nunique(dropna=True))
            entry["nunique"] = nunique

            # Store top categories sample (bounded)
            top_vals = (
                s.astype(str)
                 .replace("nan", pd.NA)
                 .dropna()
                 .value_counts()
                 .head(20)
            )
            entry["top_values"] = [{"value": v, "count": int(c)} for v, c in top_vals.items()]

        col_profiles.append(entry)

    schema = {
        "dataset": dataset_name,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "shape": {"rows": n_rows, "cols": n_cols},
        "columns": col_profiles,
    }
    return schema


def save_profile(dataset_name: str, X_path: str, y_path: str):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)

    # Merge labels into profile so target is captured too
    df = X.copy()
    for col in y.columns:
        df[col] = y[col]

    schema = profile_dataframe(df, dataset_name)

    out_path = os.path.join(PROFILE_DIR, f"{dataset_name}_schema.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote schema profile: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fraud", "titanic", "all"])
    args = ap.parse_args()

    fraud_X = os.path.join(ROOT, "data", "processed", "fraud_features.parquet")
    fraud_y = os.path.join(ROOT, "data", "processed", "fraud_labels.parquet")
    titanic_X = os.path.join(ROOT, "data", "processed", "titanic_features.parquet")
    titanic_y = os.path.join(ROOT, "data", "processed", "titanic_labels.parquet")

    if args.dataset in ("fraud", "all"):
        save_profile("fraud", fraud_X, fraud_y)

    if args.dataset in ("titanic", "all"):
        save_profile("titanic", titanic_X, titanic_y)


if __name__ == "__main__":
    main()
