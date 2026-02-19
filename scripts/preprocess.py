import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FRAUD = os.path.join(ROOT, "data", "raw", "fraudTrain.csv")
RAW_TITANIC = os.path.join(ROOT, "data", "raw", "Titanic-Dataset.csv")

OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def reduce_cardinality(s: pd.Series, top_n=50) -> pd.Series:
    s = s.astype(str).fillna("UNKNOWN")
    vc = s.value_counts(dropna=True)
    keep = set(vc.head(top_n).index)
    return s.where(s.isin(keep), other="OTHER").fillna("UNKNOWN")


def preprocess_fraud():
    df = pd.read_csv(RAW_FRAUD, low_memory=False)

    # Target
    y = df["is_fraud"].astype(int)

    # Parse datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    df["trans_hour"] = df["trans_date_trans_time"].dt.hour.fillna(0).astype(int)
    df["trans_dow"] = df["trans_date_trans_time"].dt.dayofweek.fillna(0).astype(int)
    df["trans_month"] = df["trans_date_trans_time"].dt.month.fillna(1).astype(int)
    df["is_weekend"] = (df["trans_dow"] >= 5).astype(int)

    # Customer age
    age_years = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["customer_age"] = age_years.fillna(age_years.median()).clip(lower=0, upper=120)

    # Geo distance
    df["geo_dist_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]).fillna(0)

    # Amount transforms
    df["log_amt"] = np.log1p(df["amt"].astype(float).clip(lower=0))

    # Make amt_bucket string directly (prevents categorical fill errors)
    df["amt_bucket"] = pd.cut(
        df["amt"].astype(float),
        bins=[-1, 10, 50, 200, 1000, 1e12],
        labels=["<=10", "10-50", "50-200", "200-1000", ">1000"]
    ).astype(str).replace("nan", "UNKNOWN")

    # Drop leakage/PII/identifiers
    drop_cols = [
        "is_fraud",
        "Unnamed: 0",
        "trans_num", "cc_num",
        "first", "last", "street", "zip",
        "trans_date_trans_time", "dob",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Reduce high-cardinality categorical columns
    for col in ["merchant", "job", "city"]:
        if col in X.columns:
            X[col] = reduce_cardinality(X[col], top_n=50)

    # Final missing fills (robust)
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            # For object/categorical-like columns
            X[c] = X[c].astype(str).replace("nan", "UNKNOWN").fillna("UNKNOWN")

    X_path = os.path.join(OUT_DIR, "fraud_features.parquet")
    y_path = os.path.join(OUT_DIR, "fraud_labels.parquet")

    X.to_parquet(X_path, index=False)
    y.to_frame("is_fraud").to_parquet(y_path, index=False)

    print(f"[OK] Fraud processed -> {X_path}, {y_path} | X shape={X.shape}, y shape={y.shape}")


def preprocess_titanic():
    df = pd.read_csv(RAW_TITANIC)

    y = df["Survived"].astype(int)

    # Cabin handling
    df["has_cabin"] = df["Cabin"].notna().astype(int)
    df["cabin_prefix"] = df["Cabin"].astype(str).str[0]
    df.loc[df["Cabin"].isna(), "cabin_prefix"] = "U"

    # Title extraction
    df["title"] = df["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False)
    df["title"] = df["title"].fillna("Unknown").astype(str)

    # Embarked fill
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Age imputation (Sex x Pclass)
    df["Age"] = df["Age"].astype(float)
    grp_med = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(grp_med).fillna(df["Age"].median())

    # Family features
    df["FamilySize"] = df["SibSp"].astype(int) + df["Parch"].astype(int) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    drop_cols = ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].astype(str).replace("nan", "UNKNOWN").fillna("UNKNOWN")

    X_path = os.path.join(OUT_DIR, "titanic_features.parquet")
    y_path = os.path.join(OUT_DIR, "titanic_labels.parquet")

    X.to_parquet(X_path, index=False)
    y.to_frame("Survived").to_parquet(y_path, index=False)

    print(f"[OK] Titanic processed -> {X_path}, {y_path} | X shape={X.shape}, y shape={y.shape}")


if __name__ == "__main__":
    preprocess_fraud()
    preprocess_titanic()
