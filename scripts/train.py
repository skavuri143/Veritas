import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(os.path.join(MODELS, "baselines"), exist_ok=True)
os.makedirs(os.path.join(MODELS, "champion"), exist_ok=True)
os.makedirs(os.path.join(MODELS, "metadata"), exist_ok=True)


def build_preprocessor(X: pd.DataFrame, make_dense: bool):
    """
    If make_dense=True, encoder outputs dense arrays so models like HistGradientBoosting can train.
    If make_dense=False, encoder outputs sparse matrices (more memory efficient).
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=not make_dense  
        )),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0 if make_dense else 0.3  
    )
    return pre


def eval_model(m, X_test, y_test, threshold=0.5):
    proba = m.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    pr_auc = average_precision_score(y_test, proba)
    return {"auc": float(auc), "f1": float(f1), "pr_auc": float(pr_auc)}


def train_one(dataset_name: str, X_path: str, y_path: str, target_col: str):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline preprocessor (sparse OK)
    pre_sparse = build_preprocessor(X_train, make_dense=False)
    baseline = Pipeline(steps=[
        ("pre", pre_sparse),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Champion preprocessor (dense required for HistGradientBoosting)
    pre_dense = build_preprocessor(X_train, make_dense=True)
    champion = Pipeline(steps=[
        ("pre", pre_dense),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=300,
            random_state=42
        ))
    ])

    # Fit
    baseline.fit(X_train, y_train)
    champion.fit(X_train, y_train)

    # Evaluate
    baseline_metrics = eval_model(baseline, X_test, y_test)
    champion_metrics = eval_model(champion, X_test, y_test)

    # Save models
    joblib.dump(baseline, os.path.join(MODELS, "baselines", f"{dataset_name}_logreg.pkl"))
    joblib.dump(champion, os.path.join(MODELS, "champion", f"{dataset_name}_model.pkl"))

    # Model card metadata
    card = {
        "dataset": dataset_name,
        "target": target_col,
        "rows": int(len(X)),
        "features": list(X.columns),
        "baseline_metrics": baseline_metrics,
        "champion_metrics": champion_metrics,
        "threshold": 0.5,
        "seed": 42
    }
    with open(os.path.join(MODELS, "metadata", f"{dataset_name}_model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"[OK] Trained {dataset_name}")
    print("  baseline:", baseline_metrics)
    print("  champion:", champion_metrics)


if __name__ == "__main__":
    train_one(
        dataset_name="fraud",
        X_path=os.path.join(PROC, "fraud_features.parquet"),
        y_path=os.path.join(PROC, "fraud_labels.parquet"),
        target_col="is_fraud"
    )

    train_one(
        dataset_name="titanic",
        X_path=os.path.join(PROC, "titanic_features.parquet"),
        y_path=os.path.join(PROC, "titanic_labels.parquet"),
        target_col="Survived"
    )
