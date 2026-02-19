import os
import json
import argparse
import joblib
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

import shap
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG_DIR = os.path.join(ROOT, "validation_suite", "config")
REPORTS_LATEST = os.path.join(ROOT, "reports", "latest")
REPORTS_HISTORY = os.path.join(ROOT, "reports", "history")
os.makedirs(REPORTS_LATEST, exist_ok=True)
os.makedirs(REPORTS_HISTORY, exist_ok=True)


# ------------------------- Helpers -------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for numeric arrays."""
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


def topk_overlap(list_a, list_b, k: int = 10) -> float:
    a = set(list_a[:k])
    b = set(list_b[:k])
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def explainability_reference_path(dataset_name: str) -> str:
    return os.path.join(ROOT, "models", "metadata", f"{dataset_name}_explainability_reference.json")


def load_explainability_reference(dataset_name: str):
    path = explainability_reference_path(dataset_name)
    if not os.path.exists(path):
        return None
    try:
        return json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return None


def save_explainability_reference(dataset_name: str, top_features: list):
    path = explainability_reference_path(dataset_name)
    payload = {
        "dataset": dataset_name,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "top_features": top_features[:25]
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ------------------------- Core checks -------------------------

def performance_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def drift_check(X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> dict:
    """Compute PSI for numeric features only."""
    numeric_cols = [c for c in X_ref.columns if pd.api.types.is_numeric_dtype(X_ref[c])]
    psi_scores = {}
    for c in numeric_cols:
        psi_scores[c] = psi(
            X_ref[c].to_numpy(dtype=float),
            X_cur[c].to_numpy(dtype=float),
            bins=10
        )
    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    return {"psi_by_feature": psi_scores, "max_psi": float(max_psi)}


def fairness_check(X: pd.DataFrame, y_true: pd.Series, proba: np.ndarray, group_col: str, threshold: float = 0.5) -> dict:
    """
    Fairness checks:
    - Selection rate by group (for audit)
    - TPR/FPR by group
    - TPR gap and FPR gap (governance-friendly)
    """
    df = X.copy()
    df["_y"] = y_true.values
    df["_p"] = proba
    df["_pred"] = (df["_p"] >= threshold).astype(int)

    if group_col not in df.columns:
        return {"error": f"group column '{group_col}' not found"}

    groups = sorted(df[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        return {"warning": "not enough groups to compute fairness", "groups": groups}

    selection = {}
    tpr = {}
    fpr = {}

    for g in groups:
        sub = df[df[group_col].astype(str) == g]
        if len(sub) == 0:
            selection[g] = 0.0
            tpr[g] = 0.0
            fpr[g] = 0.0
            continue

        y = sub["_y"].to_numpy()
        p = sub["_pred"].to_numpy()

        selection[g] = float(p.mean())

        tp = ((p == 1) & (y == 1)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        tn = ((p == 0) & (y == 0)).sum()

        tpr[g] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr[g] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    max_sr = max(selection.values())
    min_sr = min(selection.values())
    disparity_ratio = float(min_sr / max_sr) if max_sr > 0 else 1.0

    tpr_gap = float(max(tpr.values()) - min(tpr.values()))
    fpr_gap = float(max(fpr.values()) - min(fpr.values()))

    return {
        "group_col": group_col,
        "groups": groups,
        "selection_rate_by_group": selection,
        "disparity_ratio_min_over_max": disparity_ratio,
        "tpr_by_group": tpr,
        "fpr_by_group": fpr,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap
    }


def shap_explainability(model, X_sample: pd.DataFrame) -> dict:
    """
    SHAP feature importance (mean |shap|) for sklearn Pipeline(pre+clf).
    Uses check_additivity=False where supported to avoid TreeExplainer additivity failures.
    """
    pre = model.named_steps.get("pre", None)
    clf = model.named_steps.get("clf", None)
    if pre is None or clf is None:
        return {"error": "model is not a Pipeline(pre+clf)"}

    X_trans = pre.transform(X_sample)

    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    try:
        explainer = shap.Explainer(clf, X_trans)
        try:
            shap_values = explainer(X_trans, check_additivity=False)
        except TypeError:
            shap_values = explainer(X_trans)

        vals = np.abs(shap_values.values)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        imp = vals.mean(axis=0)

        top_idx = np.argsort(-imp)
        top_features = [{"feature": feature_names[i], "mean_abs_shap": float(imp[i])} for i in top_idx[:25]]
        return {"top_features": top_features}
    except Exception as e:
        return {"error": f"shap failed: {repr(e)}"}


# ------------------------- Gates -------------------------

def apply_gates(dataset_name: str, gates_cfg: dict, perf: dict, drift: dict, fairness: dict, expl: dict) -> dict:
    g = gates_cfg[dataset_name]
    results = {"dataset": dataset_name, "decisions": [], "final": "PASS"}

    def fail(reason, details=None):
        results["decisions"].append({"status": "FAIL", "reason": reason, "details": details})
        results["final"] = "FAIL"

    def warn(reason, details=None):
        results["decisions"].append({"status": "WARN", "reason": reason, "details": details})
        if results["final"] != "FAIL":
            results["final"] = "WARN"

    # Performance gates
    if dataset_name == "fraud":
        if perf["pr_auc"] < g["performance"]["min_pr_auc"]:
            fail("PR-AUC below minimum", {"pr_auc": perf["pr_auc"]})
        if perf["f1"] < g["performance"]["min_f1"]:
            fail("F1 below minimum", {"f1": perf["f1"]})
    else:
        if perf["auc"] < g["performance"]["min_auc"]:
            fail("AUC below minimum", {"auc": perf["auc"]})
        if perf["f1"] < g["performance"]["min_f1"]:
            fail("F1 below minimum", {"f1": perf["f1"]})

    # Drift gate
    if drift["max_psi"] > g["drift"]["max_psi_any_feature"]:
        warn("High feature drift (PSI)", {"max_psi": drift["max_psi"]})

    # Fairness gates (TPR/FPR gaps)
    if "tpr_gap" in fairness and "fpr_gap" in fairness:
        if fairness["tpr_gap"] > g["fairness"]["max_tpr_gap"]:
            warn("High TPR gap across groups (equal opportunity)", {"tpr_gap": fairness["tpr_gap"], "group": fairness.get("group_col")})
        if fairness["fpr_gap"] > g["fairness"]["max_fpr_gap"]:
            warn("High FPR gap across groups", {"fpr_gap": fairness["fpr_gap"], "group": fairness.get("group_col")})
    elif "error" in fairness:
        warn("Fairness check not computed", fairness)

    # Explainability gate
    if "topk_overlap" in expl:
        if expl["topk_overlap"] < g["explainability"]["min_topk_overlap"]:
            warn("Low explanation stability (top-k overlap)", expl)
    elif "error" in expl:
        warn("Explainability check failed", expl)

    if not results["decisions"]:
        results["decisions"].append({"status": "PASS", "reason": "All gates satisfied"})

    return results


# ------------------------- Main runner -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fraud", "titanic"])
    args = ap.parse_args()

    datasets_cfg = load_yaml(os.path.join(CFG_DIR, "datasets.yaml"))
    gates_cfg = load_yaml(os.path.join(CFG_DIR, "gates.yaml"))
    groups_cfg = load_yaml(os.path.join(CFG_DIR, "groups.yaml"))

    if args.dataset == "fraud":
        X_path = os.path.join(ROOT, datasets_cfg["processed"]["fraud_X"])
        y_path = os.path.join(ROOT, datasets_cfg["processed"]["fraud_y"])
        target = "is_fraud"
        model_path = os.path.join(ROOT, "models", "champion", "fraud_model.pkl")
        group_col = groups_cfg["fraud"]["protected_groups"][0]["column"]
    else:
        X_path = os.path.join(ROOT, datasets_cfg["processed"]["titanic_X"])
        y_path = os.path.join(ROOT, datasets_cfg["processed"]["titanic_y"])
        target = "Survived"
        model_path = os.path.join(ROOT, "models", "champion", "titanic_model.pkl")
        group_col = groups_cfg["titanic"]["protected_groups"][0]["column"]

    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)[target].astype(int)

    X_ref, X_cur, y_ref, y_cur = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    model = joblib.load(model_path)

    # Performance
    proba_cur = model.predict_proba(X_cur)[:, 1]
    perf = performance_metrics(y_cur, proba_cur, threshold=0.5)

    # Drift
    drift = drift_check(X_ref, X_cur)

    # Fairness
    fairness = fairness_check(X_cur, y_cur, proba_cur, group_col=group_col, threshold=0.5)

    # Explainability: champion vs previous champion reference
    X_sample = X_cur.sample(n=min(500, len(X_cur)), random_state=42)
    champion_expl = shap_explainability(model, X_sample)

    if "top_features" in champion_expl:
        champ_feats = [d["feature"] for d in champion_expl["top_features"]]
        ref = load_explainability_reference(args.dataset)

        if ref is None:
            save_explainability_reference(args.dataset, champion_expl["top_features"])
            expl = {
                "note": "No previous explainability reference found; created reference for future regression.",
                "topk_overlap": 1.0,
                "champion_top10": champ_feats[:10],
                "reference_top10": champ_feats[:10],
            }
        else:
            ref_feats = [d["feature"] for d in ref.get("top_features", [])]
            overlap = topk_overlap(champ_feats, ref_feats, k=10)
            expl = {
                "topk_overlap": overlap,
                "champion_top10": champ_feats[:10],
                "reference_top10": ref_feats[:10],
            }
            save_explainability_reference(args.dataset, champion_expl["top_features"])
    else:
        expl = {"error": champion_expl.get("error", "explainability failed")}

    # Gates
    gate_results = apply_gates(args.dataset, gates_cfg, perf, drift, fairness, expl)

    # ------------------------- Reporting -------------------------
    report_md = []
    report_md.append(f"# Veritas Validation Report — {args.dataset.upper()}\n")
    report_md.append(f"**Final Gate Decision:** `{gate_results['final']}`\n")

    report_md.append("## 1) Performance\n")
    report_md.append(f"- AUC: {perf['auc']:.4f}\n- F1: {perf['f1']:.4f}\n- PR-AUC: {perf['pr_auc']:.4f}\n")

    report_md.append("\n## 2) Drift (PSI)\n")
    report_md.append(f"- Max PSI: {drift['max_psi']:.4f}\n")
    top_drift = sorted(drift["psi_by_feature"].items(), key=lambda x: -x[1])[:10]
    report_md.append("\nTop drifted features:\n")
    for k, v in top_drift:
        report_md.append(f"- {k}: {v:.4f}\n")

    report_md.append("\n## 3) Fairness (Group Metrics)\n")
    if "error" in fairness:
        report_md.append(f"- Error: {fairness['error']}\n")
    else:
        report_md.append(f"- Group column: `{group_col}`\n")
        report_md.append(f"- Selection-rate disparity (min/max): {fairness['disparity_ratio_min_over_max']:.4f}\n")
        report_md.append(f"- TPR gap: {fairness['tpr_gap']:.4f}\n")
        report_md.append(f"- FPR gap: {fairness['fpr_gap']:.4f}\n")
        report_md.append("\nSelection rate by group:\n")
        for g, sr in fairness["selection_rate_by_group"].items():
            report_md.append(f"- {g}: {sr:.4f}\n")

    report_md.append("\n## 4) Explainability (SHAP)\n")
    if "error" in expl:
        report_md.append(f"- Error: {expl['error']}\n")
    else:
        report_md.append(f"- Top-10 overlap vs reference: {expl.get('topk_overlap', 0):.2f}\n")
        if "note" in expl:
            report_md.append(f"- Note: {expl['note']}\n")
        report_md.append(f"- Champion Top-10: {expl.get('champion_top10')}\n")
        report_md.append(f"- Reference Top-10: {expl.get('reference_top10')}\n")

    report_md.append("\n## 5) Gate Decisions\n")
    for d in gate_results["decisions"]:
        report_md.append(f"- **{d['status']}** — {d['reason']}\n")
        if d.get("details") is not None:
            report_md.append(f"  - details: `{json.dumps(d['details'], ensure_ascii=False)}`\n")

    # ------------------------- Write LATEST report -------------------------
    md_text = "\n".join(report_md)

    md_path = os.path.join(REPORTS_LATEST, f"{args.dataset}_validation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # -------------------------  LATEST gates JSON (combined) -------------------------
    json_path = os.path.join(REPORTS_LATEST, "gates_results.json")
    combined = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                combined = json.load(f)
        except Exception:
            combined = {}

    combined[args.dataset] = gate_results

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    # ------------------------- History snapshot (audit trail) -------------------------
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    hist_md = os.path.join(REPORTS_HISTORY, f"{args.dataset}_validation_report_{ts}.md")
    hist_json = os.path.join(REPORTS_HISTORY, f"gates_results_{ts}.json")

    with open(hist_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"[OK] Report written: {md_path}")
    print(f"[OK] Gates JSON updated: {json_path}")
    print(f"[OK] History snapshot: {hist_md}")
    print(f"[FINAL] {args.dataset.upper()} = {gate_results['final']}")

    if gate_results["final"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
