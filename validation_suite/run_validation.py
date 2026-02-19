import os
import sys
import json
import argparse
import joblib
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validation_suite.performance.metrics import compute_metrics
from validation_suite.performance.calibration_test import calibration_summary
from validation_suite.drift.feature_drift import feature_drift_report
from validation_suite.drift.prediction_drift import prediction_drift
from validation_suite.drift.stability_index import stability_index
from validation_suite.fairness.group_metrics import group_confusion_metrics, gap
from validation_suite.fairness.disparity_tests import disparity_ratio
from validation_suite.explainability.shap_validate import shap_top_features
from validation_suite.explainability.explanation_stability import topk_overlap
from validation_suite.data_validation.schema_checks import schema_check
from validation_suite.data_validation.missingness_checks import missingness_check
from validation_suite.data_validation.leakage_checks import leakage_check
from validation_suite.reporting.report_builder import build_markdown_report


CFG_DIR = os.path.join(ROOT, "validation_suite", "config")

REPORTS_LATEST = os.path.join(ROOT, "reports", "latest")
REPORTS_HISTORY = os.path.join(ROOT, "reports", "history")
os.makedirs(REPORTS_LATEST, exist_ok=True)
os.makedirs(REPORTS_HISTORY, exist_ok=True)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def explainability_reference_path(dataset_name: str) -> str:
    return os.path.join(ROOT, "models", "metadata", f"{dataset_name}_explainability_reference.json")


def load_explainability_reference(dataset_name: str):
    path = explainability_reference_path(dataset_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_explainability_reference(dataset_name: str, top_features: list):
    path = explainability_reference_path(dataset_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "top_features": top_features[:25],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def apply_gates(dataset_name: str, gates_cfg: dict, perf: dict, drift: dict, fairness: dict, expl: dict):
    g = gates_cfg[dataset_name]
    results = {"dataset": dataset_name, "decisions": [], "final": "PASS"}

    def fail(reason, details=None):
        results["decisions"].append({"status": "FAIL", "reason": reason, "details": details})
        results["final"] = "FAIL"

    def warn(reason, details=None):
        results["decisions"].append({"status": "WARN", "reason": reason, "details": details})
        if results["final"] != "FAIL":
            results["final"] = "WARN"

    # ---------------- Performance ----------------
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

    # ---------------- Drift ----------------
    if drift.get("max_psi", 0.0) > g["drift"]["max_psi_any_feature"]:
        warn("High feature drift (PSI)", {"max_psi": drift["max_psi"]})

    # ---------------- Fairness ----------------
    if "tpr_gap" in fairness and "fpr_gap" in fairness:
        if fairness["tpr_gap"] > g["fairness"]["max_tpr_gap"]:
            warn(
                "High TPR gap across groups (equal opportunity)",
                {"tpr_gap": float(fairness["tpr_gap"]), "group": fairness.get("group_col")},
            )
        if fairness["fpr_gap"] > g["fairness"]["max_fpr_gap"]:
            warn("High FPR gap across groups", {"fpr_gap": float(fairness["fpr_gap"]), "group": fairness.get("group_col")})
    elif "error" in fairness:
        warn("Fairness check not computed", fairness)

    # ---------------- Explainability ----------------
    if "topk_overlap" in expl:
        if expl["topk_overlap"] < g["explainability"]["min_topk_overlap"]:
            warn("Low explanation stability (top-k overlap)", expl)
    elif "error" in expl:
        warn("Explainability check failed", expl)

    if not results["decisions"]:
        results["decisions"].append({"status": "PASS", "reason": "All gates satisfied"})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fraud", "titanic"])
    args = ap.parse_args()

    datasets_cfg = load_yaml(os.path.join(CFG_DIR, "datasets.yaml"))
    gates_cfg = load_yaml(os.path.join(CFG_DIR, "gates.yaml"))
    groups_cfg = load_yaml(os.path.join(CFG_DIR, "groups.yaml"))

    # ---------------- dataset wiring ----------------
    if args.dataset == "fraud":
        X_path = os.path.join(ROOT, datasets_cfg["processed"]["fraud_X"])
        y_path = os.path.join(ROOT, datasets_cfg["processed"]["fraud_y"])
        target = "is_fraud"
        model_path = os.path.join(ROOT, "models", "champion", "fraud_model.pkl")
        group_col = groups_cfg["fraud"]["protected_groups"][0]["column"]

        expected_cols = datasets_cfg.get("expected_schema", {}).get("fraud_columns", None)
        forbidden_cols = datasets_cfg.get("leakage_forbidden", {}).get("fraud", ["is_fraud","trans_num","cc_num"])
    else:
        X_path = os.path.join(ROOT, datasets_cfg["processed"]["titanic_X"])
        y_path = os.path.join(ROOT, datasets_cfg["processed"]["titanic_y"])
        target = "Survived"
        model_path = os.path.join(ROOT, "models", "champion", "titanic_model.pkl")
        group_col = groups_cfg["titanic"]["protected_groups"][0]["column"]

        expected_cols = datasets_cfg.get("expected_schema", {}).get("titanic_columns", None)
        forbidden_cols = datasets_cfg.get("leakage_forbidden", {}).get("titanic", ["Survived", "PassengerId", "Name"])

    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)[target].astype(int)

        # ---------------- data validation ----------------
    dv = {"schema": None, "missingness": None, "leakage": None}

    dv["schema"] = schema_check(X, dataset_name=args.dataset)
    dv["missingness"] = missingness_check(X, max_missing_ratio=0.30, dataset_name=args.dataset)
    dv["leakage"] = leakage_check(X, forbidden_columns=forbidden_cols)

    # ---------------- reference/current split ----------------
    X_ref, X_cur, y_ref, y_cur = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    model = joblib.load(model_path)

    # ---------------- performance + calibration ----------------
    proba_cur = model.predict_proba(X_cur)[:, 1]
    perf = compute_metrics(y_cur, proba_cur, threshold=0.5)
    calib = calibration_summary(y_cur, proba_cur)

    # ---------------- drift ----------------
    drift = feature_drift_report(X_ref, X_cur)

    # optional prediction drift (ref vs cur)
    proba_ref = model.predict_proba(X_ref)[:, 1]
    pred_drift = prediction_drift(proba_ref, proba_cur)
    drift["prediction_drift_l1"] = float(pred_drift.get("hist_l1_diff", 0.0))
    drift["stability_band"] = stability_index(float(drift.get("max_psi", 0.0)))

    # ---------------- fairness ----------------
    fair = {}
    if group_col not in X_cur.columns:
        fair = {"error": f"group column '{group_col}' not found"}
    else:
        df_f = X_cur.copy()
        df_f["_y"] = y_cur.values
        df_f["_pred"] = (proba_cur >= 0.5).astype(int)

        by_group = group_confusion_metrics(df_f, group_col=group_col)
        sel_rates = {k: v["selection_rate"] for k, v in by_group.items()}

        fair = {
            "group_col": group_col,
            "groups": list(by_group.keys()),
            "metrics_by_group": by_group,
            "selection_rate_by_group": sel_rates,
            "disparity_ratio_min_over_max": disparity_ratio(sel_rates),
            "tpr_gap": float(gap(by_group, "tpr")),
            "fpr_gap": float(gap(by_group, "fpr")),
        }

    # ---------------- explainability (champion vs reference champion) ----------------
    X_sample = X_cur.sample(n=min(500, len(X_cur)), random_state=42)
    expl_raw = shap_top_features(model, X_sample, topn=25)

    if "top_features" in expl_raw:
        champ_feats = [d["feature"] for d in expl_raw["top_features"]]
        ref = load_explainability_reference(args.dataset)

        if ref is None:
            save_explainability_reference(args.dataset, expl_raw["top_features"])
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
                "topk_overlap": float(overlap),
                "champion_top10": champ_feats[:10],
                "reference_top10": ref_feats[:10],
            }
            save_explainability_reference(args.dataset, expl_raw["top_features"])
    else:
        expl = {"error": expl_raw.get("error", "explainability failed")}

    # ---------------- gates ----------------
    gate_results = apply_gates(args.dataset, gates_cfg, perf, drift, fair, expl)

    # ---------------- report ----------------
    report_text = build_markdown_report(args.dataset, perf, drift, fair, expl, gate_results)

    # Append calibration + data validation (audit appendix)
    report_text += "\n\n## 6) Calibration\n"
    report_text += f"- Brier score: {calib.get('brier_score', 0):.6f}\n"

    report_text += "\n## 7) Data Validation\n"

    # Schema
    if dv.get("schema") is None:
        report_text += "- Schema check: `SKIPPED`\n"
    else:
        report_text += f"- Schema OK: `{dv['schema'].get('ok', True)}`\n"
        if not dv["schema"].get("ok", True):
            report_text += f"  - Missing: {dv['schema'].get('missing_columns')}\n"
            report_text += f"  - Unexpected: {dv['schema'].get('unexpected_columns')}\n"

    # Missingness
    if dv.get("missingness") is None:
        report_text += "- Missingness check: `SKIPPED`\n"
    else:
        report_text += f"- Missingness OK: `{dv['missingness'].get('ok', True)}`\n"
        if not dv["missingness"].get("ok", True):
            report_text += f"  - Fail columns: {dv['missingness'].get('fail_columns')}\n"

    # Leakage
    if dv.get("leakage") is None:
        report_text += "- Leakage check: `SKIPPED`\n"
    else:
        report_text += f"- Leakage OK: `{dv['leakage'].get('ok', True)}`\n"
        if not dv["leakage"].get("ok", True):
            report_text += f"  - Forbidden present: {dv['leakage'].get('forbidden_present')}\n"


    # ---------------- write latest ----------------
    md_path = os.path.join(REPORTS_LATEST, f"{args.dataset}_validation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # latest gates json (combined)
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

    # ---------------- history snapshots ----------------
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    hist_md = os.path.join(REPORTS_HISTORY, f"{args.dataset}_validation_report_{ts}.md")
    hist_json = os.path.join(REPORTS_HISTORY, f"gates_results_{ts}.json")

    with open(hist_md, "w", encoding="utf-8") as f:
        f.write(report_text)

    # store dataset-only snapshot for history (clean audit trail)
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump({args.dataset: gate_results}, f, indent=2)

    print(f"[OK] Report written: {md_path}")
    print(f"[OK] Gates JSON updated: {json_path}")
    print(f"[OK] History snapshot: {hist_md}")
    print(f"[FINAL] {args.dataset.upper()} = {gate_results['final']}")

    if gate_results["final"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
