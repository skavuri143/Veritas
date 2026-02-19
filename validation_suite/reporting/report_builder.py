import json

def build_markdown_report(dataset_name: str, perf: dict, drift: dict, fairness: dict, expl: dict, gate_results: dict) -> str:
    lines = []
    lines.append(f"# Veritas Validation Report — {dataset_name.upper()}\n")
    lines.append(f"**Final Gate Decision:** `{gate_results['final']}`\n")

    lines.append("## 1) Performance\n")
    lines.append(f"- AUC: {perf.get('auc', 0):.4f}\n- F1: {perf.get('f1', 0):.4f}\n- PR-AUC: {perf.get('pr_auc', 0):.4f}\n")

    lines.append("\n## 2) Drift\n")
    lines.append(f"- Max PSI: {drift.get('max_psi', 0):.4f}\n")

    lines.append("\n## 3) Fairness\n")
    if "error" in fairness:
        lines.append(f"- Error: {fairness['error']}\n")
    else:
        lines.append(f"- Group column: `{fairness.get('group_col')}`\n")
        lines.append(f"- TPR gap: {fairness.get('tpr_gap', 0):.4f}\n")
        lines.append(f"- FPR gap: {fairness.get('fpr_gap', 0):.4f}\n")
        lines.append(f"- Selection-rate disparity (min/max): {fairness.get('disparity_ratio_min_over_max', 0):.4f}\n")

    lines.append("\n## 4) Explainability\n")
    if "error" in expl:
        lines.append(f"- Error: {expl['error']}\n")
    else:
        lines.append(f"- Top-10 overlap vs reference: {expl.get('topk_overlap', 0):.2f}\n")

    lines.append("\n## 5) Gate Decisions\n")
    for d in gate_results["decisions"]:
        lines.append(f"- **{d['status']}** — {d['reason']}\n")
        if d.get("details") is not None:
            lines.append(f"  - details: `{json.dumps(d['details'], ensure_ascii=False)}`\n")

    return "\n".join(lines)
