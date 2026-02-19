def performance_regression(current: dict, baseline: dict, max_drop: dict) -> dict:
    """
    max_drop: e.g. {"auc": 0.02, "f1": 0.02}
    Fails if current metric drops more than allowed compared to baseline.
    """
    failures = []
    for k, allowed in max_drop.items():
        if k in current and k in baseline:
            if baseline[k] - current[k] > allowed:
                failures.append({"metric": k, "baseline": baseline[k], "current": current[k], "allowed_drop": allowed})
    return {"ok": len(failures) == 0, "failures": failures}
