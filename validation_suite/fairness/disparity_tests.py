def disparity_ratio(selection_rates: dict) -> float:
    if not selection_rates:
        return 1.0
    mx = max(selection_rates.values())
    mn = min(selection_rates.values())
    return float(mn / mx) if mx > 0 else 1.0
