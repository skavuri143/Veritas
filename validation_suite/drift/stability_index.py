def stability_index(psi_max: float) -> str:
    if psi_max < 0.1:
        return "STABLE"
    if psi_max < 0.25:
        return "MONITOR"
    return "DRIFTED"
