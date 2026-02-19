def topk_overlap(a: list, b: list, k=10) -> float:
    sa = set(a[:k])
    sb = set(b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)
