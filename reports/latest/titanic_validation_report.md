# Veritas Validation Report — TITANIC

**Final Gate Decision:** `WARN`

## 1) Performance

- AUC: 0.9364
- F1: 0.8750
- PR-AUC: 0.9273


## 2) Drift

- Max PSI: 0.0431


## 3) Fairness

- Group column: `Sex`

- TPR gap: 0.3786

- FPR gap: 0.1378

- Selection-rate disparity (min/max): 0.1855


## 4) Explainability

- Top-10 overlap vs reference: 1.00


## 5) Gate Decisions

- **WARN** — High TPR gap across groups (equal opportunity)

  - details: `{"tpr_gap": 0.3786360698125404, "group": "Sex"}`


## 6) Calibration
- Brier score: 0.078169

## 7) Data Validation
- Schema OK: `False`
  - Missing: ['Survived']
  - Unexpected: []
- Missingness OK: `True`
- Leakage OK: `True`
