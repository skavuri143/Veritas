# Veritas Validation Report — FRAUD

**Final Gate Decision:** `PASS`

## 1) Performance

- AUC: 0.9920
- F1: 0.8702
- PR-AUC: 0.9140


## 2) Drift

- Max PSI: 0.0001


## 3) Fairness

- Group column: `gender`

- TPR gap: 0.1170

- FPR gap: 0.0001

- Selection-rate disparity (min/max): 0.7189


## 4) Explainability

- Top-10 overlap vs reference: 1.00


## 5) Gate Decisions

- **PASS** — All gates satisfied


## 6) Calibration
- Brier score: 0.001176

## 7) Data Validation
- Schema check: `SKIPPED (no expected schema configured)`
- Missingness OK: `True`
- Leakage OK: `True`
