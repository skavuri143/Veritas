# Veritas — Machine Learning Decision Validation & Trust Framework

Veritas is a **decision-trust validation framework** for machine learning systems.  It focuses on **testing ML models**, not just building them, by validating:

- **Performance regression** over time (AUC/F1/PR-AUC checks)
- **Feature drift & stability** (PSI + stability band)
- **Fairness & bias** across groups (selection rate + TPR/FPR gaps)
- **Explainability validation** using SHAP (top feature stability)
- **Deployment gates** (PASS / WARN / FAIL) with audit-ready reports

This repo supports two example datasets:
- **Fraud**: `fraudTrain.csv`
- **Titanic**: `Titanic-Dataset.csv`

---

## Project Layout

```text
Veritas/
├─ data/
│  ├─ raw/                  # place datasets here (NOT pushed to GitHub)
│  ├─ processed/            # generated parquet files (NOT pushed to GitHub)
│  └─ data_profile/         # generated schema/profile json 
├─ models/
│  ├─ champion/             
│  └─ metadata/            
├─ reports/
│  ├─ latest/              
│  └─ history/             
├─ scripts/
│  ├─ preprocess.py
│  ├─ train.py
│  └─ generate_data_profile.py
├─ validation_suite/
│  ├─ run_validation.py
│  ├─ config/
│  │  ├─ datasets.yaml
│  │  ├─ gates.yaml
│  │  └─ groups.yaml
│  ├─ data_validation/
│  ├─ performance/
│  ├─ drift/
│  ├─ fairness/
│  ├─ explainability/
│  └─ reporting/
├─ requirements.txt
├─ .gitignore
└─ README.md
```
## Setup (VS Code / Windows)

### 1) Open the project
- Open **VS Code**
- Go to `File → Open Folder…`
- Select the `Veritas/` folder (project root)

### 2) Create and activate a virtual environment (recommended)

Open a terminal in VS Code: `Terminal → New Terminal`

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### 3) Verify installation

```powershell
python -c "import pandas, numpy, sklearn, yaml; print('OK')"
```
If you see OK, your environment is ready.

## What the Validation Checks

### Performance
- **AUC**, **F1**
- **PR-AUC** (especially relevant for fraud / imbalanced data)

### Drift
- **PSI (Population Stability Index)** over numeric features
- **Prediction distribution drift** (histogram distance)
- **Stability band** summary

### Fairness (Group-based)
- **Selection rate** by group
- **TPR/FPR** by group
- **TPR gap** and **FPR gap**

### Explainability (SHAP-based)
- **Top feature importance** list
- **Stability check** vs previous reference (**top-k overlap**)

### Deployment Gates
Each dataset has thresholds in:
- `validation_suite/config/gates.yaml`

Final decisions:
- **PASS**: safe to deploy  
- **WARN**: investigate before deployment  
- **FAIL**: block deployment

## Contributing

We welcome contributions! If you'd like to improve Sentinel, feel free to fork the repository, create a new branch, and submit a pull request. Please ensure that you write tests for any new functionality and that the existing tests pass.
