# SolventIQ — Loan Default Prediction

A stacked ensemble (Random Forest + Logistic Regression/XGBoost, combined via a
meta-model) for predicting loan default risk from the LendingClub accepted-loans
dataset. See `PROBLEM_STATEMENT.txt` for the project goal and `IMPLEMENTATION.md`
(local, untracked) for the working implementation plan.

## Project structure

```text
data/
├── raw/          # accepted_loans.csv — source of truth, never modified
├── processed/    # loan_data.duckdb — cleaned DuckDB table (generated)
└── exports/      # selected_features.csv and similar exports (generated)

notebooks/        # EDA, feature selection, modeling, evaluation
util-scripts/     # DuckDB ingestion and audit scripts (named to avoid
                   # colliding with the venv's Scripts/ folder on Windows)
models/           # trained model artifacts
api/              # serving layer
```

## Pipeline so far

1. `util-scripts/01_build_database.py` — ingest `data/raw/accepted_loans.csv`
   into `data/processed/loan_data.duckdb`, keeping only Fully Paid / Charged
   Off loans and creating `default_flag`.
2. `util-scripts/02_data_audit.py` — row/column counts and target distribution.
3. `util-scripts/03_chunk_analysis.py` — chunked Pandas access over DuckDB.
4. `util-scripts/04_missingness_audit.py` — exact missingness per column.
5. `util-scripts/05_categorical_audit.py` — default rate by categorical value.
6. `util-scripts/06_correlation_audit.py` — numerical correlation with default.

## Setup

```bash
pip install -r requirements.txt
python util-scripts/01_build_database.py
```
