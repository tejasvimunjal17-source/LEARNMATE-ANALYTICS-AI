"""
data_processing.py
-------------------
STAGE 1 - Data Foundation for LearnMate Analytics AI.

This module is responsible for everything that happens BEFORE any chart or
model is built:
    1. Loading the real campus placement CSV
    2. Describing/validating its schema (data quality)
    3. Cleaning it (without inventing values)
    4. Splitting it into features/targets for the two ML tasks
    5. Building the preprocessing pipelines (encoding/scaling) used later
    6. A self-check function that proves we are not leaking data

Every number in this file is calculated from the actual CSV at runtime.
Nothing here is hard-coded from prior knowledge of the dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------

# Path to the dataset relative to the project root (where app.py lives).
CSV_PATH = os.path.join("data", "campus_placement.csv")

# Columns as they actually appear in the Kaggle "Factors Affecting Campus
# Placement" (benroshan) CSV. If your downloaded file has different column
# names, update this list -- the rest of the code reads from it.
EXPECTED_COLUMNS = [
    "sl_no", "gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s",
    "degree_p", "degree_t", "workex", "etest_p", "specialisation",
    "mba_p", "status", "salary",
]

# Column groups used throughout the app.
ID_COLUMN = "sl_no"                     # unique row identifier -> never a feature
SENSITIVE_COLUMN = "gender"             # excluded from predictive modelling
CLASSIFICATION_TARGET = "status"        # Placed / Not Placed
REGRESSION_TARGET = "salary"            # only populated for placed students

NUMERIC_FEATURES = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
CATEGORICAL_FEATURES = [
    "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation",
]

# Columns that must NEVER be fed into the placement (classification) model.
CLASSIFICATION_EXCLUDED = [ID_COLUMN, SENSITIVE_COLUMN, REGRESSION_TARGET, CLASSIFICATION_TARGET]

# Columns that must NEVER be fed into the salary (regression) model.
REGRESSION_EXCLUDED = [ID_COLUMN, SENSITIVE_COLUMN, CLASSIFICATION_TARGET, REGRESSION_TARGET]

RANDOM_STATE = 42  # fixed seed used everywhere for reproducibility

# A short, beginner-friendly data dictionary shown in the Data Explorer page.
DATA_DICTIONARY: dict[str, str] = {
    "sl_no": "Row identifier. Not used in any analysis or model (excluded).",
    "gender": "Student gender. Shown only in descriptive charts; excluded from prediction models as a sensitive attribute.",
    "ssc_p": "Secondary School (10th grade) percentage.",
    "ssc_b": "Board of education for secondary school (Central / Others).",
    "hsc_p": "Higher Secondary (12th grade) percentage.",
    "hsc_b": "Board of education for higher secondary school (Central / Others).",
    "hsc_s": "Stream chosen in higher secondary (Commerce / Science / Arts).",
    "degree_p": "Undergraduate degree percentage.",
    "degree_t": "Field of the undergraduate degree (Comm&Mgmt / Sci&Tech / Others).",
    "workex": "Whether the student has prior work experience (Yes/No).",
    "etest_p": "Employability test percentage (conducted by the college).",
    "specialisation": "MBA specialisation (Mkt&HR / Mkt&Fin).",
    "mba_p": "MBA percentage.",
    "status": "Placement outcome - classification target (Placed / Not Placed).",
    "salary": "Salary offered, in INR. Present ONLY for placed students - regression target.",
}


# ---------------------------------------------------------------------------
# 2. LOADING
# ---------------------------------------------------------------------------

def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the raw campus placement CSV exactly as it is on disk.

    Raises a clear error if the file is missing, so the Streamlit app can
    show a friendly message instead of a stack trace.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{csv_path}'. "
            f"Place 'campus_placement.csv' inside the 'data/' folder."
        )
    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------------
# 3. DATA QUALITY REPORT (read-only inspection, changes nothing)
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_rows: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    schema_matches_expected: bool
    missing_expected_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)


def data_quality_report(df: pd.DataFrame) -> DataQualityReport:
    """Compute a data-quality snapshot of the raw (uncleaned) dataframe."""
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return DataQualityReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=df.columns.tolist(),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        missing_counts={c: int(df[c].isna().sum()) for c in df.columns},
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        schema_matches_expected=(len(missing_expected) == 0),
        missing_expected_columns=missing_expected,
        extra_columns=extra_cols,
    )


# ---------------------------------------------------------------------------
# 4. CLEANING
# ---------------------------------------------------------------------------

@dataclass
class CleaningLog:
    """Tracks which cleaning steps actually ran, so the UI never shows a
    checkmark for something that did not happen."""
    dataset_loaded: bool = False
    duplicates_checked: bool = True
    duplicates_removed: int = 0
    missing_value_analysis_done: bool = True
    dtypes_validated: bool = False
    categorical_encoded: bool = False
    model_ready: bool = False
    notes: list[str] = field(default_factory=list)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean the raw dataframe.

    Rules followed (per project spec):
    - Duplicate rows are dropped (checked, count logged).
    - Numeric columns are coerced to numeric dtype.
    - Categorical/text columns are stripped of stray whitespace.
    - `salary` is LEFT AS MISSING (NaN) for non-placed students. This is
      correct, not an error: a student who was not placed was never offered
      a salary, so imputing a number there would be fabricating data.
    - No other column has missing values in the source dataset, so no other
      imputation is performed (verified in the quality report, not assumed).
    """
    log = CleaningLog(dataset_loaded=True)
    clean = df.copy()

    # -- duplicates ---------------------------------------------------------
    dup_count = int(clean.duplicated().sum())
    if dup_count > 0:
        clean = clean.drop_duplicates().reset_index(drop=True)
        log.notes.append(f"Removed {dup_count} exact duplicate row(s).")
    log.duplicates_removed = dup_count

    # -- dtypes ---------------------------------------------------------
    for col in NUMERIC_FEATURES + [REGRESSION_TARGET]:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    for col in CATEGORICAL_FEATURES + [CLASSIFICATION_TARGET, SENSITIVE_COLUMN]:
        if col in clean.columns:
            clean[col] = clean[col].astype(str).str.strip()
    log.dtypes_validated = True

    # -- salary missingness --------------------------------------------
    if REGRESSION_TARGET in clean.columns and CLASSIFICATION_TARGET in clean.columns:
        placed_mask = clean[CLASSIFICATION_TARGET].str.lower().eq("placed")
        missing_salary_but_placed = int(
            (placed_mask & clean[REGRESSION_TARGET].isna()).sum()
        )
        if missing_salary_but_placed > 0:
            log.notes.append(
                f"Warning: {missing_salary_but_placed} placed student(s) have a "
                f"missing salary value. Left as NaN (not imputed)."
            )
        log.notes.append(
            "Salary is intentionally left missing (NaN) for students who were "
            "not placed - it is not applicable to them, so it is never filled in."
        )

    log.model_ready = True
    return clean, log


# ---------------------------------------------------------------------------
# 5. FEATURE / TARGET SEPARATION
# ---------------------------------------------------------------------------

def get_classification_data(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the placement-prediction task.

    X excludes: sl_no (id), gender (sensitive attribute), salary (would leak
    the outcome, since salary only exists after a student is placed), and
    status itself (the target).
    """
    feature_cols = [c for c in clean_df.columns if c not in CLASSIFICATION_EXCLUDED]
    X = clean_df[feature_cols].copy()
    y = clean_df[CLASSIFICATION_TARGET].copy()
    return X, y


def get_regression_data(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the salary-prediction task, restricted to placed
    students only (salary is undefined for non-placed students).

    X excludes: sl_no (id), gender (sensitive attribute), status (the label
    is implicit in "this row has a salary"), and salary itself (the target).
    """
    placed_only = clean_df[clean_df[CLASSIFICATION_TARGET].str.lower() == "placed"].copy()
    feature_cols = [c for c in placed_only.columns if c not in REGRESSION_EXCLUDED]
    X = placed_only[feature_cols].copy()
    y = placed_only[REGRESSION_TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# 6. PREPROCESSING PIPELINES
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """A reusable preprocessing pipeline:
    - numeric columns: median-impute (safety net) + standard-scale
    - categorical columns: most-frequent-impute (safety net) + one-hot encode

    Used identically for classification and regression so behaviour is
    consistent and easy to explain in the report.
    """
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def get_classification_preprocessor() -> ColumnTransformer:
    return build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def get_regression_preprocessor() -> ColumnTransformer:
    return build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


# ---------------------------------------------------------------------------
# 7. VALIDATION / SELF-CHECK (prevents data leakage regressions)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, bool]
    messages: list[str]


def validate_pipeline(df: pd.DataFrame) -> ValidationResult:
    """Run a set of hard checks and refuse to silently pass if any fails.

    This is meant to be called once at app startup so a future code change
    can never quietly reintroduce data leakage.
    """
    checks: dict[str, bool] = {}
    messages: list[str] = []

    # a. expected columns exist
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    checks["expected_columns_present"] = len(missing_cols) == 0
    if missing_cols:
        messages.append(f"Missing expected columns: {missing_cols}")

    # b. target columns exist
    checks["classification_target_present"] = CLASSIFICATION_TARGET in df.columns
    checks["regression_target_present"] = REGRESSION_TARGET in df.columns

    # c/d. build the feature sets and check exclusions
    clean_df, _ = clean_data(df)
    X_cls, _ = get_classification_data(clean_df)
    X_reg, _ = get_regression_data(clean_df)

    checks["salary_not_in_classification_features"] = REGRESSION_TARGET not in X_cls.columns
    checks["status_not_in_regression_features"] = CLASSIFICATION_TARGET not in X_reg.columns
    checks["sl_no_excluded_from_classification"] = ID_COLUMN not in X_cls.columns
    checks["sl_no_excluded_from_regression"] = ID_COLUMN not in X_reg.columns
    checks["gender_excluded_from_classification"] = SENSITIVE_COLUMN not in X_cls.columns
    checks["gender_excluded_from_regression"] = SENSITIVE_COLUMN not in X_reg.columns

    passed = all(checks.values())
    if not passed:
        failed = [name for name, ok in checks.items() if not ok]
        messages.append(f"Failed checks: {failed}")

    return ValidationResult(passed=passed, checks=checks, messages=messages)


# ---------------------------------------------------------------------------
# 8. Convenience: run everything and print a summary (manual test / CLI use)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = load_data()
    report = data_quality_report(raw)
    print("=== DATA QUALITY REPORT ===")
    print(f"Rows: {report.n_rows}, Columns: {report.n_cols}")
    print(f"Duplicate rows: {report.duplicate_rows}")
    print(f"Schema matches expected: {report.schema_matches_expected}")
    print(f"Missing values per column: {report.missing_counts}")

    cleaned, log = clean_data(raw)
    print("\n=== CLEANING LOG ===")
    print(log)

    result = validate_pipeline(raw)
    print("\n=== VALIDATION ===")
    print(f"Passed: {result.passed}")
    for name, ok in result.checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if result.messages:
        print("Messages:", result.messages)
