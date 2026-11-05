"""Shared data definitions for LoanLens.

Everything that both the training script (train.py) and the app (app.py) need
lives here, so the model is trained and used on exactly the same columns:

* load_and_clean()     - reads the Kaggle CSV and fixes its formatting quirks
* FEATURES / LABELS    - the 11 input columns and their human-readable names
* RANGES               - the allowed range for every form field
* validate_applicant() - checks one applicant (a dict) before prediction
* to_frame()           - turns one applicant dict into a 1-row DataFrame
"""
from __future__ import annotations

import pandas as pd

# Target column and the two text (categorical) columns.
TARGET = "loan_status"
CATEGORICAL = ["education", "self_employed"]

# The 11 model inputs, in the same order as the CSV (after loan_id is dropped).
FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Readable names used in the UI, in SHAP factors and in the LLM prompt.
LABELS = {
    "no_of_dependents": "Number of dependents",
    "education": "Education",
    "self_employed": "Self-employed",
    "income_annum": "Annual income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "cibil_score": "CIBIL score",
    "residential_assets_value": "Residential assets",
    "commercial_assets_value": "Commercial assets",
    "luxury_assets_value": "Luxury assets",
    "bank_asset_value": "Bank assets",
}

# Columns that hold rupee amounts (shown with thousand separators in the UI).
RUPEE_FIELDS = [
    "income_annum", "loan_amount",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]

# Allowed values for every field. (min, max) for numbers, a list for choices.
# None as max means "no upper limit".
RANGES = {
    "no_of_dependents": (0, 5),
    "education": ["Graduate", "Not Graduate"],
    "self_employed": ["Yes", "No"],
    "income_annum": (1, None),          # must be positive
    "loan_amount": (1, None),           # must be positive
    "loan_term": (2, 20),               # years
    "cibil_score": (300, 900),          # official CIBIL scale
    "residential_assets_value": (0, None),
    "commercial_assets_value": (0, None),
    "luxury_assets_value": (0, None),
    "bank_asset_value": (0, None),
}


def load_and_clean(path: str = "loan_approval_dataset.csv") -> pd.DataFrame:
    """Read the Kaggle CSV and clean it (same steps as train.py has always used).

    The raw file has leading spaces in column names (" cibil_score") and in
    text values (" Approved"), so both are stripped. loan_id is only an
    identifier, so it is dropped. A few residential asset values are negative,
    which is impossible, so they are clipped to 0.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in CATEGORICAL + [TARGET]:
        df[c] = df[c].str.strip()
    df = df.drop(columns=["loan_id"])
    df["residential_assets_value"] = df["residential_assets_value"].clip(lower=0)
    return df


class ValidationError(ValueError):
    """Raised when an applicant has a missing or out-of-range value."""


def validate_applicant(applicant: dict) -> dict:
    """Check one applicant and return a cleaned copy.

    Raises ValidationError listing every problem found, e.g. a CIBIL score of
    1000 or a negative loan amount. This protects the model from inputs it
    was never trained on and gives the user a clear message instead of a crash.
    """
    errors, clean = [], {}
    for field in FEATURES:
        if field not in applicant or applicant[field] is None or applicant[field] == "":
            errors.append(f"{LABELS[field]} is missing")
            continue
        value, rule = applicant[field], RANGES[field]
        if isinstance(rule, list):                       # categorical field
            value = str(value).strip()
            if value not in rule:
                errors.append(f"{LABELS[field]} must be one of {rule}")
            clean[field] = value
            continue
        try:                                             # numeric field
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{LABELS[field]} must be a number")
            continue
        low, high = rule
        if number < low or (high is not None and number > high):
            limit = f"between {low:,} and {high:,}" if high is not None else f"at least {low:,}"
            errors.append(f"{LABELS[field]} must be {limit} (got {number:,.0f})")
        clean[field] = int(round(number))
    if errors:
        raise ValidationError("; ".join(errors))
    return clean


def format_inr(amount: float) -> str:
    """Format rupees with Indian digit grouping: 5000000 -> '50,00,000'."""
    s = str(int(round(amount)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def format_value(feature: str, value) -> str:
    """Human-readable value for the UI and the LLM prompt."""
    if feature in RUPEE_FIELDS:
        return f"Rs {format_inr(value)}"
    if feature == "loan_term":
        return f"{value} years"
    return str(value)


def to_frame(applicant: dict) -> pd.DataFrame:
    """Turn one (validated) applicant dict into a 1-row DataFrame for the model."""
    return pd.DataFrame([[applicant[f] for f in FEATURES]], columns=FEATURES)
