"""Does the model play fair? A small group-fairness audit.

Two inputs describe *who* the applicant is rather than their finances:
education (Graduate / Not Graduate) and self_employed (Yes / No). A fair
model should treat the groups similarly once the money is the same. The
audit runs on the same held-out test set as train.py (stratified 80/20,
seed 42) and checks three things per attribute:

1. Group metrics   - approval rate (actual vs predicted), accuracy,
                     true-positive rate (approved people correctly approved)
                     and false-positive rate (rejected people wrongly approved).
2. Gaps            - demographic-parity gap (difference in predicted approval
                     rate) and equal-opportunity gap (difference in TPR).
3. Flip test       - change ONLY that attribute for every applicant and see how
                     often the decision flips. A fair model should barely move.

A gap or flip rate above the tolerance (default 5 points / 2%) is flagged
"check". Note the dataset's own approval rates can differ between groups, so
a parity gap is compared with the gap already present in the labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import FEATURES, LABELS, load_and_clean

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
ATTRIBUTES = ["education", "self_employed"]
GAP_TOLERANCE = 0.05
FLIP_TOLERANCE = 0.02


def holdout_split(path: str | Path = ROOT / "loan_approval_dataset.csv"):
    """Exactly the held-out split used in train.py (X_te, y_te)."""
    df = load_and_clean(path)
    y = (df.pop("loan_status") == "Approved").astype(int)
    _, X_te, _, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    return X_te[FEATURES], y_te


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pos, neg = y_true == 1, y_true == 0
    return {
        "n": int(len(y_true)),
        "actual_approval_rate": float(y_true.mean()),
        "predicted_approval_rate": float(y_pred.mean()),
        "accuracy": float((y_true == y_pred).mean()),
        "tpr": float(y_pred[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(y_pred[neg].mean()) if neg.any() else float("nan"),
    }


def group_report(model, X: pd.DataFrame, y: pd.Series, attr: str) -> pd.DataFrame:
    """One row per group of `attr` with the metrics above."""
    pred = model.predict(X)
    rows = {}
    for group in sorted(X[attr].unique()):
        mask = (X[attr] == group).to_numpy()
        rows[group] = _rates(y.to_numpy()[mask], pred[mask])
    return pd.DataFrame(rows).T


def flip_test(model, X: pd.DataFrame, attr: str) -> dict:
    """Swap the attribute for everyone; how much do decisions move?"""
    values = sorted(X[attr].unique())
    if len(values) != 2:
        raise ValueError(f"{attr} must have exactly two groups")
    flipped = X.copy()
    flipped[attr] = X[attr].map({values[0]: values[1], values[1]: values[0]})
    p0, p1 = model.predict_proba(X)[:, 1], model.predict_proba(flipped)[:, 1]
    changed = (p0 >= 0.5) != (p1 >= 0.5)
    return {"flip_rate": float(changed.mean()), "n_flipped": int(changed.sum()),
            "mean_abs_change": float(np.abs(p1 - p0).mean()), "max_abs_change": float(np.abs(p1 - p0).max())}


def audit(model, X: pd.DataFrame, y: pd.Series, attributes=ATTRIBUTES) -> dict:
    """Full audit: per attribute -> table, gaps, flip test and a verdict."""
    out = {}
    for attr in attributes:
        table = group_report(model, X, y, attr)
        g = table.index.tolist()
        gaps = {
            "parity_gap": float(abs(table.loc[g[0], "predicted_approval_rate"] - table.loc[g[1], "predicted_approval_rate"])),
            "label_gap": float(abs(table.loc[g[0], "actual_approval_rate"] - table.loc[g[1], "actual_approval_rate"])),
            "tpr_gap": float(abs(table.loc[g[0], "tpr"] - table.loc[g[1], "tpr"])),
            "fpr_gap": float(abs(table.loc[g[0], "fpr"] - table.loc[g[1], "fpr"])),
        }
        flip = flip_test(model, X, attr)
        problems = []
        if gaps["parity_gap"] > max(GAP_TOLERANCE, gaps["label_gap"] + GAP_TOLERANCE):
            problems.append("predicted approval rates differ more than the real labels do")
        if gaps["tpr_gap"] > GAP_TOLERANCE:
            problems.append("qualified applicants are approved at different rates")
        if gaps["fpr_gap"] > GAP_TOLERANCE:
            problems.append("unqualified applicants are wrongly approved at different rates")
        if flip["flip_rate"] > FLIP_TOLERANCE:
            problems.append("changing only this attribute flips too many decisions")
        out[attr] = {"name": LABELS[attr], "table": table, "gaps": gaps, "flip": flip,
                     "verdict": "check" if problems else "ok", "problems": problems}
    return out
