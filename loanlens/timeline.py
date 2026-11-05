"""Credit-score time machine: simulate habits month by month.

Pick a few habits ("pay every bill on time", "keep card use under 30%", ...)
and a monthly saving; the simulator moves the CIBIL score and bank savings
forward one month at a time and asks the model for the approval probability
at every step, so you can see *when* the applicant would cross 50% / 60%.

IMPORTANT: the point values below are illustrative assumptions, not the real
CIBIL formula (which is not public). They are deliberately easy to edit in
the app so the user can try their own numbers. Positive effects shrink as
the score approaches 900 (diminishing returns), which is how credit scores
generally behave.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import FEATURES, validate_applicant


@dataclass(frozen=True)
class Habit:
    key: str
    label: str
    points: float          # CIBIL points (per month, or once if `month` is set)
    month: int | None = None


HABITS = [
    Habit("on_time", "Pay every EMI and card bill on time", 6),
    Habit("low_use", "Keep credit-card use under 30% of the limit", 4),
    Habit("no_new", "Avoid new loan or card applications", 1),
    Habit("clear_overdue", "Clear an overdue or settled account (month 2)", 35, month=2),
    Habit("missed", "Miss one payment (month 3)", -70, month=3),
    Habit("enquiries", "Apply for several new loans at once (month 1)", -20, month=1),
]
BY_KEY = {h.key: h for h in HABITS}


def _step(score: float, delta: float) -> float:
    """Apply one change with diminishing returns near the top of the scale."""
    if delta > 0:
        delta *= max(0.15, (900 - score) / 600)
    return min(900.0, max(300.0, score + delta))


def simulate(model, applicant: dict, habits: list[str], months: int = 24,
             monthly_saving: int = 0, points: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per month (0 = today): cibil_score, bank_asset_value, probability.

    `points` optionally overrides the default points of any habit.
    """
    a = validate_applicant(applicant)
    points = {h.key: h.points for h in HABITS} | (points or {})
    score, bank = float(a["cibil_score"]), a["bank_asset_value"]
    rows = []
    for m in range(months + 1):
        if m > 0:
            for key in habits:
                h = BY_KEY[key]
                if h.month is None or h.month == m:
                    score = _step(score, points[key])
            bank += monthly_saving
        rows.append({**a, "cibil_score": int(round(score)), "bank_asset_value": int(bank), "month": m})
    df = pd.DataFrame(rows)
    df["probability"] = model.predict_proba(df[FEATURES])[:, 1]
    return df[["month", "cibil_score", "bank_asset_value", "probability"]]


def first_month(df: pd.DataFrame, threshold: float) -> int | None:
    """First month whose probability reaches the threshold (None if never)."""
    hit = df[df["probability"] >= threshold]
    return int(hit["month"].iloc[0]) if len(hit) else None
