""""What would it take?" - smallest changes that flip a prediction to approval.

Given a rejected (or borderline) applicant, search over the inputs a person
can realistically change and return a few short plans, for example

    "Raise CIBIL score 412 -> 560"            (one change)
    "Shorten loan term 16 -> 10 years and
     borrow Rs 2,20,00,000 instead of 2,62"   (two smaller changes)

The search is brute force but cheap: every candidate is one row of a
DataFrame, and the whole grid is scored by the Random Forest in one
predict_proba call (a few thousand rows take well under a second).

Each change has an "effort" so plans can be ranked fairly:
    100 CIBIL points = 1, 20% smaller loan = 1, 20% more income = 1,
    5 years of term = 1, bank savings of 10% of the loan = 1.
Only things an applicant could plausibly change are searched; dependents,
education and self-employment are left alone.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .data import FEATURES, LABELS, format_value, validate_applicant

TARGET = 0.6          # "clear approval" in the confidence band
MAX_PLANS = 4


def _candidates(applicant: dict) -> dict[str, list]:
    """Possible new values for each actionable feature (current value excluded)."""
    a = applicant
    return {
        "cibil_score": [s for s in range(a["cibil_score"] + 10, 901, 10)],
        "loan_amount": [int(round(a["loan_amount"] * (1 - r), -4))
                        for r in np.arange(0.05, 0.61, 0.05)],
        "income_annum": [int(round(a["income_annum"] * (1 + r), -4))
                         for r in np.arange(0.05, 0.61, 0.05)],
        "loan_term": [t for t in range(2, 21) if t != a["loan_term"]],
        "bank_asset_value": [int(round(a["bank_asset_value"] + a["loan_amount"] * r, -4))
                             for r in np.arange(0.05, 0.31, 0.05)],
    }


def _effort(feature: str, old, new, applicant: dict) -> float:
    """How hard a change is, on one shared scale (1.0 = a sizeable step)."""
    if feature == "cibil_score":
        return (new - old) / 100
    if feature == "loan_amount":
        return (old - new) / old / 0.20
    if feature == "income_annum":
        return (new - old) / old / 0.20
    if feature == "loan_term":
        return abs(new - old) / 5
    if feature == "bank_asset_value":
        return (new - old) / (0.10 * applicant["loan_amount"])
    raise KeyError(feature)


def describe(feature: str, old, new) -> str:
    """Plain-English text for one change."""
    if feature == "cibil_score":
        return f"Raise CIBIL score from {old} to {new} (+{new - old} points)"
    if feature == "loan_amount":
        return (f"Borrow {format_value(feature, new)} instead of {format_value(feature, old)} "
                f"({(old - new) / old:.0%} less)")
    if feature == "income_annum":
        return (f"Show annual income of {format_value(feature, new)} instead of "
                f"{format_value(feature, old)} (+{(new - old) / old:.0%})")
    if feature == "loan_term":
        verb = "Shorten" if new < old else "Extend"
        return f"{verb} loan term from {old} to {new} years"
    if feature == "bank_asset_value":
        return f"Build bank savings to {format_value(feature, new)} (+{format_value(feature, new - old)})"
    return f"Change {LABELS[feature]} from {old} to {new}"


def _score(model, applicant: dict, changes: list[dict]) -> np.ndarray:
    rows = [{**applicant, **c} for c in changes]
    X = pd.DataFrame(rows, columns=FEATURES)
    return model.predict_proba(X)[:, 1]


def what_would_it_take(model, applicant: dict, target: float = TARGET,
                       max_plans: int = MAX_PLANS) -> dict:
    """Search for the cheapest ways to reach P(approved) >= target.

    Returns {"current", "target", "already", "plans": [{"changes": [...],
    "probability", "effort", "summary"}]}, plans sorted by effort.
    """
    a = validate_applicant(applicant)
    current = float(_score(model, a, [{}])[0])
    out = {"current": current, "target": target, "already": current >= target, "plans": []}
    if out["already"]:
        return out

    cands = _candidates(a)
    found: list[dict] = []

    # 1) one change at a time: keep the cheapest passing value per feature
    best_single: dict[str, float] = {}
    for f, values in cands.items():
        if not values:
            continue
        probs = _score(model, a, [{f: v} for v in values])
        passing = [(_effort(f, a[f], v, a), v, p) for v, p in zip(values, probs) if p >= target]
        if passing:
            e, v, p = min(passing)
            best_single[f] = e
            found.append({"changes": {f: v}, "probability": float(p), "effort": e})

    # 2) pairs of smaller changes: only keep a pair if it is cheaper than
    #    either change on its own (otherwise it adds nothing)
    for f1, f2 in combinations(cands, 2):
        grid = [{f1: v1, f2: v2} for v1 in cands[f1][::2] for v2 in cands[f2][::2]]
        if not grid:
            continue
        probs = _score(model, a, grid)
        best = None
        for change, p in zip(grid, probs):
            if p < target:
                continue
            e = _effort(f1, a[f1], change[f1], a) + _effort(f2, a[f2], change[f2], a)
            if best is None or e < best[0]:
                best = (e, change, p)
        if best and best[0] < min(best_single.get(f1, np.inf), best_single.get(f2, np.inf)):
            found.append({"changes": best[1], "probability": float(best[2]), "effort": best[0]})

    # 3) honesty check: the best probability reachable WITHOUT touching CIBIL,
    #    pulling every other lever to its limit at once
    others = {f: max(v, key=lambda x: _effort(f, a[f], x, a)) for f, v in cands.items()
              if f not in ("cibil_score", "loan_term") and v}
    terms = cands["loan_term"] + [a["loan_term"]]
    out["best_without_cibil"] = float(max(_score(model, a, [{**others, "loan_term": t} for t in terms])))

    found.sort(key=lambda plan: plan["effort"])
    for plan in found[:max_plans]:
        steps = [{"feature": f, "name": LABELS[f], "from": a[f], "to": v, "text": describe(f, a[f], v)}
                 for f, v in plan["changes"].items()]
        out["plans"].append({"changes": steps, "probability": plan["probability"],
                             "effort": round(plan["effort"], 2),
                             "summary": " and ".join(s["text"][0].lower() + s["text"][1:] if i else s["text"]
                                                     for i, s in enumerate(steps))})
    return out


def apply_plan(applicant: dict, plan: dict) -> dict:
    """The applicant after a plan's changes (used by tests and the app)."""
    return {**applicant, **{s["feature"]: s["to"] for s in plan["changes"]}}
