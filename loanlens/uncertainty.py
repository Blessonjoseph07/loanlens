""""How sure is it?" - two honest signals on top of the probability.

1. Tree agreement. A Random Forest is 200 separate decision trees. If 195
   of them say "approve", the forest is sure; if it is 110 vs 90, the 55%
   probability hides a real split. We report the share of trees on the
   majority side and the spread (std) of their individual probabilities.

2. Is this applicant like anything the model has seen? Two checks against
   the TRAINING split only (never the test set):
     * per-feature ranges  - each value vs the 1st-99th percentile of
       training data, plus two ratios that matter for lending
       (loan / income and total assets / loan);
     * nearest neighbours  - distance to the 5 closest training applicants
       in the model's own scaled feature space, compared with the 99th
       percentile of the same distance among training applicants.
   A model is only as good as the data it learned from; far away from that
   data its confident-looking numbers deserve less trust.

assess() combines them with the confidence band into high / medium / low.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from .data import FEATURES, LABELS, NUMERIC, format_value, load_and_clean, to_frame, validate_applicant

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
K = 5
LOW_AGREEMENT, HIGH_AGREEMENT = 0.70, 0.90

RATIOS = {
    "loan_to_income": ("Loan / income", lambda d: d["loan_amount"] / d["income_annum"]),
    "assets_to_loan": ("Total assets / loan", lambda d: (d["residential_assets_value"] + d["commercial_assets_value"]
                                                        + d["luxury_assets_value"] + d["bank_asset_value"])
                       / d["loan_amount"]),
}


def tree_votes(pipeline, applicant: dict) -> dict:
    """Share of trees on the majority side, and the spread of their probabilities."""
    Xt = pipeline.named_steps["pre"].transform(to_frame(validate_applicant(applicant)))
    probs = np.array([t.predict_proba(Xt)[0, 1] for t in pipeline.named_steps["clf"].estimators_])
    approve = (probs >= 0.5).mean()
    return {"agreement": float(max(approve, 1 - approve)), "approve_votes": int((probs >= 0.5).sum()),
            "n_trees": len(probs), "spread": float(probs.std())}


class FamiliarityChecker:
    """Learns what 'normal' looks like from the training split."""

    def __init__(self, pipeline, data_path: str | Path = ROOT / "loan_approval_dataset.csv"):
        df = load_and_clean(data_path)
        y = (df.pop("loan_status") == "Approved").astype(int)
        X_tr, _, _, _ = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
        X_tr = X_tr[FEATURES]
        self.pre = pipeline.named_steps["pre"]
        self.ranges = {f: (float(X_tr[f].quantile(0.01)), float(X_tr[f].quantile(0.99))) for f in NUMERIC}
        for key, (_, fn) in RATIOS.items():
            r = fn(X_tr)
            self.ranges[key] = (float(r.quantile(0.01)), float(r.quantile(0.99)))
        Z = self.pre.transform(X_tr)
        self.nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
        d, _ = self.nn.kneighbors(Z)                      # first neighbour is the point itself
        self.threshold = float(np.quantile(d[:, 1:].mean(axis=1), 0.99))

    def check(self, applicant: dict) -> dict:
        a = validate_applicant(applicant)
        unusual = []
        for f in NUMERIC:
            lo, hi = self.ranges[f]
            if not lo <= a[f] <= hi:
                unusual.append({"feature": f, "name": LABELS[f], "value": format_value(f, a[f]),
                                "range": f"{format_value(f, round(lo))} - {format_value(f, round(hi))}"})
        frame = pd.DataFrame([a])
        for key, (name, fn) in RATIOS.items():
            v, (lo, hi) = float(fn(frame).iloc[0]), self.ranges[key]
            if not lo <= v <= hi:
                unusual.append({"feature": key, "name": name, "value": f"{v:.2f}", "range": f"{lo:.2f} - {hi:.2f}"})
        d, _ = self.nn.kneighbors(self.pre.transform(to_frame(a)), n_neighbors=K)
        distance = float(d.mean())
        return {"unusual": unusual, "distance": distance, "threshold": self.threshold,
                "novel": distance > self.threshold}


def assess(pipeline, checker: FamiliarityChecker, applicant: dict, band: str) -> dict:
    """Combine band, tree agreement and familiarity into high / medium / low."""
    votes, fam = tree_votes(pipeline, applicant), checker.check(applicant)
    reasons = []
    if band == "borderline":
        reasons.append("the probability is in the 40-60% borderline band")
    if votes["agreement"] < LOW_AGREEMENT:
        reasons.append(f"only {votes['agreement']:.0%} of the trees agree")
    if fam["novel"]:
        reasons.append("this applicant is far from every training applicant")
    level = "low" if reasons else "high"
    if level == "high":
        if votes["agreement"] < HIGH_AGREEMENT:
            reasons.append(f"{votes['agreement']:.0%} of the trees agree (not unanimous)")
        if fam["unusual"]:
            reasons.append(f"{len(fam['unusual'])} value(s) are outside the usual training range")
        level = "medium" if reasons else "high"
    return {"level": level, "reasons": reasons, **votes, **fam}
