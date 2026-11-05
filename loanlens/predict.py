"""Machine-learning part of LoanLens: prediction + SHAP explanation.

model.joblib (made by train.py) is a scikit-learn Pipeline:
    "pre" : ColumnTransformer(StandardScaler on numbers, OneHotEncoder on text)
    "clf" : RandomForestClassifier(n_estimators=200)

predict(applicant) returns a dict like
    {"label": "Approved", "probability": 0.97, "band": "clear",
     "factors": [{"feature": "cibil_score", "name": "CIBIL score", "value": 778,
                  "shap": 0.31, "direction": "towards approval"}, ...]}
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import shap

from .data import LABELS, validate_applicant, to_frame

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"

# Probability limits for the confidence band. Between 0.4 and 0.6 the model is
# close to a coin flip, so we call the result "borderline".
BAND_LOW, BAND_HIGH = 0.4, 0.6


def confidence_band(p: float) -> str:
    """'clear' if p < 0.4 or p > 0.6, otherwise 'borderline'."""
    return "clear" if (p < BAND_LOW or p > BAND_HIGH) else "borderline"


def _base_feature(encoded_name: str) -> str:
    """Map a column name after preprocessing back to the original feature.

    e.g. 'num__cibil_score' -> 'cibil_score',
         'cat__education_Not Graduate' -> 'education'.
    """
    name = encoded_name.split("__", 1)[1]
    for feature in LABELS:
        if name == feature or name.startswith(feature + "_"):
            return feature
    return name


class Predictor:
    """Holds the trained pipeline and a SHAP TreeExplainer for its forest."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model = joblib.load(model_path)
        self.pre = self.model.named_steps["pre"]
        self.forest = self.model.named_steps["clf"]
        # TreeExplainer computes exact SHAP values for tree ensembles.
        self.explainer = shap.TreeExplainer(self.forest)
        self.encoded_names = list(self.pre.get_feature_names_out())

    def predict(self, applicant: dict, top_k: int = 3) -> dict:
        clean = validate_applicant(applicant)          # raises ValidationError
        X = to_frame(clean)
        p = float(self.model.predict_proba(X)[0, 1])   # P(Approved)
        return {
            "label": "Approved" if p >= 0.5 else "Rejected",
            "probability": p,
            "band": confidence_band(p),
            "factors": self.top_factors(X, clean, top_k),
            "applicant": clean,
        }

    def top_factors(self, X, applicant: dict, top_k: int = 3) -> list[dict]:
        """Top-k features that moved THIS applicant's prediction the most.

        SHAP gives each feature a signed contribution to P(Approved):
        positive = pushes towards approval, negative = towards rejection.
        One-hot columns are added back together so each original feature
        appears once, then we sort by absolute size.
        """
        Xt = self.pre.transform(X)
        sv = self.explainer.shap_values(Xt)
        # Depending on the SHAP version the output for a binary classifier is
        # a list [class0, class1] or an array (rows, features, classes).
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        contributions: dict[str, float] = {}
        for name, value in zip(self.encoded_names, sv[0]):
            feature = _base_feature(name)
            contributions[feature] = contributions.get(feature, 0.0) + float(value)
        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        return [
            {
                "feature": f,
                "name": LABELS[f],
                "value": applicant[f],
                "shap": round(v, 4),
                "direction": "towards approval" if v >= 0 else "towards rejection",
            }
            for f, v in ranked
        ]


_default: Predictor | None = None


def predict(applicant: dict) -> dict:
    """Convenience function using a model loaded once from model.joblib."""
    global _default
    if _default is None:
        _default = Predictor()
    return _default.predict(applicant)
