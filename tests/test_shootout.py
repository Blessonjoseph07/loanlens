"""Feature: model shootout results (comparison.json from compare_models.py)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def comp():
    return json.loads((ROOT / "comparison.json").read_text())


def test_all_five_models_with_sane_metrics(comp):
    assert set(comp["models"]) == {"Logistic Regression", "Random Forest", "HistGradientBoosting",
                                   "XGBoost", "LightGBM"}
    for m in comp["models"].values():
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0 <= m[k] <= 1
        assert len(m["roc"]["fpr"]) == len(m["roc"]["tpr"]) >= 2
        assert m["errors"] == round((1 - m["accuracy"]) * comp["test_size"])


def test_shared_models_match_train_py(comp):
    metrics = json.loads((ROOT / "metrics.json").read_text())
    for name in ("Logistic Regression", "Random Forest"):
        assert comp["models"][name]["accuracy"] == pytest.approx(metrics[name]["accuracy"])
        assert comp["models"][name]["roc_auc"] == pytest.approx(metrics[name]["roc_auc"])
