"""Feature: fairness audit on the held-out test set."""
import pytest

from loanlens.fairness import ATTRIBUTES, audit, flip_test, group_report, holdout_split


@pytest.fixture(scope="module")
def split():
    return holdout_split()


def test_split_matches_train_py(split):
    X, y = split
    assert len(X) == 854 and round(y.mean(), 3) == 0.622        # same numbers train.py reports


def test_group_report_rates_are_valid(shared_predictor, split):
    X, y = split
    t = group_report(shared_predictor.model, X, y, "education")
    assert set(t.index) == {"Graduate", "Not Graduate"} and t["n"].sum() == len(X)
    for col in ["actual_approval_rate", "predicted_approval_rate", "accuracy", "tpr", "fpr"]:
        assert t[col].between(0, 1).all()


def test_flip_test_and_verdicts(shared_predictor, split):
    X, y = split
    flip = flip_test(shared_predictor.model, X, "self_employed")
    assert 0 <= flip["flip_rate"] <= 1 and flip["n_flipped"] == round(flip["flip_rate"] * len(X))
    report = audit(shared_predictor.model, X, y)
    assert set(report) == set(ATTRIBUTES)
    for r in report.values():
        assert r["verdict"] in ("ok", "check") and (r["verdict"] == "ok") == (not r["problems"])
    # documented result: both attributes currently pass
    assert all(r["verdict"] == "ok" for r in report.values())


def test_flip_test_needs_two_groups(shared_predictor, split):
    X, _ = split
    with pytest.raises(ValueError):
        flip_test(shared_predictor.model, X.assign(education="Graduate"), "education")
