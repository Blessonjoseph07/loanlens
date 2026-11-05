"""Feature: "How sure is it?" meter."""
import pytest

from loanlens.uncertainty import FamiliarityChecker, assess, tree_votes

FAILURE_CASE = {
    "no_of_dependents": 3, "education": "Not Graduate", "self_employed": "Yes",
    "income_annum": 8_300_000, "loan_amount": 31_400_000, "loan_term": 6, "cibil_score": 674,
    "residential_assets_value": 1_000_000, "commercial_assets_value": 1_600_000,
    "luxury_assets_value": 17_200_000, "bank_asset_value": 6_100_000,
}


@pytest.fixture(scope="module")
def checker(shared_predictor):
    return FamiliarityChecker(shared_predictor.model)


def run(predictor, checker, applicant):
    pred = predictor.predict(applicant)
    return pred, assess(predictor.model, checker, applicant, pred["band"])


def test_tree_votes_match_forest(shared_predictor, samples):
    v = tree_votes(shared_predictor.model, samples["approved"])
    assert v["n_trees"] == 200 and 0.5 <= v["agreement"] <= 1
    assert v["approve_votes"] / 200 == pytest.approx(v["agreement"])


def test_clear_samples_are_high_and_borderline_is_low(shared_predictor, checker, samples):
    for name in ("approved", "rejected"):
        assert run(shared_predictor, checker, samples[name])[1]["level"] == "high"
    r = run(shared_predictor, checker, samples["borderline"])[1]
    assert r["level"] == "low" and r["agreement"] < 0.7


def test_known_failure_case_is_not_high(shared_predictor, checker):
    pred, r = run(shared_predictor, checker, FAILURE_CASE)
    assert pred["label"] == "Approved"                     # the model is wrong here...
    assert r["level"] != "high"                            # ...and the meter does not claim certainty
    assert any(u["feature"] == "assets_to_loan" for u in r["unusual"])


def test_far_away_applicant_is_novel(shared_predictor, checker):
    odd = {**FAILURE_CASE, "income_annum": 100_000, "loan_amount": 90_000_000}
    r = run(shared_predictor, checker, odd)[1]
    assert r["novel"] and r["level"] == "low"
