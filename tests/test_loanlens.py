"""LoanLens test cases (7 core cases).

Run from the project folder:  pytest -v
Needs model.joblib (python train.py ...) and index/ (python build_index.py).
Ollama is never called: every LLM call here uses a mock client.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loanlens.data import ValidationError, validate_applicant          # noqa: E402
from loanlens.llm import OLLAMA_DOWN_MSG, analyse, explain             # noqa: E402
from loanlens.predict import Predictor, confidence_band                # noqa: E402
from loanlens.rag import NO_SOURCE_MSG, Retriever, load_embedder       # noqa: E402

# A real applicant: the first row of the test split (X_te.iloc[0] in train.py),
# the same row train.py uses for its "cibil_sweep" edge case.
BASE = {
    "no_of_dependents": 3, "education": "Graduate", "self_employed": "No",
    "income_annum": 8_000_000, "loan_amount": 26_200_000, "loan_term": 16, "cibil_score": 890,
    "residential_assets_value": 15_800_000, "commercial_assets_value": 4_300_000,
    "luxury_assets_value": 25_000_000, "bank_asset_value": 4_000_000,
}


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_embedder())


def with_cibil(score):
    return {**BASE, "cibil_score": score}


# 1. Normal case -------------------------------------------------------------
def test_1_normal_cibil_750_is_approved(predictor):
    result = predictor.predict(with_cibil(750))
    assert result["label"] == "Approved"
    assert result["probability"] > 0.6 and result["band"] == "clear"
    assert len(result["factors"]) == 3                       # SHAP top-3 returned
    assert result["factors"][0]["name"] == "CIBIL score"     # CIBIL dominates


# 2. Edge case: same applicant, very low CIBIL ---------------------------------
def test_2_edge_cibil_350_is_rejected(predictor):
    result = predictor.predict(with_cibil(350))
    assert result["label"] == "Rejected"
    assert result["factors"][0]["direction"] == "towards rejection"


# 3. Borderline: CIBIL 550 ----------------------------------------------------
def test_3_borderline_cibil_550(predictor):
    result = predictor.predict(with_cibil(550))
    p = result["probability"]
    assert 0.4 <= p <= 0.9
    # band must follow the rule: clear if p < 0.4 or p > 0.6, else borderline
    assert result["band"] == ("clear" if (p < 0.4 or p > 0.6) else "borderline")
    assert confidence_band(0.5) == "borderline" and confidence_band(0.39) == "clear"
    assert confidence_band(0.61) == "clear"


# 4. Documented failure case ---------------------------------------------------
# This real test-set applicant was REJECTED by the bank, but the model predicts
# APPROVED. Their CIBIL score (674) is well above the ~550 cut-off the forest
# learned, and CIBIL carries ~81% of the importance, so the forest largely
# ignores that the loan (Rs 3.14 crore) is almost 4x the annual income
# (Rs 83 lakh) over only 6 years. It is kept as an expected failure (xfail) to
# document a known limitation, not hidden.
FAILURE_CASE = {
    "no_of_dependents": 3, "education": "Not Graduate", "self_employed": "Yes",
    "income_annum": 8_300_000, "loan_amount": 31_400_000, "loan_term": 6, "cibil_score": 674,
    "residential_assets_value": 1_000_000, "commercial_assets_value": 1_600_000,
    "luxury_assets_value": 17_200_000, "bank_asset_value": 6_100_000,
}


@pytest.mark.xfail(strict=True, reason="Known misclassification: actual Rejected, model predicts Approved")
def test_4_documented_failure_case(predictor):
    assert predictor.predict(FAILURE_CASE)["label"] == "Rejected"   # the true label


# 5. Invalid input -------------------------------------------------------------
def test_5_invalid_input_raises_validation_error(predictor):
    bad = {**BASE, "cibil_score": 1000, "loan_amount": -500_000}
    with pytest.raises(ValidationError) as err:
        validate_applicant(bad)
    assert "CIBIL score" in str(err.value) and "Loan amount" in str(err.value)
    with pytest.raises(ValidationError):            # the predictor validates too
        predictor.predict(bad)


# 6. Retrieval with an unrelated query ------------------------------------------
def test_6_unrelated_query_returns_no_source(predictor, retriever):
    sources = retriever.search("how to bake a chocolate cake in the oven")
    assert sources == []
    client = MagicMock()
    prediction = predictor.predict(with_cibil(750))
    result = explain(prediction["applicant"], prediction, sources, client=client)
    assert result["status"] == "no_sources"
    assert result["notice"] == NO_SOURCE_MSG == "No supporting source found"
    client.chat.assert_not_called()                 # the LLM must NOT be called


# 7. Ollama unavailable ----------------------------------------------------------
def test_7_ollama_down_still_returns_prediction_and_passages(predictor, retriever):
    client = MagicMock()
    client.chat.side_effect = ConnectionError("Failed to connect to Ollama")
    out = analyse(with_cibil(412), predictor, retriever, client=client)
    assert out["prediction"]["label"] == "Rejected"           # prediction kept
    assert len(out["sources"]) >= 1                            # passages kept
    assert all(s["score"] >= 0.35 for s in out["sources"])
    assert out["explanation"]["status"] == "ollama_unavailable"
    assert out["explanation"]["notice"].startswith(OLLAMA_DOWN_MSG)


# Extra: grounding check regenerates once, then falls back -------------------------
def test_grounding_check_regenerates_then_falls_back(predictor, retriever):
    prediction = predictor.predict(with_cibil(750))
    sources = retriever.search("approved loan application, CIBIL score 750")
    bad = {"message": {"content": "Looks fine to me."}}                  # no label, no [n]
    good = {"message": {"content": "Predicted: Approved (97%). CIBIL matters [1]."}}

    client = MagicMock()
    client.chat.side_effect = [bad, good]
    ok = explain(prediction["applicant"], prediction, sources, client=client)
    assert ok["status"] == "ok" and ok["attempts"] == 2

    client = MagicMock()
    client.chat.side_effect = [bad, bad]
    failed = explain(prediction["applicant"], prediction, sources, client=client)
    assert failed["status"] == "ungrounded" and client.chat.call_count == 2
