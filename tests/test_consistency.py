"""Feature: mismatch alarm between the LLM explanation and SHAP."""
from unittest.mock import MagicMock

from loanlens.llm import consistency_check, explain

PRED = {"label": "Rejected", "probability": 0.02, "band": "clear", "applicant": {},
        "factors": [{"feature": "cibil_score"}, {"feature": "loan_term"}, {"feature": "loan_amount"}]}


def test_all_top_factors_mentioned_is_consistent():
    text = "Rejected. Your CIBIL score is low [1], the loan term is long and the loan amount is high [2]."
    assert consistency_check(text, PRED)["status"] == "consistent"


def test_missing_lower_factor_or_extra_input_is_partial():
    c = consistency_check("Rejected mainly because of your CIBIL score [1] and loan term.", PRED)
    assert c["status"] == "partial" and c["missing"] == ["loan_amount"]
    c = consistency_check("Rejected: CIBIL score, loan term, loan amount and your income [1].", PRED)
    assert c["status"] == "partial" and c["extra"] == ["income_annum"]


def test_missing_top_factor_is_mismatch():
    c = consistency_check("Rejected because your income and savings are low [1].", PRED)
    assert c["status"] == "mismatch" and "cibil_score" in c["missing"]


def test_explain_attaches_the_check(shared_predictor, samples):
    pred = shared_predictor.predict(samples["rejected"])
    names = " and ".join(f["name"] for f in pred["factors"])
    client = MagicMock()
    client.chat.return_value = {"message": {"content": f"Rejected because of {names} [1]."}}
    src = [{"title": "T", "text": "x", "source": "u", "score": 0.5}]
    r = explain(pred["applicant"], pred, src, client=client)
    assert r["status"] == "ok" and r["consistency"]["status"] == "consistent"
