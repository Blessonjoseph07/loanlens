"""Feature: "What would it take?" counterfactual plans."""
from unittest.mock import MagicMock

from loanlens.counterfactual import TARGET, apply_plan, what_would_it_take
from loanlens.llm import OLLAMA_DOWN_MSG, explain_plan, plan_query


def test_approved_applicant_needs_no_plan(shared_predictor, samples):
    cf = what_would_it_take(shared_predictor.model, samples["approved"])
    assert cf["already"] and cf["plans"] == []


def test_rejected_applicant_gets_plans_that_really_work(shared_predictor, samples):
    a = samples["rejected"]
    cf = what_would_it_take(shared_predictor.model, a)
    assert not cf["already"] and cf["plans"]
    efforts = [p["effort"] for p in cf["plans"]]
    assert efforts == sorted(efforts)                       # cheapest first
    for plan in cf["plans"]:                                 # re-score independently
        assert shared_predictor.predict(apply_plan(a, plan))["probability"] >= TARGET
        assert all(s["feature"] not in ("education", "self_employed", "no_of_dependents")
                   for s in plan["changes"])


def test_best_without_cibil_is_reported(shared_predictor, samples):
    cf = what_would_it_take(shared_predictor.model, samples["rejected"])
    assert 0.0 <= cf["best_without_cibil"] < TARGET          # CIBIL really is required here


def test_plan_explainer_cites_sources_and_handles_ollama_down(shared_predictor, samples):
    cf = what_would_it_take(shared_predictor.model, samples["rejected"])
    assert "CIBIL score" in plan_query(cf["plans"][0])
    src = [{"title": "T", "text": "Pay EMIs on time.", "source": "u", "score": 0.6}]
    good = MagicMock(); good.chat.return_value = {"message": {"content": "Raise your score [1]."}}
    assert explain_plan(samples["rejected"], cf, src, client=good)["status"] == "ok"
    down = MagicMock(); down.chat.side_effect = ConnectionError("refused")
    r = explain_plan(samples["rejected"], cf, src, client=down)
    assert r["status"] == "ollama_unavailable" and r["notice"] == OLLAMA_DOWN_MSG
    assert explain_plan(samples["rejected"], cf, [], client=good)["status"] == "no_sources"
