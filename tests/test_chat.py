"""Feature: follow-up chat about the current decision (LLM always mocked)."""
from unittest.mock import MagicMock

import pytest

from loanlens.data import mentioned_features
from loanlens.llm import OFF_TOPIC_MSG, OLLAMA_DOWN_MSG, chat_followup
from loanlens.rag import Retriever, load_embedder


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_embedder())


@pytest.fixture(scope="module")
def prediction(shared_predictor, samples):
    return shared_predictor.predict(samples["rejected"])


def client_saying(text):
    c = MagicMock()
    c.chat.return_value = {"message": {"content": text}}
    return c


def test_feature_terms_whole_words():
    assert mentioned_features("Why does the loan term matter?") == ["loan_term"]
    assert mentioned_features("Is a long-term loan better?") == []
    assert "cibil_score" in mentioned_features("my CIBIL and tenure")


def test_on_topic_question_uses_sources_and_history(retriever, prediction):
    c = client_saying("Paying on time raises the score [1].")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    r = chat_followup("How can I improve my credit score?", prediction, retriever, history, client=c)
    assert r["status"] == "ok" and r["sources"]
    msgs = c.chat.call_args.kwargs["messages"]
    assert msgs[1]["content"] == "hi" and "QUESTION" in msgs[-1]["content"]


def test_off_topic_question_never_calls_llm(retriever, prediction):
    c = client_saying("Paris")
    r = chat_followup("What is the capital of France?", prediction, retriever, client=c)
    assert r["status"] == "no_sources" and r["notice"] == OFF_TOPIC_MSG
    c.chat.assert_not_called()


def test_question_about_an_input_is_answered_without_sources(retriever, prediction):
    c = client_saying("You have 4 dependents; the model gives that little weight.")
    r = chat_followup("Does having more dependents hurt me?", prediction, retriever, client=c)
    assert r["status"] == "ok" and r["sources"] == []


def test_uncited_answer_is_regenerated_then_rejected_and_ollama_down(retriever, prediction):
    c = client_saying("No citation here.")
    r = chat_followup("How can I improve my credit score?", prediction, retriever, client=c)
    assert r["status"] == "ungrounded" and c.chat.call_count == 2
    down = MagicMock(); down.chat.side_effect = ConnectionError
    r = chat_followup("How can I improve my credit score?", prediction, retriever, client=down)
    assert r["status"] == "ollama_unavailable" and r["notice"] == OLLAMA_DOWN_MSG and r["sources"]
    assert chat_followup("   ", prediction, retriever)["status"] == "empty"
