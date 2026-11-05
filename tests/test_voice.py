"""Feature: listen / translate the explanation."""
from unittest.mock import MagicMock

from loanlens.llm import OLLAMA_DOWN_MSG, translate
from loanlens.voice import LANGUAGES, plain_summary, speak_html, strip_for_speech


def client_saying(*texts):
    c = MagicMock()
    c.chat.side_effect = [{"message": {"content": t}} for t in texts]
    return c


def test_plain_summary_reads_the_prediction(shared_predictor, samples):
    pred = shared_predictor.predict(samples["rejected"])
    s = plain_summary(pred)
    assert s.startswith("The model predicts Rejected") and pred["factors"][0]["name"] in s
    assert "Rs " not in s                                   # spoken as "rupees"


def test_strip_for_speech_and_player():
    assert strip_for_speech("**Rejected** because of CIBIL [1].\n- pay on time [2]") == \
        "Rejected because of CIBIL .\npay on time"
    page = speak_html("Hello [1]", LANGUAGES["Malayalam"])
    assert '"ml-IN"' in page and "[1]" not in page and "speechSynthesis" in page


def test_translation_keeps_citations_or_falls_back():
    text = "Rejected because CIBIL is low [1]. Pay on time [2]."
    ok = client_saying("നിരസിച്ചു [1]. സമയത്ത് അടയ്ക്കുക [2].")
    r = translate(text, "Malayalam", client=ok)
    assert r["status"] == "ok" and "[2]" in r["text"]
    lossy = client_saying("നിരസിച്ചു.", "നിരസിച്ചു [1].")
    r = translate(text, "Malayalam", client=lossy)
    assert r["status"] == "lost_citations" and r["text"] == text and lossy.chat.call_count == 2
    down = MagicMock(); down.chat.side_effect = ConnectionError
    r = translate(text, "Hindi", client=down)
    assert r["status"] == "ollama_unavailable" and OLLAMA_DOWN_MSG in r["notice"]
    assert translate(text, "English")["status"] == "unchanged"
