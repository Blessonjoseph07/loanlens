"""Generative-AI part of LoanLens: the local LLM explanation (Ollama, llama3.2).

explain() turns the ML prediction + retrieved sources into a short, cited,
educational explanation. Safety and reliability steps:

1. No sources passed the similarity threshold -> do NOT call the LLM,
   return "No supporting source found".
2. Grounding check: the answer must mention the predicted label and cite at
   least one source as [n]. If not, regenerate once; if it fails again, show
   only the retrieved passages with a warning.
3. Ollama not running -> catch the error, keep the prediction and passages,
   and show "Explanation unavailable: Ollama is not running".

analyse() at the bottom runs the full pipeline (predict -> retrieve -> explain)
and is what the Streamlit app and the tests call.
"""
from __future__ import annotations

import re

from .data import LABELS, format_value
from .rag import NO_SOURCE_MSG, build_query

OLLAMA_HOST = "http://localhost:11434"     # local only: no data leaves the PC
OLLAMA_MODEL = "llama3.2"
TEMPERATURE = 0.2                          # low = more factual, less creative

SYSTEM_PROMPT = """You are a credit-literacy tutor. Explain a machine learning prediction to a loan applicant.
Use ONLY the facts in APPLICANT, PREDICTION and SOURCES.
Rules:
- State the predicted result and its probability first.
- Explain the top 3 factors in simple words, one sentence each.
- Give general, educational points about how such factors usually affect eligibility.
- Cite sources as [1], [2], [3]. If SOURCES is empty, say no source was found.
- Do not promise approval, do not recommend a lender or product, and do not give personal financial advice.
- End with: "This is an educational estimate, not a lending decision.\""""

OLLAMA_DOWN_MSG = "Explanation unavailable: Ollama is not running"
UNGROUNDED_MSG = ("The AI explanation failed the grounding check twice "
                  "(it did not state the result or cite a source), so only the retrieved passages are shown.")

CITATION = re.compile(r"\[\d+\]")


def build_user_message(applicant: dict, prediction: dict, sources: list[dict]) -> str:
    """The user message: three labelled blocks APPLICANT, PREDICTION, SOURCES."""
    lines = ["APPLICANT"]
    lines += [f"- {LABELS[k]}: {format_value(k, v)}" for k, v in applicant.items()]
    lines += ["", "PREDICTION",
              f"- Result: {prediction['label']}",
              f"- Probability of approval: {prediction['probability']:.0%}",
              f"- Confidence band: {prediction['band']}",
              "- Top factors (from SHAP):"]
    lines += [f"  {i}. {f['name']} = {format_value(f['feature'], f['value'])}, pushes {f['direction']}"
              for i, f in enumerate(prediction["factors"], 1)]
    lines += ["", "SOURCES"]
    if sources:
        lines += [f"[{i}] {s['title']}: {s['text']}" for i, s in enumerate(sources, 1)]
    else:
        lines.append("(none)")
    return "\n".join(lines)


def is_grounded(answer: str, label: str) -> bool:
    """Answer must contain the predicted label and at least one [n] citation."""
    return label.lower() in answer.lower() and bool(CITATION.search(answer))


def make_client(host: str = OLLAMA_HOST):
    import ollama
    return ollama.Client(host=host, timeout=180)


def _chat(client, user_message: str) -> str:
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_message}],
        options={"temperature": TEMPERATURE},
    )
    # ollama>=0.4 returns an object; older versions return a dict.
    message = response["message"] if isinstance(response, dict) else response.message
    return (message["content"] if isinstance(message, dict) else message.content).strip()


def explain(applicant: dict, prediction: dict, sources: list[dict], client=None) -> dict:
    """Return {"status", "text", "notice", "attempts"}.

    status is one of: "ok", "no_sources", "ungrounded", "ollama_unavailable".
    """
    if not sources:                                   # rule 1: no evidence, no LLM call
        return {"status": "no_sources", "text": "", "notice": NO_SOURCE_MSG, "attempts": 0}

    user_message = build_user_message(applicant, prediction, sources)
    try:
        client = client or make_client()
        for attempt in (1, 2):                        # rule 2: at most one regeneration
            answer = _chat(client, user_message)
            if is_grounded(answer, prediction["label"]):
                return {"status": "ok", "text": answer, "notice": "", "attempts": attempt}
        return {"status": "ungrounded", "text": "", "notice": UNGROUNDED_MSG, "attempts": 2}
    except Exception as exc:                          # rule 3: Ollama down / unreachable
        notice = OLLAMA_DOWN_MSG
        if "not found" in str(exc).lower():           # server up, but model missing
            notice += f" (model '{OLLAMA_MODEL}' not found - run: ollama pull {OLLAMA_MODEL})"
        return {"status": "ollama_unavailable", "text": "", "notice": notice, "attempts": 0}


def analyse(applicant: dict, predictor, retriever, client=None) -> dict:
    """Full LoanLens pipeline used by the app and the tests.

    1. ML: predict label, probability, band and top-3 SHAP factors.
    2. Retrieval: build the query from that output and fetch supporting passages.
    3. GenAI: explain with the local LLM (or return a notice instead).
    The prediction and passages are always returned, even if step 3 fails.
    """
    prediction = predictor.predict(applicant)
    query = build_query(prediction)
    sources = retriever.search(query)
    explanation = explain(prediction["applicant"], prediction, sources, client=client)
    return {"prediction": prediction, "query": query, "sources": sources, "explanation": explanation}
