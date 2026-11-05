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


def ollama_reachable(timeout: float = 0.5) -> bool:
    """Fast check that something listens on Ollama's port on this PC.

    Without it, a stopped Ollama costs ~5 s per call on Windows (the client
    tries IPv6 and IPv4 'localhost' before giving up).
    """
    import socket
    port = int(OLLAMA_HOST.rsplit(":", 1)[1])
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def make_client(host: str = OLLAMA_HOST):
    if not ollama_reachable():
        raise ConnectionError("Ollama is not running")
    import ollama
    return ollama.Client(host=host, timeout=180)


def _chat(client, user_message: str, system: str = SYSTEM_PROMPT, history: list[dict] | None = None) -> str:
    """One chat call. `history` is earlier {"role", "content"} turns (used by the follow-up chat)."""
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": system}, *(history or []),
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


PLAN_PROMPT = """You are a credit-literacy tutor. A loan model rejected an applicant, and a search found
the smallest changes that would make the model approve them (PLAN).
Use ONLY the facts in APPLICANT, PLAN and SOURCES.
Rules:
- Start with the first plan in one sentence.
- Give 2-3 practical, general steps for that change, each backed by a source cited as [1], [2] or [3].
- Mention honestly that the plan only moves this model's estimate; a real lender decides differently.
- Do not promise approval, do not recommend a lender or product, and do not give personal financial advice.
- Keep it under 140 words."""


def plan_query(plan: dict) -> str:
    """Retrieval query for a plan, e.g. 'how to improve CIBIL score for loan approval'."""
    names = [c["name"] for c in plan["changes"]]
    return "how to improve " + " and ".join(names) + " to get a loan approved"


def explain_plan(applicant: dict, cf: dict, sources: list[dict], client=None) -> dict:
    """LLM narrative for the "What would it take?" plans (same safety rules as explain())."""
    if not cf.get("plans"):
        return {"status": "no_plan", "text": "", "notice": "No plan to explain.", "attempts": 0}
    if not sources:
        return {"status": "no_sources", "text": "", "notice": NO_SOURCE_MSG, "attempts": 0}
    lines = ["APPLICANT"] + [f"- {LABELS[k]}: {format_value(k, v)}" for k, v in applicant.items()]
    lines += ["", "PLAN", f"- Current probability of approval: {cf['current']:.0%}, target {cf['target']:.0%}"]
    lines += [f"{i}. {p['summary']} -> probability {p['probability']:.0%}" for i, p in enumerate(cf["plans"], 1)]
    lines += ["", "SOURCES"] + [f"[{i}] {s['title']}: {s['text']}" for i, s in enumerate(sources, 1)]
    try:
        client = client or make_client()
        for attempt in (1, 2):
            answer = _chat(client, "\n".join(lines), system=PLAN_PROMPT)
            if CITATION.search(answer):
                return {"status": "ok", "text": answer, "notice": "", "attempts": attempt}
        return {"status": "ungrounded", "text": "", "notice": UNGROUNDED_MSG, "attempts": 2}
    except Exception:
        return {"status": "ollama_unavailable", "text": "", "notice": OLLAMA_DOWN_MSG, "attempts": 0}


CHAT_PROMPT = """You are a credit-literacy tutor answering follow-up questions about ONE loan prediction.
Use ONLY the facts in APPLICANT, PREDICTION and SOURCES (and the earlier conversation).
Rules:
- Answer the QUESTION in at most 120 words, using the applicant's own numbers where relevant.
- Cite sources as [1], [2], [3] when you use them. If SOURCES is empty, say the answer uses only the model's numbers.
- If the question cannot be answered from these facts, say so plainly instead of guessing.
- Do not promise approval, do not recommend a lender or product, and do not give personal financial advice."""

OFF_TOPIC_MSG = (NO_SOURCE_MSG + ". The question does not match the RBI/CIBIL pages or any of the model's "
                 "inputs, so the LLM was not asked.")
MAX_HISTORY = 6       # earlier chat messages sent back to the model


def chat_followup(question: str, prediction: dict, retriever, history: list[dict] | None = None,
                  client=None) -> dict:
    """Answer a follow-up question about the current prediction.

    Gate (no LLM call) unless the question either retrieves a passage with
    cosine >= threshold on its own, or is about one of the model's inputs
    (then it is answered from the applicant's numbers only).
    Returns {"status", "text", "notice", "sources", "query"}.
    """
    question = (question or "").strip()
    out = {"status": "empty", "text": "", "notice": "", "sources": [], "query": question}
    if not question:
        return out
    from .data import mentioned_features
    sources = retriever.search(question) if retriever else []
    out["sources"] = sources
    if not sources and not mentioned_features(question):
        return {**out, "status": "no_sources", "notice": OFF_TOPIC_MSG}

    message = build_user_message(prediction["applicant"], prediction, sources) + f"\n\nQUESTION\n{question}"
    past = [{"role": m["role"], "content": m["content"]} for m in (history or [])][-MAX_HISTORY:]
    try:
        client = client or make_client()
        for attempt in (1, 2):
            answer = _chat(client, message, system=CHAT_PROMPT, history=past)
            if not sources or CITATION.search(answer):     # cite when there is something to cite
                return {**out, "status": "ok", "text": answer, "attempts": attempt}
        return {**out, "status": "ungrounded", "notice": UNGROUNDED_MSG}
    except Exception:
        return {**out, "status": "ollama_unavailable", "notice": OLLAMA_DOWN_MSG}


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
