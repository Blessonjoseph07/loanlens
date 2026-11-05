"""LoanLens - Streamlit user interface.

Run with:  streamlit run app.py

Layout
  Sidebar : applicant form (11 fields, enforced ranges) + sample applicants
  Tab 1   : Prediction - result card, SHAP chart, AI explanation, sources, follow-up chat, What if?
  Tab 2   : Plan - "What would it take?" smallest changes that flip a rejection,
            and the credit-score time machine (habits simulated month by month)
  Tab 3   : Model performance - metrics table, confusion matrix, feature importance
  Tab 4   : About - problem, dataset, ML -> GenAI flow diagram, limitations
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loanlens.counterfactual import apply_plan, what_would_it_take
from loanlens.data import (FEATURES, LABELS, RANGES, RUPEE_FIELDS, ValidationError,
                           format_inr, format_value)
from loanlens.llm import OLLAMA_HOST, OLLAMA_MODEL, analyse, chat_followup, ollama_reachable, explain_plan, plan_query
from loanlens.predict import Predictor
from loanlens.timeline import HABITS, first_month, simulate
from loanlens.rag import NO_SOURCE_MSG, Retriever, THRESHOLD, build_query, load_embedder

ROOT = Path(__file__).resolve().parent
GREEN, RED, AMBER, ACCENT = "#059669", "#DC2626", "#D97706", "#4F46E5"

st.set_page_config(page_title="LoanLens", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------
# Styling: small CSS layer on top of the theme in .streamlit/config.toml
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1250px; }}
  .ll-header {{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.75rem;
               padding-bottom:1rem; border-bottom:1px solid #E5E7EB; margin-bottom:1rem; }}
  .ll-title {{ font-size:2.1rem; font-weight:800; letter-spacing:-.02em; margin:0; color:#111827; }}
  .ll-title span {{ color:{ACCENT}; }}
  .ll-sub {{ color:#4B5563; margin:.15rem 0 0 0; font-size:1rem; }}
  .ll-pill {{ display:inline-block; padding:.3rem .7rem; border-radius:999px; font-size:.78rem; font-weight:600;
             border:1px solid #E5E7EB; background:#F9FAFB; color:#374151; margin-left:.35rem; }}
  .ll-pill.warn {{ background:#FFFBEB; border-color:#FDE68A; color:#92400E; }}
  .ll-pill.ok {{ background:#ECFDF5; border-color:#A7F3D0; color:#065F46; }}
  .ll-pill.off {{ background:#FEF2F2; border-color:#FECACA; color:#991B1B; }}
  .ll-card {{ background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:1.1rem 1.25rem;
             box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom:1rem; }}
  .ll-card h4 {{ margin:0 0 .6rem 0; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; color:#6B7280; }}
  .ll-badge {{ display:inline-block; font-size:2rem; font-weight:800; padding:.35rem 1.1rem; border-radius:12px; color:white; }}
  .ll-meta {{ color:#4B5563; margin-top:.7rem; font-size:.95rem; line-height:1.6; }}
  .ll-band {{ font-weight:700; }}
  .ll-explain {{ font-size:1rem; line-height:1.7; color:#1F2937; }}
  .ll-explain sup a {{ color:{ACCENT}; text-decoration:none; font-weight:700; }}
  .ll-source {{ border-left:3px solid {ACCENT}; padding:.4rem 0 .4rem .9rem; margin-bottom:1rem; }}
  .ll-source .score {{ font-size:.8rem; color:#6B7280; }}
  .ll-source .txt {{ font-size:.9rem; color:#374151; white-space:pre-wrap; }}
  .ll-empty {{ text-align:center; padding:2.5rem 1rem; color:#6B7280; }}
  .ll-empty b {{ color:#111827; font-size:1.15rem; }}
  .ll-small {{ font-size:.82rem; color:#6B7280; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached resources: loaded once per server, shared by every user/rerun
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the Random Forest model...")
def get_predictor():
    return Predictor(ROOT / "model.joblib")


@st.cache_resource(show_spinner="Loading the sentence-transformers embedder...")
def get_embedder():
    return load_embedder()


@st.cache_resource(show_spinner="Loading the FAISS index...")
def get_retriever():
    return Retriever(get_embedder(), ROOT / "index")


@st.cache_data
def load_metrics():
    path = ROOT / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(ttl=15, show_spinner=False)
def ollama_online() -> bool:
    """Quick health check so the header can show whether the LLM is reachable."""
    if not ollama_reachable():
        return False
    try:
        import ollama
        ollama.Client(host=OLLAMA_HOST, timeout=2).list()
        return True
    except Exception:
        return False


def safe(loader):
    """Call a cached loader; return (object, error message) and never raise."""
    try:
        return loader(), None
    except FileNotFoundError as e:
        return None, f"Missing file: {Path(e.filename or str(e)).name}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Helpers for the rupee inputs (thousand separators, Indian grouping)
# --------------------------------------------------------------------------
def rupee_text(value) -> str:
    return f"₹ {format_inr(value)}"


def parse_rupees(text: str):
    """'₹ 50,00,000' -> 5000000. Returns None if it is not a whole number."""
    cleaned = re.sub(r"[₹,\s]|Rs\.?", "", str(text))
    if cleaned.startswith("-") and cleaned[1:].isdigit():
        return -int(cleaned[1:])
    return int(cleaned) if cleaned.isdigit() else None


METRICS = load_metrics()
SAMPLES = (METRICS or {}).get("sample_applicants", {})
DEFAULT = {"no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
           "income_annum": 5_000_000, "loan_amount": 12_000_000, "loan_term": 10,
           "cibil_score": 720, "residential_assets_value": 4_000_000,
           "commercial_assets_value": 2_000_000, "luxury_assets_value": 8_000_000,
           "bank_asset_value": 3_000_000}


def set_form(values: dict):
    """Write applicant values into the form widgets' session-state keys."""
    for f in FEATURES:
        v = values[f]
        st.session_state[f"in_{f}"] = rupee_text(v) if f in RUPEE_FIELDS else v


def on_sample_change():
    name = st.session_state["sample_choice"]
    if name in SAMPLES:
        set_form(SAMPLES[name])
        st.session_state["sample_loaded"] = name
        # predict straight away on the sample's exact values
        st.session_state["pending_applicant"] = {f: SAMPLES[name][f] for f in FEATURES}
        st.session_state["run_predict"] = True


def on_submit():
    """Runs before the rerun: re-format rupee fields and ask for a prediction."""
    for f in RUPEE_FIELDS:
        n = parse_rupees(st.session_state.get(f"in_{f}", ""))
        if n is not None and n >= 0:
            st.session_state[f"in_{f}"] = rupee_text(n)
    st.session_state["run_predict"] = True
    st.session_state.pop("pending_applicant", None)
    st.session_state["sample_loaded"] = None


def try_plan(applicant: dict, plan: dict):
    """Load a "What would it take?" plan into the form and predict it."""
    changed = apply_plan(applicant, plan)
    set_form(changed)
    st.session_state["pending_applicant"] = changed
    st.session_state["run_predict"] = True
    st.session_state["sample_loaded"] = None


if "in_cibil_score" not in st.session_state:
    set_form(DEFAULT)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
online = ollama_online()
st.markdown(f"""
<div class="ll-header">
  <div>
    <p class="ll-title">Loan<span>Lens</span></p>
    <p class="ll-sub">Predicts loan approval with a Random Forest, then explains the result with a local LLM grounded in RBI and CIBIL sources.</p>
  </div>
  <div>
    <span class="ll-pill {'ok' if online else 'off'}">{'● Ollama online' if online else '● Ollama offline'} · {OLLAMA_MODEL}</span>
    <span class="ll-pill warn">Educational tool, not a lending decision</span>
  </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar: applicant form
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Applicant details")
    if SAMPLES:
        st.selectbox("Load sample applicant (real test-set rows)",
                     ["- choose -"] + list(SAMPLES), key="sample_choice", on_change=on_sample_change)
    with st.form("applicant_form", border=False):
        st.number_input("Number of dependents", min_value=0, max_value=5, step=1, key="in_no_of_dependents")
        c1, c2 = st.columns(2)
        c1.selectbox("Education", RANGES["education"], key="in_education")
        c2.selectbox("Self-employed", RANGES["self_employed"], key="in_self_employed")
        st.text_input("Annual income (₹)", key="in_income_annum")
        st.text_input("Loan amount (₹)", key="in_loan_amount")
        c3, c4 = st.columns(2)
        c3.number_input("Loan term (years)", min_value=2, max_value=20, step=1, key="in_loan_term")
        c4.number_input("CIBIL score", min_value=300, max_value=900, step=1, key="in_cibil_score")
        st.markdown("**Assets (₹)**")
        c5, c6 = st.columns(2)
        c5.text_input("Residential", key="in_residential_assets_value")
        c6.text_input("Commercial", key="in_commercial_assets_value")
        c7, c8 = st.columns(2)
        c7.text_input("Luxury", key="in_luxury_assets_value")
        c8.text_input("Bank", key="in_bank_asset_value")
        st.form_submit_button("Predict", type="primary", width="stretch", on_click=on_submit)
    st.caption("Ranges: dependents 0-5 · term 2-20 years · CIBIL 300-900 · amounts in rupees, assets ≥ 0.")


def read_form() -> dict:
    """Collect the form into an applicant dict (rupee text -> int)."""
    applicant = {}
    for f in FEATURES:
        v = st.session_state[f"in_{f}"]
        if f in RUPEE_FIELDS:
            n = parse_rupees(v)
            applicant[f] = n if n is not None else v       # bad text -> validation error
        else:
            applicant[f] = v
    return applicant


# --------------------------------------------------------------------------
# Run the pipeline (only when Predict was pressed)
# --------------------------------------------------------------------------
predictor, model_error = safe(get_predictor)
retriever, index_error = safe(get_retriever)


def run_analysis(applicant: dict):
    """Predict -> retrieve -> explain. Returns (result, error message)."""
    if predictor is None:
        return None, f"The model could not be loaded ({model_error}). Run: python train.py loan_approval_dataset.csv"
    try:
        if retriever is None:   # model works but RAG index is missing: still show the prediction
            pred = predictor.predict(applicant)
            expl = {"status": "no_index", "text": "", "attempts": 0,
                    "notice": f"RAG index not available ({index_error}). Run: python build_index.py"}
            return {"prediction": pred, "query": build_query(pred), "sources": [], "explanation": expl}, None
        return analyse(applicant, predictor, retriever), None
    except ValidationError as e:
        return None, f"Please fix the form: {e}"
    except Exception as e:  # never show a traceback to the user
        return None, f"Something went wrong while analysing this applicant ({type(e).__name__})."


if st.session_state.pop("run_predict", False):
    with st.spinner("Scoring the applicant, retrieving sources and asking llama3.2 for an explanation..."):
        applicant = st.session_state.pop("pending_applicant", None) or read_form()
        result, error = run_analysis(applicant)
    st.session_state["result"], st.session_state["error"] = result, error
    st.session_state.pop("whatif", None)
    for k in ("cf", "cf_expl", "chat"):      # plans and chat belong to the previous applicant
        st.session_state.pop(k, None)
    if result:   # reset the What-if sliders to this applicant
        st.session_state["wi_cibil"] = result["prediction"]["applicant"]["cibil_score"]
        st.session_state["wi_term"] = result["prediction"]["applicant"]["loan_term"]
        st.session_state["result_sample"] = st.session_state.get("sample_loaded")
    st.session_state.pop("sample_loaded", None)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def colour_for(pred: dict) -> str:
    return GREEN if pred["label"] == "Approved" else RED


def gauge(p: float, height: int = 230) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Probability of approval", "font": {"size": 14, "color": "#6B7280"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9CA3AF"},
               "bar": {"color": GREEN if p >= 0.5 else RED, "thickness": 0.3},
               "bgcolor": "white", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "#FEE2E2"},
                         {"range": [40, 60], "color": "#FEF3C7"},
                         {"range": [60, 100], "color": "#D1FAE5"}],
               "threshold": {"line": {"color": "#111827", "width": 3}, "thickness": 0.8, "value": 50}}))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def shap_chart(factors: list[dict]) -> go.Figure:
    rows = list(reversed(factors))   # biggest factor on top
    labels = [f"{f['name']}<br><span style='font-size:11px;color:#6B7280'>{format_value(f['feature'], f['value'])}</span>"
              for f in rows]
    fig = go.Figure(go.Bar(
        x=[f["shap"] for f in rows], y=labels, orientation="h",
        marker_color=[GREEN if f["shap"] >= 0 else RED for f in rows],
        text=[f"{f['shap']:+.3f}" for f in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{x:+.3f}<extra></extra>"))
    span = max(abs(f["shap"]) for f in factors) * 1.35 or 0.1
    fig.update_layout(height=260, margin=dict(l=10, r=30, t=10, b=40), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      xaxis=dict(range=[-span, span], zeroline=True, zerolinecolor="#9CA3AF",
                                 gridcolor="#F3F4F6", title="SHAP contribution to approval probability"))
    return fig


def render_answer(text: str) -> str:
    """Escape the LLM text, then render **bold**, bullets and [n] as superscript links."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"^\s*[-*•]\s+", "• ", out, flags=re.M)
    out = re.sub(r"\[(\d+)\]", r'<sup><a href="#src-\1">[\1]</a></sup>', out)
    return out.replace("\n", "<br>")


def result_card(pred: dict, sample_name: str | None):
    colour = colour_for(pred)
    band_colour = AMBER if pred["band"] == "borderline" else "#374151"
    actual = ""
    if sample_name and sample_name in SAMPLES:
        actual = (f"<br>Loaded sample: <b>{html.escape(sample_name)}</b> · "
                  f"actual outcome in dataset: <b>{SAMPLES[sample_name]['actual']}</b>")
    st.markdown(f"""
    <div class="ll-card">
      <h4>Predicted result</h4>
      <span class="ll-badge" style="background:{colour}">{pred['label']}</span>
      <div class="ll-meta">
        Probability of approval: <b>{pred['probability']:.1%}</b><br>
        Confidence band: <span class="ll-band" style="color:{band_colour}">{pred['band'].title()}</span>
        <span class="ll-small">(clear if &lt; 40% or &gt; 60%, otherwise borderline)</span>{actual}
      </div>
    </div>""", unsafe_allow_html=True)


def explanation_block(result: dict, key: str):
    """AI explanation card + Sources expander (shared by main result and What-if)."""
    expl, sources = result["explanation"], result["sources"]
    status = expl["status"]
    if status == "ok":
        retry = " · regenerated once after a failed grounding check" if expl["attempts"] == 2 else ""
        st.markdown(f"""
        <div class="ll-card">
          <h4>AI explanation · {OLLAMA_MODEL} via Ollama{retry}</h4>
          <div class="ll-explain">{render_answer(expl['text'])}</div>
        </div>""", unsafe_allow_html=True)
    elif status == "no_sources":
        st.markdown(f"""
        <div class="ll-card"><h4>AI explanation</h4>
          <div class="ll-empty"><b>{NO_SOURCE_MSG}</b><br>
          No knowledge-base passage reached the {THRESHOLD} similarity threshold, so the LLM was not called.</div>
        </div>""", unsafe_allow_html=True)
    elif status == "ungrounded":
        st.warning(expl["notice"])
    else:   # ollama_unavailable or no_index
        st.warning(expl["notice"] + (". The prediction and the retrieved passages are still shown below."
                                     if status == "ollama_unavailable" else ""))

    with st.expander(f"Sources ({len(sources)} retrieved)", expanded=status != "ok"):
        st.markdown(f"<p class='ll-small'>Retrieval query built from the ML output: "
                    f"<i>“{html.escape(result['query'])}”</i></p>", unsafe_allow_html=True)
        if not sources:
            st.markdown(f"**{NO_SOURCE_MSG}**")
        for i, s in enumerate(sources, 1):
            st.markdown(f"""
            <div class="ll-source" id="src-{i}">
              <b>[{i}] {html.escape(s['title'])}</b>
              <span class="score"> · cosine similarity {s['score']:.3f}</span><br>
              <a href="{html.escape(s['source'])}" target="_blank">{html.escape(s['source'])}</a>
              <div class="txt">{html.escape(s['text'][:1200])}{'…' if len(s['text']) > 1200 else ''}</div>
            </div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_pred, tab_plan, tab_perf, tab_about = st.tabs(["Prediction", "Plan", "Model performance", "About"])

with tab_pred:
    if model_error:
        st.error(f"Model not available ({model_error}). Run `python train.py loan_approval_dataset.csv` first.")
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    result = st.session_state.get("result")

    if not result:
        st.markdown("""
        <div class="ll-card ll-empty">
          <b>Fill in the applicant form and press Predict</b><br>
          or pick a real test-set applicant from <i>Load sample applicant</i> in the sidebar.<br><br>
          LoanLens will show the Random Forest prediction, the three factors that moved it most (SHAP),
          and a short explanation written by a local LLM that cites RBI and CIBIL pages.
        </div>""", unsafe_allow_html=True)
    else:
        pred = result["prediction"]
        c1, c2 = st.columns([1, 1.1], gap="large")
        with c1:
            result_card(pred, st.session_state.get("result_sample"))
        with c2:
            st.plotly_chart(gauge(pred["probability"]), width="stretch",
                            config={"displayModeBar": False}, key="gauge_main")

        st.markdown("#### Top 3 factors for this applicant")
        c3, c4 = st.columns([1.6, 1], gap="large")
        with c3:
            st.plotly_chart(shap_chart(pred["factors"]), width="stretch",
                            config={"displayModeBar": False}, key="shap_main")
        with c4:
            for f in pred["factors"]:
                arrow = "▲" if f["shap"] >= 0 else "▼"
                colour = GREEN if f["shap"] >= 0 else RED
                st.markdown(f"<p style='margin:.35rem 0'><span style='color:{colour}'>{arrow}</span> "
                            f"<b>{f['name']}</b> = {format_value(f['feature'], f['value'])}<br>"
                            f"<span class='ll-small'>pushes {f['direction']} ({f['shap']:+.3f})</span></p>",
                            unsafe_allow_html=True)
            st.caption("SHAP values show how much each input moved this applicant's approval "
                       "probability away from the average prediction.")

        explanation_block(result, "main")

        # ---------------- Chat with the decision ----------------
        st.markdown("#### Ask about this decision")
        st.caption("Follow-up questions are answered from this applicant's numbers and the RBI/CIBIL pages, "
                   "with citations. Off-topic questions are refused without calling the LLM.")
        chat = st.session_state.setdefault("chat", [])

        def ask(question: str):
            history = [m for m in st.session_state["chat"] if m.get("content")]
            reply = chat_followup(question, st.session_state["result"]["prediction"], retriever, history)
            if reply["status"] == "empty":
                return
            st.session_state["chat"] += [{"role": "user", "content": question},
                                         {"role": "assistant", "content": reply["text"],
                                          "notice": reply["notice"], "sources": reply["sources"]}]

        for m in chat:
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    st.markdown(html.escape(m["content"]))
                    continue
                if m["content"]:
                    st.markdown(f"<div class='ll-explain'>{render_answer(m['content'])}</div>",
                                unsafe_allow_html=True)
                if m.get("notice"):
                    st.warning(m["notice"])
                if m.get("sources"):
                    st.markdown("<p class='ll-small'>" + " · ".join(
                        f"[{i}] <a href='{html.escape(s['source'])}' target='_blank'>{html.escape(s['title'])}</a>"
                        f" ({s['score']:.2f})" for i, s in enumerate(m["sources"], 1)) + "</p>",
                        unsafe_allow_html=True)

        suggestions = ["Why does the loan term matter?", "How can I improve my CIBIL score?",
                       "Which of my inputs matters most?"]
        sc = st.columns(len(suggestions))
        for col, s in zip(sc, suggestions):
            col.button(s, key=f"sugg_{s}", on_click=ask, args=(s,), width="stretch")
        with st.form("chat_form", clear_on_submit=True, border=False):
            q1, q2 = st.columns([5, 1], vertical_alignment="bottom")
            q1.text_input("Your question", key="chat_q", placeholder="e.g. Would a shorter tenure help me?")
            if q2.form_submit_button("Ask", width="stretch"):
                ask(st.session_state.get("chat_q", ""))
                st.rerun()

        # ---------------- What if? ----------------
        st.markdown("#### What if?")
        st.caption("Change the CIBIL score or loan term, keep everything else the same, and re-run the "
                   "prediction and explanation.")
        w1, w2, w3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        w1.slider("CIBIL score", 300, 900, key="wi_cibil")
        w2.slider("Loan term (years)", 2, 20, key="wi_term")
        if w3.button("Run what-if", width="stretch"):
            changed = {**pred["applicant"], "cibil_score": st.session_state["wi_cibil"],
                       "loan_term": st.session_state["wi_term"]}
            with st.spinner("Re-running prediction and explanation..."):
                st.session_state["whatif"], st.session_state["whatif_error"] = run_analysis(changed)

        if st.session_state.get("whatif_error"):
            st.error(st.session_state["whatif_error"])
        wi = st.session_state.get("whatif")
        if wi:
            new = wi["prediction"]
            o1, o2 = st.columns(2, gap="large")
            with o1:
                st.markdown(f"""<div class="ll-card"><h4>Original</h4>
                  CIBIL {pred['applicant']['cibil_score']} · term {pred['applicant']['loan_term']} years<br>
                  <span class="ll-badge" style="background:{colour_for(pred)};font-size:1.3rem">{pred['label']}</span>
                  <div class="ll-meta">Probability of approval <b>{pred['probability']:.1%}</b> · {pred['band']}</div>
                </div>""", unsafe_allow_html=True)
            with o2:
                delta = new["probability"] - pred["probability"]
                st.markdown(f"""<div class="ll-card"><h4>What if</h4>
                  CIBIL {new['applicant']['cibil_score']} · term {new['applicant']['loan_term']} years<br>
                  <span class="ll-badge" style="background:{colour_for(new)};font-size:1.3rem">{new['label']}</span>
                  <div class="ll-meta">Probability of approval <b>{new['probability']:.1%}</b> · {new['band']}
                  · <b style="color:{GREEN if delta >= 0 else RED}">{delta:+.1%}</b></div>
                </div>""", unsafe_allow_html=True)
            st.plotly_chart(shap_chart(new["factors"]), width="stretch",
                            config={"displayModeBar": False}, key="shap_whatif")
            explanation_block(wi, "whatif")

with tab_plan:
    result = st.session_state.get("result")
    if not result:
        st.markdown("""<div class="ll-card ll-empty"><b>Run a prediction first</b><br>
          The Plan tab searches for the smallest realistic changes that would turn this applicant into a
          clear approval (probability ≥ 60%).</div>""", unsafe_allow_html=True)
    else:
        pred = result["prediction"]
        if "cf" not in st.session_state:
            with st.spinner("Searching a few thousand what-if variations..."):
                st.session_state["cf"] = what_would_it_take(predictor.model, pred["applicant"])
        cf = st.session_state["cf"]
        st.markdown("#### What would it take?")
        st.caption("Every combination of the inputs a person can actually change (CIBIL score, loan amount, "
                   "income, loan term, bank savings) is scored by the model; the cheapest plans that reach a "
                   "60% approval probability are shown. Effort scale: 100 CIBIL points ≈ 20% smaller loan ≈ "
                   "20% more income ≈ 5 years of term ≈ savings worth 10% of the loan.")
        if cf["already"]:
            st.markdown(f"""<div class="ll-card"><h4>Nothing to change</h4>
              <span class="ll-badge" style="background:{GREEN};font-size:1.3rem">Clear approval</span>
              <div class="ll-meta">This applicant is already at <b>{cf['current']:.1%}</b>, above the
              {cf['target']:.0%} target.</div></div>""", unsafe_allow_html=True)
        elif not cf["plans"]:
            st.warning("No combination of realistic changes reaches a 60% approval probability for this applicant.")
        else:
            cols = st.columns(min(len(cf["plans"]), 2), gap="large")
            for i, plan in enumerate(cf["plans"]):
                with cols[i % len(cols)]:
                    steps = "".join(f"<li>{html.escape(s['text'])}</li>" for s in plan["changes"])
                    st.markdown(f"""<div class="ll-card"><h4>Plan {i + 1} · effort {plan['effort']:.1f}</h4>
                      <ul style="margin:.2rem 0 .6rem 1.1rem;padding:0">{steps}</ul>
                      <div class="ll-meta">Approval probability <b>{cf['current']:.1%}</b> →
                      <b style="color:{GREEN}">{plan['probability']:.1%}</b></div></div>""",
                                unsafe_allow_html=True)
                    st.button("Try this plan", key=f"try_plan_{i}", on_click=try_plan,
                              args=(pred["applicant"], plan))
            if all(any(s["feature"] == "cibil_score" for s in p["changes"]) for p in cf["plans"]):
                st.info(f"Every plan needs a higher CIBIL score. Even with a 60% smaller loan, 60% more income, "
                        f"30% extra savings and the best loan term, this model only reaches "
                        f"**{cf['best_without_cibil']:.1%}** without touching CIBIL — another sign of how "
                        "much it leans on that one number.")

            if st.button("Explain plan 1 with AI", type="primary"):
                with st.spinner("Retrieving sources and asking llama3.2..."):
                    q = plan_query(cf["plans"][0])
                    srcs = retriever.search(q) if retriever else []
                    st.session_state["cf_expl"] = {"query": q, "sources": srcs,
                                                   "explanation": explain_plan(pred["applicant"], cf, srcs)}
            if st.session_state.get("cf_expl"):
                explanation_block(st.session_state["cf_expl"], "plan")

        # ---------------- Credit-score time machine ----------------
        st.markdown("#### Credit-score time machine")
        st.caption("Pick habits and a monthly saving; the CIBIL score and bank savings move forward one month at "
                   "a time and the model re-scores the applicant at every step. The point values are "
                   "illustrative assumptions (the real CIBIL formula is not public) - edit them below.")
        t1, t2, t3 = st.columns([2.2, 1, 1], vertical_alignment="bottom")
        labels = {h.label: h.key for h in HABITS}
        chosen = t1.multiselect("Habits", list(labels), default=list(labels)[:3], key="tm_habits")
        months = t2.slider("Months", 6, 48, 24, key="tm_months")
        saving = t3.number_input("Monthly saving (₹)", 0, 10_000_000, 0, step=10_000, key="tm_saving")
        with st.expander("Assumptions (CIBIL points)"):
            edited = st.data_editor(
                pd.DataFrame({"Habit": [h.label for h in HABITS], "Points": [h.points for h in HABITS],
                              "When": ["every month" if h.month is None else f"once, month {h.month}"
                                       for h in HABITS]}),
                disabled=["Habit", "When"], hide_index=True, width="stretch", key="tm_points")
            st.caption("Positive effects shrink as the score nears 900 (diminishing returns).")
        pts = {h.key: float(v) for h, v in zip(HABITS, edited["Points"])}
        sim = simulate(predictor.model, pred["applicant"], [labels[c] for c in chosen], months, int(saving), pts)
        m50, m60 = first_month(sim, 0.5), first_month(sim, 0.6)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim["month"], y=sim["probability"] * 100, name="Approval probability",
                                 mode="lines+markers", line=dict(color=ACCENT, width=3), marker=dict(size=5),
                                 hovertemplate="Month %{x}: %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=sim["month"], y=sim["cibil_score"], name="CIBIL score", yaxis="y2",
                                 mode="lines", line=dict(color="#9CA3AF", dash="dot"),
                                 hovertemplate="Month %{x}: CIBIL %{y}<extra></extra>"))
        for level, colour in ((50, AMBER), (60, GREEN)):
            fig.add_hline(y=level, line=dict(color=colour, dash="dash", width=1),
                          annotation_text=f"{level}%", annotation_position="top right")
        if m60 is not None and m60 > 0:
            fig.add_vline(x=m60, line=dict(color=GREEN, width=1))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=40), plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1),
                          xaxis=dict(title="Month", gridcolor="#F3F4F6"),
                          yaxis=dict(title="Approval probability (%)", range=[0, 102], gridcolor="#F3F4F6"),
                          yaxis2=dict(title="CIBIL score", overlaying="y", side="right", range=[300, 900],
                                      tickvals=[300, 450, 550, 650, 750, 900], showgrid=False))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="time_machine")
        end = sim.iloc[-1]
        if m60 == 0:
            msg = "Already a clear approval today."
        elif m60 is not None:
            row = sim[sim["month"] == m60].iloc[0]
            msg = (f"Crosses a **clear approval (60%) in month {m60}**, when the simulated CIBIL score reaches "
                   f"**{int(row['cibil_score'])}**.")
        else:
            msg = (f"Does **not** reach 60% within {months} months (ends at {end['probability']:.1%}, "
                   f"CIBIL {int(end['cibil_score'])}). Try more months or other habits.")
        st.markdown(msg)

with tab_perf:
    if not METRICS:
        st.info("metrics.json not found. Run `python train.py loan_approval_dataset.csv` first.")
    else:
        models = ["Majority class (baseline)", "Logistic Regression", "Random Forest"]
        table = pd.DataFrame({
            m: {"Accuracy": METRICS[m]["accuracy"], "Precision": METRICS[m]["precision"],
                "Recall": METRICS[m]["recall"], "F1": METRICS[m]["f1"], "ROC-AUC": METRICS[m]["roc_auc"],
                "CV F1 (5-fold)": f"{METRICS[m]['cv_f1_mean']:.3f} ± {METRICS[m]['cv_f1_std']:.3f}"}
            for m in models}).T
        st.markdown("#### Model comparison (test set, 854 applicants)")
        st.dataframe(table.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]})
                     .highlight_max(subset=["Accuracy", "F1", "ROC-AUC"], color="#E0E7FF"),
                     width="stretch")
        rf, lr, base = (METRICS[m] for m in reversed(models))
        st.caption(f"Interpretation: the Random Forest reaches {rf['accuracy']:.1%} accuracy, well above Logistic "
                   f"Regression ({lr['accuracy']:.1%}) and the majority-class baseline ({base['accuracy']:.1%}), "
                   "because trees can learn the sharp CIBIL cut-off that a linear model cannot.")

        cm = rf["confusion_matrix"]
        fi = METRICS["feature_importance"]
        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.markdown("#### Confusion matrix (Random Forest)")
            st.image(str(ROOT / "confusion_matrix.png"), width="stretch")
            st.caption(f"Interpretation: of {sum(map(sum, cm))} test applicants only {cm[0][1]} rejected loans were "
                       f"predicted as approved (false positives) and {cm[1][0]} approved loans were predicted as "
                       "rejected (false negatives).")
        with p2:
            st.markdown("#### Feature importance (Random Forest)")
            st.image(str(ROOT / "feature_importance.png"), width="stretch")
            st.caption(f"Interpretation: CIBIL score alone accounts for {fi['cibil_score']:.1%} of the forest's "
                       f"importance; loan term is a distant second at {fi['loan_term']:.1%}.")

        st.markdown("#### Same applicant, different CIBIL score")
        sweep = METRICS.get("cibil_sweep", {})
        st.dataframe(pd.DataFrame({"CIBIL score": list(sweep), "Probability of approval": list(sweep.values())}),
                     hide_index=True)
        st.caption("Interpretation: holding every other input fixed, the approval probability jumps once CIBIL "
                   "passes roughly 550, which shows how strongly the model depends on this one feature.")

with tab_about:
    a1, a2 = st.columns([1.15, 1], gap="large")
    with a1:
        st.markdown("""
#### Problem
Banks receive many loan applications and must decide which ones to approve. LoanLens predicts whether an
application is likely to be **approved or rejected** from 11 applicant details, and then explains the
prediction in plain language so anyone can understand *why*.

#### Dataset
[Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
by Archit Sharma on Kaggle: 4,269 applications, 11 input features (dependents, education, self-employment,
income, loan amount, loan term, CIBIL score and four asset values) and the target `loan_status`.

#### How ML and GenAI are connected
1. **ML:** a Random Forest (200 trees) predicts the label and probability; SHAP finds the
   three inputs that moved *this* applicant's prediction most.
2. **Retrieval:** a query is built **from the ML output** (label, CIBIL score, top factors), embedded with
   all-MiniLM-L6-v2 and searched in a FAISS index of RBI and CIBIL credit-awareness pages.
   Only passages with cosine similarity ≥ 0.35 are kept.
3. **GenAI:** llama3.2 running locally in Ollama writes a short, cited explanation using only
   the applicant, prediction and sources. A grounding check verifies the label and a [n] citation.
""")
    with a2:
        st.graphviz_chart("""
digraph {
  rankdir=TB; bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#C7D2FE" fillcolor="#EEF2FF"];
  edge [color="#6B7280" fontname="Helvetica" fontsize=9];
  form  [label="Applicant form\\n(11 fields, validated)"];
  rf    [label="Random Forest\\nlabel + probability" fillcolor="#E0E7FF"];
  shap  [label="SHAP\\ntop-3 factors" fillcolor="#E0E7FF"];
  query [label="Query built from\\nML output"];
  faiss [label="FAISS search\\n(MiniLM, cos >= 0.35)"];
  kb    [label="RBI + CIBIL pages" shape=cylinder fillcolor="#F3F4F6" color="#D1D5DB"];
  gate  [label="Any source?" shape=diamond fillcolor="#FEF3C7" color="#FDE68A"];
  none  [label="No supporting\\nsource found" fillcolor="#FEE2E2" color="#FECACA"];
  llm   [label="llama3.2 (Ollama)\\ncited explanation" fillcolor="#D1FAE5" color="#A7F3D0"];
  check [label="Grounding check\\n(label + [n])"];
  form -> rf -> shap -> query -> faiss; kb -> faiss; faiss -> gate;
  gate -> none [label="no"]; gate -> llm [label="yes"]; llm -> check;
}
""", width="stretch")

    st.markdown("""
#### Limitations
- **CIBIL score dominates:** it carries **80.7%** of the Random Forest's feature importance, so the model is
  close to a CIBIL threshold rule and the 98% accuracy partly reflects that one feature.
- **Dataset is likely synthetic:** values such as loan term and asset amounts look generated, so results may
  not transfer to real Indian lending data.
- The explanation is only as good as the retrieved pages; the LLM can still phrase things imperfectly.
- **Learning project only.** LoanLens does not make lending decisions and does not give personal financial advice.
""")
    st.markdown("<p class='ll-small'>LoanLens · Random Forest + SHAP + FAISS RAG + Llama 3.2 (Ollama), all running locally</p>",
                unsafe_allow_html=True)
