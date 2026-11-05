"""Feature: downloadable PDF decision summary."""
import io

from loanlens.counterfactual import what_would_it_take
from loanlens.pdf_report import _latin1, build_pdf


def _result(pred, status="ok"):
    expl = ({"status": "ok", "text": "**Rejected** because the CIBIL score is low [1]. Costs ₹ 5,000 – “ok”.",
             "consistency": {"status": "partial", "missing": ["loan_term"], "extra": []}}
            if status == "ok" else {"status": "ollama_unavailable", "notice": "Explanation unavailable"})
    return {"prediction": pred, "explanation": expl, "query": "q",
            "sources": [{"title": "Loan Approval Process | CIBIL", "source": "https://www.cibil.com/x",
                         "text": "t", "score": 0.58}]}


def _text(pdf_bytes):
    import pypdf
    return " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages)


def test_pdf_contains_result_sources_and_plan(shared_predictor, samples):
    pred = shared_predictor.predict(samples["rejected"])
    cf = what_would_it_take(shared_predictor.model, samples["rejected"])
    pdf = build_pdf(_result(pred), {"level": "high", "reasons": []}, cf)
    assert pdf.startswith(b"%PDF")
    text = _text(pdf)
    for needle in ("LoanLens decision summary", "Rejected", "CIBIL score", "Loan Approval Process",
                   "What would it take", "Certainty: high", "Rs 5,000"):
        assert needle in text, needle


def test_pdf_without_llm_or_extras(shared_predictor, samples):
    pred = shared_predictor.predict(samples["approved"])
    text = _text(build_pdf(_result(pred, status="down")))
    assert "Approved" in text and "Explanation unavailable" in text and "What would it take" not in text


def test_latin1_cleanup():
    assert _latin1("₹ 5 – “x” → y ₹9") == "Rs 5 - \"x\" -> y Rs 9"
    assert _latin1("മലയാളം") == "??????"
