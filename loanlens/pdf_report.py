"""One-page PDF summary of a decision (fpdf2 + a matplotlib chart).

build_pdf() takes the same result dict the app shows (prediction, sources,
explanation) plus the optional certainty meter and "What would it take?"
plans, and returns the PDF as bytes for a download button or an API.

The core PDF fonts only cover Latin-1, so text is passed through _latin1()
(rupee sign -> "Rs", curly quotes -> straight quotes, and so on). The PDF is
always in English; translations stay in the app.
"""
from __future__ import annotations

import io
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402

from .data import LABELS, format_value  # noqa: E402

GREEN, RED, INDIGO, GREY = (5, 150, 105), (220, 38, 38), (79, 70, 229), (107, 114, 128)
_SUBS = {"₹ ": "Rs ", "₹": "Rs ", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "•": "-", "…": "...",
         "≥": ">=", "≤": "<=", "→": "->", "×": "x", " ": " "}


def _latin1(text: str) -> str:
    for a, b in _SUBS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _plain(text: str) -> str:
    """LLM markdown -> plain text (keeps [n] citations)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"^\s*[-*•]\s+", "- ", text, flags=re.M)


def shap_png(factors: list[dict]) -> bytes:
    rows = list(reversed(factors))
    fig, ax = plt.subplots(figsize=(6.2, 1.9), dpi=160)
    ax.barh([f"{f['name']} = {format_value(f['feature'], f['value'])}" for f in rows],
            [f["shap"] for f in rows], color=["#059669" if f["shap"] >= 0 else "#DC2626" for f in rows])
    ax.axvline(0, color="#9CA3AF", lw=0.8)
    span = max(abs(f["shap"]) for f in factors) * 1.3 or 0.1
    ax.set_xlim(-span, span)
    ax.set_xlabel("SHAP contribution to approval probability", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


class _PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 5, _latin1("LoanLens - an educational estimate, not a lending decision or financial advice. "
                                f"Page {self.page_no()}"), align="C")


def build_pdf(result: dict, certainty: dict | None = None, plans: dict | None = None) -> bytes:
    pred, expl, sources = result["prediction"], result["explanation"], result["sources"]
    pdf = _PDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(text):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*INDIGO)
        pdf.cell(0, 7, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    def para(text, size=9.5, style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 5, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 9, "LoanLens decision summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, _latin1(f"Generated {datetime.now():%d %b %Y, %H:%M} · Random Forest + SHAP + RAG"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # result badge + numbers
    colour = GREEN if pred["label"] == "Approved" else RED
    pdf.set_fill_color(*colour)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(42, 11, pred["label"], align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin + 48)
    line = f"Probability of approval: {pred['probability']:.1%}   ·   Confidence band: {pred['band']}"
    if certainty:
        line += f"   ·   Certainty: {certainty['level']}"
    pdf.cell(0, 11, _latin1(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if certainty and certainty.get("reasons"):
        para("Why: " + "; ".join(certainty["reasons"]) + ".", size=8.5, style="I")

    # applicant table
    heading("Applicant")
    pdf.set_font("Helvetica", "", 9)
    items = [(LABELS[k], format_value(k, v)) for k, v in pred["applicant"].items()]
    half = (len(items) + 1) // 2
    col = W / 4
    for i in range(half):
        for k, v in [items[i]] + ([items[i + half]] if i + half < len(items) else []):
            pdf.set_text_color(*GREY)
            pdf.cell(col, 5.5, _latin1(k))
            pdf.set_text_color(0, 0, 0)
            pdf.cell(col, 5.5, _latin1(v))
        pdf.ln(5.5)

    # SHAP
    heading("Top 3 factors (SHAP)")
    pdf.image(io.BytesIO(shap_png(pred["factors"])), w=W * 0.85)

    # explanation
    heading("Explanation")
    if expl.get("status") == "ok":
        para(_plain(expl["text"]))
        c = expl.get("consistency")
        if c:
            para(f"Mismatch alarm: {c['status']}"
                 + (f" (missing: {', '.join(LABELS[f] for f in c['missing'])})" if c["missing"] else ""),
                 size=8.5, style="I")
    else:
        para(expl.get("notice") or "No AI explanation available.", style="I")

    if sources:
        heading("Sources")
        for i, s in enumerate(sources, 1):
            para(f"[{i}] {s['title']} - {s['source']} (cosine {s['score']:.2f})", size=8.5)

    if plans and plans.get("plans"):
        heading("What would it take? (cheapest plans to reach a clear approval)")
        for i, p in enumerate(plans["plans"], 1):
            para(f"{i}. {p['summary']} -> {p['probability']:.0%} (effort {p['effort']:.1f})", size=9)
    return bytes(pdf.output())
