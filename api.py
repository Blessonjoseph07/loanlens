"""LoanLens REST API (FastAPI).

Run:   uvicorn api:app --port 8000
Docs:  http://localhost:8000/docs  (interactive Swagger UI)

Endpoints
  GET  /health          model loaded? Ollama reachable?
  POST /predict         label, probability, band, SHAP top-3, certainty meter
  POST /what-it-takes   cheapest changes that reach a clear approval
  POST /explain         full pipeline: prediction + retrieved sources + LLM explanation
  POST /report          the same, as a one-page PDF

The same Python modules power the Streamlit app, so both always agree.
The embedder and FAISS index are only loaded on the first /explain call,
which keeps /predict fast and lets the API start in about a second.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from loanlens.counterfactual import what_would_it_take
from loanlens.data import ValidationError
from loanlens.llm import analyse, ollama_reachable
from loanlens.pdf_report import build_pdf
from loanlens.predict import Predictor
from loanlens.uncertainty import FamiliarityChecker, assess

ROOT = Path(__file__).resolve().parent
STATE: dict = {}


class Applicant(BaseModel):
    """One loan applicant. Ranges match the app's form (loanlens/data.py RANGES)."""
    no_of_dependents: int = Field(ge=0, le=5, examples=[2])
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    income_annum: int = Field(ge=1, examples=[5_000_000], description="Annual income in rupees")
    loan_amount: int = Field(ge=1, examples=[12_000_000], description="Rupees")
    loan_term: int = Field(ge=2, le=20, examples=[10], description="Years")
    cibil_score: int = Field(ge=300, le=900, examples=[720])
    residential_assets_value: int = Field(ge=0, examples=[4_000_000])
    commercial_assets_value: int = Field(ge=0, examples=[2_000_000])
    luxury_assets_value: int = Field(ge=0, examples=[8_000_000])
    bank_asset_value: int = Field(ge=0, examples=[3_000_000])


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor = Predictor(ROOT / "model.joblib")
    STATE["predictor"] = predictor
    STATE["checker"] = FamiliarityChecker(predictor.model, ROOT / "loan_approval_dataset.csv")
    yield
    STATE.clear()


app = FastAPI(title="LoanLens API", version="1.0",
              description="Loan approval prediction with SHAP factors, a certainty meter, counterfactual plans "
                          "and a locally generated, cited explanation.",
              lifespan=lifespan)


def _retriever():
    if "retriever" not in STATE:
        from loanlens.rag import Retriever, load_embedder
        STATE["retriever"] = Retriever(load_embedder(), ROOT / "index")
    return STATE["retriever"]


def _certainty(applicant: dict, band: str) -> dict:
    s = assess(STATE["predictor"].model, STATE["checker"], applicant, band)
    return {k: s[k] for k in ("level", "reasons", "agreement", "approve_votes", "n_trees", "novel", "unusual")}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in STATE, "ollama_reachable": ollama_reachable()}


@app.post("/predict")
def predict(applicant: Applicant):
    try:
        pred = STATE["predictor"].predict(applicant.model_dump())
    except ValidationError as e:                       # defensive: pydantic already checks ranges
        raise HTTPException(status_code=422, detail=str(e))
    return {**pred, "certainty": _certainty(pred["applicant"], pred["band"])}


@app.post("/what-it-takes")
def what_it_takes(applicant: Applicant, target: float = 0.6):
    if not 0.5 <= target <= 0.99:
        raise HTTPException(status_code=422, detail="target must be between 0.5 and 0.99")
    return what_would_it_take(STATE["predictor"].model, applicant.model_dump(), target=target)


@app.post("/explain")
def explain(applicant: Applicant):
    """Prediction + sources + LLM explanation. Works without Ollama (explanation.status says why)."""
    result = analyse(applicant.model_dump(), STATE["predictor"], _retriever())
    result["prediction"]["certainty"] = _certainty(result["prediction"]["applicant"], result["prediction"]["band"])
    return result


@app.post("/report", response_class=Response, responses={200: {"content": {"application/pdf": {}}}})
def report(applicant: Applicant):
    """One-page PDF: result, certainty, applicant, SHAP chart, explanation, sources, cheapest plan."""
    a = applicant.model_dump()
    result = analyse(a, STATE["predictor"], _retriever())
    pred = result["prediction"]
    pdf = build_pdf(result, assess(STATE["predictor"].model, STATE["checker"], pred["applicant"], pred["band"]),
                    what_would_it_take(STATE["predictor"].model, a))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="loanlens_{pred["label"].lower()}.pdf"'})
