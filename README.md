# LoanLens

**Loan Approval Prediction with a Generative AI-Based Loan Eligibility Explainer**

A weekend experiment in gluing a classic ML model to a small local RAG + LLM stack: can a Random Forest's
prediction be explained in plain language, with citations, without sending anything to the cloud?

LoanLens predicts whether a loan application is likely to be approved (Random Forest, 98.2% test accuracy),
shows the three factors that moved the prediction most (SHAP), and then asks a local LLM (Llama 3.2 in Ollama)
to explain the result in plain language, citing passages retrieved from public RBI and TransUnion CIBIL
credit-awareness pages (RAG with FAISS). It's a learning project, not a lending decision.

## Project structure

```
loanlens/
  app.py                 Streamlit UI (Prediction / Model performance / About tabs)
  train.py               trains the models -> model.joblib, metrics.json, 2 charts
  build_index.py         chunks + embeds the knowledge base -> index/
  loanlens/
    data.py              cleaning, feature list, input ranges, validation
    predict.py           load model, predict, confidence band, SHAP top-3 factors
    rag.py               query from ML output, MiniLM embeddings, FAISS search, 0.35 threshold
    llm.py               Ollama call, system prompt, grounding check, fallbacks, full pipeline
  knowledge_base/        8 RBI / CIBIL pages saved as text (SOURCE + TITLE header lines)
  index/                 faiss.index + chunks.json (generated)
  tests/test_loanlens.py 7 pytest cases
  .streamlit/config.toml light theme, indigo accent, tracebacks hidden
```

## Setup (Windows, Python 3.11+)

1. Create and activate a virtual environment, then install the packages:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

2. Install [Ollama](https://ollama.com/download) and pull the model (about 2 GB):

```bash
ollama pull llama3.2
```

3. Download `loan_approval_dataset.csv` from
[Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) into this folder.

## Run

```bash
python train.py loan_approval_dataset.csv
```

Expected: Random Forest accuracy 0.982, F1 0.986, ROC-AUC 0.999; Logistic Regression accuracy 0.915;
majority baseline 0.622.

```bash
python compare_models.py loan_approval_dataset.csv
```

Optional: writes `comparison.json` for the model shootout.

```bash
python build_index.py
```

```bash
streamlit run app.py
```

Then open http://localhost:8501. Keep Ollama running (it starts automatically after installation; otherwise run `ollama serve`).

```bash
pytest -v
```

The first run of `build_index.py` downloads the `all-MiniLM-L6-v2` embedder (~90 MB) once.

## How it works

1. **Predict** - the form values are validated (`data.validate_applicant`) and scored by the Random Forest
   pipeline. Probability < 0.4 or > 0.6 is a *clear* result, otherwise *borderline*.
2. **Explain the model** - `shap.TreeExplainer` gives each feature's contribution for this applicant;
   the top 3 are shown with their direction (towards approval / rejection).
3. **Retrieve** - a query is built from the ML output, e.g. *"rejected loan application, CIBIL score 412,
   main factors CIBIL score, Loan term and Loan amount"*, embedded and searched in FAISS. Only chunks with
   cosine similarity ≥ 0.35 are kept (top 3). If none pass, the app shows **"No supporting source found"**
   and does not call the LLM.
4. **Generate** - llama3.2 (temperature 0.2, localhost only) writes the explanation from the APPLICANT,
   PREDICTION and SOURCES blocks. A grounding check requires the predicted label and at least one `[n]`
   citation; it regenerates once, then falls back to showing only the passages with a warning.
5. **Fallback** - if Ollama is not running, the prediction and passages are still shown with
   *"Explanation unavailable: Ollama is not running"*.

## Extra features

- **What would it take? (Plan tab)** - `loanlens/counterfactual.py` scores a few thousand variations of the
  inputs a person can actually change (CIBIL, loan amount, income, term, bank savings) in one batch and
  returns the cheapest plans that reach a 60% approval probability, ranked on a shared effort scale.
  It also reports the best probability reachable *without* touching CIBIL. "Try this plan" loads a plan
  into the form; "Explain plan 1 with AI" retrieves RBI/CIBIL passages and asks llama3.2 for cited steps.
- **Credit-score time machine (Plan tab)** - `loanlens/timeline.py` applies chosen habits (on-time payments,
  low card use, a missed payment, ...) and a monthly saving one month at a time, re-scores the applicant at
  every step, and charts when the approval probability crosses 50% and 60%. The CIBIL point values are
  illustrative assumptions with diminishing returns near 900, and are editable in the app.
- **Chat with the decision (Prediction tab)** - `chat_followup()` in `loanlens/llm.py` answers follow-up
  questions with the applicant's numbers, the prediction and freshly retrieved passages, sending the last few
  turns as history. A question is only sent to the LLM if it retrieves a passage on its own (cosine ≥ 0.35)
  or names one of the model's inputs (`mentioned_features()` in `loanlens/data.py`); anything else is refused
  without an LLM call. A fast port check makes the "Ollama is not running" fallback instant.
- **Listen in your language (Prediction tab)** - `loanlens/voice.py` reads the explanation aloud with the
  browser's own Web Speech API (nothing leaves the device). `translate()` in `loanlens/llm.py` asks llama3.2
  for Malayalam, Hindi or Tamil and only accepts the result if every `[n]` citation survives (one retry,
  then English). Without Ollama, a summary built from the prediction is read instead.
- **Fairness tab** - `loanlens/fairness.py` re-creates train.py's exact test split and compares Graduate vs
  Not Graduate and self-employed vs not on approval rate, accuracy, true- and false-positive rate, plus a flip
  test (change only that attribute and count flipped decisions). Both attributes currently pass; the
  write-up points out the one number worth watching (false-positive rate for non-graduates).
- **How sure is it? (Prediction tab)** - `loanlens/uncertainty.py` adds two signals to the probability:
  how many of the 200 trees agree, and whether the applicant looks like the training data (1st-99th
  percentile ranges, loan/income and assets/loan ratios, and the distance to the 5 nearest training
  applicants in the model's scaled space). The documented failure case - wrongly approved at 90% - is
  flagged *medium* because its assets-to-loan ratio is outside the training range.
- **Mismatch alarm** - `consistency_check()` in `loanlens/llm.py` finds which model inputs the LLM's
  explanation talks about and compares them with SHAP's top 3. The explanation card shows a badge:
  *matches*, *partly matches* (a lower factor missing or other inputs discussed) or *mismatch* (SHAP's
  number-one factor never mentioned). It is stricter than the grounding check, which only needs a citation.
- **Model shootout (Model performance tab)** - `compare_models.py` trains Logistic Regression, the Random
  Forest, HistGradientBoosting, XGBoost and LightGBM on train.py's exact split and writes `comparison.json`
  (quality, CV F1, error count, training time, single-prediction latency, model size, ROC points). The tab
  shows the table, zoomed ROC curves and an errors-vs-speed chart. LightGBM makes 12 errors vs 15 for the
  forest and is several times faster, but the app keeps the forest because SHAP and the certainty meter use it.

## Tests

| # | Test | Expected |
|---|------|----------|
| 1 | Normal applicant, CIBIL 750 | Approved |
| 2 | Same applicant, CIBIL 350 | Rejected |
| 3 | Same applicant, CIBIL 550 | probability 0.4-0.9, band computed correctly |
| 4 | Documented failure: CIBIL 674, income 83 lakh, loan 3.14 crore, term 6 | actually Rejected, model says Approved (xfail) |
| 5 | Invalid input: CIBIL 1000, negative loan | ValidationError |
| 6 | Unrelated query | empty retrieval, "No supporting source found", LLM not called |
| 7 | Ollama unavailable (mocked) | prediction + passages + notice |

## Screenshots

`take_screenshots.py` captures the main views into `screenshots/` (app must be running). Other views worth checking by hand:

1. **Approved result + explanation** - load *Approved applicant*, Predict, capture the result card, SHAP chart and AI explanation.
2. **Rejected result with Sources expanded** - load *Rejected applicant*, Predict, open the *Sources* panel.
3. **"No supporting source found"** - temporarily raise `THRESHOLD` in `loanlens/rag.py` to `0.9`, restart the app and Predict (set it back to `0.35` afterwards).
4. **App with Ollama stopped** - quit Ollama from the system tray, Predict, capture the notice with the passages still shown.
5. **Model performance tab** - the comparison table, confusion matrix and feature importance chart.

## Knowledge base sources

All files in `knowledge_base/` are the readable text of public pages, copied from the live sites (only menus,
footers and "Return to top" links removed): seven TransUnion CIBIL pages (what is a CIBIL score, score and
report FAQ, credit score and loan approvals, loan approval process, personal loans, home loans, why the score
matters beyond loans) and the Reserve Bank of India *Financial Awareness Messages (FAME)* booklet, 4th edition,
February 2024. The first two lines of each file give its URL and title.

## Limitations

- CIBIL score carries 80.7% of the Random Forest's feature importance, so the model is close to a CIBIL rule.
- The dataset is likely synthetic; results may not transfer to real lending.
- Learning project only, not a lending decision or personal financial advice.
