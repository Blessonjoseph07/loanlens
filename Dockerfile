# LoanLens API (FastAPI) - CPU only.
#   docker build -t loanlens .
#   docker run -p 8000:8000 loanlens            -> http://localhost:8000/docs
# The Streamlit app is in the same image:
#   docker run -p 8501:8501 loanlens streamlit run app.py --server.address 0.0.0.0
# Explanations need Ollama on the host; on Linux add --network host so
# "localhost:11434" inside the container reaches it. Without it, /explain
# still returns the prediction and sources with an "Ollama is not running" notice.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface
# libgomp is needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
# bake the sentence-transformers model into the image so /explain works offline
RUN python -c "from loanlens.rag import load_embedder; load_embedder()"

RUN useradd --create-home loanlens && chown -R loanlens /app
USER loanlens
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
