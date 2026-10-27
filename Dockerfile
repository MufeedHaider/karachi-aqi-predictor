# Two images from one file.
#
#   Dashboard (default):  docker build -t karachi-aqi .
#                         docker run -p 8501:8501 karachi-aqi
#                         -> http://localhost:8501
#
#   Full pipeline:        docker build --target pipeline -t karachi-aqi-pipeline .
#                         docker run -e OPENAQ_API_KEY=... karachi-aqi-pipeline
#                         (runs the tests, then the same steps as the nightly job)
#
# The dashboard image installs only what the dashboard imports (no XGBoost or
# SHAP) and reads the artefacts committed in data/ and models/.

FROM python:3.13-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS pipeline
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["sh", "-c", "python -m pytest tests -q && python src/fetch_data.py && python src/ground_truth.py --update && python src/feature_engineering.py && python src/train_model.py && python src/forecast_model.py && python src/accuracy_gate.py && python src/explain_model.py && python src/drift.py && python src/live_scoring.py"]

FROM base AS dashboard
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install -r dashboard/requirements.txt
COPY .streamlit .streamlit
COPY src src
COPY dashboard dashboard
COPY data data
COPY models models
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
