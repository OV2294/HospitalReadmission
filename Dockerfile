FROM python:3.11-slim

WORKDIR /app

# Install only what the prediction service needs at runtime (lighter image
# than the full training environment)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY models/best_model.pkl ./models/best_model.pkl
COPY models/preprocessor.pkl ./models/preprocessor.pkl

ENV MODEL_PATH=/app/models/best_model.pkl
ENV PREPROCESSOR_PATH=/app/models/preprocessor.pkl
ENV MODEL_VERSION=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
