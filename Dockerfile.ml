# Dockerfile.ml  (FastAPI + PyTorch model)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY api/ ./api/

WORKDIR /app/api
ENV MODEL_PATH=/app/ml/checkpoints/best_model.pt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]