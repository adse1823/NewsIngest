FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    pydantic==2.7.3 \
    mlflow==2.13.0 \
    lightgbm==4.3.0 \
    shap==0.45.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    prometheus-client==0.20.0 \
    python-dotenv==1.0.1

COPY serving/main.py .
COPY .env* ./

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
