# ============================================================
# API 网关 - Dockerfile
# ============================================================
FROM python:3.11-slim

LABEL maintainer="途景汇团队"
LABEL service="gateway"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/checkpoints

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV GATEWAY_PORT=8000

EXPOSE 8000

WORKDIR /app/gateway
CMD ["python", "gateway.py"]