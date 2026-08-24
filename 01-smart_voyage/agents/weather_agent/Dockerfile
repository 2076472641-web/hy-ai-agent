# ============================================================
# 天气查询智能体 - Dockerfile
# ============================================================
FROM python:3.11-slim

LABEL maintainer="途景汇团队"
LABEL agent="weather_agent"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/checkpoints

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENT_NAME=weather_agent
ENV WEATHER_AGENT_PORT=8101

EXPOSE 8101

WORKDIR /app/agents/weather_agent
CMD ["python", "main.py"]