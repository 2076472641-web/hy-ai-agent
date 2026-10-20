# ============================================================
# MCP 工具服务 - Dockerfile
# ============================================================
FROM python:3.11-slim

LABEL maintainer="途景汇团队"
LABEL service="mcp_server"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/checkpoints

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MCP_SERVER_HOST=0.0.0.0
ENV MCP_SERVER_PORT=8001

EXPOSE 8001

WORKDIR /app/mcp_server
CMD ["python", "server.py"]