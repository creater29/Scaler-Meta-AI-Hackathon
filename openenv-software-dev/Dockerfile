FROM python:3.11-slim AS base

LABEL maintainer="openenv-software-dev"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install ".[llm,hub]"

COPY server/ ./server/
COPY client/ ./client/
COPY training/ ./training/
COPY inference.py ./
COPY openenv.yaml ./

# Non-root user for security
RUN useradd --create-home --shell /bin/bash agent
USER agent

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
