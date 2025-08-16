FROM python:3.10-slim AS base

# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY . .

ENV PATH="/app/.venv/bin:${PATH}"

ENV PYTHONUNBUFFERED=1

EXPOSE 10200

ENV THTTS_LOG_LEVEL=DEBUG
ENV THTTS_HOST=0.0.0.0
ENV THTTS_PORT=10200
ENV THTTS_MODEL=VIZINTZOR/MMS-TTS-THAI-FEMALEV2

ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]
