FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --bin-dir /usr/local/bin

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
