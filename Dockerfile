FROM python:3.10-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl ca-certificates xz-utils \
 && rm -rf /var/lib/apt/lists/*

ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz

RUN set -eux; \
  curl -L "$FFMPEG_URL" -o /tmp/ffmpeg.tar.xz; \
  mkdir -p /tmp/ffmpeg; \
  tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1; \
  cp /tmp/ffmpeg/bin/ffmpeg  /usr/local/bin/ffmpeg; \
  cp /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe; \
  rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

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
