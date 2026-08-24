FROM python:3.10-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini ca-certificates curl xz-utils espeak \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl --fail --location --show-error "$FFMPEG_URL" -o /tmp/ffmpeg.tar.xz; \
    mkdir -p /tmp/ffmpeg; \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1; \
    install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg; \
    install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe; \
    ffmpeg -version; \
    rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/huggingface
ENV CACHED_PATH_CACHE_ROOT=/data/cached-path
ENV XDG_CACHE_HOME=/data/cache

EXPOSE 10200

ENV THTTS_BACKEND=vits
ENV THTTS_LOG_LEVEL=INFO
ENV THTTS_HOST=0.0.0.0
ENV THTTS_PORT=10200

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["thtts-healthcheck", "--host", "127.0.0.1", "--port", "10200", "--timeout", "3"]

ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]
