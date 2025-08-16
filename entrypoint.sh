#!/usr/bin/env bash
set -e

exec uv run python src/wyoming_thai_vits.py \
  --log-level "${THTTS_LOG_LEVEL}" \
  --host "${THTTS_HOST}" \
  --port "${THTTS_PORT}" \
  --model-id "${THTTS_MODEL}"
