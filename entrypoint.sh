#!/usr/bin/env bash
set -e

exec uv run python src/wyoming_thai_vits.py \
  --log-level "${WY_LOG_LEVEL}" \
  --host "${WY_HOST}" \
  --port "${WY_PORT}" \
  --model-id "${WY_MODEL}"
