#!/usr/bin/env bash
set -Eeuo pipefail

# -------- Common env (with defaults) --------
: "${THTTS_HOST:=0.0.0.0}"
: "${THTTS_PORT:=10200}"
: "${THTTS_LOG_LEVEL:=INFO}"

# VITS-specific
: "${THTTS_MODEL:=VIZINTZOR/MMS-TTS-THAI-FEMALEV2}"  # can be overridden

# F5-specific (optional overrides)
: "${THTTS_REF_AUDIO:=hf_sample}"
: "${THTTS_REF_TEXT:=}"             # empty = let backend decide/ASR if supported
: "${THTTS_DEVICE:=auto}"           # auto|cpu|cuda
: "${THTTS_SPEED:=1.0}"
: "${THTTS_NFE_STEPS:=32}"
: "${THTTS_MAX_CONCURRENT:=2}"
: "${THTTS_CKPT_FILE:=}"            # optional override
: "${THTTS_VOCAB_FILE:=}"           # optional override

BACKEND="${THTTS_BACKEND:-VITS}"
BACKEND_UPPER="$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')"

run_vits () {
  exec uv run python src/wyoming_thai_vits.py \
    --log-level "${THTTS_LOG_LEVEL}" \
    --host "${THTTS_HOST}" \
    --port "${THTTS_PORT}" \
    --model-id "${THTTS_MODEL}"
}

run_f5 () {
  local version="$1"    # v1 or v2
  local ckpt="${THTTS_CKPT_FILE}"
  local vocab="${THTTS_VOCAB_FILE}"

  if [[ -z "$ckpt" || -z "$vocab" ]]; then
    if [[ "$version" == "v1" ]]; then
      ckpt="${ckpt:-hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt}"
      vocab="${vocab:-hf://VIZINTZOR/F5-TTS-THAI/vocab.txt}"
    else
      ckpt="${ckpt:-hf://VIZINTZOR/F5-TTS-TH-V2/model_250000.pt}"
      vocab="${vocab:-hf://VIZINTZOR/F5-TTS-TH-V2/vocab.txt}"
    fi
  fi

  # Build args safely as an array
  args=(
    --log-level "${THTTS_LOG_LEVEL}"
    --host "${THTTS_HOST}"
    --port "${THTTS_PORT}"
    --model-version "${version}"
    --ckpt-file "${ckpt}"
    --vocab-file "${vocab}"
    --ref-audio "${THTTS_REF_AUDIO}"
    --device "${THTTS_DEVICE}"
    --speed "${THTTS_SPEED}"
    --nfe-steps "${THTTS_NFE_STEPS}"
    --max-concurrent "${THTTS_MAX_CONCURRENT}"
  )
  # Only pass --ref-text if provided (avoid empty string ambiguity)
  if [[ -n "${THTTS_REF_TEXT}" ]]; then
    args+=( --ref-text "${THTTS_REF_TEXT}" )
  fi

  exec uv run python src/wyoming_thai_f5.py "${args[@]}"
}

case "$BACKEND_UPPER" in
  VITS)
    echo "[entrypoint] Using backend: VITS"
    run_vits
    ;;
  F5_V1|F5-THV1|F5TH|V1)
    echo "[entrypoint] Using backend: F5 v1"
    run_f5 "v1"
    ;;
  F5_V2|F5-THV2|V2)
    echo "[entrypoint] Using backend: F5 v2"
    run_f5 "v2"
    ;;
  *)
    echo "[entrypoint] ERROR: Unknown THTTS_BACKEND='$BACKEND'. Use VITS, F5_V1, or F5_V2." >&2
    exit 1
    ;;
esac
