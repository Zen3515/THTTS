# THTTS

THTTS is a local Thai text-to-speech Wyoming service for Home Assistant. One
process loads one selected backend and serves it on TCP port `10200` by default.

| Backend ID | Model/profile | Notes |
| --- | --- | --- |
| `vits` | VIZINTZOR Thai VITS | Choose female/male with `THTTS_VITS_MODEL`. |
| `f5-v1` | VIZINTZOR F5-TTS Thai v1 | Recommended current F5 profile. |
| `f5-v2` | VIZINTZOR F5-TTS Thai v2 | Uses the Thai IPA inference path. |

Run separate THTTS instances on separate ports when Home Assistant needs more
than one backend. This is intentionally not a per-request model router.

Model weights are provided by [VIZINTZOR](https://huggingface.co/VIZINTZOR):
[VITS female](https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-FEMALEV2),
[VITS male](https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-MALEV2),
[F5 v1](https://huggingface.co/VIZINTZOR/F5-TTS-THAI), and
[F5 v2](https://huggingface.co/VIZINTZOR/F5-TTS-TH-V2). Review each model's
license before use or redistribution.

## Run

```bash
uv sync
THTTS_BACKEND=f5-v1 THTTS_DEVICE=auto thtts
```

The selected model is loaded before the listener opens. Query its Wyoming
discovery response with:

```bash
thtts-healthcheck --host 127.0.0.1 --port 10200
```

The service supports both one-shot `synthesize` and
`synthesize-start`/`synthesize-chunk`/`synthesize-stop`. Text-input streaming
buffers completed segments; it is not token-by-token neural audio streaming.
A streamed request has no artificial leading silence and produces one
`audio-start`/`audio-stop` pair when it generated audio.

Home Assistant's compatibility `synthesize` mirror within a streamed request
is accepted and ignored: chunks remain the source of synthesis, so the full
message is never spoken twice.

## Configuration

Resolution order is **CLI > canonical environment variable > legacy alias >
default**. Empty values are unset. A conflicting canonical and legacy value is
a startup error. Every non-empty legacy name emits one startup warning, even
when its canonical counterpart has the same value. Legacy names and upper-case
backend values are accepted only for this compatibility release.

| Canonical variable | Default | Legacy alias / meaning |
| --- | --- | --- |
| `THTTS_BACKEND` | `vits` | Use `vits`, `f5-v1`, or `f5-v2`. `VITS`, `F5_V1`, `F5_V2`, `F5-THV1`, `F5-THV2`, `F5TH`, `V1`, and `V2` are temporary aliases. |
| `THTTS_HOST` | `0.0.0.0` | Unchanged. |
| `THTTS_PORT` | `10200` | Unchanged. |
| `THTTS_LOG_LEVEL` | `INFO` | Unchanged. |
| `THTTS_DEVICE` | `auto` | Unchanged; now applies to VITS and F5. |
| `THTTS_MAX_CONCURRENT_SYNTHESES` | `1` | `THTTS_MAX_CONCURRENT`; F5 currently rejects values other than `1` until a dedicated concurrency stress test approves them. |
| `THTTS_MAX_QUEUED_SYNTHESES` | `0` | New; `0` fails fast when busy. |
| `THTTS_MAX_QUEUE_SECONDS` | `30` | New maximum wait for admitted work. |
| `THTTS_VITS_MODEL` | `VIZINTZOR/MMS-TTS-THAI-FEMALEV2` | `THTTS_MODEL` (VITS only) |
| `THTTS_F5_CHECKPOINT_FILE` | selected profile | `THTTS_CKPT_FILE` |
| `THTTS_F5_VOCAB_FILE` | selected profile | `THTTS_VOCAB_FILE` |
| `THTTS_F5_REFERENCE_AUDIO` | selected profile sample | `THTTS_REF_AUDIO` |
| `THTTS_F5_REFERENCE_TEXT` | selected profile transcript | `THTTS_REF_TEXT` |
| `THTTS_F5_SPEED` | `1.0` | `THTTS_SPEED`, `THTTS_SPEAK_SPEED` |
| `THTTS_F5_NFE_STEPS` | `24` | `THTTS_NFE_STEPS`; set `32` to retain the previous sampling/quality setting. |
| `THTTS_VOICES_FILE` | unset | `THTTS_VOICES_YAML` |
| `THTTS_STREAM_IDLE_FLUSH_MS` | `220` | `THTTS_MAX_WAIT_MS` |
| `THTTS_STREAM_MIN_SEGMENT_CHARS` | `15` | `THTTS_MIN_SENT_CHARS` |
| `THTTS_STREAM_TARGET_CHARS` | `48` | New; formerly hard-coded. |
| `THTTS_STREAM_MAX_SEGMENT_CHARS` | `180` | New; formerly hard-coded. |
| `THTTS_SHUTDOWN_GRACE_SECONDS` | `15` | New maximum time to let active requests finish after SIGTERM. |

CLI equivalents are available through `thtts --help`. Startup summaries and
deprecation warnings never include source text, reference transcripts, or
model/reference paths.

When a legacy name is present, boot logs a copyable migration line for that
exact key. The right-hand side deliberately uses a shell/Compose placeholder,
so the log does not reveal a private value:

```text
Deprecated configuration: THTTS_CKPT_FILE will be removed in the next breaking release. Replace it with: THTTS_F5_CHECKPOINT_FILE=${THTTS_CKPT_FILE}
Deprecated configuration: THTTS_BACKEND=F5_V1 will be removed in the next breaking release. Replace it with: THTTS_BACKEND=f5-v1
```

For the first example, replace the old line with
`THTTS_F5_CHECKPOINT_FILE=${THTTS_CKPT_FILE}` in a shell-style configuration,
or use the same source value under `THTTS_F5_CHECKPOINT_FILE` in an
environment map. Do not retain the old key merely because the new key is also
set: it will still warn until removed.

### Upcoming-version migration

Replace a former F5 v1 deployment such as this:

```bash
# THTTS_BACKEND=F5_V1
# THTTS_MAX_CONCURRENT=1
# THTTS_CKPT_FILE=/models/f5.pt
# THTTS_REF_AUDIO=/voices/default.wav
```

with this exact upcoming compatibility-release configuration:

```bash
THTTS_BACKEND=f5-v1 \
THTTS_HOST=0.0.0.0 \
THTTS_PORT=10200 \
THTTS_DEVICE=auto \
THTTS_MAX_CONCURRENT_SYNTHESES=1 \
THTTS_F5_CHECKPOINT_FILE=/models/f5.pt \
THTTS_F5_REFERENCE_AUDIO=/voices/default.wav \
thtts
```

See [CHANGELOG.md](CHANGELOG.md) for the operator-facing migration notice.

### F5 voice list

Set `THTTS_VOICES_FILE` to a YAML list to expose multiple F5 reference voices:

```yaml
- name: default
  attribution:
    name: VIZINTZOR/F5-TTS-THAI
    url: https://huggingface.co/VIZINTZOR/F5-TTS-THAI
  languages: ["th", "th-TH"]
  description: Default Thai voice
  version: "1.0"
  ref_sound_path: /voices/default.wav
  ref_sound_sentence: ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบาย
```

An explicitly configured voice file is fail-closed: invalid YAML, duplicate
names, missing required fields, and missing reference audio stop startup rather
than silently using another voice. The built-in F5 `default` and
`thai-default` compatibility names remain available when no file is supplied.

## Docker

The images use `thtts` directly and probe `describe` for readiness. Mount
`/data` to retain model downloads: `HF_HOME`, `CACHED_PATH_CACHE_ROOT`, and
`XDG_CACHE_HOME` are already set beneath that directory.

```yaml
services:
  thtts:
    image: ghcr.io/zen3515/thtts:latest
    restart: unless-stopped
    ports: ["10200:10200"]
    volumes: ["./thtts-data:/data"]
    environment:
      THTTS_BACKEND: f5-v1
      THTTS_DEVICE: auto
      THTTS_MAX_CONCURRENT_SYNTHESES: "1"
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
```

For Pascal GPUs, use `ghcr.io/zen3515/thtts:cuda126-sm61` (also tagged
`pascal`); it remains `linux/amd64` only. Configure NVIDIA Container Toolkit
on the host before requesting a GPU.

Wyoming TCP has no built-in authentication or TLS. Bind it to a trusted local
network, firewall it, or put an appropriate authenticated tunnel/proxy in front
of it; do not expose the listener directly to the public Internet.

## Compatibility and testing

For one compatibility release, the old script paths forward to the packaged
service and emit a deprecation warning:

```bash
python src/wyoming_thai_vits.py --model-id VIZINTZOR/MMS-TTS-THAI-MALEV2
python src/wyoming_thai_f5.py --model-version v2
```

Run the non-model validation lane without downloads:

```bash
uv lock --check
uv run ruff check src/thtts tests
uv run pytest -q
uv build
```

The opt-in real-model oracle suite is deliberately separate. It requires a
local JSON manifest that records model/reference hashes, platform, fixed Thai
input, exact PCM hashes when deterministic, or calibrated signal bounds when
not. Start from `tests/integration/model-oracles.example.json`, keep approved
audio artifacts private when their license disallows redistribution, and run:

```bash
THTTS_RUN_MODEL_TESTS=1 \
THTTS_MODEL_ORACLE_MANIFEST=/secure/model-oracles.json \
uv run pytest -m integration -q
```
