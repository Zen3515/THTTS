# Thai TTS (TH TTS)

## Model Attribution

All model weights are provided by [VIZINTZOR](https://huggingface.co/VIZINTZOR) via Hugging Face:

- **VITS Thai Female/Male**:  
  [MMS-TTS-THAI-FEMALEV2](https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-FEMALEV2),  
  [MMS-TTS-THAI-MALEV2](https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-MALEV2)
- **F5-TTS Thai**:  
  [F5-TTS-THAI](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)  
  [F5-TTS-TH-V2](https://huggingface.co/VIZINTZOR/F5-TTS-TH-V2)

Please acknowledge and cite VIZINTZOR if you use these models in your work.

---

## Recommended Model

**For best quality and performance, use F5-TTS v1.**

---

## How to Run

You can run the server using either direct `uv` commands or the provided `entrypoint.sh` script (recommended for Docker and easy switching).

### 1. Using `uv` Directly

#### VITS Thai (Female/Male)

```bash
uv run python src/wyoming_thai_vits.py --log-level INFO --host 0.0.0.0 --port 10200 \
  --model-id VIZINTZOR/MMS-TTS-THAI-FEMALEV2

uv run python src/wyoming_thai_vits.py --log-level INFO --host 0.0.0.0 --port 10200 \
  --model-id VIZINTZOR/MMS-TTS-THAI-MALEV2
```

#### F5-TTS Thai v1 (**Recommended**)

```bash
uv run python src/wyoming_thai_f5.py --log-level INFO --host 0.0.0.0 --port 10200 \
  --model-version v1
```

#### F5-TTS Thai v2

```bash
uv run python src/wyoming_thai_f5.py --log-level INFO --host 0.0.0.0 --port 10200 \
  --model-version v2
```

### 2. Using `entrypoint.sh` (Recommended)

Set the backend via `THTTS_BACKEND` environment variable:

- `VITS` for VITS model
- `F5_V1` for F5-TTS v1 (**recommended**)
- `F5_V2` for F5-TTS v2

Example:

```bash
THTTS_BACKEND=F5_V1 ./entrypoint.sh
```

You can override other parameters via environment variables (see below).

---

## Environment Variables

| Variable              | Default Value                                 | Description                                      |
|-----------------------|-----------------------------------------------|--------------------------------------------------|
| `THTTS_BACKEND`       | `VITS`                                        | Model backend: `VITS`, `F5_V1`, or `F5_V2`       |
| `THTTS_HOST`          | `0.0.0.0`                                     | Bind address                                     |
| `THTTS_PORT`          | `10200`                                       | Port to listen on                                |
| `THTTS_LOG_LEVEL`     | `INFO`                                        | Log level (`DEBUG`, `INFO`, etc.)                |
| `THTTS_MODEL`         | `VIZINTZOR/MMS-TTS-THAI-FEMALEV2`             | VITS model ID                                    |
| `THTTS_REF_AUDIO`     | `hf_sample`                                   | F5 reference audio path                          |
| `THTTS_REF_TEXT`      | *(empty)*                                     | F5 reference transcript                          |
| `THTTS_DEVICE`        | `auto`                                        | `auto`, `cpu`, or `cuda`                         |
| `THTTS_SPEED`         | `1.0`                                         | F5 speech speed multiplier                       |
| `THTTS_NFE_STEPS`     | `32`                                          | F5 denoising steps                               |
| `THTTS_MAX_CONCURRENT`| `1`                                           | Max concurrent synth requests                    |
| `THTTS_CKPT_FILE`     | *(auto-selected by backend)*                  | F5 checkpoint file path                          |
| `THTTS_VOCAB_FILE`    | *(auto-selected by backend)*                  | F5 vocab file path                               |


## 3. Docker Compose (NVIDIA GPU)

```yaml
services:
  thtts:
    image: ghcr.io/zen3515/thtts:latest
    container_name: thtts
    restart: unless-stopped
    shm_size: "2g" # please adjust
    environment:
      - THTTS_BACKEND=F5_V1
      - THTTS_HOST=0.0.0.0
      - THTTS_PORT=10200
      - THTTS_LOG_LEVEL=INFO
      - THTTS_DEVICE=auto
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    ports:
      - "10200:10200"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Note:**  
- Make sure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
- Adjust the `THTTS_BACKEND` and other environment variables as needed.

---

## How to Test

### Query Info

```bash
printf '{"type":"describe","data":{}}\n' | nc 127.0.0.1 10200
```

### Synthesize Speech

Just connect it to homeassistant, it's probably the most up to spec with wyoming protocol

---


## License

See individual model pages on Hugging Face for license details.
