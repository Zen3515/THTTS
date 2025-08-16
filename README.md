# TH TTS

## How to run

```bash
uv run python src/wyoming_thai_vits.py --log-level DEBUG --host 0.0.0.0 --port 10200 \
  --model-id VIZINTZOR/MMS-TTS-THAI-FEMALEV2

uv run python src/wyoming_thai_vits.py --log-level DEBUG --host 0.0.0.0 --port 10200 \
  --model-id VIZINTZOR/MMS-TTS-THAI-MALEV2
```

## How to test

### tool

```bash
go install github.com/john-pettigrew/wyoming-cli@latest
```

### info
```bash
printf '{"type":"describe","data":{}}\n' | nc 127.0.0.1 10200
```

### synth
> Connect to HA seems to work much better, wyoming-cli only managed to get describe, so just let people in UFW
```bash
sudo ufw allow 10200/tcp
sudo ufw delete allow 10200/tcp
```

```bash
wyoming-cli tts -voice-name 'thai-female' -addr 'localhost:10200' -text 'สวัสดีชาวโลก' -output_file './hello.wav'
```

```bash
( printf '{"type":"synthesize","data":{"text":"สวัสดีครับ ยินดีที่ได้รู้จัก","voice":"thai-female"}}\n'; ) \
| nc 127.0.0.1 10200 \
| tee responses.ndjson \
| jq -r 'select(.type=="audio-start") or select(.type=="audio-chunk") or select(.type=="audio-stop")' > audio_events.ndjson

# Extract audio chunks (base64) -> raw PCM
jq -r 'select(.type=="audio-chunk") | .data.audio' audio_events.ndjson | base64 -d > out.pcm

# Convert PCM (s16le, 22.05kHz, mono) -> WAV (use either ffmpeg or sox)
ffmpeg -f s16le -ar 22050 -ac 1 -i out.pcm out.wav -y
# or:
sox -t raw -r 22050 -e signed -b 16 -c 1 out.pcm out.wav
```