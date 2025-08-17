#!/usr/bin/env python3
import argparse
import asyncio
import logging
import re
import os
from typing import List
import numpy as np

# ---- Wyoming imports (official lib) ----
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer

from wyoming.tts import (
    Synthesize,
    SynthesizeStart,
    SynthesizeChunk,
    SynthesizeStop,
    SynthesizeStopped,
)

from wyoming.audio import (
    AudioStart,
    AudioChunk,
    AudioStop,
)

from wyoming.info import Info, Describe, TtsProgram, TtsVoice, Attribution
from wyoming.error import Error

# ---- F5-TTS (Thai) imports ----
import torch
from importlib.resources import files
from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    mel_spec_type,             # "vocos" (24 kHz)
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed as default_speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

# -----------------------
# Utils
# -----------------------


def float32_to_int16_pcm(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    x16 = (x * 32767.0).astype(np.int16)
    return x16.tobytes()


def split_sentences_th(text: str) -> List[str]:
    parts = re.split(r'([.!?。\n])', text)
    chunks, buf = [], ""
    for p in parts:
        if p is None:
            continue
        buf += p
        if p in {".", "!", "?", "。", "\n"}:
            s = buf.strip()
            if s:
                chunks.append(s)
            buf = ""
    tail = buf.strip()
    if tail:
        chunks.append(tail)
    return [c for c in (s.strip() for s in chunks) if c]


# -----------------------
# F5-TTS Thai Engine
# -----------------------
class ThaiF5Engine:
    """
    Wraps F5-TTS (DiT + vocos) with Thai finetuned checkpoint.
    Produces 24 kHz mono float32 waveform via infer_process().
    """

    def __init__(
        self,
        model_version: str | None,
        ckpt_file: str | None,
        vocab_file: str | None,
        ref_audio: str | None,
        ref_text: str | None,
        device: str = "auto",
        speed: float = default_speed,
        nfe_steps: int = nfe_step,
    ):
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Resolve Thai finetune paths (HF) if not provided
        # Hugging Face repo: VIZINTZOR/F5-TTS-THAI (model_1000000.pt, vocab.txt, sample/ref_audio.wav)
        self.ckpt_file = str(cached_path(ckpt_file or "hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt"))
        self.vocab_file = str(cached_path(vocab_file or "hf://VIZINTZOR/F5-TTS-THAI/vocab.txt"))

        # Model base config from f5_tts package
        # model_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/F5-TTS-THAI/F5TTS_Base_train.yaml')
        # model_cfg = OmegaConf.load(model_cfg_path).model.arch

        # Vocoder + model (vocos @ 24 kHz)
        # self.mel_type = "vocos"   # force vocoder to VOCOS regardless of package default
        # self.output_sr = None     # set to an int (e.g. 22050) to downsample for legacy sinks
        # self.vocoder = load_vocoder(vocoder_name=self.mel_type, is_local=False, local_path="") # "../checkpoints/vocos-mel-24khz")
        self.vocoder = load_vocoder()
        if model_version == "v1":
            F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
        else:
            F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True, conv_layers=4, pe_attn_head=None)
        self.model = load_model(DiT, F5TTS_model_cfg, self.ckpt_file, vocab_file=self.vocab_file, use_ema=True)

        # Reference audio/text (voice prompt)
        if not ref_audio or ref_audio == "hf_sample":
            self.ref_audio = str(cached_path("hf://VIZINTZOR/F5-TTS-THAI/sample/ref_audio.wav"))
            # Any natural Thai line works; better if it exactly matches the sample voice
            self.ref_text = ref_text or "ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบาย"
        else:
            self.ref_audio = ref_audio
            # If ref_text is empty, preprocess_ref_audio_text will attempt ASR on the prompt (more VRAM)
            self.ref_text = ref_text or ""

        self.ref_audio_p, self.ref_text_p = preprocess_ref_audio_text(self.ref_audio, self.ref_text)

        self.speed = float(speed)
        self.nfe_steps = int(nfe_steps)

        # F5-TTS + vocos emit 24 kHz mono
        self.sr = 24000
        logging.info("Engine ready: device=%s sr=%d", self.device, self.sr)

    @torch.inference_mode()
    def synth_blocking(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        logging.debug("Synth start (blocking): %r", text)

        result = infer_process(
            self.ref_audio_p,
            self.ref_text_p,
            text,
            self.model,
            self.vocoder,
            mel_spec_type=mel_spec_type,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=self.nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=self.speed,
            fix_duration=fix_duration,
        )
        if len(result) == 3:
            audio, sr, _ = result
        elif len(result) == 2:
            audio, sr = result
        else:
            raise ValueError("infer_process returned unexpected number of values")

        # Ensure float32 numpy
        if sr != self.sr:
            # (Shouldn't happen with vocos; if it does, resample or just trust sr)
            pass
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        return audio.astype(np.float32, copy=False)


# -----------------------
# Wyoming Handler
# -----------------------
class ThaiF5Handler(AsyncEventHandler):
    def __init__(self, *args, engine: ThaiF5Engine, sem: asyncio.Semaphore, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming = False
        self.engine = engine
        self.sem = sem
        peer = getattr(self, "writer", None)
        try:
            addr = peer.get_extra_info("peername") if peer else None
        except Exception:
            addr = None
        logging.info("New handler created for peer=%s", addr)

    async def handle_event(self, event: Event) -> bool:
        logging.debug("Received event: type=%s size=%s", event.type, len(event.data or b""))

        try:
            if Describe.is_type(event.type):
                logging.info("Describe requested")
                info = Info(
                    tts=[
                        TtsProgram(
                            name="thai-f5-tts",
                            attribution=Attribution(
                                name="VIZINTZOR/F5-TTS-THAI",
                                url="https://huggingface.co/VIZINTZOR/F5-TTS-THAI",
                            ),
                            voices=[
                                TtsVoice(
                                    name="thai-default",
                                    attribution=Attribution(
                                        name="VIZINTZOR/F5-TTS-THAI",
                                        url="https://huggingface.co/VIZINTZOR/F5-TTS-THAI",
                                    ),
                                    languages=["th", "th-TH"],
                                    description="Thai female (F5-TTS finetune)",
                                    installed=True,
                                    version="1.0",
                                ),
                                TtsVoice(
                                    name="default",
                                    attribution=Attribution(
                                        name="VIZINTZOR/F5-TTS-THAI",
                                        url="https://huggingface.co/VIZINTZOR/F5-TTS-THAI",
                                    ),
                                    languages=["th", "th-TH"],
                                    description="Alias of thai-default",
                                    installed=True,
                                    version="1.0",
                                ),
                            ],
                            installed=True,
                            description="Thai TTS via F5-TTS (DiT + vocos, 24 kHz)",
                            version="1.0",
                            supports_synthesize_streaming=True,
                        )
                    ]
                )
                await self.write_event(info.event())
                logging.debug("Describe replied with: %s", info.to_dict())
                return True

            if Synthesize.is_type(event.type):
                if self._streaming:
                    logging.debug("Ignoring legacy one-shot 'synthesize' (streaming already in progress/done)")
                    return True
                syn: Synthesize = Synthesize.from_event(event)
                text = (syn.text or "").strip()
                logging.info("Synthesize (oneshot): %r", text)
                await self._speak_text(text)
                return True

            if SynthesizeStart.is_type(event.type):
                self._streaming = True
                logging.info("Synthesize streaming START: %s", event)
                return True

            if SynthesizeChunk.is_type(event.type):
                chunk = SynthesizeChunk.from_event(event)
                text = (chunk.text or "").strip()
                if not text:
                    logging.debug("Empty chunk")
                    return True
                sents = split_sentences_th(text)
                logging.info("Synthesize streaming CHUNK: %d sentences", len(sents))
                for sentence in sents:
                    await self._speak_text(sentence)
                return True

            if SynthesizeStop.is_type(event.type):
                logging.info("Synthesize streaming STOP")
                await self.write_event(SynthesizeStopped().event())
                self._streaming = False
                return True

            msg = f"Unhandled event type: {event.type}"
            logging.warning(msg)
            await self.write_event(Error(text=msg).event())

        except Exception as e:
            logging.exception("Exception in handle_event: %s", e)
            await self.write_event(Error(text=f"Server error: {e}").event())
        return False

    async def _speak_text(self, text: str):
        if not text:
            logging.debug("Empty text; skip speak")
            return
        rate, width, channels = self.engine.sr, 2, 1  # 16-bit, mono
        logging.debug("AudioStart: rate=%d width=%d channels=%d", rate, width, channels)
        await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())

        loop = asyncio.get_running_loop()
        async with self.sem:
            wav = await loop.run_in_executor(None, self.engine.synth_blocking, text)

        samples_per_chunk = int(self.engine.sr * 0.2) or 1  # ~200 ms per chunk
        total_bytes = 0
        for i in range(0, len(wav), samples_per_chunk):
            chunk = wav[i:i + samples_per_chunk]
            payload = float32_to_int16_pcm(chunk)
            total_bytes += len(payload)
            await self.write_event(AudioChunk(rate=rate, width=width, channels=channels, audio=payload).event())
        await self.write_event(AudioStop().event())
        logging.info("Streamed audio: text_len=%d samples=%d bytes=%d",
                     len(text), len(wav), total_bytes)


# -----------------------
# Main
# -----------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10200)

    # Thai F5-TTS options
    ap.add_argument("--model-version", default="v1", help="v1 or v2")
    ap.add_argument("--ckpt-file", default=None, help="Path to Thai finetune checkpoint (.pt/.safetensors). Default: download from HF.")
    ap.add_argument("--vocab-file", default=None, help="Path to Thai vocab.txt. Default: download from HF.")
    ap.add_argument("--ref-audio", default="hf_sample", help='Reference voice audio path. Use "hf_sample" to use the model’s bundled sample.')
    ap.add_argument("--ref-text", default=None, help="Transcript for the reference audio. If omitted with a local file, ASR may be attempted.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--speed", type=float, default=default_speed, help="Speech speed multiplier.")
    ap.add_argument("--nfe-steps", type=int, default=nfe_step, help="Denoising steps.")
    ap.add_argument("--max-concurrent", type=int, default=2)

    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Starting Thai F5-TTS server on %s:%d", args.host, args.port)

    engine = ThaiF5Engine(
        model_version=args.model_version,
        ckpt_file=args.ckpt_file,
        vocab_file=args.vocab_file,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        device=args.device,
        speed=args.speed,
        nfe_steps=args.nfe_steps,
    )
    sem = asyncio.Semaphore(args.max_concurrent)

    uri = f"tcp://{args.host}:{args.port}"
    server = AsyncServer.from_uri(uri)
    logging.info("Listening at %s", uri)

    class BoundThaiF5Handler(ThaiF5Handler):
        def __init__(self, reader, writer):
            super().__init__(reader=reader, writer=writer, engine=engine, sem=sem)

    await server.run(handler_factory=lambda r, w: BoundThaiF5Handler(r, w))


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
