#!/usr/bin/env python3
import argparse
import asyncio
import logging
import re
from typing import List
import numpy as np
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

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


class ThaiVITSEngine:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id
        logging.info("Loading tokenizer/model: %s", model_id)
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id)
        self.model.eval()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        set_seed(456)
        self.sr = int(getattr(self.model.config, "sampling_rate", 22050))
        logging.info("Engine ready: device=%s sr=%d", self.device, self.sr)

    @torch.inference_mode()
    def synth_blocking(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        logging.debug("Synth start (blocking): %r", text)
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        wav = out.waveform[0].detach().cpu().numpy().astype(np.float32)
        logging.debug("Synth done: samples=%d duration=%.3fs",
                      len(wav), len(wav) / max(self.sr, 1))
        return wav


class ThaiVITSHandler(AsyncEventHandler):
    def __init__(self, *args, engine: ThaiVITSEngine, sem: asyncio.Semaphore, **kwargs):
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
                            name="thai-vits-female",
                            attribution=Attribution(
                                name="Hugging Face",
                                url="https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-FEMALEV2",
                            ),
                            voices=[
                                TtsVoice(
                                    name="thai-female",
                                    attribution=Attribution(
                                        name="Hugging Face",
                                        url="https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-FEMALEV2",
                                    ),
                                    languages=["th", "th-TH"],
                                    description=None,
                                    installed=True,
                                    version="1.0",
                                ),
                                TtsVoice(  # alias for clients expecting a default voice
                                    name="default",
                                    attribution=Attribution(
                                        name="Hugging Face",
                                        url="https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-FEMALEV2",
                                    ),
                                    languages=["th", "th-TH"],
                                    description="Default Thai female (alias)",
                                    installed=True,
                                    version="1.0",
                                ),
                            ],
                            installed=True,
                            description="Thai VITS female voice (Hugging Face)",
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
                    # This means that we are doing a streaming synthesize,
                    # one shot synthesize is recieved for legacy clients, we can ignore this message.
                    logging.debug("Ignoring legacy one-shot 'synthesize' (streaming already in progress/done)")
                    return True
                syn: Synthesize = Synthesize.from_event(event)
                text = (syn.text or "").strip()
                logging.info("Synthesize (oneshot): %r", text)
                await self._speak_text(text)
                # if self._streaming:
                #     await self.write_event(SynthesizeStopped().event())
                # self._streaming = False
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
        rate, width, channels = self.engine.sr, 2, 1
        logging.debug("AudioStart: rate=%d width=%d channels=%d", rate, width, channels)
        await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())

        loop = asyncio.get_running_loop()
        async with self.sem:
            wav = await loop.run_in_executor(None, self.engine.synth_blocking, text)

        samples_per_chunk = int(self.engine.sr * 0.2) or 1
        total_bytes = 0
        for i in range(0, len(wav), samples_per_chunk):
            chunk = wav[i:i + samples_per_chunk]
            payload = float32_to_int16_pcm(chunk)
            total_bytes += len(payload)
            await self.write_event(AudioChunk(rate=rate, width=width, channels=channels, audio=payload).event())
        await self.write_event(AudioStop().event())
        logging.info("Streamed audio: text_len=%d samples=%d bytes=%d",
                     len(text), len(wav), total_bytes)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10200)
    ap.add_argument("--model-id", default="VIZINTZOR/MMS-TTS-THAI-FEMALEV2")
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--device", default="auto")  # "cpu" or "cuda" or "auto"
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Starting Thai VITS server on %s:%d", args.host, args.port)
    engine = ThaiVITSEngine(args.model_id, device=args.device)
    sem = asyncio.Semaphore(args.max_concurrent)

    uri = f"tcp://{args.host}:{args.port}"
    server = AsyncServer.from_uri(uri)
    logging.info("Listening at %s", uri)

    class BoundThaiVITSHandler(ThaiVITSHandler):
        def __init__(self, reader, writer):
            super().__init__(reader=reader, writer=writer, engine=engine, sem=sem)

    await server.run(handler_factory=lambda r, w: BoundThaiVITSHandler(r, w))


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
