#!/usr/bin/env python3
from f5_tts.infer.utils_infer import (
    mel_spec_type,             # "vocos" (24 kHz)
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    # speed as default_speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT
from cached_path import cached_path
import torch
from wyoming.error import Error
from wyoming.info import Info, Describe, TtsProgram, TtsVoice, Attribution
from wyoming.audio import (
    AudioStart,
    AudioChunk,
    AudioStop,
)
from wyoming.tts import (
    Synthesize,
    SynthesizeStart,
    SynthesizeChunk,
    SynthesizeStop,
    SynthesizeStopped,
)
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.event import Event
from pythainlp.tokenize import sent_tokenize
import argparse
import asyncio
import logging
import os
import re
import unicodedata
from typing import List
import numpy as np

from util.cleantext import process_thai_repeat, replace_numbers_with_thai

SPEAK_SPEED = float(os.getenv("THTTS_SPEAK_SPEED", "0.8"))        # 0.5x, 1x, 1.5x, 2x
MIN_CHARS = 48                                                    # flush when buffer reaches this length
MAX_WAIT_MS = int(os.getenv("THTTS_MAX_WAIT_MS", "220"))          # flush if idle for this many ms
MAX_SENT_LEN = 180                                                # if a 'sentence' is too long, treat as complete to avoid stalling
TERMINATORS = {"।", "?", "!", "…", "\n"}                          # Thai often lacks punctuation; timeout/length still cover
MIN_SENT_CHARS = int(os.getenv("THTTS_MIN_SENT_CHARS", "15"))     # do not emit a sentence shorter than this unless final flush
EMOJI_PATTERN = re.compile(
    "["                       # Common emoji blocks + variation selectors
    "\U0001F300-\U0001F6FF"   # Misc Symbols and Pictographs, Transport & Map
    "\U0001F900-\U0001F9FF"   # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"   # Symbols and Pictographs Extended-A
    "\u2600-\u27BF"           # Misc symbols
    "\uFE0E\uFE0F"            # variation selectors
    "]+",
    flags=re.UNICODE
)
CONTROL_PATTERN = re.compile(r"[\u0000-\u001F\u007F]")
ZW_PATTERN = re.compile(r"[\u200B-\u200D\u2060]")  # zero-width chars
MULTISPACE = re.compile(r"[ \t\u00A0]{2,}")
THAI_SENT_END = re.compile(r"([\.!\?…]|[\u0E2F])")  # ., !, ?, …, ฯ

# -----------------------
# Utils
# -----------------------


def float32_to_int16_pcm(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    x16 = (x * 32767.0).astype(np.int16)
    return x16.tobytes()


def normalize_thai_text(text: str) -> str:
    """
    Conservative normalization aligned with F5-TTS-THAI WebUI:
      - NFC normalize
      - unify common quotes/dashes
      - strip emoji, zero-width & control chars
      - collapse excessive whitespace
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    t = t.replace("–", "-").replace("—", "-")
    t = EMOJI_PATTERN.sub("", t)
    t = ZW_PATTERN.sub("", t)
    t = CONTROL_PATTERN.sub("", t)
    t = MULTISPACE.sub(" ", t)
    return t.strip()


def preprocess_thai(text: str) -> str:
    """Full Thai preprocessing (normalize + optional Thai-digit mapping)."""
    t = replace_numbers_with_thai(text)
    t = process_thai_repeat(t)
    t = normalize_thai_text(t)
    return t


def split_sentences_th(text: str) -> List[str]:
    splitted = sent_tokenize(text, keep_whitespace=False, engine="thaisum")
    # logging.debug(f"Splitted sentences to: len={len(splitted)} {splitted}")
    return splitted


def _split_ready_vs_tail(text: str, *, final: bool = False) -> tuple[list[str], str]:
    """
    Tokenize Thai into sentences and return (ready_sentences, tail_remainder).
    Strategy:
      - If >=2 sentences, treat all but the last as ready; keep last as tail.
      - If only 1 sentence:
          - If it ends with a terminator or is very long, treat as ready.
          - Else keep as tail (incomplete).
      - Additionally, never emit a ready sentence with length < MIN_SENT_CHARS
        by coalescing it with the next sentence — unless final=True.
    """
    sents = split_sentences_th(text)
    if not sents:
        return [], ""

    if len(sents) >= 2:
        base = sents[:-1]
        last = sents[-1]
        # Coalesce short sentences in 'base' so we only emit items >= MIN_SENT_CHARS (unless final).
        ready: list[str] = []
        acc = ""
        for s in base:
            if not final and (len(s) < MIN_SENT_CHARS):
                acc += s
                if len(acc) >= MIN_SENT_CHARS:
                    ready.append(acc)
                    acc = ""
            else:
                if acc:
                    # prefer to attach short acc to this sentence if it keeps it coherent
                    merged = acc + s
                    if not final and len(merged) < MIN_SENT_CHARS:
                        acc = merged
                    else:
                        ready.append(merged)
                        acc = ""
                else:
                    ready.append(s)
        # Whatever remains in acc is too short; push it into the tail.
        tail = (acc + last)
        # If tail is obviously complete, we may emit it too.
        if tail and (tail[-1] in TERMINATORS or len(tail) >= MAX_SENT_LEN or final):
            if final or len(tail) >= MIN_SENT_CHARS:
                ready.append(tail)
                tail = ""
        return ready, tail

    # single sentence
    s = sents[0]
    if s and (s[-1] in TERMINATORS or len(s) >= MAX_SENT_LEN or final or len(s) >= MIN_CHARS):
        # Only emit if it's long enough, unless final=True
        if final or len(s) >= MIN_SENT_CHARS:
            return [s], ""
        else:
            # too short and not final → keep waiting
            return [], s
    return [], s


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
        speed: float = SPEAK_SPEED,
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
        self._buf: list[str] = []   # list of chunk strings
        self._flush_task = None     # asyncio.Task or None
        # Single audio stream per request (low-latency continuous playback)
        self._audio_started = False
        self._rate = self.engine.sr
        self._width = 2
        self._channels = 1
        self._chunk_samples = int(self.engine.sr * 0.2) or 1  # ~200ms per chunk
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
                text = preprocess_thai(syn.text or "")
                logging.info("Synthesize (oneshot): %r", text)
                await self._speak_text(text, standalone=True)
                return True

            if SynthesizeStart.is_type(event.type):
                self._streaming = True
                self._reset_buffer()
                self._audio_started = False
                logging.info("Synthesize streaming START: %s", event)
                # Prime playback immediately so the player opens.
                await self._ensure_audio_started()
                import numpy as _np
                _sil = _np.zeros(int(self._rate * 0.12), dtype=_np.float32)  # ~120 ms
                await self.write_event(AudioChunk(
                    rate=self._rate, width=self._width, channels=self._channels,
                    audio=float32_to_int16_pcm(_sil)
                ).event())
                return True

            if SynthesizeChunk.is_type(event.type):
                chunk = SynthesizeChunk.from_event(event)
                # IMPORTANT: do NOT normalize/strip here.
                # We must preserve '\n' so the sentence splitter can use it.
                text = chunk.text or ""
                if text == "":
                    logging.debug("Empty chunk")
                    return True

                # Accumulate
                self._buf.append(text)
                buf_str = "".join(self._buf)
                # logging.debug("Accumulated chunk; buffer_len=%d (just got: %r)", len(buf_str), text)

                # Peek at sentence segmentation to see if we have a *complete* sentence
                sents = split_sentences_th(buf_str)
                if len(sents) >= 2:
                    # We have at least one complete sentence; flush the ready part(s) now
                    await self._flush_buffer()
                elif buf_str and buf_str[-1] in TERMINATORS:
                    # Single sentence but explicitly terminated; safe to flush
                    await self._flush_buffer()
                else:
                    # Still constructing the first/only sentence → do NOT flush now.
                    # Re-arm the idle timer so we don't stall if the producer pauses.
                    self._schedule_idle_flush()

                return True

            if SynthesizeStop.is_type(event.type):
                logging.info("Synthesize streaming STOP")
                # Flush any remaining text first
                await self._flush_buffer(force_all=True)
                # Close the single audio stream if we opened it
                if self._audio_started:
                    await self.write_event(AudioStop().event())
                    self._audio_started = False
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

    async def _ensure_audio_started(self):
        if not self._audio_started:
            await self.write_event(AudioStart(rate=self._rate, width=self._width, channels=self._channels).event())
            self._audio_started = True

    async def _speak_text(self, text: str, *, standalone: bool = False):
        if not text:
            logging.debug("Empty text; skip speak")
            return
        rate, width, channels = self._rate, self._width, self._channels
        # Make sure no pending idle task fires mid-speak (streaming path)
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            self._flush_task = None
        # Preprocess fully (incl. digit mapping) right before synth
        text = preprocess_thai(text)
        logging.debug(f"Preprocessed text: {text}")

        if standalone:
            # one-shot mode: its own AudioStart/Stop
            await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        else:
            # streaming mode: ensure single AudioStart
            await self._ensure_audio_started()

        loop = asyncio.get_running_loop()
        async with self.sem:
            wav = await loop.run_in_executor(None, self.engine.synth_blocking, text)

        total_bytes = 0
        for i in range(0, len(wav), self._chunk_samples):
            chunk = wav[i:i + self._chunk_samples]
            payload = float32_to_int16_pcm(chunk)
            total_bytes += len(payload)
            await self.write_event(AudioChunk(rate=rate, width=width, channels=channels, audio=payload).event())
        if standalone:
            await self.write_event(AudioStop().event())

        logging.info("Streamed audio chunk(s): text_len=%d samples=%d bytes=%d standalone=%s",
                     len(text), len(wav), total_bytes, standalone)

    def _reset_buffer(self):
        self._buf = []
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = None
        logging.debug("Resetted buffer")

    def _schedule_idle_flush(self):
        # Cancel any previous idle flush task and schedule a new one
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

        self._flush_task = asyncio.create_task(self._idle_wait_and_flush())

    async def _idle_wait_and_flush(self):
        try:
            await asyncio.sleep(MAX_WAIT_MS / 1000.0)
            # Only flush if we truly have ready sentences; otherwise keep waiting.
            if not self._buf:
                return
            buf_str = "".join(self._buf)
            sents = split_sentences_th(buf_str)
            if (
                len(sents) >= 2
                or (buf_str and buf_str[-1] in TERMINATORS)
                or len(buf_str) >= MAX_SENT_LEN
                or len(buf_str) >= MIN_CHARS
            ):
                await self._flush_buffer()
            else:
                # Not ready yet; re-arm the timer to check again later.
                self._schedule_idle_flush()
        except asyncio.CancelledError:
            pass

    async def _flush_buffer(self, force_all: bool = False):
        """
        Flush accumulated text:
          - If force_all=True, synth everything in the buffer (no remainder).
          - Else, synth only full sentences and keep tail remainder.
        """
        logging.debug(f"Flushing buffer force_all={force_all}")
        if not self._buf:
            return

        buf_str = "".join(self._buf)
        ready_sents: list[str]
        tail: str

        if force_all:
            # Treat the entire buffer as ready (split just to get clean sentences)
            # final=True allows emitting < MIN_SENT_CHARS at end of stream
            ready_sents, tail = _split_ready_vs_tail(buf_str, final=True)
            tail = ""  # by definition, final
        else:
            ready_sents, tail = _split_ready_vs_tail(buf_str, final=False)

        if ready_sents:
            logging.info("Flushing %d ready sentence(s)", len(ready_sents))
            # Prevent a racing idle task from re-flushing the same sentences
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                self._flush_task = None
            # Move tail back to buffer BEFORE we speak, so any concurrent timer sees only the tail
            self._buf = [tail] if tail else []
            for sentence in ready_sents:
                await self._speak_text(sentence, standalone=False)

        else:
            # No ready sentences; keep the current buffer as-is
            pass

        # If there is still a tail, keep the idle flush armed (so it won't stall forever)
        if self._buf:
            self._schedule_idle_flush()


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
    ap.add_argument("--speed", type=float, default=SPEAK_SPEED, help="Speech speed multiplier.")
    ap.add_argument("--nfe-steps", type=int, default=nfe_step, help="Denoising steps.")
    ap.add_argument("--max-concurrent", type=int, default=1, help="Legacy params, do not change")

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
    sem = asyncio.Semaphore(args.max_concurrent)  # TODO: more than 1 is broken

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
