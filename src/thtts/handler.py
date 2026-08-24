"""Common Wyoming TTS event handler for every selected backend."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .audio import iter_pcm_chunks
from .backends.base import (
    BackendBusyError,
    BackendError,
    BackendQueueTimeoutError,
    SynthesisResult,
    TtsBackend,
    UnknownVoiceError,
)
from .config import StreamSettings
from .text import TextSegmenter

_LOGGER = logging.getLogger(__name__)
_MAX_PENDING_STREAM_SEGMENTS = 4


class TtsEventHandler(AsyncEventHandler):
    """Owns one client connection and all request-local streaming state."""

    def __init__(
        self,
        wyoming_info: Info,
        backend: TtsBackend,
        *args: Any,
        stream_settings: StreamSettings,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info_event = wyoming_info.event()
        self._backend = backend
        self._stream_settings = stream_settings
        self._streaming = False
        self._voice_name: str | None = None
        self._segmenter: TextSegmenter | None = None
        self._audio_started = False
        self._audio_sample_rate: int | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._stream_queue: asyncio.Queue[str | None] | None = None
        self._stream_worker: asyncio.Task[None] | None = None
        self._stream_failure: BackendError | None = None
        self._stream_closing = False
        self._state_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self._send_event(self._info_event)
                return True
            if Synthesize.is_type(event.type):
                synthesize = Synthesize.from_event(event)
                if self._streaming:
                    return await self._handle_stream_compatibility_synthesize(synthesize)
                return await self._handle_oneshot(synthesize)
            if SynthesizeStart.is_type(event.type):
                return await self._handle_start(SynthesizeStart.from_event(event))
            if SynthesizeChunk.is_type(event.type):
                return await self._handle_chunk(SynthesizeChunk.from_event(event))
            if SynthesizeStop.is_type(event.type):
                return await self._handle_stop()

            await self._write_error("Unsupported Wyoming TTS event", "tts-invalid-event")
            return True
        except (KeyError, TypeError, ValueError):
            _LOGGER.warning("Malformed Wyoming TTS event")
            await self._write_error("Malformed Wyoming TTS event", "tts-invalid-event")
            await self._reset()
            return True
        except BackendBusyError:
            await self._write_error("TTS inference queue is full; retry later", "tts-busy")
            await self._reset()
            return True
        except BackendQueueTimeoutError:
            await self._write_error("Timed out waiting for TTS capacity", "tts-timeout")
            await self._reset()
            return True
        except UnknownVoiceError:
            await self._write_error("Unknown TTS voice", "tts-voice-not-found")
            await self._reset()
            return True
        except BackendError:
            _LOGGER.error("TTS backend failed")
            await self._write_error("TTS synthesis failed", "tts-stream-failed")
            await self._reset()
            return True

    async def disconnect(self) -> None:
        await self._reset()

    async def _handle_oneshot(self, synthesize: Synthesize) -> bool:
        async with self._state_lock:
            if self._streaming:
                await self._write_error(
                    "Cannot use one-shot synthesis while a streamed request is active",
                    "tts-invalid-event-order",
                )
                return True
            voice_name = synthesize.voice.name if synthesize.voice else None
            result = await self._backend.synthesize(synthesize.text or "", voice_name=voice_name)
            await self._write_result(result, close_stream=True)
            await self._reset_audio_state()
        return True

    async def _handle_start(self, start: SynthesizeStart) -> bool:
        async with self._state_lock:
            if self._streaming:
                await self._write_error(
                    "A streamed synthesis request is already active", "tts-invalid-event-order"
                )
                return True
            self._streaming = True
            self._stream_closing = False
            self._stream_failure = None
            self._voice_name = start.voice.name if start.voice else None
            self._segmenter = TextSegmenter(
                minimum_chars=self._stream_settings.min_segment_chars,
                target_chars=self._stream_settings.target_chars,
                maximum_chars=self._stream_settings.max_segment_chars,
            )
            await self._reset_audio_state()
            self._stream_queue = asyncio.Queue(maxsize=_MAX_PENDING_STREAM_SEGMENTS)
            self._stream_worker = asyncio.create_task(
                self._run_stream_worker(self._stream_queue, self._voice_name)
            )
        return True

    async def _handle_stream_compatibility_synthesize(self, synthesize: Synthesize) -> bool:
        """Ignore Wyoming's full-text mirror sent inside a streamed request.

        Home Assistant sends this legacy one-shot event after all streamed
        chunks and before ``synthesize-stop`` so older servers can still
        synthesize the complete message. This handler has already accepted the
        chunks, so processing the mirror would duplicate audio.
        """

        async with self._state_lock:
            if self._streaming:
                _LOGGER.debug("Ignoring compatibility synthesize event within active stream")
                return True
        return await self._handle_oneshot(synthesize)

    async def _handle_chunk(self, chunk: SynthesizeChunk) -> bool:
        async with self._state_lock:
            if not self._streaming or self._stream_closing or self._segmenter is None:
                await self._write_error(
                    "Received synthesize-chunk without synthesize-start",
                    "tts-invalid-event-order",
                )
                return True
            if self._stream_failure is not None:
                await self._reset()
                return True
            await self._cancel_idle_task()
            self._queue_stream_segments(self._segmenter.add(chunk.text or ""))
            if self._segmenter.pending_text:
                self._idle_task = asyncio.create_task(self._idle_flush())
        return True

    async def _handle_stop(self) -> bool:
        async with self._state_lock:
            if not self._streaming or self._segmenter is None:
                await self._write_error(
                    "Received synthesize-stop without synthesize-start",
                    "tts-invalid-event-order",
                )
                return True
            self._stream_closing = True
            await self._cancel_idle_task()
            final_segments = self._segmenter.final_flush()
            queue = self._stream_queue
            worker = self._stream_worker

        if queue is not None and worker is not None and self._stream_failure is None:
            for text in final_segments:
                if not await self._enqueue_stream_segment_when_ready(queue, worker, text):
                    break
            await self._enqueue_stream_segment_when_ready(queue, worker, None)

        if worker is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        if self._stream_failure is not None:
            await self._reset()
            return True
        if self._audio_started:
            await self._send_event(AudioStop().event())
        await self._send_event(SynthesizeStopped().event())
        await self._reset()
        return True

    async def _idle_flush(self) -> None:
        try:
            await asyncio.sleep(self._stream_settings.idle_flush_ms / 1000.0)
            async with self._state_lock:
                if (
                    not self._streaming
                    or self._stream_closing
                    or self._stream_failure is not None
                    or self._segmenter is None
                ):
                    return
                queue = self._stream_queue
                if queue is None:
                    return
                if queue.full():
                    self._idle_task = asyncio.create_task(self._idle_flush())
                    return
                self._queue_stream_segments(self._segmenter.idle_flush())
        except asyncio.CancelledError:
            raise

    def _queue_stream_segments(self, texts: list[str]) -> None:
        queue = self._stream_queue
        if queue is None:
            raise BackendError("TTS stream worker is unavailable")
        for text in texts:
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull as err:
                raise BackendBusyError("TTS stream segment queue is full") from err

    async def _enqueue_stream_segment_when_ready(
        self, queue: asyncio.Queue[str | None], worker: asyncio.Task[None], text: str | None
    ) -> bool:
        """Wait only while stopping; normal chunks never block the input reader."""

        while not worker.done():
            try:
                queue.put_nowait(text)
                return True
            except asyncio.QueueFull:
                await asyncio.sleep(0.005)
        return False

    async def _run_stream_worker(
        self, queue: asyncio.Queue[str | None], voice_name: str | None
    ) -> None:
        """Serialize synthesis/output while the handler keeps accepting text chunks."""

        try:
            while True:
                text = await queue.get()
                try:
                    if text is None:
                        return
                    result = await self._backend.synthesize(text, voice_name=voice_name)
                    await self._write_result(result, close_stream=False)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except BackendError as err:
            await self._record_stream_failure(err)

    async def _record_stream_failure(self, error: BackendError) -> None:
        async with self._state_lock:
            self._stream_failure = error
        if isinstance(error, BackendBusyError):
            await self._write_error("TTS inference queue is full; retry later", "tts-busy")
        elif isinstance(error, BackendQueueTimeoutError):
            await self._write_error("Timed out waiting for TTS capacity", "tts-timeout")
        elif isinstance(error, UnknownVoiceError):
            await self._write_error("Unknown TTS voice", "tts-voice-not-found")
        else:
            _LOGGER.error("TTS backend failed")
            await self._write_error("TTS synthesis failed", "tts-stream-failed")

    async def _write_result(self, result: SynthesisResult, *, close_stream: bool) -> None:
        waveform = np.asarray(result.waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise BackendError("TTS backend returned non-mono audio")
        if not np.isfinite(waveform).all():
            raise BackendError("TTS backend returned non-finite audio")
        if len(waveform) == 0:
            return
        if self._audio_sample_rate is None:
            self._audio_sample_rate = result.sample_rate
        elif self._audio_sample_rate != result.sample_rate:
            raise BackendError("Backend changed sample rate during one synthesis stream")

        if not self._audio_started:
            await self._send_event(AudioStart(rate=result.sample_rate, width=2, channels=1).event())
            self._audio_started = True
        for payload in iter_pcm_chunks(
            waveform,
            sample_rate=result.sample_rate,
            chunk_milliseconds=self._stream_settings.output_chunk_milliseconds,
        ):
            await self._send_event(
                AudioChunk(rate=result.sample_rate, width=2, channels=1, audio=payload).event()
            )
        if close_stream and self._audio_started:
            await self._send_event(AudioStop().event())

    async def _cancel_idle_task(self) -> None:
        task = self._idle_task
        self._idle_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _reset_audio_state(self) -> None:
        self._audio_started = False
        self._audio_sample_rate = None

    async def _reset(self) -> None:
        await self._cancel_idle_task()
        worker = self._stream_worker
        self._stream_worker = None
        self._stream_queue = None
        self._streaming = False
        self._stream_closing = False
        self._stream_failure = None
        self._voice_name = None
        self._segmenter = None
        await self._reset_audio_state()
        if worker is not None and worker is not asyncio.current_task() and not worker.done():
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker

    async def _send_event(self, event: Event) -> None:
        async with self._write_lock:
            await self.write_event(event)

    async def _write_error(self, text: str, code: str) -> None:
        try:
            await self._send_event(Event(type="error", data={"text": text, "code": code}))
        except (BrokenPipeError, ConnectionError):
            pass
