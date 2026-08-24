"""Protocol state-machine tests with a fake backend, no model required."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from wyoming.event import Event
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeVoice,
)

from thtts.backends.base import BackendMetadata, SynthesisResult, UnknownVoiceError, VoiceMetadata
from thtts.config import StreamSettings
from thtts.handler import TtsEventHandler
from thtts.info import make_info


class _FakeBackend:
    metadata = BackendMetadata(
        backend_id="fake",
        program_name="fake",
        program_description="fake backend",
        attribution_name="test",
        attribution_url="https://example.test",
    )
    voices = (
        VoiceMetadata(
            name="default",
            description="default test voice",
            attribution_name="test",
            attribution_url="https://example.test",
        ),
    )
    resolved_device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def load(self) -> None:
        return None

    async def synthesize(self, text: str, *, voice_name: str | None) -> SynthesisResult:
        self.calls.append((text, voice_name))
        if voice_name not in {None, "default"}:
            raise UnknownVoiceError(voice_name)
        return SynthesisResult(
            waveform=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            sample_rate=24_000,
            queue_seconds=0.0,
            inference_seconds=0.0,
        )


class _NonFiniteBackend(_FakeBackend):
    async def synthesize(self, text: str, *, voice_name: str | None) -> SynthesisResult:
        return SynthesisResult(
            waveform=np.array([float("nan")], dtype=np.float32),
            sample_rate=24_000,
            queue_seconds=0.0,
            inference_seconds=0.0,
        )


class _SlowBackend(_FakeBackend):
    async def synthesize(self, text: str, *, voice_name: str | None) -> SynthesisResult:
        await asyncio.sleep(0.05)
        return await super().synthesize(text, voice_name=voice_name)


class _Handler(TtsEventHandler):
    def __init__(self, backend: _FakeBackend) -> None:
        self.events: list[Event] = []
        super().__init__(
            make_info(backend),
            backend,
            None,
            None,
            stream_settings=StreamSettings(
                idle_flush_ms=1000,
                min_segment_chars=15,
                target_chars=48,
                max_segment_chars=180,
                output_chunk_milliseconds=200,
            ),
        )

    async def write_event(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_describe_uses_backend_metadata() -> None:
    backend = _FakeBackend()
    handler = _Handler(backend)
    assert await handler.handle_event(Event(type="describe"))
    assert handler.events[0].type == "info"


@pytest.mark.asyncio
async def test_oneshot_writes_one_complete_audio_stream() -> None:
    backend = _FakeBackend()
    handler = _Handler(backend)
    event = Synthesize(text="hello", voice=SynthesizeVoice(name="default")).event()
    await handler.handle_event(event)
    assert backend.calls == [("hello", "default")]
    assert [event.type for event in handler.events] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]


@pytest.mark.asyncio
async def test_streaming_short_terminated_text_uses_one_audio_stream() -> None:
    backend = _FakeBackend()
    handler = _Handler(backend)
    await handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="default")).event())
    await handler.handle_event(SynthesizeChunk(text="สวัสดี!").event())
    await handler.handle_event(SynthesizeStop().event())
    assert backend.calls == [("สวัสดี!", "default")]
    assert [event.type for event in handler.events] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
        "synthesize-stopped",
    ]


@pytest.mark.asyncio
async def test_streaming_ignores_home_assistant_legacy_synthesize_mirror() -> None:
    backend = _FakeBackend()
    handler = _Handler(backend)
    await handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="default")).event())
    await handler.handle_event(SynthesizeChunk(text="สวัสดี!").event())
    # Home Assistant sends the full one-shot request after chunks for legacy
    # Wyoming servers. The modern stream must not synthesize it twice.
    await handler.handle_event(
        Synthesize(text="สวัสดี!", voice=SynthesizeVoice(name="default")).event()
    )
    await handler.handle_event(SynthesizeStop().event())
    assert backend.calls == [("สวัสดี!", "default")]
    assert [event.type for event in handler.events] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
        "synthesize-stopped",
    ]


@pytest.mark.asyncio
async def test_streaming_accepts_following_text_while_prior_segment_synthesizes() -> None:
    backend = _SlowBackend()
    handler = _Handler(backend)
    await handler.handle_event(SynthesizeStart().event())

    await handler.handle_event(SynthesizeChunk(text="first.").event())
    started = asyncio.get_running_loop().time()
    await handler.handle_event(SynthesizeChunk(text="second.").event())
    accepted_in = asyncio.get_running_loop().time() - started
    await handler.handle_event(SynthesizeStop().event())

    assert accepted_in < 0.025
    assert backend.calls == [("first.", None), ("second.", None)]


@pytest.mark.asyncio
async def test_streaming_empty_input_has_no_placeholder_audio() -> None:
    handler = _Handler(_FakeBackend())
    await handler.handle_event(SynthesizeStart().event())
    await handler.handle_event(SynthesizeStop().event())
    assert [event.type for event in handler.events] == ["synthesize-stopped"]


@pytest.mark.asyncio
async def test_unknown_voice_becomes_stable_error() -> None:
    handler = _Handler(_FakeBackend())
    await handler.handle_event(Synthesize(text="hello", voice=SynthesizeVoice(name="nope")).event())
    assert handler.events[-1].type == "error"
    assert handler.events[-1].data["code"] == "tts-voice-not-found"


@pytest.mark.asyncio
async def test_nonfinite_backend_audio_becomes_a_safe_stream_error() -> None:
    handler = _Handler(_NonFiniteBackend())
    await handler.handle_event(Synthesize(text="hello").event())
    assert [event.type for event in handler.events] == ["error"]
    assert handler.events[-1].data["code"] == "tts-stream-failed"


@pytest.mark.asyncio
async def test_stream_chunk_without_start_is_protocol_error() -> None:
    handler = _Handler(_FakeBackend())
    await handler.handle_event(SynthesizeChunk(text="hello").event())
    assert handler.events[-1].data["code"] == "tts-invalid-event-order"


@pytest.mark.asyncio
async def test_disconnect_cancels_idle_flush_task() -> None:
    handler = _Handler(_FakeBackend())
    await handler.handle_event(SynthesizeStart().event())
    await handler.handle_event(SynthesizeChunk(text="short").event())
    await handler.disconnect()
    await asyncio.sleep(0)
    assert handler.events == []


@pytest.mark.asyncio
async def test_idle_flush_processes_a_complete_unterminated_segment_once() -> None:
    backend = _FakeBackend()
    handler = _Handler(backend)
    handler._stream_settings = StreamSettings(  # type: ignore[attr-defined]
        idle_flush_ms=1,
        min_segment_chars=15,
        target_chars=48,
        max_segment_chars=180,
    )
    await handler.handle_event(SynthesizeStart().event())
    await handler.handle_event(SynthesizeChunk(text="ก" * 15).event())
    await asyncio.sleep(0.02)
    await handler.handle_event(SynthesizeStop().event())
    assert backend.calls == [("ก" * 15, None)]
    assert [event.type for event in handler.events] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
        "synthesize-stopped",
    ]
