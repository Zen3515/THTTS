"""Serialized TCP protocol checks, including a captured Wyoming 1.7.2 input."""

from __future__ import annotations

import asyncio
from functools import partial

import numpy as np
import pytest
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop

from thtts.backends.base import BackendMetadata, SynthesisResult, VoiceMetadata
from thtts.config import StreamSettings
from thtts.handler import TtsEventHandler
from thtts.info import make_info


class _ProtocolBackend:
    metadata = BackendMetadata(
        backend_id="fake",
        program_name="fake",
        program_description="fake backend",
        attribution_name="test",
        attribution_url="https://example.test",
    )
    voices = (VoiceMetadata("default", "test", "test", "https://example.test"),)
    resolved_device = "cpu"

    async def load(self) -> None:
        return None

    async def synthesize(self, text: str, *, voice_name: str | None) -> SynthesisResult:
        return SynthesisResult(
            waveform=np.array([0.0, 0.5], dtype=np.float32),
            sample_rate=24_000,
            queue_seconds=0.0,
            inference_seconds=0.0,
        )


async def _start_server() -> tuple[AsyncServer, int]:
    backend = _ProtocolBackend()
    server = AsyncServer.from_uri("tcp://127.0.0.1:0")
    factory = partial(
        TtsEventHandler,
        make_info(backend),
        backend,
        stream_settings=StreamSettings(),
    )
    await server.start(factory)
    listener = server._server
    assert listener is not None
    return server, listener.sockets[0].getsockname()[1]


async def _stop_server(server: AsyncServer) -> None:
    listener = getattr(server, "_server", None)
    await server.stop()
    if listener is not None:
        await listener.wait_closed()


@pytest.mark.asyncio
async def test_tcp_describe_round_trip() -> None:
    server, port = await _start_server()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        await async_write_event(Describe().event(), writer)
        event = await asyncio.wait_for(async_read_event(reader), 1.0)
        assert event is not None
        assert Info.is_type(event.type)
    finally:
        writer.close()
        await writer.wait_closed()
        await _stop_server(server)


@pytest.mark.asyncio
async def test_tcp_replays_legacy_1_7_wire_fixture() -> None:
    server, port = await _start_server()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        # Wyoming's server accepts this structural capture from the previous
        # locked client version; the protocol reader intentionally ignores the
        # client version while preserving the event contract.
        writer.write(b'{"type":"describe","version":"1.7.2"}\n')
        await writer.drain()
        event = await asyncio.wait_for(async_read_event(reader), 1.0)
        assert event is not None and Info.is_type(event.type)
    finally:
        writer.close()
        await writer.wait_closed()
        await _stop_server(server)


@pytest.mark.asyncio
async def test_tcp_oneshot_audio_contract() -> None:
    server, port = await _start_server()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        await async_write_event(Synthesize(text="hello").event(), writer)
        events = [await asyncio.wait_for(async_read_event(reader), 1.0) for _ in range(3)]
        assert [event.type for event in events if event is not None] == [
            "audio-start",
            "audio-chunk",
            "audio-stop",
        ]
    finally:
        writer.close()
        await writer.wait_closed()
        await _stop_server(server)


@pytest.mark.asyncio
async def test_tcp_home_assistant_streaming_sequence_does_not_duplicate_audio() -> None:
    """Accept HA's legacy complete-text mirror within a streaming request."""

    server, port = await _start_server()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        await async_write_event(SynthesizeStart().event(), writer)
        await async_write_event(SynthesizeChunk(text="สวัสดี!").event(), writer)
        await async_write_event(Synthesize(text="สวัสดี!").event(), writer)
        await async_write_event(SynthesizeStop().event(), writer)
        events = [await asyncio.wait_for(async_read_event(reader), 1.0) for _ in range(4)]
        assert [event.type for event in events if event is not None] == [
            "audio-start",
            "audio-chunk",
            "audio-stop",
            "synthesize-stopped",
        ]
    finally:
        writer.close()
        await writer.wait_closed()
        await _stop_server(server)
