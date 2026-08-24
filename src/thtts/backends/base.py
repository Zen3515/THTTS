"""Stable backend contract shared by every THTTS inference method."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

_T = TypeVar("_T")


class BackendError(RuntimeError):
    """Base error returned by an inference backend."""


class BackendBusyError(BackendError):
    """Raised when bounded synthesis admission rejects a request."""


class BackendQueueTimeoutError(BackendError):
    """Raised when an admitted request waits too long for inference."""


class UnknownVoiceError(BackendError):
    """Raised when a request selects a voice unavailable from this backend."""


@dataclass(frozen=True)
class VoiceMetadata:
    name: str
    description: str | None
    attribution_name: str
    attribution_url: str
    languages: tuple[str, ...] = ("th", "th-TH")
    version: str = "1.0"


@dataclass(frozen=True)
class BackendMetadata:
    backend_id: str
    program_name: str
    program_description: str
    attribution_name: str
    attribution_url: str
    version: str = "1.0"
    supports_synthesize_streaming: bool = True


@dataclass(frozen=True)
class SynthesisResult:
    waveform: np.ndarray
    sample_rate: int
    queue_seconds: float
    inference_seconds: float


class TtsBackend(Protocol):
    """Interface consumed by Wyoming metadata and the event handler."""

    @property
    def metadata(self) -> BackendMetadata: ...

    @property
    def voices(self) -> tuple[VoiceMetadata, ...]: ...

    @property
    def resolved_device(self) -> str | None: ...

    async def load(self) -> None: ...

    async def synthesize(self, text: str, *, voice_name: str | None) -> SynthesisResult: ...


class AdmissionController:
    """Bound model execution without allowing an unbounded waiting queue."""

    def __init__(
        self, *, maximum_active: int, maximum_queued: int, maximum_wait_seconds: float
    ) -> None:
        if maximum_active < 1:
            raise ValueError("maximum_active must be at least one")
        if maximum_queued < 0:
            raise ValueError("maximum_queued cannot be negative")
        if maximum_wait_seconds <= 0:
            raise ValueError("maximum_wait_seconds must be positive")
        self._maximum_pending = maximum_active + maximum_queued
        self._maximum_wait_seconds = maximum_wait_seconds
        self._semaphore = asyncio.Semaphore(maximum_active)
        self._admission_lock = asyncio.Lock()
        self._pending = 0

    async def run(self, operation: Callable[[], _T]) -> tuple[_T, float, float]:
        await self._admit()
        acquired = False
        release_pending = True
        queued_at = time.monotonic()
        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=self._maximum_wait_seconds
                )
                acquired = True
            except TimeoutError as err:
                raise BackendQueueTimeoutError("Timed out waiting for TTS capacity") from err

            inference_started = time.monotonic()
            operation_task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                # A cancelled client must not release model capacity while its
                # synchronous vendor call is still running in a worker thread.
                result = await asyncio.shield(operation_task)
            except asyncio.CancelledError:
                acquired = False
                release_pending = False
                operation_task.add_done_callback(self._release_cancelled_operation)
                raise
            return result, inference_started - queued_at, time.monotonic() - inference_started
        finally:
            if acquired:
                self._semaphore.release()
            if release_pending:
                await self._release()

    def _release_cancelled_operation(self, operation_task: asyncio.Task[object]) -> None:
        """Release admission only after an abandoned worker-thread call ends."""

        with contextlib.suppress(asyncio.CancelledError, Exception):
            operation_task.result()
        self._semaphore.release()
        asyncio.create_task(self._release())

    async def _admit(self) -> None:
        async with self._admission_lock:
            if self._pending >= self._maximum_pending:
                raise BackendBusyError("TTS inference queue is full")
            self._pending += 1

    async def _release(self) -> None:
        async with self._admission_lock:
            self._pending -= 1
