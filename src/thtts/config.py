"""Typed runtime settings and public THTTS environment migration rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10200
DEFAULT_BACKEND = "vits"
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_CONCURRENT_SYNTHESES = 1
DEFAULT_MAX_QUEUED_SYNTHESES = 0
DEFAULT_MAX_QUEUE_SECONDS = 30.0
DEFAULT_VITS_MODEL = "VIZINTZOR/MMS-TTS-THAI-FEMALEV2"
DEFAULT_F5_SPEED = 1.0
DEFAULT_F5_NFE_STEPS = 24
DEFAULT_STREAM_IDLE_FLUSH_MS = 220
DEFAULT_STREAM_MIN_SEGMENT_CHARS = 15
DEFAULT_STREAM_TARGET_CHARS = 48
DEFAULT_STREAM_MAX_SEGMENT_CHARS = 180
DEFAULT_OUTPUT_CHUNK_MILLISECONDS = 200
DEFAULT_SHUTDOWN_GRACE_SECONDS = 15.0


@dataclass(frozen=True)
class F5Settings:
    checkpoint_file: str | None = None
    vocab_file: str | None = None
    reference_audio: str | None = None
    reference_text: str | None = None
    speed: float = DEFAULT_F5_SPEED
    nfe_steps: int = DEFAULT_F5_NFE_STEPS
    voices_file: Path | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.speed) or self.speed <= 0:
            raise ValueError("F5 speed must be a finite value greater than zero")
        if self.nfe_steps < 1:
            raise ValueError("F5 nfe_steps must be at least one")


@dataclass(frozen=True)
class StreamSettings:
    idle_flush_ms: int = DEFAULT_STREAM_IDLE_FLUSH_MS
    min_segment_chars: int = DEFAULT_STREAM_MIN_SEGMENT_CHARS
    target_chars: int = DEFAULT_STREAM_TARGET_CHARS
    max_segment_chars: int = DEFAULT_STREAM_MAX_SEGMENT_CHARS
    output_chunk_milliseconds: int = DEFAULT_OUTPUT_CHUNK_MILLISECONDS

    def __post_init__(self) -> None:
        if self.idle_flush_ms < 1:
            raise ValueError("stream idle flush must be at least one millisecond")
        if self.min_segment_chars < 1:
            raise ValueError("stream min segment chars must be at least one")
        if self.target_chars < self.min_segment_chars:
            raise ValueError("stream target chars must be at least min segment chars")
        if self.max_segment_chars < self.target_chars:
            raise ValueError("stream max segment chars must be at least target chars")
        if self.output_chunk_milliseconds < 1:
            raise ValueError("output chunk milliseconds must be at least one")


@dataclass(frozen=True)
class Settings:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    backend: str = DEFAULT_BACKEND
    device: str = DEFAULT_DEVICE
    log_level: str = "INFO"
    max_concurrent_syntheses: int = DEFAULT_MAX_CONCURRENT_SYNTHESES
    max_queued_syntheses: int = DEFAULT_MAX_QUEUED_SYNTHESES
    max_queue_seconds: float = DEFAULT_MAX_QUEUE_SECONDS
    vits_model: str = DEFAULT_VITS_MODEL
    f5: F5Settings = F5Settings()
    stream: StreamSettings = StreamSettings()
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS
    deprecations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.backend not in {"vits", "f5-v1", "f5-v2"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unsupported device: {self.device}")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"unsupported log level: {self.log_level}")
        if self.max_concurrent_syntheses < 1:
            raise ValueError("max concurrent syntheses must be at least one")
        if self.max_queued_syntheses < 0:
            raise ValueError("max queued syntheses cannot be negative")
        if not isfinite(self.max_queue_seconds) or self.max_queue_seconds <= 0:
            raise ValueError("max queue seconds must be positive")
        if not isfinite(self.shutdown_grace_seconds) or self.shutdown_grace_seconds < 0:
            raise ValueError("shutdown grace seconds cannot be negative")

    def safe_summary(self) -> str:
        """Return safe startup fields without model/reference paths or text."""

        return (
            f"backend={self.backend} listener={self.host}:{self.port} "
            f"device={self.device} active_limit={self.max_concurrent_syntheses} "
            f"queued_limit={self.max_queued_syntheses} "
            f"shutdown_grace_seconds={self.shutdown_grace_seconds:g}"
        )
