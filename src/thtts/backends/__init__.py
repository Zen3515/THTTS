"""THTTS backend contracts and lazy registry exports."""

from __future__ import annotations

from ..config import Settings
from .base import BackendBusyError, BackendError, BackendMetadata, SynthesisResult, TtsBackend


def create_backend(settings: Settings) -> TtsBackend:
    """Import heavy vendor adapters only when a process actually starts one."""

    from .registry import create_backend as create

    return create(settings)


def available_backends() -> tuple[str, ...]:
    """Return stable backend names without importing model/text dependencies."""

    return ("f5-v1", "f5-v2", "vits")


__all__ = [
    "BackendBusyError",
    "BackendError",
    "BackendMetadata",
    "SynthesisResult",
    "available_backends",
    "create_backend",
]
