"""Explicit THTTS backend registry."""

from __future__ import annotations

from ..config import Settings
from .base import TtsBackend
from .f5 import F5Backend
from .vits import VitsBackend


def available_backends() -> tuple[str, ...]:
    return ("f5-v1", "f5-v2", "vits")


def create_backend(settings: Settings) -> TtsBackend:
    if settings.backend == "vits":
        return VitsBackend(settings)
    if settings.backend in {"f5-v1", "f5-v2"}:
        return F5Backend(settings)
    raise ValueError(f"Unsupported backend: {settings.backend}")
