"""Canonical audio conversion and Wyoming PCM framing."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def float32_to_int16_pcm(waveform: np.ndarray) -> bytes:
    """Clip a float waveform and encode signed little-endian 16-bit PCM."""

    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2", copy=False).tobytes()


def iter_pcm_chunks(
    waveform: np.ndarray, *, sample_rate: int, chunk_milliseconds: int
) -> Iterator[bytes]:
    """Yield non-empty PCM chunks with a bounded, time-based frame size."""

    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if chunk_milliseconds < 1:
        raise ValueError("chunk_milliseconds must be positive")

    samples_per_chunk = max(1, sample_rate * chunk_milliseconds // 1000)
    for start in range(0, len(waveform), samples_per_chunk):
        payload = float32_to_int16_pcm(waveform[start : start + samples_per_chunk])
        if payload:
            yield payload
