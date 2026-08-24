"""Transformers VITS implementation of the THTTS backend contract."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from ..config import Settings
from .base import (
    AdmissionController,
    BackendError,
    BackendMetadata,
    SynthesisResult,
    UnknownVoiceError,
    VoiceMetadata,
)


class VitsBackend:
    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.vits_model
        self._requested_device = settings.device
        self._controller = AdmissionController(
            maximum_active=settings.max_concurrent_syntheses,
            maximum_queued=settings.max_queued_syntheses,
            maximum_wait_seconds=settings.max_queue_seconds,
        )
        self._load_lock = asyncio.Lock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None
        self._sample_rate = 22_050
        label = "male" if "MALE" in self.model_id.upper() else "female"
        attribution_url = f"https://huggingface.co/{self.model_id}"
        self._metadata = BackendMetadata(
            backend_id="vits",
            program_name="thai-vits",
            program_description=f"Thai VITS ({label}) via {self.model_id}",
            attribution_name="VIZINTZOR",
            attribution_url=attribution_url,
        )
        self._voices = (
            VoiceMetadata(
                name="default",
                description=f"Thai VITS {label} ({self.model_id})",
                attribution_name="VIZINTZOR",
                attribution_url=attribution_url,
            ),
        )

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    @property
    def voices(self) -> tuple[VoiceMetadata, ...]:
        return self._voices

    @property
    def resolved_device(self) -> str | None:
        return self._resolved_device

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_sync)

    async def synthesize(self, text: str, *, voice_name: str | None) -> SynthesisResult:
        await self.load()
        if voice_name not in {None, "", "default"}:
            raise UnknownVoiceError(f"Unknown VITS voice: {voice_name}")
        if not text.strip():
            return SynthesisResult(
                waveform=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                queue_seconds=0.0,
                inference_seconds=0.0,
            )
        waveform, queue_seconds, inference_seconds = await self._controller.run(
            lambda: self._synthesize_sync(text)
        )
        return SynthesisResult(
            waveform=waveform,
            sample_rate=self._sample_rate,
            queue_seconds=queue_seconds,
            inference_seconds=inference_seconds,
        )

    def _load_sync(self) -> None:
        try:
            import torch
            from transformers import VitsModel, VitsTokenizer, set_seed
        except ImportError as err:  # pragma: no cover - dependency packaging guard
            raise BackendError("VITS dependencies are unavailable; run uv sync") from err

        if self._requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._requested_device == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but torch.cuda.is_available() is false")
        else:
            device = self._requested_device

        try:
            tokenizer = VitsTokenizer.from_pretrained(self.model_id)
            model = VitsModel.from_pretrained(self.model_id)
            model.eval()
            model.to(device)
            set_seed(456)
        except Exception:
            raise BackendError(
                "Unable to load the configured VITS model on the requested device"
            ) from None

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = device
        self._sample_rate = int(getattr(model.config, "sampling_rate", 22_050))

    def _synthesize_sync(self, text: str) -> np.ndarray:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise BackendError("VITS model was not loaded")
        try:
            with self._torch.inference_mode():
                inputs = self._tokenizer(text=text, return_tensors="pt").to(self._resolved_device)
                output = self._model(**inputs)
                return output.waveform[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            raise BackendError("VITS synthesis failed") from None
