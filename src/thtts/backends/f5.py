"""F5-TTS implementation with immutable, request-selected voice contexts."""

from __future__ import annotations

import asyncio
import contextlib
import io
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..text import preprocess_f5_text
from .base import (
    AdmissionController,
    BackendError,
    BackendMetadata,
    SynthesisResult,
    UnknownVoiceError,
    VoiceMetadata,
)
from .f5_infer import custom_infer_process, prepare_reference
from .profiles import F5_PROFILES, F5Profile


@dataclass(frozen=True)
class _VoiceContext:
    metadata: VoiceMetadata
    reference_audio: str
    reference_text: str
    prepared_reference: Any | None = None


class F5Backend:
    def __init__(self, settings: Settings) -> None:
        if settings.max_concurrent_syntheses != 1:
            raise BackendError(
                "F5 supports exactly one active synthesis until a dedicated "
                "concurrency stress test approves more"
            )
        self._settings = settings
        self._profile: F5Profile = F5_PROFILES[settings.backend]
        self._requested_device = settings.device
        self._controller = AdmissionController(
            maximum_active=settings.max_concurrent_syntheses,
            maximum_queued=settings.max_queued_syntheses,
            maximum_wait_seconds=settings.max_queue_seconds,
        )
        self._load_lock = asyncio.Lock()
        self._model: Any | None = None
        self._vocoder: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None
        self._sample_rate = 24_000
        self._inference: Any | None = None
        self._inference_defaults: dict[str, Any] = {}
        self._contexts: dict[str, _VoiceContext] = {}
        self._metadata = BackendMetadata(
            backend_id=self._profile.backend_id,
            program_name=f"thai-{self._profile.backend_id}",
            program_description=(f"Thai TTS via {self._profile.attribution_name} (DiT + vocos)"),
            attribution_name=self._profile.attribution_name,
            attribution_url=self._profile.attribution_url,
        )

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    @property
    def voices(self) -> tuple[VoiceMetadata, ...]:
        return tuple(context.metadata for context in self._contexts.values())

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
        context = self._voice_context(voice_name)
        prepared_text = preprocess_f5_text(text)
        if not prepared_text:
            return SynthesisResult(
                waveform=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                queue_seconds=0.0,
                inference_seconds=0.0,
            )
        (waveform, sample_rate), queue_seconds, inference_seconds = await self._controller.run(
            lambda: self._synthesize_sync(prepared_text, context)
        )
        return SynthesisResult(
            waveform=waveform,
            sample_rate=sample_rate,
            queue_seconds=queue_seconds,
            inference_seconds=inference_seconds,
        )

    def _load_sync(self) -> None:
        try:
            import torch
            from cached_path import cached_path
            from f5_tts.infer.utils_infer import (
                cfg_strength,
                cross_fade_duration,
                fix_duration,
                infer_process,
                load_model,
                load_vocoder,
                mel_spec_type,
                nfe_step,
                preprocess_ref_audio_text,
                sway_sampling_coef,
                target_rms,
            )
            from f5_tts.model import DiT
        except ImportError as err:  # pragma: no cover - packaging guard
            raise BackendError("F5 dependencies are unavailable; run uv sync") from err

        if self._requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._requested_device == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but torch.cuda.is_available() is false")
        else:
            device = self._requested_device

        checkpoint_source = self._settings.f5.checkpoint_file or self._profile.checkpoint_uri
        vocab_source = self._settings.f5.vocab_file or self._profile.vocab_uri
        try:
            # The vendor helper prints model/reference paths and transcripts.
            # Keep those private and publish only the safe THTTS lifecycle log.
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint_file = str(cached_path(checkpoint_source))
                vocab_file = str(cached_path(vocab_source))
                vocoder = load_vocoder(device=device)
                model = load_model(
                    DiT,
                    self._profile.model_config,
                    checkpoint_file,
                    vocab_file=vocab_file,
                    use_ema=True,
                    device=device,
                )
                contexts = self._load_voice_contexts(
                    preprocess_ref_audio_text,
                    resolve_reference_audio=lambda source: str(cached_path(source)),
                )
                if self._profile.uses_ipa_inference:
                    contexts = self._prepare_v2_voice_contexts(
                        contexts,
                        target_rms=target_rms,
                        device=device,
                    )
        except BackendError:
            raise
        except Exception:
            raise BackendError(
                f"Unable to load {self._profile.backend_id} on the requested device"
            ) from None

        self._torch = torch
        self._resolved_device = device
        self._model = model
        self._vocoder = vocoder
        self._contexts = contexts
        self._inference = (
            custom_infer_process if self._profile.uses_ipa_inference else infer_process
        )
        self._inference_defaults = {
            "mel_spec_type": mel_spec_type,
            "target_rms": target_rms,
            "cross_fade_duration": cross_fade_duration,
            "nfe_step": self._settings.f5.nfe_steps or nfe_step,
            "cfg_strength": cfg_strength,
            "sway_sampling_coef": sway_sampling_coef,
            "speed": self._settings.f5.speed,
            "fix_duration": fix_duration,
            "device": device,
        }

    def _prepare_v2_voice_contexts(
        self,
        contexts: dict[str, _VoiceContext],
        *,
        target_rms: float,
        device: str,
    ) -> dict[str, _VoiceContext]:
        """Cache immutable audio/IPA state shared by aliases of one voice."""

        prepared_by_source: dict[tuple[str, str], Any] = {}
        prepared_contexts: dict[str, _VoiceContext] = {}
        for name, context in contexts.items():
            source = (context.reference_audio, context.reference_text)
            prepared = prepared_by_source.get(source)
            if prepared is None:
                prepared = prepare_reference(
                    context.reference_audio,
                    context.reference_text,
                    target_rms=target_rms,
                    device=device,
                )
                prepared_by_source[source] = prepared
            prepared_contexts[name] = replace(context, prepared_reference=prepared)
        return prepared_contexts

    def _load_voice_contexts(
        self, prepare: Any, *, resolve_reference_audio: Any = str
    ) -> dict[str, _VoiceContext]:
        if self._settings.f5.voices_file is None:
            reference_audio = self._settings.f5.reference_audio
            if reference_audio in {None, "", "hf_sample"}:
                reference_audio = self._profile.reference_audio_uri
            reference_text = self._settings.f5.reference_text or self._profile.reference_text
            resolved_audio = resolve_reference_audio(reference_audio)
            prepared_audio, prepared_text = prepare(resolved_audio, reference_text)
            default = _VoiceContext(
                metadata=VoiceMetadata(
                    name="default",
                    description=f"Default {self._profile.backend_id} Thai voice",
                    attribution_name=self._profile.attribution_name,
                    attribution_url=self._profile.attribution_url,
                ),
                reference_audio=prepared_audio,
                reference_text=prepared_text,
            )
            # Preserve the discovery alias that existing clients may use.
            return {
                "default": default,
                "thai-default": _VoiceContext(
                    metadata=VoiceMetadata(
                        name="thai-default",
                        description="Alias of default",
                        attribution_name=self._profile.attribution_name,
                        attribution_url=self._profile.attribution_url,
                    ),
                    reference_audio=prepared_audio,
                    reference_text=prepared_text,
                ),
            }

        path = self._settings.f5.voices_file
        assert path is not None
        if not path.is_file():
            raise BackendError(f"Configured voices file does not exist: {path}")
        try:
            import yaml

            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as err:
            raise BackendError(f"Unable to read voices file {path}: {err}") from err
        if not isinstance(raw, list) or not raw:
            raise BackendError("Voices file must contain a non-empty top-level list")

        contexts: dict[str, _VoiceContext] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise BackendError(f"Voice entry #{index} must be a mapping")
            name = item.get("name")
            reference_audio = item.get("ref_sound_path")
            reference_text = item.get("ref_sound_sentence")
            values = (name, reference_audio, reference_text)
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise BackendError(
                    f"Voice entry #{index} requires name, ref_sound_path, and ref_sound_sentence"
                )
            if name in contexts:
                raise BackendError(f"Duplicate voice name: {name}")
            if not Path(reference_audio).is_file():
                raise BackendError(f"Reference audio for voice '{name}' does not exist")
            prepared_audio, prepared_text = prepare(reference_audio, reference_text)
            attribution = item.get("attribution")
            attribution_name = (
                attribution.get("name")
                if isinstance(attribution, dict) and isinstance(attribution.get("name"), str)
                else self._profile.attribution_name
            )
            attribution_url = (
                attribution.get("url")
                if isinstance(attribution, dict) and isinstance(attribution.get("url"), str)
                else self._profile.attribution_url
            )
            languages = item.get("languages", ["th", "th-TH"])
            if not isinstance(languages, list) or not all(
                isinstance(lang, str) for lang in languages
            ):
                raise BackendError(f"Voice entry #{index} has invalid languages")
            description = item.get("description")
            version = item.get("version")
            contexts[name] = _VoiceContext(
                metadata=VoiceMetadata(
                    name=name,
                    description=description if isinstance(description, str) else None,
                    attribution_name=attribution_name,
                    attribution_url=attribution_url,
                    languages=tuple(languages),
                    version=version if isinstance(version, str) else "1.0",
                ),
                reference_audio=prepared_audio,
                reference_text=prepared_text,
            )
        return contexts

    def _voice_context(self, voice_name: str | None) -> _VoiceContext:
        selected = voice_name or "default"
        try:
            return self._contexts[selected]
        except KeyError as err:
            raise UnknownVoiceError(f"Unknown F5 voice: {selected}") from err

    def _synthesize_sync(self, text: str, context: _VoiceContext) -> tuple[np.ndarray, int]:
        if self._model is None or self._vocoder is None or self._inference is None:
            raise BackendError("F5 model was not loaded")
        try:
            kwargs = dict(self._inference_defaults)
            # Upstream inference helpers default to printing generated source
            # text and progress. THTTS must not expose caller text in logs.
            kwargs["show_info"] = lambda _: None
            kwargs["progress"] = None
            if self._profile.uses_ipa_inference:
                kwargs["use_ipa"] = True
                kwargs["prepared_reference"] = context.prepared_reference
            with contextlib.redirect_stdout(io.StringIO()):
                result = self._inference(
                    context.reference_audio,
                    context.reference_text,
                    text,
                    self._model,
                    self._vocoder,
                    **kwargs,
                )
            if len(result) == 3:
                waveform, sample_rate, _ = result
            elif len(result) == 2:
                waveform, sample_rate = result
            else:
                raise BackendError("F5 inference returned an unexpected result shape")
            if waveform is None:
                return np.zeros(0, dtype=np.float32), int(sample_rate)
            return np.asarray(waveform, dtype=np.float32), int(sample_rate)
        except BackendError:
            raise
        except Exception:
            raise BackendError("F5 synthesis failed") from None
