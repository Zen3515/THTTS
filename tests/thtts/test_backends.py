"""Unit tests for backend registrations and bounded inference behavior."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from thtts.backends.base import (
    AdmissionController,
    BackendBusyError,
    BackendError,
    UnknownVoiceError,
    VoiceMetadata,
)
from thtts.backends.f5 import F5Backend, _VoiceContext
from thtts.backends.profiles import F5_PROFILES
from thtts.backends.registry import available_backends, create_backend
from thtts.backends.vits import VitsBackend
from thtts.config import F5Settings, Settings


def test_registry_exposes_three_selectable_backends() -> None:
    assert available_backends() == ("f5-v1", "f5-v2", "vits")
    assert isinstance(create_backend(Settings(backend="vits")), VitsBackend)
    assert isinstance(create_backend(Settings(backend="f5-v1")), F5Backend)
    assert isinstance(create_backend(Settings(backend="f5-v2")), F5Backend)


def test_f5_profiles_do_not_share_v1_checkpoint() -> None:
    assert F5_PROFILES["f5-v1"].checkpoint_uri != F5_PROFILES["f5-v2"].checkpoint_uri
    assert not F5_PROFILES["f5-v1"].uses_ipa_inference
    assert F5_PROFILES["f5-v2"].uses_ipa_inference


def test_f5_rejects_unsafe_concurrency_until_stress_tested() -> None:
    with pytest.raises(BackendError, match="exactly one"):
        F5Backend(Settings(backend="f5-v1", max_concurrent_syntheses=2))


def test_f5_voice_context_is_selected_without_shared_mutation() -> None:
    backend = F5Backend(Settings(backend="f5-v1"))
    first = _VoiceContext(
        metadata=VoiceMetadata("first", None, "test", "https://example.test"),
        reference_audio="first.wav",
        reference_text="first",
    )
    second = _VoiceContext(
        metadata=VoiceMetadata("second", None, "test", "https://example.test"),
        reference_audio="second.wav",
        reference_text="second",
    )
    backend._contexts = {"first": first, "second": second}  # type: ignore[attr-defined]
    assert backend._voice_context("first") is first  # type: ignore[attr-defined]
    assert backend._voice_context("second") is second  # type: ignore[attr-defined]
    with pytest.raises(UnknownVoiceError):
        backend._voice_context("missing")  # type: ignore[attr-defined]


def test_f5_v2_prepares_shared_reference_context_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = F5Backend(Settings(backend="f5-v2"))
    metadata = VoiceMetadata("default", None, "test", "https://example.test")
    first = _VoiceContext(metadata, "reference.wav", "reference text")
    second = _VoiceContext(metadata, "reference.wav", "reference text")
    calls: list[tuple[str, str, float, str]] = []
    prepared = object()

    def fake_prepare(audio: str, text: str, *, target_rms: float, device: str) -> object:
        calls.append((audio, text, target_rms, device))
        return prepared

    monkeypatch.setattr("thtts.backends.f5.prepare_reference", fake_prepare)
    contexts = backend._prepare_v2_voice_contexts(  # type: ignore[attr-defined]
        {"default": first, "thai-default": second}, target_rms=0.1, device="cpu"
    )

    assert calls == [("reference.wav", "reference text", 0.1, "cpu")]
    assert contexts["default"].prepared_reference is prepared
    assert contexts["thai-default"].prepared_reference is prepared


def test_f5_explicit_missing_voices_file_fails_closed(tmp_path) -> None:
    backend = F5Backend(
        Settings(backend="f5-v1", f5=F5Settings(voices_file=tmp_path / "missing.yaml"))
    )
    with pytest.raises(BackendError, match="does not exist"):
        backend._load_voice_contexts(lambda audio, text: (audio, text))  # type: ignore[attr-defined]


def test_f5_loader_passes_resolved_device_to_vendor_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    fake_cached_path = ModuleType("cached_path")
    fake_cached_path.cached_path = lambda source: f"cached:{source}"
    fake_infer = ModuleType("f5_tts.infer.utils_infer")
    fake_infer.cfg_strength = 2.0
    fake_infer.cross_fade_duration = 0.0
    fake_infer.fix_duration = None
    fake_infer.mel_spec_type = "vocos"
    fake_infer.nfe_step = 32
    fake_infer.sway_sampling_coef = -1.0
    fake_infer.target_rms = 0.1
    fake_infer.infer_process = object()
    fake_infer.load_vocoder = lambda *, device: calls.setdefault("vocoder_device", device)

    def load_model(model_class, model_config, checkpoint, *, vocab_file, use_ema, device):
        calls["model"] = (model_class, model_config, checkpoint, vocab_file, use_ema, device)
        return object()

    fake_infer.load_model = load_model
    fake_infer.preprocess_ref_audio_text = lambda audio, text: (audio, text)
    fake_model = ModuleType("f5_tts.model")
    fake_model.DiT = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cached_path", fake_cached_path)
    monkeypatch.setitem(sys.modules, "f5_tts.infer.utils_infer", fake_infer)
    monkeypatch.setitem(sys.modules, "f5_tts.model", fake_model)

    backend = F5Backend(Settings(backend="f5-v1", device="cuda"))
    backend._load_sync()  # type: ignore[attr-defined]

    assert calls["vocoder_device"] == "cuda"
    assert calls["model"][-1] == "cuda"  # type: ignore[index]
    assert backend._contexts["default"].reference_audio.startswith("cached:hf://")  # type: ignore[attr-defined]
    assert backend.resolved_device == "cuda"


def test_f5_inference_suppresses_vendor_source_text_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = F5Backend(Settings(backend="f5-v2"))
    context = _VoiceContext(
        metadata=VoiceMetadata("default", None, "test", "https://example.test"),
        reference_audio="reference.wav",
        reference_text="private reference transcript",
    )
    backend._model = object()  # type: ignore[attr-defined]
    backend._vocoder = object()  # type: ignore[attr-defined]
    backend._inference_defaults = {}  # type: ignore[attr-defined]

    def noisy_inference(*args, **kwargs):
        print(args[2])
        assert kwargs["use_ipa"] is True
        assert kwargs["progress"] is None
        kwargs["show_info"]("also private")
        return np.array([0.0, 0.5], dtype=np.float32), 24_000, None

    backend._inference = noisy_inference  # type: ignore[attr-defined]
    waveform, sample_rate = backend._synthesize_sync("private caller text", context)  # type: ignore[attr-defined]

    assert sample_rate == 24_000
    assert waveform.tolist() == [0.0, 0.5]
    captured = capsys.readouterr()
    assert "private caller text" not in captured.out
    assert "also private" not in captured.out


def test_f5_v2_passes_cached_reference_to_custom_inference() -> None:
    backend = F5Backend(Settings(backend="f5-v2"))
    cached_reference = object()
    context = _VoiceContext(
        metadata=VoiceMetadata("default", None, "test", "https://example.test"),
        reference_audio="reference.wav",
        reference_text="private reference transcript",
        prepared_reference=cached_reference,
    )
    backend._model = object()  # type: ignore[attr-defined]
    backend._vocoder = object()  # type: ignore[attr-defined]
    backend._inference_defaults = {}  # type: ignore[attr-defined]
    seen: dict[str, object] = {}

    def inference(*args, **kwargs):
        seen.update(kwargs)
        return np.array([0.0, 0.5], dtype=np.float32), 24_000, None

    backend._inference = inference  # type: ignore[attr-defined]
    backend._synthesize_sync("caller text", context)  # type: ignore[attr-defined]

    assert seen["prepared_reference"] is cached_reference
    assert seen["use_ipa"] is True


@pytest.mark.asyncio
async def test_admission_controller_rejects_unbounded_work() -> None:
    controller = AdmissionController(maximum_active=1, maximum_queued=0, maximum_wait_seconds=1.0)
    # Model execution itself is covered by adapter tests. Set the admission
    # state directly here so the rejection path has no worker-thread timing.
    controller._pending = 1  # type: ignore[attr-defined]
    with pytest.raises(BackendBusyError):
        await controller.run(lambda: "nope")


@pytest.mark.asyncio
async def test_cancelled_client_does_not_release_running_model_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AdmissionController(maximum_active=1, maximum_queued=0, maximum_wait_seconds=1.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def deferred_to_thread(operation):
        started.set()
        await release.wait()
        return operation()

    monkeypatch.setattr(asyncio, "to_thread", deferred_to_thread)

    task = asyncio.create_task(controller.run(lambda: "done"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    try:
        with pytest.raises(BackendBusyError):
            await controller.run(lambda: "must not overlap")
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(10):
        await asyncio.sleep(0)
        if controller._pending == 0:  # type: ignore[attr-defined]
            break
    result, _, _ = await controller.run(lambda: "now safe")
    assert result == "now safe"


@pytest.mark.asyncio
async def test_vits_synthesis_uses_backend_sample_rate_without_loading_vendor_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = VitsBackend(Settings(backend="vits"))
    backend._model = object()  # type: ignore[attr-defined]
    backend._sample_rate = 16_000  # type: ignore[attr-defined]
    monkeypatch.setattr(
        backend,
        "_synthesize_sync",
        lambda text: np.array([0.0, 0.5], dtype=np.float32),
    )

    async def immediate_run(operation):
        return operation(), 0.0, 0.0

    monkeypatch.setattr(backend._controller, "run", immediate_run)  # type: ignore[attr-defined]
    result = await backend.synthesize("test", voice_name="default")
    assert result.sample_rate == 16_000
    assert result.waveform.tolist() == [0.0, 0.5]
