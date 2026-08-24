"""Opt-in, artifact-pinned real-model audio oracle tests.

The normal test lane never imports model weights. Enable this module only on a
prepared CPU/GPU runner by setting ``THTTS_RUN_MODEL_TESTS=1`` and pointing
``THTTS_MODEL_ORACLE_MANIFEST`` at a reviewed private manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thtts.audio import float32_to_int16_pcm
from thtts.backends.registry import create_backend
from thtts.config import F5Settings, Settings

pytestmark = pytest.mark.integration


def _manifest() -> dict[str, Any]:
    if os.getenv("THTTS_RUN_MODEL_TESTS") != "1":
        pytest.skip("set THTTS_RUN_MODEL_TESTS=1 to run real-model audio oracle tests")
    raw_path = os.getenv("THTTS_MODEL_ORACLE_MANIFEST")
    if not raw_path:
        pytest.skip("set THTTS_MODEL_ORACLE_MANIFEST to an approved local manifest")
    path = Path(raw_path)
    if not path.is_file():
        pytest.fail(f"model oracle manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("backends"), dict):
        pytest.fail("model oracle manifest must use schema_version 1 and a backends mapping")
    return manifest


def _settings(backend_id: str, spec: dict[str, Any]) -> Settings:
    f5_spec = spec.get("f5", {})
    if not isinstance(f5_spec, dict):
        pytest.fail(f"{backend_id}: f5 settings must be a mapping")
    return Settings(
        backend=backend_id,
        device=str(spec.get("device", "cpu")),
        vits_model=str(spec.get("vits_model", Settings().vits_model)),
        f5=F5Settings(
            checkpoint_file=_optional_string(f5_spec, "checkpoint_file"),
            vocab_file=_optional_string(f5_spec, "vocab_file"),
            reference_audio=_optional_string(f5_spec, "reference_audio"),
            reference_text=_optional_string(f5_spec, "reference_text"),
            speed=float(f5_spec.get("speed", 1.0)),
            nfe_steps=int(f5_spec.get("nfe_steps", 32)),
        ),
    )


def _optional_string(values: dict[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        pytest.fail(f"manifest {key} must be a non-empty string when set")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def _spectral_centroid_hz(waveform: np.ndarray, sample_rate: int) -> float:
    magnitudes = np.abs(np.fft.rfft(waveform.astype(np.float64)))
    total = float(magnitudes.sum())
    if total == 0:
        return 0.0
    frequencies = np.fft.rfftfreq(len(waveform), d=1 / sample_rate)
    return float(np.dot(frequencies, magnitudes) / total)


def _assert_range(name: str, value: float, bounds: Any) -> None:
    if not isinstance(bounds, list) or len(bounds) != 2:
        pytest.fail(f"oracle signal.{name} must be [minimum, maximum]")
    lower, upper = (float(item) for item in bounds)
    assert lower <= value <= upper, f"{name}={value} outside [{lower}, {upper}]"


@pytest.mark.asyncio
async def test_real_model_audio_oracles() -> None:
    manifest = _manifest()
    all_backend_ids = {"vits", "f5-v1", "f5-v2"}
    requested = os.getenv("THTTS_MODEL_ORACLE_BACKENDS")
    selected_backend_ids = (
        {item.strip() for item in requested.split(",") if item.strip()}
        if requested
        else all_backend_ids
    )
    assert selected_backend_ids <= all_backend_ids
    assert selected_backend_ids <= set(manifest["backends"])
    if not requested:
        assert set(manifest["backends"]) == all_backend_ids
    lock_sha256 = manifest.get("uv_lock_sha256")
    if lock_sha256 is not None:
        assert lock_sha256 == _sha256_file(Path("uv.lock"))

    for backend_id in sorted(selected_backend_ids):
        raw_spec = manifest["backends"][backend_id]
        assert backend_id in {"vits", "f5-v1", "f5-v2"}
        assert isinstance(raw_spec, dict)
        spec: dict[str, Any] = raw_spec
        for artifact, expected_hash in spec.get("artifacts", {}).items():
            artifact_path = Path(artifact)
            assert artifact_path.is_file(), f"{backend_id}: missing artifact {artifact_path}"
            assert _sha256_file(artifact_path) == expected_hash

        expected_platform = spec.get("platform", {})
        if expected_platform:
            assert expected_platform.get("system") == platform.system()
            assert expected_platform.get("machine") == platform.machine()

        text = spec.get("text")
        assert isinstance(text, str) and text.strip(), f"{backend_id}: text must be non-empty"
        repeats = int(spec.get("repeats", 2))
        assert repeats >= 2
        backend = create_backend(_settings(backend_id, spec))
        await backend.load()
        results = [await backend.synthesize(text, voice_name="default") for _ in range(repeats)]
        sample_rates = {result.sample_rate for result in results}
        assert len(sample_rates) == 1
        waveforms = [np.asarray(result.waveform, dtype=np.float32) for result in results]
        assert all(waveform.ndim == 1 and len(waveform) > 0 for waveform in waveforms)
        assert all(np.isfinite(waveform).all() for waveform in waveforms)
        pcm_values = [float32_to_int16_pcm(waveform) for waveform in waveforms]
        pcm_hashes = [hashlib.sha256(pcm).hexdigest() for pcm in pcm_values]

        oracle = spec.get("oracle")
        assert isinstance(oracle, dict), f"{backend_id}: oracle mapping is required"
        assert results[0].sample_rate == int(oracle["sample_rate"])
        if oracle.get("byte_stable"):
            assert len(set(pcm_hashes)) == 1, f"{backend_id}: determinism probe changed PCM"
            assert pcm_hashes[0] == oracle["pcm_sha256"]
            continue

        sequence_hashes = oracle.get("sequence_pcm_sha256")
        if sequence_hashes is not None:
            assert isinstance(sequence_hashes, list)
            assert pcm_hashes == sequence_hashes
            continue

        signal = oracle.get("signal")
        assert isinstance(signal, dict), f"{backend_id}: non-stable backends require signal bounds"
        first = waveforms[0]
        _assert_range("frame_count", float(len(first)), signal["frame_count"])
        _assert_range("peak", float(np.max(np.abs(first))), signal["peak"])
        _assert_range("rms", _rms(first), signal["rms"])
        _assert_range("dc", float(np.mean(first)), signal["dc"])
        _assert_range(
            "spectral_centroid_hz",
            _spectral_centroid_hz(first, results[0].sample_rate),
            signal["spectral_centroid_hz"],
        )
        minimum_correlation = float(signal["minimum_repeat_correlation"])
        for waveform in waveforms[1:]:
            common_length = min(len(first), len(waveform))
            correlation = float(np.corrcoef(first[:common_length], waveform[:common_length])[0, 1])
            assert correlation >= minimum_correlation
