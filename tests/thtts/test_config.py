"""Public CLI/environment migration contract tests."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from thtts.cli import parse_settings


def test_defaults_are_safe_and_canonical() -> None:
    settings = parse_settings([], environ={})
    assert settings.backend == "vits"
    assert settings.host == "0.0.0.0"
    assert settings.port == 10200
    assert settings.max_concurrent_syntheses == 1
    assert settings.f5.nfe_steps == 24
    assert settings.deprecations == ()


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [("VITS", "vits"), ("F5_V1", "f5-v1"), ("F5_V2", "f5-v2")],
)
def test_legacy_backend_values_are_canonicalized(legacy: str, canonical: str) -> None:
    settings = parse_settings([], environ={"THTTS_BACKEND": legacy})
    assert settings.backend == canonical
    assert settings.deprecations


def test_matching_canonical_and_legacy_environment_warns_with_a_copyable_migration() -> None:
    settings = parse_settings(
        [],
        environ={
            "THTTS_F5_SPEED": "0.8",
            "THTTS_SPEED": "0.8",
            "THTTS_BACKEND": "f5-v1",
        },
    )
    assert settings.f5.speed == 0.8
    assert settings.deprecations == (
        "Deprecated configuration: THTTS_SPEED will be removed in the next breaking "
        "release. Replace it with: THTTS_F5_SPEED=${THTTS_SPEED}",
    )


def test_legacy_environment_alias_maps_and_warns() -> None:
    settings = parse_settings(
        [],
        environ={"THTTS_BACKEND": "F5_V1", "THTTS_CKPT_FILE": "/models/f5.pt"},
    )
    assert settings.backend == "f5-v1"
    assert settings.f5.checkpoint_file == "/models/f5.pt"
    assert settings.deprecations == (
        "Deprecated configuration: THTTS_BACKEND=F5_V1 will be removed in the next "
        "breaking release. Replace it with: THTTS_BACKEND=f5-v1",
        "Deprecated configuration: THTTS_CKPT_FILE will be removed in the next breaking "
        "release. Replace it with: THTTS_F5_CHECKPOINT_FILE=${THTTS_CKPT_FILE}",
    )


def test_each_matching_legacy_key_gets_its_own_migration_notice() -> None:
    settings = parse_settings(
        [],
        environ={
            "THTTS_F5_SPEED": "0.8",
            "THTTS_SPEED": "0.8",
            "THTTS_SPEAK_SPEED": "0.8",
        },
    )

    assert settings.deprecations == (
        "Deprecated configuration: THTTS_SPEED will be removed in the next breaking "
        "release. Replace it with: THTTS_F5_SPEED=${THTTS_SPEED}",
        "Deprecated configuration: THTTS_SPEAK_SPEED will be removed in the next breaking "
        "release. Replace it with: THTTS_F5_SPEED=${THTTS_SPEAK_SPEED}",
    )


@pytest.mark.parametrize(
    ("legacy_name", "legacy_value", "canonical_name"),
    [
        ("THTTS_MODEL", "example/model", "THTTS_VITS_MODEL"),
        ("THTTS_MAX_CONCURRENT", "1", "THTTS_MAX_CONCURRENT_SYNTHESES"),
        ("THTTS_CKPT_FILE", "/models/f5.pt", "THTTS_F5_CHECKPOINT_FILE"),
        ("THTTS_VOCAB_FILE", "/models/vocab.txt", "THTTS_F5_VOCAB_FILE"),
        ("THTTS_REF_AUDIO", "/voices/default.wav", "THTTS_F5_REFERENCE_AUDIO"),
        ("THTTS_REF_TEXT", "private reference", "THTTS_F5_REFERENCE_TEXT"),
        ("THTTS_SPEED", "0.8", "THTTS_F5_SPEED"),
        ("THTTS_SPEAK_SPEED", "0.8", "THTTS_F5_SPEED"),
        ("THTTS_NFE_STEPS", "32", "THTTS_F5_NFE_STEPS"),
        ("THTTS_VOICES_YAML", "/voices/voices.yaml", "THTTS_VOICES_FILE"),
        ("THTTS_MAX_WAIT_MS", "220", "THTTS_STREAM_IDLE_FLUSH_MS"),
        ("THTTS_MIN_SENT_CHARS", "15", "THTTS_STREAM_MIN_SEGMENT_CHARS"),
    ],
)
def test_each_legacy_environment_key_has_a_copyable_migration_notice(
    legacy_name: str, legacy_value: str, canonical_name: str
) -> None:
    settings = parse_settings([], environ={legacy_name: legacy_value})

    assert settings.deprecations == (
        f"Deprecated configuration: {legacy_name} will be removed in the next breaking "
        f"release. Replace it with: {canonical_name}=${{{legacy_name}}}",
    )
    assert legacy_value not in settings.deprecations[0]


@pytest.mark.parametrize(
    "environ",
    [
        {"THTTS_F5_SPEED": "1.0", "THTTS_SPEED": "0.8"},
        {"THTTS_SPEED": "1.0", "THTTS_SPEAK_SPEED": "0.8"},
    ],
)
def test_conflicting_canonical_or_legacy_values_fail(environ: dict[str, str]) -> None:
    with pytest.raises(SystemExit):
        parse_settings([], environ=environ)


def test_cli_overrides_environment() -> None:
    settings = parse_settings(
        ["--backend", "vits", "--port", "11000"],
        environ={"THTTS_BACKEND": "F5_V1", "THTTS_PORT": "10200"},
    )
    assert settings.backend == "vits"
    assert settings.port == 11000


def test_safe_summary_redacts_reference_values() -> None:
    settings = parse_settings(
        [],
        environ={
            "THTTS_BACKEND": "f5-v1",
            "THTTS_F5_REFERENCE_AUDIO": "/private/voice.wav",
            "THTTS_F5_REFERENCE_TEXT": "secret transcript",
        },
    )
    summary = settings.safe_summary()
    assert "/private/voice.wav" not in summary
    assert "secret transcript" not in summary


def test_documented_f5_migration_example_resolves_without_legacy_warnings() -> None:
    settings = parse_settings(
        [],
        environ={
            "THTTS_BACKEND": "f5-v1",
            "THTTS_HOST": "0.0.0.0",
            "THTTS_PORT": "10200",
            "THTTS_DEVICE": "auto",
            "THTTS_MAX_CONCURRENT_SYNTHESES": "1",
            "THTTS_F5_CHECKPOINT_FILE": "/models/f5.pt",
            "THTTS_F5_REFERENCE_AUDIO": "/voices/default.wav",
        },
    )
    assert settings.backend == "f5-v1"
    assert settings.f5.checkpoint_file == "/models/f5.pt"
    assert settings.f5.reference_audio == "/voices/default.wav"
    assert settings.deprecations == ()


def test_shutdown_grace_accepts_zero_for_immediate_exit() -> None:
    settings = parse_settings([], environ={"THTTS_SHUTDOWN_GRACE_SECONDS": "0"})
    assert settings.shutdown_grace_seconds == 0


def test_legacy_warning_redacts_its_value() -> None:
    settings = parse_settings(
        [],
        environ={"THTTS_CKPT_FILE": "/private/very-secret-model.pt"},
    )
    warnings = " ".join(settings.deprecations)
    assert "/private/very-secret-model.pt" not in warnings
    assert "THTTS_F5_CHECKPOINT_FILE=${THTTS_CKPT_FILE}" in warnings


def test_help_does_not_need_pythainlp_or_a_writable_model_cache(tmp_path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHAINLP_DATA_DIR", None)
    environment["HOME"] = str(tmp_path / "read-only-home")
    (tmp_path / "read-only-home").mkdir()
    (tmp_path / "read-only-home").chmod(0o500)
    result = subprocess.run(
        [sys.executable, "-m", "thtts", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0
    assert "--backend" in result.stdout
