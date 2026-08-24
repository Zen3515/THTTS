"""Compatibility tests for the old script paths retained for one release."""

from __future__ import annotations

import pytest

from wyoming_thai_f5 import translate_legacy_args as translate_f5
from wyoming_thai_vits import translate_legacy_args as translate_vits


def test_legacy_vits_arguments_forward_to_the_unified_cli() -> None:
    assert translate_vits(["--model-id", "example/model", "--max-concurrent=2"]) == [
        "--backend",
        "vits",
        "--vits-model",
        "example/model",
        "--max-concurrent-syntheses=2",
    ]


def test_legacy_f5_v2_arguments_forward_to_the_unified_cli() -> None:
    assert translate_f5(
        [
            "--model-version",
            "v2",
            "--ckpt-file=/models/f5.pt",
            "--ref-audio",
            "/voices/default.wav",
            "--max-concurrent",
            "1",
        ]
    ) == [
        "--backend",
        "f5-v2",
        "--f5-checkpoint-file=/models/f5.pt",
        "--f5-reference-audio",
        "/voices/default.wav",
        "--max-concurrent-syntheses",
        "1",
    ]


@pytest.mark.parametrize("version", ["", "v3"])
def test_legacy_f5_rejects_invalid_model_version(version: str) -> None:
    with pytest.raises(SystemExit):
        translate_f5([f"--model-version={version}"])
