#!/usr/bin/env python3
"""Deprecated compatibility launcher for the pre-0.2 VITS server path."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence

from thtts.__main__ import run

_RENAMED_FLAGS = {
    "--model-id": "--vits-model",
    "--max-concurrent": "--max-concurrent-syntheses",
}


def translate_legacy_args(argv: Sequence[str]) -> list[str]:
    """Map the documented legacy VITS options onto the unified CLI."""

    translated = ["--backend", "vits"]
    for argument in argv:
        for old, new in _RENAMED_FLAGS.items():
            if argument == old:
                argument = new
                break
            if argument.startswith(f"{old}="):
                argument = f"{new}={argument[len(old) + 1:]}"
                break
        translated.append(argument)
    return translated


def main(argv: Sequence[str] | None = None) -> None:
    warnings.warn(
        "src/wyoming_thai_vits.py is deprecated; use 'thtts --backend vits' instead.",
        FutureWarning,
        stacklevel=2,
    )
    run(translate_legacy_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
