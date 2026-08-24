#!/usr/bin/env python3
"""Deprecated compatibility launcher for the pre-0.2 F5 server path."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence

from thtts.__main__ import run

_RENAMED_FLAGS = {
    "--ckpt-file": "--f5-checkpoint-file",
    "--vocab-file": "--f5-vocab-file",
    "--ref-audio": "--f5-reference-audio",
    "--ref-text": "--f5-reference-text",
    "--speed": "--f5-speed",
    "--nfe-steps": "--f5-nfe-steps",
    "--max-concurrent": "--max-concurrent-syntheses",
    "--voices-yaml": "--voices-file",
}


def translate_legacy_args(argv: Sequence[str]) -> list[str]:
    """Map documented F5 v1/v2 launcher arguments to the unified CLI."""

    backend = "f5-v1"
    translated: list[str] = []
    position = 0
    while position < len(argv):
        argument = argv[position]
        if argument == "--model-version":
            if position + 1 >= len(argv):
                raise SystemExit("--model-version requires v1 or v2")
            version = argv[position + 1].lower()
            position += 2
        elif argument.startswith("--model-version="):
            version = argument.partition("=")[2].lower()
            position += 1
        else:
            for old, new in _RENAMED_FLAGS.items():
                if argument == old:
                    argument = new
                    break
                if argument.startswith(f"{old}="):
                    argument = f"{new}={argument[len(old) + 1:]}"
                    break
            translated.append(argument)
            position += 1
            continue

        if version not in {"v1", "v2"}:
            raise SystemExit("--model-version must be v1 or v2")
        backend = f"f5-{version}"

    return ["--backend", backend, *translated]


def main(argv: Sequence[str] | None = None) -> None:
    warnings.warn(
        "src/wyoming_thai_f5.py is deprecated; use 'thtts --backend f5-v1' or "
        "'thtts --backend f5-v2' instead.",
        FutureWarning,
        stacklevel=2,
    )
    run(translate_legacy_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
