"""Single source of truth for THTTS CLI, environment, and deprecations."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from . import __version__
from .config import (
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_F5_NFE_STEPS,
    DEFAULT_F5_SPEED,
    DEFAULT_HOST,
    DEFAULT_MAX_CONCURRENT_SYNTHESES,
    DEFAULT_MAX_QUEUE_SECONDS,
    DEFAULT_MAX_QUEUED_SYNTHESES,
    DEFAULT_PORT,
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    DEFAULT_STREAM_IDLE_FLUSH_MS,
    DEFAULT_STREAM_MAX_SEGMENT_CHARS,
    DEFAULT_STREAM_MIN_SEGMENT_CHARS,
    DEFAULT_STREAM_TARGET_CHARS,
    DEFAULT_VITS_MODEL,
    F5Settings,
    Settings,
    StreamSettings,
)

_T = TypeVar("_T")

_BACKEND_ALIASES = {
    "vits": "vits",
    "f5-v1": "f5-v1",
    "f5-v2": "f5-v2",
    "f5_v1": "f5-v1",
    "f5_v2": "f5-v2",
    "f5-thv1": "f5-v1",
    "f5-thv2": "f5-v2",
    "f5th": "f5-v1",
    "v1": "f5-v1",
    "v2": "f5-v2",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thai Wyoming text-to-speech service with selectable backends"
    )
    parser.add_argument("--host", default=argparse.SUPPRESS)
    parser.add_argument("--port", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--backend", default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=argparse.SUPPRESS)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--max-concurrent-syntheses", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--max-queued-syntheses", type=_nonnegative_int, default=argparse.SUPPRESS)
    parser.add_argument("--max-queue-seconds", type=_positive_float, default=argparse.SUPPRESS)
    parser.add_argument("--vits-model", default=argparse.SUPPRESS)
    parser.add_argument("--f5-checkpoint-file", default=argparse.SUPPRESS)
    parser.add_argument("--f5-vocab-file", default=argparse.SUPPRESS)
    parser.add_argument("--f5-reference-audio", default=argparse.SUPPRESS)
    parser.add_argument("--f5-reference-text", default=argparse.SUPPRESS)
    parser.add_argument("--f5-speed", type=_positive_float, default=argparse.SUPPRESS)
    parser.add_argument("--f5-nfe-steps", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--voices-file", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--stream-idle-flush-ms", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--stream-min-segment-chars", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--stream-target-chars", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--stream-max-segment-chars", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--shutdown-grace-seconds", type=_nonnegative_float, default=argparse.SUPPRESS
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def parse_settings(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None
) -> Settings:
    parser = build_parser()
    args = parser.parse_args(argv)
    env = os.environ if environ is None else environ
    deprecations: list[str] = []

    backend_raw = _resolve(
        parser, args, env, "backend", "THTTS_BACKEND", DEFAULT_BACKEND, deprecations
    )
    backend = _canonical_backend(parser, str(backend_raw), deprecations)

    vits_model = _resolve(
        parser,
        args,
        env,
        "vits_model",
        "THTTS_VITS_MODEL",
        DEFAULT_VITS_MODEL,
        deprecations,
        legacy=("THTTS_MODEL",),
    )
    if backend != "vits" and _nonempty(env.get("THTTS_MODEL")):
        deprecations.append(
            "THTTS_MODEL is VITS-only and is ignored because the selected backend is F5"
        )

    try:
        return Settings(
            host=_resolve(parser, args, env, "host", "THTTS_HOST", DEFAULT_HOST, deprecations),
            port=_resolve(
                parser,
                args,
                env,
                "port",
                "THTTS_PORT",
                DEFAULT_PORT,
                deprecations,
                parser_type=_positive_int,
            ),
            backend=backend,
            device=_resolve(
                parser, args, env, "device", "THTTS_DEVICE", DEFAULT_DEVICE, deprecations
            ),
            log_level=_resolve(
                parser, args, env, "log_level", "THTTS_LOG_LEVEL", "INFO", deprecations
            ).upper(),
            max_concurrent_syntheses=_resolve(
                parser,
                args,
                env,
                "max_concurrent_syntheses",
                "THTTS_MAX_CONCURRENT_SYNTHESES",
                DEFAULT_MAX_CONCURRENT_SYNTHESES,
                deprecations,
                parser_type=_positive_int,
                legacy=("THTTS_MAX_CONCURRENT",),
            ),
            max_queued_syntheses=_resolve(
                parser,
                args,
                env,
                "max_queued_syntheses",
                "THTTS_MAX_QUEUED_SYNTHESES",
                DEFAULT_MAX_QUEUED_SYNTHESES,
                deprecations,
                parser_type=_nonnegative_int,
            ),
            max_queue_seconds=_resolve(
                parser,
                args,
                env,
                "max_queue_seconds",
                "THTTS_MAX_QUEUE_SECONDS",
                DEFAULT_MAX_QUEUE_SECONDS,
                deprecations,
                parser_type=_positive_float,
            ),
            vits_model=str(vits_model),
            f5=F5Settings(
                checkpoint_file=_resolve(
                    parser,
                    args,
                    env,
                    "f5_checkpoint_file",
                    "THTTS_F5_CHECKPOINT_FILE",
                    None,
                    deprecations,
                    legacy=("THTTS_CKPT_FILE",),
                ),
                vocab_file=_resolve(
                    parser,
                    args,
                    env,
                    "f5_vocab_file",
                    "THTTS_F5_VOCAB_FILE",
                    None,
                    deprecations,
                    legacy=("THTTS_VOCAB_FILE",),
                ),
                reference_audio=_resolve(
                    parser,
                    args,
                    env,
                    "f5_reference_audio",
                    "THTTS_F5_REFERENCE_AUDIO",
                    None,
                    deprecations,
                    legacy=("THTTS_REF_AUDIO",),
                ),
                reference_text=_resolve(
                    parser,
                    args,
                    env,
                    "f5_reference_text",
                    "THTTS_F5_REFERENCE_TEXT",
                    None,
                    deprecations,
                    legacy=("THTTS_REF_TEXT",),
                ),
                speed=_resolve(
                    parser,
                    args,
                    env,
                    "f5_speed",
                    "THTTS_F5_SPEED",
                    DEFAULT_F5_SPEED,
                    deprecations,
                    parser_type=_positive_float,
                    legacy=("THTTS_SPEED", "THTTS_SPEAK_SPEED"),
                ),
                nfe_steps=_resolve(
                    parser,
                    args,
                    env,
                    "f5_nfe_steps",
                    "THTTS_F5_NFE_STEPS",
                    DEFAULT_F5_NFE_STEPS,
                    deprecations,
                    parser_type=_positive_int,
                    legacy=("THTTS_NFE_STEPS",),
                ),
                voices_file=_resolve(
                    parser,
                    args,
                    env,
                    "voices_file",
                    "THTTS_VOICES_FILE",
                    None,
                    deprecations,
                    parser_type=Path,
                    legacy=("THTTS_VOICES_YAML",),
                ),
            ),
            stream=StreamSettings(
                idle_flush_ms=_resolve(
                    parser,
                    args,
                    env,
                    "stream_idle_flush_ms",
                    "THTTS_STREAM_IDLE_FLUSH_MS",
                    DEFAULT_STREAM_IDLE_FLUSH_MS,
                    deprecations,
                    parser_type=_positive_int,
                    legacy=("THTTS_MAX_WAIT_MS",),
                ),
                min_segment_chars=_resolve(
                    parser,
                    args,
                    env,
                    "stream_min_segment_chars",
                    "THTTS_STREAM_MIN_SEGMENT_CHARS",
                    DEFAULT_STREAM_MIN_SEGMENT_CHARS,
                    deprecations,
                    parser_type=_positive_int,
                    legacy=("THTTS_MIN_SENT_CHARS",),
                ),
                target_chars=_resolve(
                    parser,
                    args,
                    env,
                    "stream_target_chars",
                    "THTTS_STREAM_TARGET_CHARS",
                    DEFAULT_STREAM_TARGET_CHARS,
                    deprecations,
                    parser_type=_positive_int,
                ),
                max_segment_chars=_resolve(
                    parser,
                    args,
                    env,
                    "stream_max_segment_chars",
                    "THTTS_STREAM_MAX_SEGMENT_CHARS",
                    DEFAULT_STREAM_MAX_SEGMENT_CHARS,
                    deprecations,
                    parser_type=_positive_int,
                ),
            ),
            shutdown_grace_seconds=_resolve(
                parser,
                args,
                env,
                "shutdown_grace_seconds",
                "THTTS_SHUTDOWN_GRACE_SECONDS",
                DEFAULT_SHUTDOWN_GRACE_SECONDS,
                deprecations,
                parser_type=_nonnegative_float,
            ),
            deprecations=tuple(deprecations),
        )
    except ValueError as err:
        parser.error(str(err))
        raise AssertionError("argparse.error must exit") from err


def _resolve(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    env: Mapping[str, str],
    argument: str,
    canonical_env: str,
    default: _T,
    deprecations: list[str],
    *,
    parser_type: Callable[[str], _T] | type[Path] | None = None,
    legacy: tuple[str, ...] = (),
) -> _T:
    if hasattr(args, argument):
        for legacy_name in legacy:
            if _nonempty(env.get(legacy_name)) is not None:
                deprecations.append(_legacy_migration_notice(legacy_name, canonical_env))
        return getattr(args, argument)

    canonical = _nonempty(env.get(canonical_env))
    legacy_values = {name: value for name in legacy if (value := _nonempty(env.get(name)))}
    distinct_legacy = set(legacy_values.values())
    if len(distinct_legacy) > 1:
        parser.error(
            f"conflicting legacy configuration for {canonical_env}: "
            + ", ".join(sorted(legacy_values))
        )
    if canonical is not None and distinct_legacy and canonical not in distinct_legacy:
        parser.error(
            f"conflicting configuration: {canonical_env} differs from "
            + ", ".join(sorted(legacy_values))
        )

    if canonical is not None:
        value: object = canonical
    elif legacy_values:
        legacy_name, value = next(iter(legacy_values.items()))
    else:
        return default

    for legacy_name in legacy_values:
        deprecations.append(_legacy_migration_notice(legacy_name, canonical_env))

    if parser_type is None:
        return value  # type: ignore[return-value]
    try:
        return parser_type(str(value))  # type: ignore[call-arg,return-value]
    except (TypeError, ValueError, argparse.ArgumentTypeError) as err:
        parser.error(f"invalid {canonical_env}: {err}")
        raise AssertionError("argparse.error must exit") from err


def _canonical_backend(parser: argparse.ArgumentParser, raw: str, deprecations: list[str]) -> str:
    normalized = raw.strip().lower()
    try:
        backend = _BACKEND_ALIASES[normalized]
    except KeyError:
        parser.error("THTTS_BACKEND/--backend must be one of: vits, f5-v1, f5-v2")
        raise AssertionError("argparse.error must exit") from None
    if raw != backend:
        deprecations.append(
            f"Deprecated configuration: THTTS_BACKEND={raw} will be removed in the next "
            f"breaking release. Replace it with: THTTS_BACKEND={backend}"
        )
    return backend


def _legacy_migration_notice(legacy_name: str, canonical_env: str) -> str:
    """Return a copy-paste-safe migration notice without exposing env values."""

    return (
        f"Deprecated configuration: {legacy_name} will be removed in the next breaking "
        f"release. Replace it with: {canonical_env}=${{{legacy_name}}}"
    )


def _nonempty(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value
