"""Lifecycle entry point for the unified THTTS Wyoming service."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Sequence
from functools import partial

from wyoming.server import AsyncServer

from .backends.base import BackendError
from .cli import parse_settings

_LOGGER = logging.getLogger(__name__)


async def main(argv: Sequence[str] | None = None) -> None:
    settings = parse_settings(argv)
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for warning in settings.deprecations:
        _LOGGER.warning("%s", warning)
    _LOGGER.info("Starting THTTS with %s", settings.safe_summary())

    # Keep `thtts --help` and `--version` free of vendor/PyThaiNLP imports.
    from .backends import create_backend
    from .handler import TtsEventHandler
    from .info import make_info

    backend = create_backend(settings)
    _LOGGER.info("Loading TTS backend %s", settings.backend)
    await backend.load()
    _LOGGER.info("TTS backend ready: device=%s", backend.resolved_device)

    server = AsyncServer.from_uri(f"tcp://{settings.host}:{settings.port}")
    handler_factory = partial(
        TtsEventHandler,
        make_info(backend),
        backend,
        stream_settings=settings.stream,
    )
    await _serve_until_cancelled(
        server, handler_factory, shutdown_grace_seconds=settings.shutdown_grace_seconds
    )


async def _serve_until_cancelled(
    server: AsyncServer, handler_factory: object, *, shutdown_grace_seconds: float
) -> None:
    """Stop accepting work on SIGTERM/SIGINT and close active connections."""

    await server.start(handler_factory)  # type: ignore[arg-type]
    _LOGGER.info("Serving Wyoming TTS")
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    registered: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
            registered.append(signum)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await stop_event.wait()
    finally:
        for signum in registered:
            loop.remove_signal_handler(signum)
        listener = getattr(server, "_server", None)
        if listener is not None:
            listener.close()
            await listener.wait_closed()
        handlers = getattr(server, "_handlers", {})
        if shutdown_grace_seconds and handlers:
            _LOGGER.info(
                "Allowing active Wyoming requests up to %ss to finish", shutdown_grace_seconds
            )
            deadline = asyncio.get_running_loop().time() + shutdown_grace_seconds
            while handlers and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
        await server.stop()


def run(argv: Sequence[str] | None = None) -> None:
    try:
        asyncio.run(main(argv))
    except KeyboardInterrupt:
        pass
    except (BackendError, ValueError) as err:
        raise SystemExit(f"Unable to start THTTS: {err}") from err


if __name__ == "__main__":
    run()
