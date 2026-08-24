"""Small Wyoming-aware readiness probe for the selected THTTS listener."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info


async def check(host: str, port: int, timeout: float) -> bool:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        try:
            await asyncio.wait_for(async_write_event(Describe().event(), writer), timeout)
            event = await asyncio.wait_for(async_read_event(reader), timeout)
            return event is not None and Info.is_type(event.type)
        finally:
            writer.close()
            await writer.wait_closed()
    except (ConnectionError, OSError, TimeoutError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a local THTTS Wyoming listener")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=10200, type=int)
    parser.add_argument("--timeout", default=5.0, type=float)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not asyncio.run(check(args.host, args.port, args.timeout)):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
