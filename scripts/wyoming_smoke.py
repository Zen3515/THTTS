#!/usr/bin/env python3
"""Fail fast unless a THTTS Wyoming listener answers a `describe` request."""

from __future__ import annotations

from thtts.healthcheck import run

if __name__ == "__main__":
    run()
