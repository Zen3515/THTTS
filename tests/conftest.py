"""Shared test setup for the legacy and refactored service suites."""

from __future__ import annotations

import os

# PyThaiNLP and Matplotlib otherwise create caches below the read-only home
# directory used by CI and the development sandbox.
os.environ.setdefault("PYTHAINLP_DATA_DIR", "/tmp/thtts-pythainlp-data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thtts-matplotlib")
