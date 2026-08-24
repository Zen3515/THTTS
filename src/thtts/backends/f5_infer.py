"""Compatibility boundary for THTTS's local F5 v2 inference extension."""

from __future__ import annotations

# The implementation remains in the vendored utility module for this release.
# Keeping this import boundary lets the backend stop depending on its legacy
# location in the next compatibility release without changing model behavior.
from util.custom_infer import PreparedReference, custom_infer_process, prepare_reference

__all__ = ["PreparedReference", "custom_infer_process", "prepare_reference"]
