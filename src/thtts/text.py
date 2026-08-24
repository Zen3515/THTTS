"""Thai preprocessing and request-local streaming text segmentation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from util.cleantext import process_thai_repeat, replace_numbers_with_thai

_EMOJI_PATTERN = re.compile(
    "[\U0001f300-\U0001f6ff\U0001f900-\U0001f9ff\U0001fa70-\U0001faff\u2600-\u27bf\ufe0e\ufe0f]+",
    flags=re.UNICODE,
)
_CONTROL_PATTERN = re.compile(r"[\u0000-\u001F\u007F]")
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\u2060]")
_MULTISPACE_PATTERN = re.compile(r"[ \t\u00A0]{2,}")
_TERMINATORS = frozenset({".", "!", "?", "…", "।", "ฯ", "\n"})


def normalize_thai_text(text: str) -> str:
    """Normalize visible text without changing its language-specific content."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = (
        normalized.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    normalized = _EMOJI_PATTERN.sub("", normalized)
    normalized = _ZERO_WIDTH_PATTERN.sub("", normalized)
    normalized = _CONTROL_PATTERN.sub("", normalized)
    return _MULTISPACE_PATTERN.sub(" ", normalized).strip()


def preprocess_f5_text(text: str) -> str:
    """Preserve the existing F5 Thai number/repetition preprocessing path."""

    return normalize_thai_text(process_thai_repeat(replace_numbers_with_thai(text)))


@dataclass
class TextSegmenter:
    """Buffers streamed text and yields safe request-local synthesis segments.

    An explicit terminator always flushes, including a very short Thai answer.
    The minimum threshold only avoids emitting an unterminated partial segment.
    """

    minimum_chars: int
    target_chars: int
    maximum_chars: int
    _buffer: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.minimum_chars < 1:
            raise ValueError("minimum_chars must be positive")
        if self.target_chars < self.minimum_chars:
            raise ValueError("target_chars must be at least minimum_chars")
        if self.maximum_chars < self.target_chars:
            raise ValueError("maximum_chars must be at least target_chars")

    @property
    def pending_text(self) -> str:
        return self._buffer

    def add(self, text: str) -> list[str]:
        self._buffer += text
        return self._take_ready()

    def idle_flush(self) -> list[str]:
        if len(self._buffer.strip()) >= self.minimum_chars:
            return self._take_all()
        return []

    def final_flush(self) -> list[str]:
        return self._take_all()

    def _take_ready(self) -> list[str]:
        ready: list[str] = []
        start = 0
        for index, char in enumerate(self._buffer):
            if char in _TERMINATORS:
                candidate = self._buffer[start : index + 1].strip()
                if candidate:
                    ready.append(candidate)
                start = index + 1

        if start:
            self._buffer = self._buffer[start:]

        if len(self._buffer.strip()) >= self.maximum_chars:
            ready.extend(self._take_all())
        elif len(self._buffer.strip()) >= self.target_chars:
            ready.extend(self._take_all())
        return ready

    def _take_all(self) -> list[str]:
        text = self._buffer.strip()
        self._buffer = ""
        return [text] if text else []
