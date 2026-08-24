"""Characterization tests for text behavior that the refactor must preserve."""

from __future__ import annotations

import numpy as np
import pytest

from thtts.audio import float32_to_int16_pcm
from thtts.text import normalize_thai_text, preprocess_f5_text
from util.cleantext import process_thai_repeat, replace_numbers_with_thai
from util.custom_infer import custom_chunk_text, words_to_frame


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0", "ศูนย์"),
        ("1", "หนึ่ง"),
        ("10", "สิบ"),
        ("11", "สิบเอ็ด"),
        ("20", "ยี่สิบ"),
        ("21", "ยี่สิบเอ็ด"),
        ("101", "หนึ่งร้อยหนึ่ง"),
        ("1000000", "หนึ่งล้าน"),
        ("1,234", "หนึ่งพันสองร้อยสามสิบสี่"),
        ("12.34", "สิบสองจุด สาม สี่"),
        ("12345678", "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด"),
        ("abc123", "a b c หนึ่ง สอง สาม"),
        # These deliberately record legacy behavior, including imperfect
        # linguistic handling, so it cannot change unnoticed during the move.
        ("-42", "- สี่ สอง"),
        ("12..3", "หนึ่ง สอง . . สาม"),
        ("๐๑๒", "สิบสอง"),
        ("ราคา 12 บาท", "ราคา สิบสอง บาท"),
    ],
)
def test_legacy_number_expansion(source: str, expected: str) -> None:
    assert replace_numbers_with_thai(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("คำ ๆ", "คำคำ"),
        ("คำๆ", "คำคำ"),
        ("hello", "hello"),
        ("", ""),
    ],
)
def test_legacy_thai_repeat_expansion(source: str, expected: str) -> None:
    assert process_thai_repeat(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  สวัสดี  โลก ", "สวัสดี โลก"),
        ("“ทดสอบ”—🙂\u200b", '"ทดสอบ"-'),
        ("a\x00b\u00a0\u00a0c", "ab c"),
        ("", ""),
    ],
)
def test_legacy_normalization(source: str, expected: str) -> None:
    assert normalize_thai_text(source) == expected


def test_legacy_preprocess_composes_number_repeat_and_normalization() -> None:
    assert preprocess_f5_text("ราคา 12 บาท คำๆ 🙂") == "ราคา สิบสอง บาท คำคำ"


def test_legacy_custom_f5_helpers() -> None:
    assert custom_chunk_text("hello world", max_chars=5) == ["hello", "world"]
    assert custom_chunk_text("ไทยทดสอบ", max_chars=5) == ["ไทยทดสอบ"]
    assert words_to_frame("สวัสดี hello,", 10) == 50


def test_legacy_pcm_conversion_is_shared_and_clipped() -> None:
    waveform = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    expected = np.array([-32767, -32767, -16383, 0, 16383, 32767, 32767], dtype=np.int16).tobytes()
    assert float32_to_int16_pcm(waveform) == expected
