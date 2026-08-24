"""Target streaming behavior and shared audio helper tests."""

from __future__ import annotations

import numpy as np

from thtts.audio import float32_to_int16_pcm, iter_pcm_chunks
from thtts.text import TextSegmenter, normalize_thai_text, preprocess_f5_text


def test_explicit_terminator_flushes_short_target_segment() -> None:
    segmenter = TextSegmenter(minimum_chars=15, target_chars=48, maximum_chars=180)
    assert segmenter.add("สวัสดี!") == ["สวัสดี!"]
    assert segmenter.pending_text == ""


def test_unterminated_short_text_waits_until_final_flush() -> None:
    segmenter = TextSegmenter(minimum_chars=15, target_chars=48, maximum_chars=180)
    assert segmenter.add("สวัสดี") == []
    assert segmenter.idle_flush() == []
    assert segmenter.final_flush() == ["สวัสดี"]


def test_segmenter_buffers_across_transport_chunks() -> None:
    segmenter = TextSegmenter(minimum_chars=15, target_chars=48, maximum_chars=180)
    assert segmenter.add("คำตอบส") == []
    assert segmenter.add("ั้น.") == ["คำตอบสั้น."]


def test_shared_f5_preprocess_preserves_legacy_result() -> None:
    assert preprocess_f5_text("ราคา 12 บาท คำๆ 🙂") == "ราคา สิบสอง บาท คำคำ"
    assert normalize_thai_text("“ทดสอบ”—🙂\u200b") == '"ทดสอบ"-'


def test_pcm_chunks_are_deterministic_and_nonempty() -> None:
    waveform = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    chunks = list(iter_pcm_chunks(waveform, sample_rate=10, chunk_milliseconds=300))
    assert [len(chunk) for chunk in chunks] == [6, 6, 6, 2]
    assert b"".join(chunks) == float32_to_int16_pcm(waveform)
