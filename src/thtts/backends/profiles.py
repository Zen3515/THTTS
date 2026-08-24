"""Verified-compatible F5 model profiles selected by registry entry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class F5Profile:
    backend_id: str
    checkpoint_uri: str
    vocab_uri: str
    reference_audio_uri: str
    reference_text: str
    model_config: dict[str, int | bool | None]
    uses_ipa_inference: bool
    attribution_name: str
    attribution_url: str


_REFERENCE_TEXT = "ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบาย"
_V1_REFERENCE_AUDIO = "hf://VIZINTZOR/F5-TTS-THAI/sample/ref_audio.wav"

F5_PROFILES: dict[str, F5Profile] = {
    "f5-v1": F5Profile(
        backend_id="f5-v1",
        checkpoint_uri="hf://VIZINTZOR/F5-TTS-THAI/model_1000000.pt",
        vocab_uri="hf://VIZINTZOR/F5-TTS-THAI/vocab.txt",
        reference_audio_uri=_V1_REFERENCE_AUDIO,
        reference_text=_REFERENCE_TEXT,
        model_config={
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "text_mask_padding": False,
            "conv_layers": 4,
            "pe_attn_head": 1,
        },
        uses_ipa_inference=False,
        attribution_name="VIZINTZOR/F5-TTS-THAI",
        attribution_url="https://huggingface.co/VIZINTZOR/F5-TTS-THAI",
    ),
    "f5-v2": F5Profile(
        backend_id="f5-v2",
        checkpoint_uri="hf://VIZINTZOR/F5-TTS-TH-V2/model_250000.pt",
        vocab_uri="hf://VIZINTZOR/F5-TTS-TH-V2/vocab.txt",
        # Existing v2 deployment behavior uses the v1 bundled reference sample.
        # Keep this compatibility default until an integration fixture validates a
        # v2-native reference asset and transcript together.
        reference_audio_uri=_V1_REFERENCE_AUDIO,
        reference_text=_REFERENCE_TEXT,
        model_config={
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "text_mask_padding": True,
            "conv_layers": 4,
            "pe_attn_head": None,
        },
        uses_ipa_inference=True,
        attribution_name="VIZINTZOR/F5-TTS-TH-V2",
        attribution_url="https://huggingface.co/VIZINTZOR/F5-TTS-TH-V2",
    ),
}
