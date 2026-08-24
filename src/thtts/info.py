"""Wyoming discovery response built from the selected TTS backend."""

from __future__ import annotations

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

from . import __version__
from .backends.base import TtsBackend


def make_info(backend: TtsBackend) -> Info:
    metadata = backend.metadata
    return Info(
        tts=[
            TtsProgram(
                name=metadata.program_name,
                attribution=Attribution(
                    name=metadata.attribution_name, url=metadata.attribution_url
                ),
                voices=[
                    TtsVoice(
                        name=voice.name,
                        attribution=Attribution(
                            name=voice.attribution_name, url=voice.attribution_url
                        ),
                        languages=list(voice.languages),
                        description=voice.description,
                        installed=True,
                        version=voice.version,
                    )
                    for voice in backend.voices
                ],
                installed=True,
                description=metadata.program_description,
                version=__version__,
                supports_synthesize_streaming=metadata.supports_synthesize_streaming,
            )
        ]
    )
