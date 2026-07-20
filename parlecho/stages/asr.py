"""Stage 2: ASR. Transcribes the vocals stem with faster-whisper,
returning timed segments that drive translation and TTS alignment."""
import gc
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None   # filled in by the diarization stage


def transcribe(
    vocals_path: Path,
    model_size: str = "small",
    language: str | None = None,   # None = auto-detect
    device: str = "cpu",
    model=None,                    # pass a loaded WhisperModel to reuse it
) -> tuple[list[Segment], str]:
    """Transcribe audio. Returns (segments, detected_language)."""
    from faster_whisper import WhisperModel

    owns_model = model is None
    if owns_model:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",   # fast on M-series CPU; harmless elsewhere
        )

    raw_segments, info = model.transcribe(
        str(vocals_path),
        language=language,
        vad_filter=True,               # skip non-speech, big win post-separation
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
    )

    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in raw_segments
        if s.text.strip()
    ]

    if owns_model:
        del model
        gc.collect()

    return segments, info.language