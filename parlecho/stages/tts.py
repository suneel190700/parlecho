"""Stage 5: TTS. XTTS-v2 speaks each translated segment in the original
speaker's cloned voice, using the reference clips from diarization."""
import gc
import os
from pathlib import Path

from parlecho.config import lang
from parlecho.stages.asr import Segment

# XTTS-v2 is CPML-licensed; this env var accepts the terms non-interactively
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def synthesize(
    segments: list[Segment],
    speaker_refs: dict[str, Path],
    target_lang: str,
    output_dir: Path,
    device: str = "cpu",
) -> list[tuple[Segment, Path]]:
    """Generate one WAV per segment in the matching speaker's cloned voice.
    Returns [(segment, wav_path), ...] in order."""
    from TTS.api import TTS

    output_dir.mkdir(parents=True, exist_ok=True)
    xtts_lang = lang(target_lang, "xtts")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    results: list[tuple[Segment, Path]] = []
    fallback_ref = next(iter(speaker_refs.values()))

    for i, seg in enumerate(segments):
        if not seg.text.strip():
            continue
        ref = speaker_refs.get(seg.speaker, fallback_ref)
        out_path = output_dir / f"seg_{i:04d}.wav"
        tts.tts_to_file(
            text=seg.text,
            speaker_wav=str(ref),
            language=xtts_lang,
            file_path=str(out_path),
        )
        results.append((seg, out_path))
        print(f"  [{i+1}/{len(segments)}] {seg.speaker}: {seg.text[:50]}")

    del tts
    gc.collect()

    return results