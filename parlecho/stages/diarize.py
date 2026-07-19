"""Stage 3: speaker diarization. Tags ASR segments with speaker labels
and exports one reference audio clip per speaker for voice cloning."""
import gc
from pathlib import Path

import torchaudio

from parlecho.config import HF_TOKEN
from parlecho.stages.asr import Segment


def diarize(
    vocals_path: Path,
    segments: list[Segment],
    output_dir: Path,
    device: str = "cpu",
    min_ref_seconds: float = 6.0,
    max_ref_seconds: float = 20.0,
) -> tuple[list[Segment], dict[str, Path]]:
    """Assign speakers to segments and export per-speaker reference WAVs.
    Returns (tagged_segments, {speaker_label: reference_wav_path})."""
    import torch
    from pyannote.audio import Pipeline

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing — put it in .env")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    pipeline.to(torch.device(device))

    result = pipeline(str(vocals_path))
    # pyannote 4.x returns a wrapper; 3.x returns the Annotation directly
    diarization = getattr(result, "speaker_diarization", result)

    # Collect speaker turns: list of (start, end, label)
    turns = [
        (turn.start, turn.end, label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]

    # Tag each ASR segment with the speaker who overlaps it most
    for seg in segments:
        best_label, best_overlap = "SPEAKER_00", 0.0
        for t_start, t_end, label in turns:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        seg.speaker = best_label

    # Build a reference clip per speaker for XTTS voice cloning:
    # concatenate that speaker's segments until we have enough audio
    output_dir.mkdir(parents=True, exist_ok=True)
    wav, sr = torchaudio.load(str(vocals_path))

    refs: dict[str, Path] = {}
    for speaker in sorted({s.speaker for s in segments}):
        chunks, total = [], 0.0
        for seg in segments:
            if seg.speaker != speaker:
                continue
            dur = seg.end - seg.start
            chunk = wav[:, int(seg.start * sr): int(seg.end * sr)]
            chunks.append(chunk)
            total += dur
            if total >= max_ref_seconds:
                break
        if not chunks:
            continue
        ref = torchaudio.functional.resample(
            torch.cat(chunks, dim=1), sr, 24000  # XTTS wants 24k mono-ish input
        )
        ref_path = output_dir / f"{speaker}.wav"
        torchaudio.save(str(ref_path), ref, 24000)
        if total < min_ref_seconds:
            print(f"warning: {speaker} has only {total:.1f}s of reference audio — "
                  f"clone quality will suffer")
        refs[speaker] = ref_path

    del pipeline
    gc.collect()

    return segments, refs