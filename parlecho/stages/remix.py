"""Stage 6: remix. Time-stretch each generated line to fit its original slot,
place it on a timeline, and mix with the accompaniment stem."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from parlecho.stages.asr import Segment


def remix(
    tts_results: list[tuple[Segment, Path]],
    accompaniment_path: Path,
    output_path: Path,
    sample_rate: int = 44100,
    max_stretch: float = 1.35,
    min_stretch: float = 0.75,
) -> Path:
    """Assemble the dubbed track: stretched TTS lines over the music stem."""
    music, _ = librosa.load(str(accompaniment_path), sr=sample_rate, mono=False)
    if music.ndim == 1:
        music = np.stack([music, music])
    total_samples = music.shape[1]

    dialogue = np.zeros_like(music)

    for seg, wav_path in tts_results:
        speech, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)

        slot = seg.end - seg.start                     # original duration
        actual = len(speech) / sample_rate             # generated duration
        rate = actual / slot                           # >1 = too long, speed up

        # Clamp: beyond these bounds stretched speech sounds bad, so we
        # accept overflow into the following silence instead
        rate = float(np.clip(rate, min_stretch, max_stretch))
        if abs(rate - 1.0) > 0.02:
            speech = librosa.effects.time_stretch(speech, rate=rate)

        start = int(seg.start * sample_rate)
        end = min(start + len(speech), total_samples)
        if start >= total_samples:
            continue
        dialogue[:, start:end] += speech[: end - start]

    # Peak-normalize dialogue to sit clearly above the music
    peak = np.abs(dialogue).max()
    if peak > 0:
        dialogue = dialogue / peak * 0.9

    mix = dialogue + music * 0.85       # slight music duck, crude but effective
    mix = np.clip(mix, -1.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix.T, sample_rate)
    return output_path