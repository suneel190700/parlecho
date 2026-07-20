"""Streaming step 1: real-time chunk reader + VAD utterance segmentation.
Feeds 32ms frames through Silero VAD; emits complete utterances on
end-of-speech. Stream time is tracked independently of wall time so
latency can be measured whether or not playback is simulated live."""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512          # 32ms — the frame size silero-vad expects at 16k


@dataclass
class Utterance:
    audio: np.ndarray          # float32 mono 16k
    start_s: float             # stream-time start of speech
    end_s: float               # stream-time end of speech
    detected_wall: float       # wall-clock moment VAD emitted the end event


def stream_utterances(
    input_path: Path,
    realtime: bool = True,
    min_silence_ms: int = 400,
    pad_ms: int = 100,
):
    """Yield Utterances from an audio file replayed as a stream."""
    import librosa
    import torch
    from silero_vad import load_silero_vad, VADIterator

    audio, _ = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
    model = load_silero_vad()
    vad = VADIterator(model, sampling_rate=SAMPLE_RATE,
                      min_silence_duration_ms=min_silence_ms)

    pad = int(SAMPLE_RATE * pad_ms / 1000)
    frame_dur = FRAME_SAMPLES / SAMPLE_RATE
    speech_start: int | None = None
    t0 = time.perf_counter()

    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        frame = audio[i : i + FRAME_SAMPLES]
        stream_now = (i + FRAME_SAMPLES) / SAMPLE_RATE

        if realtime:
            # sleep until this frame "arrives"
            target = t0 + stream_now
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        event = vad(torch.from_numpy(frame), return_seconds=False)
        if event is None:
            continue
        if "start" in event:
            speech_start = event["start"]
        elif "end" in event and speech_start is not None:
            s = max(0, speech_start - pad)
            e = min(len(audio), event["end"] + pad)
            yield Utterance(
                audio=audio[s:e],
                start_s=s / SAMPLE_RATE,
                end_s=event["end"] / SAMPLE_RATE,
                detected_wall=time.perf_counter() - t0,
            )
            speech_start = None

    # flush: file ended mid-speech
    if speech_start is not None:
        s = max(0, speech_start - pad)
        yield Utterance(
            audio=audio[s:],
            start_s=s / SAMPLE_RATE,
            end_s=len(audio) / SAMPLE_RATE,
            detected_wall=time.perf_counter() - t0,
        )
    vad.reset_states()