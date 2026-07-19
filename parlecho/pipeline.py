"""Orchestrator: runs all six stages in sequence with per-stage timing."""
import time
from pathlib import Path

from parlecho.config import PipelineConfig
from parlecho.stages.separate import separate
from parlecho.stages.asr import transcribe
from parlecho.stages.diarize import diarize
from parlecho.stages.translate import translate
from parlecho.stages.tts import synthesize
from parlecho.stages.remix import remix


def dub(
    input_path: Path,
    target_lang: str,
    source_lang: str | None = None,   # None = auto-detect from ASR
    output_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> Path:
    cfg = config or PipelineConfig()
    out = output_dir or (Path("outputs") / input_path.stem)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[name] = time.perf_counter() - t0
        print(f"[{name}] {timings[name]:.1f}s")
        return result

    stems = timed("separate", separate, input_path, out / "stems",
                  model_name=cfg.demucs_model, device=cfg.devices.separate)

    segs, detected = timed("asr", transcribe, stems["vocals"],
                           model_size=cfg.whisper_model, language=source_lang,
                           device=cfg.devices.asr)
    src = source_lang or detected
    print(f"source language: {src} ({len(segs)} segments)")

    segs, refs = timed("diarize", diarize, stems["vocals"], segs, out / "refs",
                       device=cfg.devices.diarize)

    segs = timed("translate", translate, segs, src, target_lang,
                 model_name=cfg.nllb_model, device=cfg.devices.translate)

    tts_out = timed("tts", synthesize, segs, refs, target_lang, out / "tts",
                    device=cfg.devices.tts)

    final = timed("remix", remix, tts_out, stems["accompaniment"],
                  out / "dubbed.wav", sample_rate=cfg.sample_rate,
                  max_stretch=cfg.max_stretch, min_stretch=cfg.min_stretch)

    total = sum(timings.values())
    print(f"\ntotal: {total:.1f}s -> {final}")
    return final