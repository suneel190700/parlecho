"""Streaming step 2: resident-model worker. Processes one utterance at a
time — ASR -> CT2 translate -> XTTS — with all models held in memory and
per-stage timings recorded for the latency report.

Streaming scope (v1): no Demucs, no clustering diarization. The clone
reference is a rolling buffer of the first ~10s of detected speech."""
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from parlecho.config import lang
from parlecho.streaming.chunker import Utterance, SAMPLE_RATE

os.environ.setdefault("COQUI_TOS_AGREED", "1")


@dataclass
class DubbedChunk:
    utterance: Utterance
    src_text: str
    tgt_text: str
    wav_path: Path | None          # None if the utterance had no speech text
    timings: dict[str, float]      # per-stage seconds


class StreamWorker:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        workdir: Path,
        whisper_model: str = "small",
        ct2_dir: Path = Path("models/nllb-600M-ct2"),
        device: str = "cpu",
        ref_target_s: float = 10.0,
    ):
        from faster_whisper import WhisperModel
        import ctranslate2
        from transformers import AutoTokenizer
        from TTS.api import TTS

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.workdir = workdir
        workdir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        self.asr = WhisperModel(whisper_model, device=device, compute_type="int8")
        self.translator = ctranslate2.Translator(
            str(ct2_dir), device=device, compute_type="int8"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M", src_lang=lang(source_lang, "nllb")
        )
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.load_s = time.perf_counter() - t0

        # rolling clone reference: first ref_target_s of detected speech
        self.ref_target = int(ref_target_s * SAMPLE_RATE)
        self._ref_audio = np.zeros(0, dtype=np.float32)
        self._ref_path = workdir / "ref.wav"
        self._counter = 0

    def _update_ref(self, audio: np.ndarray) -> Path:
        if len(self._ref_audio) < self.ref_target:
            self._ref_audio = np.concatenate([self._ref_audio, audio])
            sf.write(str(self._ref_path), self._ref_audio, SAMPLE_RATE)
        return self._ref_path

    def process(self, utt: Utterance) -> DubbedChunk:
        timings: dict[str, float] = {}
        ref = self._update_ref(utt.audio)

        t = time.perf_counter()
        segs, _ = self.asr.transcribe(
            utt.audio,
            language=self.source_lang,
            vad_filter=False,          # VAD already ran in the chunker
            beam_size=1,
        )
        src_text = " ".join(s.text for s in segs).strip()
        timings["asr"] = time.perf_counter() - t

        if not src_text:
            return DubbedChunk(utt, "", "", None, timings)

        t = time.perf_counter()
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(src_text))
        result = self.translator.translate_batch(
            [tokens],
            target_prefix=[[lang(self.target_lang, "nllb")]],
            beam_size=1,               # greedy: latency over marginal quality
        )
        hyp = result[0].hypotheses[0][1:]
        tgt_text = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(hyp), skip_special_tokens=True
        ).strip()
        timings["translate"] = time.perf_counter() - t

        t = time.perf_counter()
        out_path = self.workdir / f"chunk_{self._counter:04d}.wav"
        self._counter += 1
        self.tts.tts_to_file(
            text=tgt_text,
            speaker_wav=str(ref),
            language=lang(self.target_lang, "xtts"),
            file_path=str(out_path),
        )
        timings["tts"] = time.perf_counter() - t

        return DubbedChunk(utt, src_text, tgt_text, out_path, timings)