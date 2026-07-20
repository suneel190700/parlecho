# Parlecho

Speech translation with voice-cloned dubbing. Takes an audio or video file in one
language and produces the same audio in another language — same speakers' voices,
same background music, translated dialogue.

```
parlecho dub interview.mp3 --to en
```

## Status

| Module | Status | Evidence |
|--------|--------|----------|
| Offline pipeline (6 stages, CLI) | done | `parlecho dub`, timing in Performance section |
| WER/CER + BLEU benchmark, 5 pairs | done | `eval_results.md`, `eval_results_large.md` |
| Speaker similarity benchmark | done | `eval_speaker_sim.md` |
| CT2 int8 NLLB (CPU speedup + memory) | done | `eval_quant.md` |
| VRAM measurement (CUDA) | not started | needs GPU session |
| Chunked streaming + latency instrumentation | not started | — |
| Context-aware short-segment translation | not started | — |
| Sidechain music ducking | not started | — |

"Done" means the evidence file is committed in this repo; a module flips to
done in the same commit that adds its evidence.

## How it works

Six-stage cascade, each model loaded sequentially so the pipeline runs on 16GB
unified memory:

1. **Separate** — Demucs (htdemucs) splits the input into a vocals stem and an
   accompaniment stem. Translation only touches dialogue; music and effects
   survive untouched.
2. **ASR** — faster-whisper transcribes the vocals stem with word-level
   timestamps and VAD filtering.
3. **Diarize** — pyannote speaker-diarization-3.1 tags each segment with a
   speaker and exports a 6–20s reference clip per speaker for voice cloning.
4. **Translate** — NLLB-200 (distilled 600M) translates each segment,
   preserving timing and speaker attribution. Two backends: HuggingFace fp32
   and CTranslate2 int8 (see quantization results).
5. **TTS** — XTTS-v2 speaks each translated line, cloned from that speaker's
   reference clip.
6. **Remix** — each generated line is time-stretched (clamped to 0.75–1.35x)
   to fit its original slot, placed on the timeline, and mixed over the
   accompaniment stem.

## Benchmark results

All quality metrics are measured on the FLEURS test split, n=100 sentences per
pair (n=15 for speaker similarity), evaluated on Apple M-series CPU. Scripts to
reproduce every table are in `parlecho/eval/`.

### ASR + translation quality (whisper-large-v3, NLLB-200-600M)

| Pair | n | WER % | CER % | BLEU (cascade) | BLEU (oracle MT) |
|------|---|-------|-------|----------------|------------------|
| es→en | 100 | 2.9 | 1.2 | 29.7 | 32.1 |
| fr→en | 100 | 5.8 | 1.8 | 38.1 | 43.0 |
| de→en | 100 | 4.4 | 1.7 | 37.0 | 41.5 |
| hi→en | 100 | 26.4 | 9.1 | 22.7 | 35.9 |
| ja→en | 100 | 91.6* | 7.7 | 20.2 | 24.9 |

*Word-level WER is not meaningful for unsegmented scripts — read CER for ja
(and prefer CER for hi). WER/CER computed after normalizing case, punctuation,
and whitespace on both reference and hypothesis (as in the Whisper paper).

"Cascade" BLEU scores translations of the ASR output — the number the actual
pipeline achieves. "Oracle" BLEU scores translations of the gold transcript —
what the MT stage achieves with perfect ASR. The gap between them quantifies
how much ASR errors cost downstream.

### ASR model comparison: where the bottleneck is

| Pair | WER % (small) | WER % (large-v3) | Cascade BLEU (small) | Cascade BLEU (large-v3) |
|------|---------------|------------------|----------------------|-------------------------|
| es→en | 5.7 | 2.9 | 26.8 | 29.7 |
| fr→en | 14.1 | 5.8 | 32.5 | 38.1 |
| de→en | 8.5 | 4.4 | 34.4 | 37.0 |
| hi→en | 54.1 | 26.4 | 12.3 | 22.7 |
| ja→en | 15.4 CER | 7.7 CER | 15.3 | 20.2 |

Key finding: for lower-resource languages, ASR — not translation — is the
bottleneck. Upgrading only the ASR model halved Hindi WER and recovered +10.4
cascade BLEU, closing most of the gap to oracle MT. European pairs were already
near their oracle ceiling with the small model.

### Speaker similarity (XTTS-v2 cross-lingual clones)

| Pair | n | Same-speaker cosine | Std | Different-speaker baseline |
|------|---|---------------------|-----|----------------------------|
| es→en | 15 | 0.46 | 0.076 | 0.048 |
| fr→en | 15 | 0.531 | 0.072 | 0.219 |
| de→en | 15 | 0.58 | 0.074 | 0.267 |
| hi→en | 15 | 0.447 | 0.076 | 0.081 |
| ja→en | 15 | 0.439 | 0.085 | 0.195 |

Cosine similarity between ECAPA-TDNN embeddings of the original recording and
English XTTS output cloned from that recording. The different-speaker baseline
scores mismatched (original, clone) pairs; the same/different separation on
every pair shows cross-lingual cloning preserves speaker identity. Absolute
values in the 0.4–0.6 range are typical for cross-lingual cloning, which scores
below same-language cloning.

### Quantization (NLLB-600M: HF fp32 vs CTranslate2 int8)

Matched greedy decoding (beam 1) on both backends, identical inputs, Apple
M-series CPU.

| Backend | Pair | n | BLEU | Translate time (s) | Sentences/s | Peak RSS (GB) |
|---------|------|---|------|--------------------|-------------|----------------|
| HF fp32 | es→en | 100 | 32.1 | 36.9 | 2.71 | 3.67 |
| CT2 int8 | es→en | 100 | 32.9 | 13.6 | 7.38 | 2.33 |

int8 quantization: 2.7x faster, no BLEU degradation, 36% lower peak process
memory, and 599MB on disk vs ~2.4GB fp32 (75% smaller). In an unmatched run,
CT2 int8 with beam_size=4 (32.1s) still outran HF fp32 greedy (36.9s) — int8
buys back the full cost of 4-beam search. VRAM comparison on CUDA is pending
(see Status).

## Setup

Requirements: Python 3.11, ffmpeg on PATH, a HuggingFace token with terms
accepted for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio
pip install faster-whisper demucs "pyannote.audio>=3.1"
pip install coqui-tts
pip install "transformers>=4.57,<5" sentencepiece python-dotenv
pip install soundfile librosa
pip install datasets "sacrebleu[ja]" jiwer speechbrain   # eval only
pip install -e .
echo "HF_TOKEN=hf_your_token" > .env
```

To build the CT2 int8 translation backend:

```
ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
  --quantization int8 --output_dir models/nllb-600M-ct2
```

## Usage

```
parlecho dub input.mp3 --to en              # auto-detects source language
parlecho dub input.mp4 --to en --from es    # or pin it
```

Output lands in `outputs/<input-name>/dubbed.wav`, with intermediate stems,
per-speaker reference clips, and per-segment TTS in the same directory.
Per-stage timing is printed on every run.

Reproduce the benchmarks:

```
python -m parlecho.eval.run_eval --pairs es,fr,de,hi,ja --n 100 --whisper large-v3
python -m parlecho.eval.speaker_sim --pairs es,fr,de,hi,ja --n 15
python -m parlecho.eval.quant_bench --backend hf --n 100
python -m parlecho.eval.quant_bench --backend ct2 --n 100
```

## Performance

On an Apple M-series laptop (CPU/MPS, sequential model loading), a 58.5s
Spanish clip dubs to English in 82.5s end to end with whisper-small:
separation 7.5s, ASR 7.4s, diarization 12.3s, translation 6.1s, TTS 48.8s,
remix 0.3s. TTS dominates because XTTS-v2 runs on CPU (unstable on MPS);
faster-whisper runs CTranslate2 int8 on CPU (no MPS backend). Demucs, pyannote,
and NLLB run on MPS.

## Current limitations

- Short-utterance translation: isolated one-word segments ("Paulo") can
  hallucinate in NLLB without sentence context. Context-window translation is
  planned (see Status).
- Music ducking is static (fixed gain), not sidechained.
- Overlapping dialogue: lines that exceed the stretch clamp spill into
  following silence and can collide in fast conversation.
- Speaker similarity is measured at n=15 per pair (CPU TTS cost); the
  same/different separation is unambiguous but per-pair means are noisy.

## License note

XTTS-v2 is distributed under the Coqui Public Model License (non-commercial).
This project is a research/portfolio pipeline; swap the TTS stage for a
commercially licensed model before any commercial use.