"""WER + BLEU benchmark over FLEURS. Usage:
python -m parlecho.eval.run_eval --pairs es,fr,de,hi,ja --n 100
"""
import argparse
import tempfile
import time
from pathlib import Path

import soundfile as sf

from parlecho.eval.datasets import load_pair
from parlecho.stages.asr import transcribe, Segment
from parlecho.stages.translate import translate


def eval_pair(source: str, n: int, whisper_model: str, device: str) -> dict:
    import jiwer
    import sacrebleu
    from faster_whisper import WhisperModel

    model = WhisperModel(whisper_model, device=device, compute_type="int8")

    refs_wer, hyps_wer = [], []
    refs_bleu, asr_texts, gold_texts = [], [], []

    t0 = time.perf_counter()
    for ex in load_pair(source, "en", n=n):
        # ASR expects a file path; FLEURS gives arrays
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, ex["audio"], ex["sr"])
            segs, _ = transcribe(Path(f.name), language=source,
                                 device=device, model=model)
        asr_text = " ".join(s.text for s in segs).strip()

        refs_wer.append(ex["src_transcript"])
        hyps_wer.append(asr_text)
        refs_bleu.append(ex["tgt_raw"])
        asr_texts.append(asr_text)
        gold_texts.append(ex["src_raw"])

    del model

    def translate_batch(texts):
        # empty ASR outputs become a single space so batch positions stay
        # aligned; an empty hypothesis scores as garbage instead of shifting rows
        segs = [Segment(0.0, 1.0, t if t else " ") for t in texts]
        out = translate(segs, source, "en", device=device)
        return [s.text for s in out]

    hyps_cascade = translate_batch(asr_texts)
    hyps_oracle = translate_batch(gold_texts)

    # Normalize both sides before WER/CER: FLEURS references are lowercased
    # and unpunctuated, Whisper output is cased and punctuated — scoring raw
    # text counts cosmetic differences as errors (Whisper paper does the same)
    norm = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    refs_n = [norm(r) for r in refs_wer]
    hyps_n = [norm(h) for h in hyps_wer]
    wer = jiwer.wer(refs_n, hyps_n)
    cer = jiwer.cer(refs_n, hyps_n)

    bleu_cascade = sacrebleu.corpus_bleu(hyps_cascade, [refs_bleu]).score
    bleu_oracle = sacrebleu.corpus_bleu(hyps_oracle, [refs_bleu]).score

    return {
        "pair": f"{source}->en", "n": len(refs_wer),
        "wer": round(wer * 100, 1),
        "cer": round(cer * 100, 1),
        "bleu_cascade": round(bleu_cascade, 1),
        "bleu_oracle": round(bleu_oracle, 1),
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="es,fr,de,hi,ja")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--whisper", default="small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("eval_results.md"))
    args = p.parse_args()

    rows = []
    for source in args.pairs.split(","):
        print(f"=== {source}->en (n={args.n}, whisper={args.whisper}) ===")
        row = eval_pair(source.strip(), args.n, args.whisper, args.device)
        print(row)
        rows.append(row)

    lines = [
        f"# Parlecho benchmark — FLEURS test, whisper-{args.whisper}, NLLB-600M, n={args.n}/pair",
        "",
        "| Pair | n | WER % | CER % | BLEU (cascade) | BLEU (oracle MT) | Runtime (min) |",
        "|------|---|-------|-------|----------------|------------------|---------------|",
    ]
    for r in rows:
        lines.append(f"| {r['pair']} | {r['n']} | {r['wer']} | {r['cer']} | "
                     f"{r['bleu_cascade']} | {r['bleu_oracle']} | {r['minutes']} |")
    lines.append("")
    lines.append("Note: for ja (unsegmented script), word-level WER is not "
                 "meaningful — read CER for ja and hi.")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()