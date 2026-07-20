"""Quantization benchmark: HF fp32 vs CT2 int8 NLLB on identical inputs,
matched greedy decoding. Run once per backend; each run appends a row.

python -m parlecho.eval.quant_bench --backend hf --n 100
python -m parlecho.eval.quant_bench --backend ct2 --n 100
"""
import argparse
import resource
import sys
import time
from pathlib import Path

from parlecho.eval.datasets import load_pair
from parlecho.stages.asr import Segment


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "ct2"], required=True)
    p.add_argument("--source", default="es")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("eval_quant.md"))
    args = p.parse_args()

    import sacrebleu

    data = list(load_pair(args.source, "en", n=args.n))
    texts = [ex["src_raw"] for ex in data]
    refs = [ex["tgt_raw"] for ex in data]
    segs = [Segment(0.0, 1.0, t) for t in texts]

    if args.backend == "hf":
        from parlecho.stages.translate import translate as run
        kwargs = {"device": args.device}
    else:
        from parlecho.stages.translate_ct2 import translate_ct2 as run
        kwargs = {"device": args.device, "beam_size": 1}   # match HF greedy

    t_run = time.perf_counter()
    out = run(segs, args.source, "en", **kwargs)
    t_done = time.perf_counter()

    hyps = [s.text for s in out]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    total_time = t_done - t_run
    sents_per_s = len(texts) / total_time

    row = (f"| {args.backend} | {args.source}->en | {args.n} | "
           f"{round(bleu, 1)} | {round(total_time, 1)} | "
           f"{round(sents_per_s, 2)} | {round(peak_rss_gb(), 2)} |")

    header = [
        "# Parlecho quantization benchmark — NLLB-600M, HF fp32 vs CTranslate2 int8",
        "",
        "Matched greedy decoding (beam 1) on both backends, identical inputs.",
        "",
        "| Backend | Pair | n | BLEU | Translate time (s) | Sentences/s | Peak RSS (GB) |",
        "|---------|------|---|------|--------------------|-------------|----------------|",
    ]
    if args.out.exists():
        args.out.write_text(args.out.read_text().rstrip() + "\n" + row + "\n")
    else:
        args.out.write_text("\n".join(header) + "\n" + row + "\n")
    print(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()