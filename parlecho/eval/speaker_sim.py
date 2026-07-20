"""Speaker-similarity eval: ECAPA cosine between original FLEURS speaker
audio and XTTS output cloned from that speaker. Usage:
python -m parlecho.eval.speaker_sim --pairs es,fr,de,hi,ja --n 15
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from parlecho.eval.datasets import load_pair

os.environ.setdefault("COQUI_TOS_AGREED", "1")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed(classifier, path: Path) -> np.ndarray:
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)              # mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return classifier.encode_batch(wav).squeeze().detach().cpu().numpy()


def eval_pair(source: str, n: int, workdir: Path, tts, classifier) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    orig_embs, tts_embs = [], []

    t0 = time.perf_counter()
    for i, ex in enumerate(load_pair(source, "en", n=n)):
        ref = workdir / f"{source}_{i:03d}_orig.wav"
        sf.write(str(ref), ex["audio"], ex["sr"])

        out = workdir / f"{source}_{i:03d}_tts.wav"
        # speak the parallel English sentence, cloned from the original speaker;
        # using FLEURS's own English text keeps MT quality out of this metric
        tts.tts_to_file(text=ex["tgt_raw"], speaker_wav=str(ref),
                        language="en", file_path=str(out))

        orig_embs.append(embed(classifier, ref))
        tts_embs.append(embed(classifier, out))
        print(f"  {source}: {i+1}/{n}")

    sims = [cosine(o, t) for o, t in zip(orig_embs, tts_embs)]
    # negative baseline: original vs a clone of a DIFFERENT speaker (shift by 1)
    negs = [cosine(orig_embs[i], tts_embs[(i + 1) % len(tts_embs)])
            for i in range(len(orig_embs))]

    return {
        "pair": f"{source}->en", "n": len(sims),
        "sim_mean": round(float(np.mean(sims)), 3),
        "sim_std": round(float(np.std(sims)), 3),
        "neg_mean": round(float(np.mean(negs)), 3),
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="es,fr,de,hi,ja")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("eval_speaker_sim.md"))
    args = p.parse_args()

    from TTS.api import TTS
    from speechbrain.inference.speaker import EncoderClassifier

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(args.device)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/ecapa",
    )

    rows = []
    for source in args.pairs.split(","):
        print(f"=== {source} (n={args.n}) ===")
        rows.append(eval_pair(source.strip(), args.n,
                              Path("outputs/speaker_sim"), tts, classifier))
        print(rows[-1])

    lines = [
        f"# Parlecho speaker similarity — ECAPA-TDNN cosine, XTTS-v2 clones, n={args.n}/pair",
        "",
        "| Pair | n | Same-speaker sim | Std | Diff-speaker baseline | Runtime (min) |",
        "|------|---|------------------|-----|----------------------|---------------|",
    ]
    for r in rows:
        lines.append(f"| {r['pair']} | {r['n']} | {r['sim_mean']} | {r['sim_std']} | "
                     f"{r['neg_mean']} | {r['minutes']} |")
    lines.append("")
    lines.append("Similarity is cosine between ECAPA embeddings of the original "
                 "recording and English XTTS output cloned from it. "
                 "Diff-speaker baseline uses mismatched clone pairs.")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()