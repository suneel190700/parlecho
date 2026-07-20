"""Streaming step 3: orchestrator. Runs the VAD chunker (real-time paced)
and the resident-model worker concurrently; measures per-utterance lag
(stream-time end of speech -> dubbed chunk ready) and writes the latency
report. Model loading happens before the stream starts, as in a live system.

python -m parlecho.streaming.orchestrator input.mp3 --from es --to en
"""
import argparse
import queue
import statistics
import threading
import time
from pathlib import Path

from parlecho.streaming.chunker import stream_utterances
from parlecho.streaming.worker import StreamWorker


def run(input_path: Path, source: str, target: str, workdir: Path,
        whisper_model: str, device: str, realtime: bool,
        out_report: Path) -> None:
    worker = StreamWorker(source, target, workdir, whisper_model=whisper_model,
                          device=device)
    print(f"models resident in {worker.load_s:.1f}s — starting stream")

    q: queue.Queue = queue.Queue()
    rows: list[dict] = []
    t0_holder: dict[str, float] = {}

    def consume():
        while True:
            item = q.get()
            if item is None:
                return
            utt, enq_abs = item
            # reconstruct the chunker's absolute t0 from the first arrival
            t0_holder.setdefault("t0", enq_abs - utt.detected_wall)
            t0 = t0_holder["t0"]

            queued_at = enq_abs - t0
            chunk = worker.process(utt)
            ready_at = time.perf_counter() - t0

            rows.append({
                "end_s": round(utt.end_s, 2),
                "audio_s": round(utt.end_s - utt.start_s, 2),
                "lag": round(ready_at - utt.end_s, 2),
                "queue_wait": round(queued_at - utt.detected_wall +
                                    max(0.0, ready_at - queued_at
                                        - sum(chunk.timings.values())), 2),
                **{k: round(v, 2) for k, v in chunk.timings.items()},
                "text": chunk.tgt_text[:50],
            })
            print(f"[{utt.end_s:6.2f}] lag {rows[-1]['lag']:5.2f}s  "
                  f"{rows[-1]['text']}")

    t = threading.Thread(target=consume, daemon=True)
    t.start()

    for utt in stream_utterances(input_path, realtime=realtime):
        q.put((utt, time.perf_counter()))
    q.put(None)
    t.join()

    lags = sorted(r["lag"] for r in rows)
    p50 = lags[len(lags) // 2]
    p95 = lags[min(len(lags) - 1, int(len(lags) * 0.95))]

    lines = [
        f"# Parlecho streaming latency — whisper-{whisper_model} + CT2 int8 NLLB "
        f"+ XTTS-v2, greedy decoding",
        "",
        f"Input: {input_path.name}, {source}->{target}, device={device}, "
        f"{'real-time paced' if realtime else 'unpaced'}. "
        "Lag = stream-time end of speech to dubbed chunk ready; includes "
        "~0.42s VAD end-of-speech detection. Models resident before stream "
        f"start (load: {worker.load_s:.1f}s, excluded).",
        "",
        f"**n={len(rows)} utterances — p50 lag {p50:.2f}s, p95 lag {p95:.2f}s**",
        "",
        "| End (s) | Audio (s) | Lag (s) | Queue (s) | ASR | Translate | TTS | Output |",
        "|---------|-----------|---------|-----------|-----|-----------|-----|--------|",
    ]
    for r in rows:
        lines.append(f"| {r['end_s']} | {r['audio_s']} | {r['lag']} | "
                     f"{r['queue_wait']} | {r.get('asr','-')} | "
                     f"{r.get('translate','-')} | {r.get('tts','-')} | "
                     f"{r['text']} |")
    lines.append("")
    lines.append(f"Streaming v1 scope: no source separation, single rolling "
                 "clone reference, greedy decoding.")
    out_report.write_text("\n".join(lines) + "\n")
    print(f"\np50 {p50:.2f}s  p95 {p95:.2f}s  ->  wrote {out_report}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--to", dest="target", required=True)
    p.add_argument("--whisper", default="small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workdir", type=Path, default=Path("outputs/stream"))
    p.add_argument("--no-realtime", action="store_true")
    p.add_argument("--out", type=Path, default=Path("eval_latency.md"))
    args = p.parse_args()
    run(args.input, args.source, args.target, args.workdir,
        args.whisper, args.device, not args.no_realtime, args.out)


def _unused():
    pass


if __name__ == "__main__":
    main()