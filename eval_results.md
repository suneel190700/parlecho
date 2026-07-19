# Parlecho benchmark — FLEURS test, whisper-small, NLLB-600M, n=100/pair

| Pair | n | WER % | CER % | BLEU (cascade) | BLEU (oracle MT) | Runtime (min) |
|------|---|-------|-------|----------------|------------------|---------------|
| es->en | 100 | 5.7 | 2.0 | 26.8 | 32.1 | 4.1 |
| fr->en | 100 | 14.1 | 4.9 | 32.5 | 43.0 | 4.4 |
| de->en | 100 | 8.5 | 2.6 | 34.4 | 41.5 | 4.6 |
| hi->en | 100 | 54.1 | 22.6 | 12.3 | 35.9 | 10.3 |
| ja->en | 100 | 98.1 | 15.4 | 15.3 | 24.9 | 5.8 |

Note: for ja (unsegmented script), word-level WER is not meaningful — read CER for ja and hi.
