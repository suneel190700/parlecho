# Parlecho benchmark — FLEURS test, whisper-large-v3, NLLB-600M, n=100/pair

| Pair | n | WER % | CER % | BLEU (cascade) | BLEU (oracle MT) | Runtime (min) |
|------|---|-------|-------|----------------|------------------|---------------|
| es->en | 100 | 2.9 | 1.2 | 29.7 | 32.1 | 25.7 |
| fr->en | 100 | 5.8 | 1.8 | 38.1 | 43.0 | 26.8 |
| de->en | 100 | 4.4 | 1.7 | 37.0 | 41.5 | 26.7 |
| hi->en | 100 | 26.4 | 9.1 | 22.7 | 35.9 | 45.0 |
| ja->en | 100 | 91.6 | 7.7 | 20.2 | 24.9 | 21.3 |

Note: for ja (unsegmented script), word-level WER is not meaningful — read CER for ja and hi.
