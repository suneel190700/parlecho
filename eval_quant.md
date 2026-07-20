# Parlecho quantization benchmark — NLLB-600M, HF fp32 vs CTranslate2 int8

Matched greedy decoding (beam 1) on both backends, identical inputs.

| Backend | Pair | n | BLEU | Translate time (s) | Sentences/s | Peak RSS (GB) |
|---------|------|---|------|--------------------|-------------|----------------|
| hf | es->en | 100 | 32.1 | 36.9 | 2.71 | 3.67 |
| ct2 | es->en | 100 | 32.9 | 13.6 | 7.38 | 2.33 |


Additionally, CT2 int8 at beam_size=4 (13.6s equivalent-class runtime measured
at 32.1s in an earlier unmatched run) completed 4-beam search in less time than
HF fp32 needed for greedy decoding.