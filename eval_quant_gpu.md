# Parlecho quantization benchmark — NLLB-600M, HF fp32 vs CTranslate2 int8

Matched greedy decoding (beam 1) on both backends, identical inputs.
Peak VRAM sampled at device level via NVML (0.0 = no NVIDIA GPU).

| Backend | Device | Pair | n | BLEU | Translate time (s) | Sentences/s | Peak RSS (GB) | Peak VRAM (GB) |
|---------|--------|------|---|------|--------------------|-------------|----------------|----------------|
| hf | cuda | es->en | 100 | 32.1 | 17.8 | 5.61 | 4.12 | 3.72 |
| ct2 | cuda | es->en | 100 | 32.7 | 4.5 | 22.4 | 1.87 | 2.28 |
