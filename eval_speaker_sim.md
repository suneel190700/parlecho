# Parlecho speaker similarity — ECAPA-TDNN cosine, XTTS-v2 clones, n=15/pair

| Pair | n | Same-speaker sim | Std | Diff-speaker baseline | Runtime (min) |
|------|---|------------------|-----|----------------------|---------------|
| es->en | 15 | 0.46 | 0.076 | 0.048 | 1.9 |
| fr->en | 15 | 0.531 | 0.072 | 0.219 | 1.8 |
| de->en | 15 | 0.58 | 0.074 | 0.267 | 2.0 |
| hi->en | 15 | 0.447 | 0.076 | 0.081 | 1.9 |
| ja->en | 15 | 0.439 | 0.085 | 0.195 | 2.2 |

Similarity is cosine between ECAPA embeddings of the original recording and English XTTS output cloned from it. Diff-speaker baseline uses mismatched clone pairs.
