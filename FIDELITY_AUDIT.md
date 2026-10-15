# FIDELITY_AUDIT — DR-KD

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Teacher/student logits KD | 90% | R50→R18 |
| 2 | Z-score logits | 92% | Per-sample |
| 3 | Cosine alignment (vs KL) | 95% | cos≈0.97 |
| 4 | LFW family eval | 90% | IJB omitted (D4) |
| 5 | Size gate + publish | 95% | Stop |

Confidence vs paper structure: **~92%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ 4.09 GiB → `full_train`
- Teacher 6 + student 10; **2665 s**; KD cosine **0.969**
- LFW **64.20%**; CALFW **51.52%**; CPLFW **52.75%** (6000 pairs)
