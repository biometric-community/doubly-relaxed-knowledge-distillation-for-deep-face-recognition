# Implementation Plan — DR-KD

1. Teacher ResNet-50 CE on CelebA; student ResNet-18.
2. DR-KD loss: Z-score logits + cosine alignment + student CE.
3. Eval LFW / CALFW / CPLFW; smoke → size gate → full → publish.
