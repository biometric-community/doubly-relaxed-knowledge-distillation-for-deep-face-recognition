# Doubly Relaxed Knowledge Distillation for Deep Face Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of **DR-KD**: Z-score standardized logits + cosine alignment for logits-based face recognition distillation.

**Repository:** https://github.com/biometric-community/doubly-relaxed-knowledge-distillation-for-deep-face-recognition

**Paper:** [10.1109/TBIOM.2026.3671881](https://doi.org/10.1109/TBIOM.2026.3671881)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
