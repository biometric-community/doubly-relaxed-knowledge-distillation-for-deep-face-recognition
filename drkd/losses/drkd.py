"""DR-KD: Z-score logits + cosine alignment (doubly relaxed vs KL/softmax)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def zscore_logits(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample Z-score over the class dimension (paper Fig. 2)."""
    mu = z.mean(dim=1, keepdim=True)
    std = z.std(dim=1, keepdim=True).clamp_min(eps)
    return (z - mu) / std


def drkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_kd: float = 1.0,
    lambda_ce: float = 1.0,
    target_std: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Doubly relaxed distillation:
      1) Z-score both logit vectors (relax softmax / variance coupling)
      2) Cosine similarity alignment (relax linear relationship of KL)
    plus student CE on identity.
    """
    ce = F.cross_entropy(student_logits, labels)
    zs = zscore_logits(student_logits) * target_std
    zt = zscore_logits(teacher_logits.detach()) * target_std
    # maximize cosine → minimize 1 - cos
    cos = F.cosine_similarity(zs, zt, dim=1).mean()
    kd = 1.0 - cos
    total = lambda_ce * ce + lambda_kd * kd
    return total, {
        "total": float(total.detach()),
        "ce": float(ce.detach()),
        "kd": float(kd.detach()),
        "cos": float(cos.detach()),
    }
