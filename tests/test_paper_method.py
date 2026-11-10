"""DR-KD loss properties stated in the abstract, on synthetic logits."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from drkd.eval import best_acc_eer
from drkd.losses.drkd import drkd_loss, zscore_logits


def logits(seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(8, 20), torch.randn(8, 20), torch.randint(0, 20, (8,))


def test_zscore_standardises_each_sample():
    z = zscore_logits(torch.randn(4, 30) * 5 + 3)
    assert torch.allclose(z.mean(dim=1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(z.std(dim=1), torch.ones(4), atol=1e-5)


def test_transfer_term_is_one_minus_pearson_correlation_of_raw_logits():
    """Abstract: cosine similarity of Z-scored logits equals the Pearson correlation."""
    s, t, y = logits()
    _, parts = drkd_loss(s, t, y, lambda_ce=0.0)
    pearson = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(s.numpy(), t.numpy())])
    assert parts["kd"] == pytest.approx(1.0 - pearson, abs=1e-5)


def test_variance_constraint_is_relaxed():
    """Abstract: the student does not have to match the teacher's logit scale or offset."""
    s, t, y = logits(1)
    base = drkd_loss(s, t, y, lambda_ce=0.0)[1]["kd"]
    scaled = drkd_loss(3.0 * s + 2.0, t, y, lambda_ce=0.0)[1]["kd"]
    assert scaled == pytest.approx(base, abs=1e-5)


def test_kd_term_is_zero_for_a_matching_student():
    s, t, y = logits(2)
    assert drkd_loss(t.clone(), t, y, lambda_ce=0.0)[1]["kd"] == pytest.approx(0.0, abs=1e-5)


def test_teacher_gets_no_gradient_and_student_ce_is_included():
    s, t, y = logits(3)
    s.requires_grad_(True)
    t.requires_grad_(True)
    total, parts = drkd_loss(s, t, y, lambda_ce=1.0, lambda_kd=1.0)
    total.backward()
    assert t.grad is None
    assert parts["ce"] == pytest.approx(F.cross_entropy(s.detach(), y).item(), rel=1e-5)


def test_teacher_and_student_networks(default_cfg, targets):
    assert default_cfg["model"]["teacher"] == targets["teacher"]
    assert default_cfg["model"]["student"] == targets["student"]


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
