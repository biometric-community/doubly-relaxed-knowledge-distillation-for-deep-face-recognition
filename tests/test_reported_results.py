"""REPORT.md, the committed output of the full run."""

from __future__ import annotations

import re


def accuracy(report: str, benchmark: str) -> float:
    match = re.search(rf"\| {benchmark} \| ([\d.]+)% \|", report)
    assert match, f"{benchmark} row missing from REPORT.md"
    return float(match.group(1))


def test_reported_networks_are_the_paper_pair(report, targets):
    assert f"`{targets['teacher']}`" in report and f"`{targets['student']}`" in report


def test_distilled_student_is_a_working_recogniser(report, targets):
    """Abstract: DR-KD gives a student competitive with the state of the art."""
    assert accuracy(report, "LFW") >= targets["clean_lfw_accuracy_floor"]
