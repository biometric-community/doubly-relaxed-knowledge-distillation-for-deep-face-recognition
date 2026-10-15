"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drkd.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(train_s, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    th = train_s.get("teacher_history", [])
    sh = train_s.get("student_history", [])
    if th:
        ax.plot(
            [h["epoch"] for h in th],
            [h["loss"] for h in th],
            color=MULTI_SERIES_COLORS[0],
            linewidth=LINEWIDTH_MAIN,
            label="teacher CE",
        )
    if sh:
        ax.plot(
            [h["epoch"] for h in sh],
            [h["total"] for h in sh],
            color=MULTI_SERIES_COLORS[1],
            linewidth=LINEWIDTH_MAIN,
            label="student DR-KD",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DR-KD training")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_bench(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = [k for k in ("lfw", "calfw", "cplfw") if k in eval_res]
    acc = [eval_res[n]["accuracy"] * 100 for n in names]
    x = np.arange(len(names))
    ax.bar(x, acc, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Student verification accuracy")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        f"# REPORT: Doubly Relaxed KD (DR-KD) — {protocol}",
        "",
        f"**Protocol:** `{protocol}`  ",
        f"**Teacher:** `{cfg['model'].get('teacher')}` → **Student:** `{cfg['model'].get('student')}`  ",
        "",
        "## Summary",
        "",
        "| Benchmark | Accuracy | EER | Pairs |",
        "|-----------|----------|-----|-------|",
    ]
    for name in ("lfw", "calfw", "cplfw"):
        if name not in eval_res:
            continue
        r = eval_res[name]
        lines.append(
            f"| {name.upper()} | {r['accuracy']*100:.2f}% | {r['eer']*100:.2f}% | {r['n_pairs']} |"
        )
    lines += [
        "",
        "## Training",
        "",
        f"- Teacher epochs: {train_s.get('teacher_epochs')}",
        f"- Student epochs: {train_s.get('student_epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed: {train_s.get('elapsed_sec', 0):.2f} s",
        f"- Checkpoint: `{train_s.get('checkpoint')}`",
        "",
        "## Figures",
        "",
        "### `train_loss.svg`",
        "",
        "![train_loss](figures/train_loss.svg)",
        "",
        "### `bench_acc.svg`",
        "",
        "![bench_acc](figures/bench_acc.svg)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/smoke.sh",
        "bash scripts/train_full.sh",
        "```",
        "",
        "## Notes",
        "",
        "- DR-KD: Z-score logits + cosine alignment + student CE (see DEVIATIONS.md).",
        "- Trained on CelebA; CASIA-WebFace / WebFace omitted.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval_results.json").read_text())
    plot_train(train_s, fig_dir / "train_loss.svg")
    plot_bench(eval_res, fig_dir / "bench_acc.svg")
    tracked = _ROOT / "figures"
    tracked.mkdir(exist_ok=True)
    for name in ("train_loss.svg", "bench_acc.svg"):
        src = fig_dir / name
        if src.exists():
            (tracked / name).write_bytes(src.read_bytes())
    write_report(cfg, train_s, eval_res, _ROOT / "REPORT.md")
    print(f"Wrote {_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
