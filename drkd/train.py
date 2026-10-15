"""Train teacher (CE) then student with DR-KD on CelebA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drkd.data.celeba import CelebAIdDataset, resolve_celeba_root
from drkd.losses.drkd import drkd_loss
from drkd.models.fr_net import FaceResNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _loader(ds, cfg, device, train: bool):
    return DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=train,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=train,
    )


def _train_ce(model, loader, device, epochs, lr, desc):
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    hist = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        pbar = tqdm(loader, desc=f"{desc} {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        mean = float(np.mean(losses))
        hist.append({"epoch": epoch, "loss": mean})
        print(f"[{desc}] epoch={epoch} loss={mean:.4f}")
    return hist


def _train_student_drkd(student, teacher, loader, device, cfg):
    epochs = int(cfg["train"].get("student_epochs", 10))
    opt = torch.optim.SGD(
        student.parameters(),
        lr=float(cfg["train"].get("student_lr", 0.05)),
        momentum=0.9,
        weight_decay=5e-4,
    )
    teacher.eval()
    hist = []
    for epoch in range(1, epochs + 1):
        parts = []
        student.train()
        pbar = tqdm(loader, desc=f"student {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits, _ = teacher(images)
            s_logits, _ = student(images)
            loss, d = drkd_loss(
                s_logits,
                t_logits,
                labels,
                lambda_kd=float(cfg["train"].get("lambda_kd", 1.0)),
                lambda_ce=float(cfg["train"].get("lambda_ce", 1.0)),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            parts.append(d)
            pbar.set_postfix(loss=f"{d['total']:.3f}")
        mean = {k: float(np.mean([x[k] for x in parts])) for k in parts[0]}
        mean["epoch"] = epoch
        hist.append(mean)
        print(
            f"[student] epoch={epoch} total={mean['total']:.4f} "
            f"ce={mean['ce']:.4f} kd={mean['kd']:.4f} cos={mean['cos']:.4f}"
        )
    return hist


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    train_ds = CelebAIdDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = _loader(train_ds, cfg, device, train=True)

    teacher = FaceResNet(
        train_ds.num_classes,
        backbone=str(cfg["model"].get("teacher", "resnet50")),
        pretrained=bool(cfg["model"].get("pretrained", True)),
    ).to(device)
    student = FaceResNet(
        train_ds.num_classes,
        backbone=str(cfg["model"].get("student", "resnet18")),
        pretrained=bool(cfg["model"].get("pretrained", True)),
    ).to(device)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    teacher_hist = _train_ce(
        teacher,
        loader,
        device,
        epochs=int(cfg["train"].get("teacher_epochs", 6)),
        lr=float(cfg["train"].get("teacher_lr", 0.05)),
        desc="teacher",
    )
    student_hist = _train_student_drkd(student, teacher, loader, device, cfg)
    elapsed = time.time() - t0

    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "num_classes": train_ds.num_classes,
            "config": cfg,
        },
        ckpt_dir / "last.pt",
    )
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "teacher_epochs": int(cfg["train"].get("teacher_epochs", 6)),
        "student_epochs": int(cfg["train"].get("student_epochs", 10)),
        "elapsed_sec": elapsed,
        "teacher_history": teacher_hist,
        "student_history": student_hist,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
