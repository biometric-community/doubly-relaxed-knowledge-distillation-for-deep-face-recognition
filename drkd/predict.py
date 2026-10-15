"""Embed one image with the student."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drkd.models.fr_net import FaceResNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / "last.pt",
        map_location=device,
        weights_only=False,
    )
    model = FaceResNet(
        int(ckpt["num_classes"]),
        backbone=str(cfg["model"].get("student", "resnet18")),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["student"])
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((int(cfg["model"].get("image_size", 112)),) * 2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.embed(x).cpu().numpy()[0].tolist()
    out = {"dim": len(emb), "embed": emb}
    if args.out:
        Path(args.out).write_text(json.dumps(out) + "\n")
    print(json.dumps({"dim": len(emb)}))


if __name__ == "__main__":
    main()
