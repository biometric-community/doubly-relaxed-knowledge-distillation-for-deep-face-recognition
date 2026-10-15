"""ResNet teacher / student for face identity (shared Softmax head dim)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceResNet(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            feat = 2048
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            feat = 512
        else:
            raise ValueError(backbone)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(feat, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor):
        emb = self.embed(x)
        logits = self.classifier(emb)
        return logits, emb
