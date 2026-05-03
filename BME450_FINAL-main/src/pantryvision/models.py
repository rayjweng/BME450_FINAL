from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if architecture == "resnet18":
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        else:
            for parameter in model.features[-1].parameters():
                parameter.requires_grad = True
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

    return model
