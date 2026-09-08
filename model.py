"""Model definitions for text-region classification."""
import torch
import torch.nn as nn
from torchvision import models

DEFAULT_BACKBONE = "efficientnet_b0"


def build_model(
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a classifier from a torchvision backbone.

    Default: MobileNetV3-Large with 4 classes:
      ar_handwritten, ar_printed, en_handwritten, en_printed
    """
    backbone = backbone.lower()

    if backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return net


def freeze_backbone_layers(model: nn.Module, unfreeze_last_n: int = 20) -> None:
    params = list(model.parameters())
    for p in params[:-unfreeze_last_n]:
        p.requires_grad = False
