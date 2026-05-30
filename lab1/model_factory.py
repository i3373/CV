import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    densenet121,
    efficientnet_b0,
)

from models_resnet50_scratch import resnet50_scratch


ARCH_CHOICES = ("resnet50_scratch", "efficientnet_b0_ft", "densenet121_ft")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet50_scratch":
        return resnet50_scratch(num_classes)

    if arch == "efficientnet_b0_ft":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "densenet121_ft":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unknown architecture: {arch}. Choices: {', '.join(ARCH_CHOICES)}")
