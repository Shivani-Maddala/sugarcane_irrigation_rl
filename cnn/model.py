"""
Sugarcane leaf-condition CNN: ResNet18 backbone (ImageNet-pretrained),
fine-tuned with a new classification head for our 3 condition levels.

Why freeze early layers: ResNet18's early convolutional layers learn very
general features (edges, colour gradients, textures) that are useful for
almost any image task, sugarcane leaves included. Freezing them means we
only train the later, more task-specific layers plus our new head -- this
is what makes transfer learning effective with a few thousand images
instead of the millions a from-scratch CNN would need.
"""

import torch.nn as nn
from torchvision import models

from . import config


def build_model(num_classes=None, freeze_backbone=True, pretrained=True):
    """
    pretrained=True (default, use for real training): downloads ImageNet
    weights -- requires an environment with unrestricted internet access.

    pretrained=False: random initialization. ONLY for offline code
    smoke-testing (e.g. in network-restricted sandboxes) -- never use this
    for actual training, since it discards the entire point of transfer
    learning.
    """
    num_classes = num_classes or config.NUM_CLASSES

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer (originally 1000 ImageNet
    # classes) with a small head for our 3 condition levels. This new layer
    # is always trainable, even when the backbone is frozen.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),          # reduces overfitting given our small dataset
        nn.Linear(128, num_classes),
    )

    return model
