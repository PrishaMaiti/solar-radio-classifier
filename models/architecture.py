"""
CNN architecture for solar radio burst classification.
"""

from __future__ import annotations

import torch
from torch import nn


CLASS_NAMES = (
    "type_2",
    "type_3",
    "type_4",
    "type_5",
    "type_6",
    "type_7",
    "type_8",
    "no_burst",
)


class ConvBlock(nn.Module):
    """Convolution, normalization, nonlinearity, and downsampling block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SolarRadioClassifier(nn.Module):
    """
    Convolutional Neural Network for classifying solar radio burst spectrograms.

    The network returns raw logits. Use CrossEntropyLoss during training and
    apply softmax only when class probabilities are needed for reporting or
    inference.
    """

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        in_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64, 128, 256),
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        conv_layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in conv_channels:
            conv_layers.append(ConvBlock(current_channels, out_channels, dropout=dropout / 2))
            current_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        shared_layers: list[nn.Module] = []
        current_dim = current_channels * 4 * 4
        for hidden_dim in hidden_dims:
            shared_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        self.classifier = nn.Sequential(*shared_layers)
        self.solar_head = nn.Linear(current_dim, num_classes)
        self.rfi_head = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor shaped (batch, channels, height, width).

        Returns:
            Dict with solar class logits and binary RFI logits.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return {
            "solar_logits": self.solar_head(x),
            "rfi_logits": self.rfi_head(x).squeeze(1),
        }

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return solar class probabilities for inference/reporting."""
        outputs = self.forward(x)
        return torch.softmax(outputs["solar_logits"], dim=1)

    def predict_rfi_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return RFI-present probabilities for inference/reporting."""
        outputs = self.forward(x)
        return torch.sigmoid(outputs["rfi_logits"])
