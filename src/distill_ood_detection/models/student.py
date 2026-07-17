"""Student model definitions."""

from __future__ import annotations

from functools import reduce
from operator import mul

import torch
from torch import nn
from torchvision.ops import MLP

from distill_ood_detection.config import StudentConfig


class LinearStudent(nn.Module):
    """A single linear classifier over flattened image pixels."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        input_dim = reduce(mul, input_shape, 1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return student logits for a batch of images."""

        return self.classifier(torch.flatten(images, start_dim=1))


class MLPStudent(nn.Module):
    """A configurable multilayer perceptron over flattened image pixels."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        input_dim = reduce(mul, input_shape, 1)
        self.classifier = MLP(input_dim, [*hidden_channels, num_classes])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return student logits for a batch of images."""

        return self.classifier(torch.flatten(images, start_dim=1))


class BottleneckAutoencoderStudent(nn.Module):
    """Non-residual MLP autoencoder for pooled feature coordinates."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(input_shape) != 1:
            raise ValueError("autoencoder input_shape must contain one feature dimension")
        dimensions = (input_shape[0], *hidden_channels, input_shape[0])
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend([nn.Linear(input_dim, output_dim), nn.GELU()])
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.autoencoder = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct flattened feature coordinates."""

        return self.autoencoder(torch.flatten(features, start_dim=1))


class ResidualConvBlock(nn.Module):
    """Small residual convolutional block for feature-map reconstruction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return residual block outputs."""

        return self.activation(features + self.block(features))


class FeatureReconstructorStudent(nn.Module):
    """Residual CNN that reconstructs teacher feature maps."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                "feature_reconstructor input_shape must contain channels, height, and width"
            )
        input_channels = input_shape[0]
        hidden = hidden_channels[0] if hidden_channels else max(1, input_channels // 2)
        block_count = hidden_channels[1] if len(hidden_channels) > 1 else 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualConvBlock(hidden) for _ in range(block_count))
        )
        self.decoder = nn.Conv2d(hidden, input_channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return reconstructed feature maps."""

        residual = self.decoder(self.blocks(self.encoder(features)))
        return features + residual


class SpatialTokenPredictorStudent(nn.Module):
    """Transformer predictor from visible feature tokens to target feature tokens."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                "spatial_token_predictor input_shape must contain channels, "
                "height, and width"
            )
        input_channels, height, width = input_shape
        hidden_dim = hidden_channels[0] if hidden_channels else 256
        layer_count = hidden_channels[1] if len(hidden_channels) > 1 else 2
        head_count = hidden_channels[2] if len(hidden_channels) > 2 else 4
        if hidden_dim % head_count != 0:
            raise ValueError("spatial_token_predictor hidden width must divide heads")
        token_count = height * width
        self.token_projection = nn.Linear(input_channels, hidden_dim)
        self.position_embedding = nn.Embedding(token_count, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=head_count,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layer_count,
        )
        self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=head_count,
            dropout=0.0,
            batch_first=True,
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_channels),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict target feature tokens from visible feature tokens."""

        visible_tokens = batch["visible_tokens"]
        visible_positions = batch["visible_positions"].long()
        visible_padding_mask = batch["visible_padding_mask"].bool()
        target_positions = batch["target_positions"].long()
        visible = (
            self.token_projection(visible_tokens)
            + self.position_embedding(visible_positions)
        )
        encoded_visible = self.context_encoder(
            visible,
            src_key_padding_mask=visible_padding_mask,
        )
        queries = (
            self.target_token.expand(target_positions.shape[0], target_positions.shape[1], -1)
            + self.position_embedding(target_positions)
        )
        attended_targets, _attention = self.cross_attention(
            query=queries,
            key=encoded_visible,
            value=encoded_visible,
            key_padding_mask=visible_padding_mask,
            need_weights=False,
        )
        return self.predictor(attended_targets)


def build_student(config: StudentConfig) -> nn.Module:
    """Create the configured student model."""

    if config.kind == "linear":
        return LinearStudent(config.input_shape, config.num_classes)
    if config.kind == "mlp":
        return MLPStudent(
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            hidden_channels=config.hidden_channels,
        )
    if config.kind == "feature_reconstructor":
        return FeatureReconstructorStudent(
            input_shape=config.input_shape,
            hidden_channels=config.hidden_channels,
        )
    if config.kind == "spatial_token_predictor":
        return SpatialTokenPredictorStudent(
            input_shape=config.input_shape,
            hidden_channels=config.hidden_channels,
        )
    if config.kind == "autoencoder":
        return BottleneckAutoencoderStudent(
            input_shape=config.input_shape,
            hidden_channels=config.hidden_channels,
        )
    raise ValueError(f"Unsupported student kind: {config.kind}")
