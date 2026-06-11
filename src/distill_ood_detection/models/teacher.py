"""Teacher model loading from Hugging Face."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

from distill_ood_detection.config import TeacherConfig


class CifarResNet18(ResNet):
    """ResNet-18 variant with the CIFAR stem used by the HF checkpoint."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.maxpool = nn.Identity()


class TeacherFeatureExtractor(nn.Module):
    """Return teacher logits and intermediate features from one forward pass."""

    def __init__(self, teacher: nn.Module, feature_layer: str) -> None:
        super().__init__()
        self.teacher = teacher
        self.feature_layer = feature_layer
        self._features: torch.Tensor | None = None
        try:
            layer = teacher.get_submodule(feature_layer)
        except AttributeError as error:
            msg = f"Teacher does not have feature layer: {feature_layer}"
            raise ValueError(msg) from error
        self._hook = layer.register_forward_hook(self._capture_features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(teacher_logits, features)`` for a batch of images."""

        self._features = None
        logits = self.teacher(images)
        if self._features is None:
            raise RuntimeError(f"Feature hook did not run for layer: {self.feature_layer}")
        return logits, self._features

    def close(self) -> None:
        """Remove the feature hook."""

        self._hook.remove()

    def _capture_features(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        self._features = output


class ResNetFeatureForwarder(nn.Module):
    """Run a ResNet teacher to and from a named residual feature layer."""

    _feature_layers = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, teacher: nn.Module, feature_layer: str) -> None:
        super().__init__()
        if feature_layer not in self._feature_layers:
            supported = ", ".join(self._feature_layers)
            raise ValueError(
                f"Perturbation feature_layer must be one of {supported}; "
                f"got {feature_layer!r}"
            )
        for name in (
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "avgpool",
            "fc",
            *self._feature_layers,
        ):
            if not hasattr(teacher, name):
                raise ValueError(f"Teacher is missing ResNet submodule: {name}")
        self.teacher = teacher
        self.feature_layer = feature_layer

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(teacher_logits, features)`` for the configured layer."""

        features = self.forward_to_features(images)
        logits = self.forward_from_features(features)
        return logits, features

    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return teacher activations at the configured feature layer."""

        x = self.teacher.conv1(images)
        x = self.teacher.bn1(x)
        x = self.teacher.relu(x)
        x = self.teacher.maxpool(x)
        for layer_name in self._feature_layers:
            x = getattr(self.teacher, layer_name)(x)
            if layer_name == self.feature_layer:
                return x
        raise RuntimeError(f"Feature layer was not reached: {self.feature_layer}")

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Continue the teacher forward pass from configured-layer features."""

        x = features
        start_index = self._feature_layers.index(self.feature_layer) + 1
        for layer_name in self._feature_layers[start_index:]:
            x = getattr(self.teacher, layer_name)(x)
        x = self.teacher.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.teacher.fc(x)


def load_teacher(config: TeacherConfig, device: torch.device) -> nn.Module:
    """Load the pretrained ResNet-18 CIFAR teacher from Hugging Face."""

    model = CifarResNet18(num_classes=config.num_classes)
    state_dict = _download_state_dict(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _download_state_dict(config: TeacherConfig) -> dict[str, torch.Tensor]:
    try:
        path = hf_hub_download(
            repo_id=config.hf_model_id,
            filename="model.safetensors",
            revision=config.revision,
        )
        return load_file(path)
    except EntryNotFoundError:
        path = hf_hub_download(
            repo_id=config.hf_model_id,
            filename="pytorch_model.bin",
            revision=config.revision,
        )
        return torch.load(path, map_location="cpu", weights_only=True)
