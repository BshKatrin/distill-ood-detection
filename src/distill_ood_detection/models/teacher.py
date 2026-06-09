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
