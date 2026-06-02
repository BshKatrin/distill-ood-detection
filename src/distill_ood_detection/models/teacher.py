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
    """ResNet-18 variant with the CIFAR-10 stem used by the HF checkpoint."""

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


def load_teacher(config: TeacherConfig, device: torch.device) -> nn.Module:
    """Load the pretrained ResNet-18 CIFAR-10 teacher from Hugging Face."""

    model = CifarResNet18(num_classes=10)
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
