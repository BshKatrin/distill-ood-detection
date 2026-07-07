"""Teacher model loading from Hugging Face."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

from distill_ood_detection.config import TeacherConfig
from distill_ood_detection.utils import get_hf_token


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


class CifarResNet50(ResNet):
    """ResNet-50 variant with the CIFAR stem used by the HF checkpoint."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
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


class HuggingFaceImageClassifier(nn.Module):
    """Thin wrapper returning logits from a Hugging Face image classifier."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return classifier logits."""

        outputs = self.model(images)
        return outputs.logits

    def forward_with_hidden_states(self, images: torch.Tensor) -> object:
        """Return Hugging Face outputs including hidden states."""

        return self.model(images, output_hidden_states=True)


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


class VitClsFeatureForwarder(nn.Module):
    """Return CLS-token features from a configured ViT hidden layer."""

    def __init__(self, teacher: HuggingFaceImageClassifier, feature_layer: str) -> None:
        super().__init__()
        layer_index = _vit_layer_index(feature_layer)
        if layer_index < 1:
            raise ValueError("ViT feature_layer must refer to encoder layer 1 or later")
        self.teacher = teacher
        self.feature_layer = feature_layer
        self.layer_index = layer_index

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(teacher_logits, cls_features)`` for the configured layer."""

        outputs = self.teacher.forward_with_hidden_states(images)
        return outputs.logits, self._cls_from_outputs(outputs)

    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return CLS-token activations at the configured hidden layer."""

        outputs = self.teacher.forward_with_hidden_states(images)
        return self._cls_from_outputs(outputs)

    def _cls_from_outputs(self, outputs: object) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("ViT teacher did not return hidden states")
        if self.layer_index >= len(hidden_states):
            max_layer = len(hidden_states) - 1
            raise ValueError(
                f"ViT feature_layer {self.feature_layer!r} is out of range; "
                f"maximum encoder layer is layer{max_layer}"
            )
        return hidden_states[self.layer_index][:, 0, :]


def build_feature_forwarder(teacher: nn.Module, feature_layer: str) -> nn.Module:
    """Create a feature forwarder matching the teacher architecture."""

    if isinstance(teacher, HuggingFaceImageClassifier):
        return VitClsFeatureForwarder(teacher, feature_layer)
    return ResNetFeatureForwarder(teacher, feature_layer)


def load_teacher(config: TeacherConfig, device: torch.device) -> nn.Module:
    """Load a pretrained CIFAR teacher from Hugging Face."""

    model = build_teacher_model(config)
    architecture = _infer_architecture_from_hf_model_id(config.hf_model_id)
    if architecture in {"resnet18", "resnet50"}:
        state_dict = _download_state_dict(config)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def build_teacher_model(config: TeacherConfig) -> nn.Module:
    """Build one teacher model inferred from the Hugging Face model id."""

    architecture = _infer_architecture_from_hf_model_id(config.hf_model_id)
    if architecture == "resnet18":
        return CifarResNet18(num_classes=config.num_classes)
    if architecture == "resnet50":
        return CifarResNet50(num_classes=config.num_classes)
    if architecture == "vit":
        try:
            from transformers import AutoModelForImageClassification
        except ImportError as error:
            raise ImportError(
                "ViT teachers require the 'transformers' package in the active uv env"
            ) from error
        token = get_hf_token()
        model = AutoModelForImageClassification.from_pretrained(
            config.hf_model_id,
            revision=config.revision,
            token=token,
        )
        return HuggingFaceImageClassifier(model)
    raise AssertionError(f"Unsupported inferred architecture: {architecture}")


def _download_state_dict(config: TeacherConfig) -> dict[str, torch.Tensor]:
    token = get_hf_token()
    try:
        path = hf_hub_download(
            repo_id=config.hf_model_id,
            filename="model.safetensors",
            revision=config.revision,
            token=token,
        )
        return load_file(path)
    except EntryNotFoundError:
        path = hf_hub_download(
            repo_id=config.hf_model_id,
            filename="pytorch_model.bin",
            revision=config.revision,
            token=token,
        )
        return torch.load(path, map_location="cpu", weights_only=True)


def _infer_architecture_from_hf_model_id(hf_model_id: str) -> str:
    """Infer the teacher architecture from the configured Hugging Face model id."""

    model_id = hf_model_id.lower()
    if "resnet18" in model_id:
        return "resnet18"
    if "resnet50" in model_id:
        return "resnet50"
    if "vit" in model_id or "patch16" in model_id:
        return "vit"
    msg = (
        "Could not infer teacher architecture from hf_model_id. "
        "Expected a model id containing 'resnet18', 'resnet50', or 'vit', got "
        f"{hf_model_id!r}."
    )
    raise ValueError(msg)


def _vit_layer_index(feature_layer: str) -> int:
    if not feature_layer.startswith("layer"):
        raise ValueError(
            f"ViT feature_layer must use names like 'layer3'; got {feature_layer!r}"
        )
    raw_index = feature_layer.removeprefix("layer")
    try:
        return int(raw_index)
    except ValueError as error:
        raise ValueError(
            f"ViT feature_layer must use names like 'layer3'; got {feature_layer!r}"
        ) from error
