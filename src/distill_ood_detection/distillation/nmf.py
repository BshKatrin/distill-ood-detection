"""Global non-negative concept projection for Feature Denoising."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.decomposition import MiniBatchNMF


@dataclass(frozen=True)
class NmfConceptProjector:
    """Fixed non-negative concept directions fitted from ID activations."""

    components: torch.Tensor
    encoding_max_iter: int = 200
    fit_reconstruction_error: float | None = None
    fit_n_iter: int | None = None

    @property
    def concept_count(self) -> int:
        """Return the number of fitted concept directions."""

        return self.components.shape[0]

    @property
    def channel_count(self) -> int:
        """Return the feature-channel dimension of each concept direction."""

        return self.components.shape[1]

    def to(self, device: torch.device) -> NmfConceptProjector:
        """Move the concept directions to a device."""

        return NmfConceptProjector(
            components=self.components.to(device),
            encoding_max_iter=self.encoding_max_iter,
            fit_reconstruction_error=self.fit_reconstruction_error,
            fit_n_iter=self.fit_n_iter,
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Encode ``NCHW`` activations as non-negative ``NHWK`` concept scores.

        The fixed-basis multiplicative updates minimize the same Frobenius
        residual used to fit the NMF basis, with the concept directions held
        fixed. Spatial rows are ordered by converting NCHW to NHWC before the
        two-dimensional matrix factorization.
        """

        _validate_feature_map(features, self.channel_count)
        minimum = float(features.min())
        if minimum < -1.0e-6:
            raise ValueError(
                "NMF concept projection requires post-ReLU non-negative "
                f"activations; observed minimum {minimum:.6g}"
            )

        batch_size, channels, height, width = features.shape
        values = (
            features.clamp_min(0.0)
            .permute(0, 2, 3, 1)
            .reshape(batch_size * height * width, channels)
        )
        components = self.components.to(device=features.device, dtype=features.dtype)
        numerator = values @ components.T
        component_gram = components @ components.T
        initial_value = torch.sqrt(
            values.mean().clamp_min(0.0) / float(self.concept_count)
        )
        codes = torch.full_like(numerator, initial_value)
        epsilon = torch.finfo(features.dtype).eps
        for _ in range(self.encoding_max_iter):
            denominator = (codes @ component_gram).clamp_min(epsilon)
            codes.mul_(numerator / denominator)
        return codes.reshape(batch_size, height, width, self.concept_count)

    def inverse_transform(self, concept_scores: torch.Tensor) -> torch.Tensor:
        """Reconstruct an NCHW feature map from NHWK concept scores."""

        if concept_scores.ndim != 4:
            raise ValueError("NMF concept scores must have shape (B, H, W, K)")
        if concept_scores.shape[-1] != self.concept_count:
            raise ValueError(
                "NMF concept-score width does not match the fitted concept count"
            )
        components = self.components.to(
            device=concept_scores.device,
            dtype=concept_scores.dtype,
        )
        batch_size, height, width, _ = concept_scores.shape
        values = concept_scores.reshape(-1, self.concept_count) @ components
        return (
            values.reshape(batch_size, height, width, self.channel_count)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


def fit_nmf_concept_projector_from_activations(
    activation_path: Path,
    n_components: int,
    batch_size: int,
    max_iter: int,
    encoding_max_iter: int,
    random_state: int,
    expected_dataset: str | None = None,
    expected_layer: str | None = None,
) -> NmfConceptProjector:
    """Fit a global MiniBatchNMF basis from complete ID training activations."""

    artifact = torch.load(activation_path, map_location="cpu", weights_only=False)
    if expected_dataset is not None and (
        artifact.get("dataset") != expected_dataset or artifact.get("split") != "train"
    ):
        raise ValueError(
            "NMF must be fitted from the complete ID training-split activation "
            f"artifact {expected_dataset!r}; got dataset={artifact.get('dataset')!r}, "
            f"split={artifact.get('split')!r}: {activation_path}"
        )
    if expected_layer is not None and artifact.get("layer") != expected_layer:
        raise ValueError(
            f"NMF activation layer mismatch: expected {expected_layer!r}, "
            f"got {artifact.get('layer')!r}: {activation_path}"
        )
    activations = artifact.get("activations")
    if not torch.is_tensor(activations):
        raise ValueError(
            f"Activation artifact is missing tensor 'activations': {activation_path}"
        )
    if activations.ndim != 4:
        raise ValueError(
            "NMF concept fitting requires activation maps with shape (N, C, H, W)"
        )
    minimum = float(activations.min())
    if minimum < -1.0e-6:
        raise ValueError(
            "NMF concept fitting requires post-ReLU non-negative activations; "
            f"observed minimum {minimum:.6g}: {activation_path}"
        )
    channel_count = activations.shape[1]
    if n_components > channel_count:
        raise ValueError(
            f"nmf_components={n_components} exceeds activation channels {channel_count}"
        )

    values = (
        activations.float()
        .clamp_min_(0.0)
        .permute(0, 2, 3, 1)
        .reshape(-1, channel_count)
        .numpy()
    )
    model = MiniBatchNMF(
        n_components=n_components,
        init="nndsvda",
        batch_size=batch_size,
        beta_loss="frobenius",
        max_iter=max_iter,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        random_state=random_state,
    )
    model.fit(values)
    return NmfConceptProjector(
        components=torch.from_numpy(model.components_.copy()).float(),
        encoding_max_iter=encoding_max_iter,
        fit_reconstruction_error=float(model.reconstruction_err_),
        fit_n_iter=int(model.n_iter_),
    )


def save_nmf_concept_projector(
    path: Path,
    projector: NmfConceptProjector,
    metadata: dict[str, object],
) -> None:
    """Save a fitted global NMF concept projector and its provenance."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "components": projector.components.cpu(),
            "encoding_max_iter": projector.encoding_max_iter,
            "fit_reconstruction_error": projector.fit_reconstruction_error,
            "fit_n_iter": projector.fit_n_iter,
        },
        path,
    )


def load_nmf_concept_projector(
    path: Path,
    device: torch.device | None = None,
) -> NmfConceptProjector:
    """Load a fitted global NMF concept projector."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    components = artifact.get("components")
    if not torch.is_tensor(components) or components.ndim != 2:
        raise ValueError(f"NMF projector artifact is missing matrix 'components': {path}")
    projector = NmfConceptProjector(
        components=components.float(),
        encoding_max_iter=int(artifact.get("encoding_max_iter", 200)),
        fit_reconstruction_error=artifact.get("fit_reconstruction_error"),
        fit_n_iter=artifact.get("fit_n_iter"),
    )
    if device is not None:
        projector = projector.to(device)
    return projector


def nmf_concept_projector_path(experiment_dir: Path) -> Path:
    """Return the standard NMF projector artifact path for an experiment."""

    return experiment_dir / "nmf_concept_projector.pt"


def _validate_feature_map(features: torch.Tensor, channel_count: int) -> None:
    if features.ndim != 4:
        raise ValueError("NMF concept projection expects features with shape (B, C, H, W)")
    if features.shape[1] != channel_count:
        raise ValueError(
            "NMF feature channels do not match the fitted concept directions: "
            f"{features.shape[1]} != {channel_count}"
        )
