"""OOD detection scores from teacher and student outputs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_PROBABILITY_DIFFERENCE = "max_probability_difference"
ABSOLUTE_MAX_PROBABILITY_DIFFERENCE = "absolute_max_probability_difference"
STUDENT_TEACHER_KL_DIVERGENCE = "student_teacher_kl_divergence"
LOGIT_L2_DISTANCE = "logit_l2_distance"
ENERGY = "energy"
ENERGY_GAP = "energy_gap"
ABSOLUTE_ENERGY_GAP = "absolute_energy_gap"
STUDENT_MSP = "student_msp"
STUDENT_ENERGY = "student_energy"
FEATURE_DENOISING_PCA_RECONSTRUCTION_ERROR = "feature_denoising_pca_reconstruction_error"
FEATURE_DENOISING_SPATIAL_RECONSTRUCTION_ERROR = "feature_denoising_spatial_reconstruction_error"
FEATURE_DENOISING_CHANNEL_RECONSTRUCTION_ERROR = "feature_denoising_channel_reconstruction_error"
FEATURE_DENOISING_SPATIAL_TOKEN_PREDICTION_ERROR = "feature_denoising_spatial_token_prediction_error"
FEATURE_DENOISING_PIXEL_EMBEDDING_PREDICTION_ERROR = "feature_denoising_pixel_embedding_prediction_error"
FEATURE_DENOISING_PIXEL_AUGMENTED_EMBEDDING_PREDICTION_ERROR = (
    "feature_denoising_pixel_augmented_embedding_prediction_error"
)
FEATURE_DENOISING_PIXEL_MULTILAYER_PREDICTION_ERROR = "feature_denoising_pixel_multilayer_prediction_error"
FEATURE_DENOISING_PIXEL_MULTILAYER_L234_PREDICTION_ERROR = "feature_denoising_pixel_multilayer_l234_prediction_error"

# Signs convert raw scores to the repository convention used by OOD metrics:
# ID is the positive class, so larger signed scores are more ID-like.
SIGNS: Mapping[str, int] = {
    MAX_PROBABILITY_DIFFERENCE: +1,
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE: -1,
    STUDENT_TEACHER_KL_DIVERGENCE: -1,
    LOGIT_L2_DISTANCE: -1,
    ENERGY: +1,
    ENERGY_GAP: +1,
    ABSOLUTE_ENERGY_GAP: -1,
    STUDENT_MSP: +1,
    STUDENT_ENERGY: +1,
    FEATURE_DENOISING_PCA_RECONSTRUCTION_ERROR: -1,
    FEATURE_DENOISING_SPATIAL_RECONSTRUCTION_ERROR: -1,
    FEATURE_DENOISING_CHANNEL_RECONSTRUCTION_ERROR: -1,
    FEATURE_DENOISING_SPATIAL_TOKEN_PREDICTION_ERROR: -1,
    FEATURE_DENOISING_PIXEL_EMBEDDING_PREDICTION_ERROR: -1,
    FEATURE_DENOISING_PIXEL_AUGMENTED_EMBEDDING_PREDICTION_ERROR: -1,
    FEATURE_DENOISING_PIXEL_MULTILAYER_PREDICTION_ERROR: -1,
    FEATURE_DENOISING_PIXEL_MULTILAYER_L234_PREDICTION_ERROR: -1,
}


def _maybe_signed(
    scores: NDArray[np.float64],
    score_name: str,
    signed: bool,
) -> NDArray[np.float64]:
    if not signed:
        return scores
    try:
        sign = SIGNS[score_name]
    except KeyError as error:
        msg = f"Missing sign for OOD score: {score_name}"
        raise ValueError(msg) from error
    return sign * scores


def _maybe_average_draws(
    scores: NDArray[np.float64],
    average_draws: bool,
) -> NDArray[np.float64]:
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2 and average_draws:
        return scores.mean(axis=1)
    if scores.ndim == 2:
        return scores
    msg = f"scores must be one- or two-dimensional, got {scores.ndim} dimensions."
    raise ValueError(msg)


def _max_probabilities(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    teacher, student = _probability_matrices(
        teacher_probabilities,
        student_probabilities,
    )
    return teacher.max(axis=-1), student.max(axis=-1)


def _probability_matrices(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    teacher = _as_probability_matrix(teacher_probabilities, "teacher_probabilities")
    student = _as_probability_matrix(student_probabilities, "student_probabilities")
    if teacher.shape != student.shape:
        msg = (
            "teacher_probabilities and student_probabilities must have the same "
            f"shape, got {teacher.shape} and {student.shape}."
        )
        raise ValueError(msg)
    return teacher, student


def _logit_matrices(
    teacher_logits: ArrayLike,
    student_logits: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    teacher = _as_output_matrix(teacher_logits, "teacher_logits")
    student = _as_output_matrix(student_logits, "student_logits")
    if teacher.shape != student.shape:
        msg = (
            "teacher_logits and student_logits must have the same "
            f"shape, got {teacher.shape} and {student.shape}."
        )
        raise ValueError(msg)
    return teacher, student


def _center_outputs(outputs: NDArray[np.float64]) -> NDArray[np.float64]:
    return outputs - outputs.mean(axis=-1, keepdims=True)


def _logsumexp(outputs: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    maxima = outputs.max(axis=axis, keepdims=True)
    summed = np.exp(outputs - maxima).sum(axis=axis)
    return np.squeeze(maxima, axis=axis) + np.log(summed)


def _validate_temperature(temperature: float) -> float:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        msg = f"temperature must be a positive finite value, got {temperature}."
        raise ValueError(msg)
    return temperature


def _as_probability_matrix(
    probabilities: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    return _as_output_matrix(probabilities, name)


def _as_output_matrix(
    outputs: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(outputs, dtype=float)
    if array.ndim not in (2, 3):
        msg = f"{name} must be a two- or three-dimensional array."
        raise ValueError(msg)
    if array.shape[-1] == 0:
        msg = f"{name} must contain at least one class output."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return array


def _kl(p: ArrayLike, q: ArrayLike) -> NDArray[np.float64]:
    """Kullback-Leibler divergence D(P || Q) for discrete distributions.
        P : true probability distribution
        Q : approximating probability distribution

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    positive_support = p > 0.0
    if np.any(positive_support & (q == 0.0)):
        msg = "q must be positive where p is positive to compute KL divergence."
        raise ValueError(msg)
    terms = np.zeros_like(p, dtype=float)
    terms[positive_support] = (
        p[positive_support] * np.log(p[positive_support] / q[positive_support])
    )
    return np.sum(terms, axis=-1)


def max_probability_difference(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute max_proba_teacher - max_proba_student for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed score per sample.
    """

    teacher_max, student_max = _max_probabilities(
        teacher_probabilities,
        student_probabilities,
    )
    scores = _maybe_average_draws(teacher_max - student_max, average_draws)
    return _maybe_signed(scores, MAX_PROBABILITY_DIFFERENCE, signed)


def absolute_max_probability_difference(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute abs(max_proba_teacher - max_proba_student) for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed score per sample.
    """

    scores = np.abs(
        max_probability_difference(
            teacher_probabilities,
            student_probabilities,
            average_draws=False,
        )
    )
    scores = _maybe_average_draws(scores, average_draws)
    return _maybe_signed(scores, ABSOLUTE_MAX_PROBABILITY_DIFFERENCE, signed)


def student_teacher_kl_divergence(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute KL(teacher || student) for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed divergence value per sample.

    Raises:
        ValueError: If inputs have different shapes, contain invalid values, or
            student probabilities are zero where teacher probabilities are
            positive.
    """

    teacher, student = _probability_matrices(
        teacher_probabilities,
        student_probabilities,
    )
    scores = _maybe_average_draws(_kl(teacher, student), average_draws)
    return _maybe_signed(scores, STUDENT_TEACHER_KL_DIVERGENCE, signed)


def logit_l2_distance(
    teacher_logits: ArrayLike,
    student_logits: ArrayLike,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute L2 distance between centered teacher and student logits.

    Args:
        teacher_logits: Teacher logits with shape ``(n_samples, n_classes)``.
        student_logits: Student logits with shape ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed L2 distance per sample.

    Raises:
        ValueError: If inputs have different shapes or contain invalid values.
    """

    teacher, student = _logit_matrices(teacher_logits, student_logits)
    centered_teacher = _center_outputs(teacher)
    centered_student = _center_outputs(student)
    scores = np.linalg.norm(centered_teacher - centered_student, ord=2, axis=-1)
    scores = _maybe_average_draws(scores, average_draws)
    return _maybe_signed(scores, LOGIT_L2_DISTANCE, signed)


def energy(
    logits: ArrayLike,
    temperature: float = 1.0,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute the sign-adjusted energy confidence score for each sample.

    This returns ``T * logsumexp(logits / T)``, which is the negative of the
    free energy from energy-based OOD detection. The sign follows this
    repository's convention where larger scores are more ID-like.

    Args:
        logits: Model logits with shape ``(n_samples, n_classes)``.
        temperature: Positive energy temperature.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed energy score per sample.
    """

    temperature = _validate_temperature(temperature)
    outputs = _as_output_matrix(logits, "logits")
    scores = temperature * _logsumexp(outputs / temperature, axis=-1)
    scores = _maybe_average_draws(scores, average_draws)
    return _maybe_signed(scores, ENERGY, signed)


def student_msp(
    probabilities: ArrayLike,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute the student Maximum Softmax Probability for each sample."""

    outputs = _as_probability_matrix(probabilities, "student_probabilities")
    scores = _maybe_average_draws(outputs.max(axis=-1), average_draws)
    return _maybe_signed(scores, STUDENT_MSP, signed)


def student_energy(
    logits: ArrayLike,
    temperature: float = 1.0,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute the student-only energy OOD Score for each sample."""

    temperature = _validate_temperature(temperature)
    outputs = _as_output_matrix(logits, "student_logits")
    scores = temperature * _logsumexp(outputs / temperature, axis=-1)
    scores = _maybe_average_draws(scores, average_draws)
    return _maybe_signed(scores, STUDENT_ENERGY, signed)


def energy_gap(
    teacher_logits: ArrayLike,
    student_logits: ArrayLike,
    temperature: float = 1.0,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute teacher energy minus student energy for each sample.

    Args:
        teacher_logits: Teacher logits with shape ``(n_samples, n_classes)``.
        student_logits: Student logits with shape ``(n_samples, n_classes)``.
        temperature: Positive energy temperature.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed energy gap score per sample.
    """

    teacher, student = _logit_matrices(teacher_logits, student_logits)
    scores = energy(
        teacher,
        temperature=temperature,
        average_draws=False,
    ) - energy(
        student,
        temperature=temperature,
        average_draws=False,
    )
    scores = _maybe_average_draws(scores, average_draws)
    return _maybe_signed(scores, ENERGY_GAP, signed)


def absolute_energy_gap(
    teacher_logits: ArrayLike,
    student_logits: ArrayLike,
    temperature: float = 1.0,
    signed: bool = False,
    average_draws: bool = True,
) -> NDArray[np.float64]:
    """Compute absolute teacher-student energy gap for each sample.

    Args:
        teacher_logits: Teacher logits with shape ``(n_samples, n_classes)``.
        student_logits: Student logits with shape ``(n_samples, n_classes)``.
        temperature: Positive energy temperature.
        signed: If True, apply this score's sign so higher values are more
            ID-like.
        average_draws: If True and inputs have shape
            ``(n_samples, n_draws, n_classes)``, average per-draw scores for
            each sample.

    Returns:
        One raw or signed absolute energy gap score per sample.
    """

    scores = np.abs(
        energy_gap(
            teacher_logits,
            student_logits,
            temperature=temperature,
            average_draws=False,
        )
    )
    scores = _maybe_average_draws(scores, average_draws)
    return _maybe_signed(scores, ABSOLUTE_ENERGY_GAP, signed)
