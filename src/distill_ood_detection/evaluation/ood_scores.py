"""OOD detection scores from teacher and student probabilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def max_probability_difference(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
) -> NDArray[np.float64]:
    """Compute max_proba_teacher - max_proba_student for each sample.

    The returned values follow the repository's OOD convention: ID samples are
    the positive class and OOD samples are the negative class.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.

    Returns:
        One score per sample. Higher scores are treated as more ID-like by
        ``ood_detection_metrics``.
    """

    teacher_max, student_max = _max_probabilities(
        teacher_probabilities,
        student_probabilities,
    )
    return teacher_max - student_max


def absolute_max_probability_difference(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
) -> NDArray[np.float64]:
    """Compute abs(max_proba_teacher - max_proba_student) for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.

    Returns:
        One score per sample. Higher scores are treated as more ID-like by
        ``ood_detection_metrics``.
    """

    return np.abs(
        max_probability_difference(
            teacher_probabilities,
            student_probabilities,
        )
    )


def confidence_ratio(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
) -> NDArray[np.float64]:
    """Compute log(max_proba_teacher) - log(max_proba_student) per sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.

    Returns:
        One score per sample. Higher scores are treated as more ID-like by
        ``ood_detection_metrics``.

    Raises:
        ValueError: If either model has a maximum probability less than or equal
            to zero for any sample.
    """

    teacher_max, student_max = _max_probabilities(
        teacher_probabilities,
        student_probabilities,
    )
    if np.any(teacher_max <= 0.0) or np.any(student_max <= 0.0):
        msg = "confidence_ratio requires positive maximum probabilities."
        raise ValueError(msg)
    return np.log(teacher_max) - np.log(student_max)


def _max_probabilities(
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
    return teacher.max(axis=1), student.max(axis=1)


def _as_probability_matrix(
    probabilities: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2:
        msg = f"{name} must be a two-dimensional array."
        raise ValueError(msg)
    if array.shape[1] == 0:
        msg = f"{name} must contain at least one class probability."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return array
