"""OOD detection scores from teacher and student probabilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_PROBABILITY_DIFFERENCE = "max_probability_difference"
ABSOLUTE_MAX_PROBABILITY_DIFFERENCE = "absolute_max_probability_difference"
STUDENT_TEACHER_KL_DIVERGENCE = "student_teacher_kl_divergence"

# Signs convert raw scores to the repository convention used by OOD metrics:
# ID is the positive class, so larger signed scores are more ID-like.
SIGNS: Mapping[str, int] = {
    MAX_PROBABILITY_DIFFERENCE: +1,
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE: -1,
    STUDENT_TEACHER_KL_DIVERGENCE: -1,
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


def _max_probabilities(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    teacher, student = _probability_matrices(
        teacher_probabilities,
        student_probabilities,
    )
    return teacher.max(axis=1), student.max(axis=1)


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
    return np.sum(terms, axis=1)


def max_probability_difference(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
    signed: bool = False,
) -> NDArray[np.float64]:
    """Compute max_proba_teacher - max_proba_student for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.

    Returns:
        One raw or signed score per sample.
    """

    teacher_max, student_max = _max_probabilities(
        teacher_probabilities,
        student_probabilities,
    )
    scores = teacher_max - student_max
    return _maybe_signed(scores, MAX_PROBABILITY_DIFFERENCE, signed)


def absolute_max_probability_difference(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
    signed: bool = False,
) -> NDArray[np.float64]:
    """Compute abs(max_proba_teacher - max_proba_student) for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.

    Returns:
        One raw or signed score per sample.
    """

    scores = np.abs(
        max_probability_difference(
            teacher_probabilities,
            student_probabilities,
        )
    )
    return _maybe_signed(scores, ABSOLUTE_MAX_PROBABILITY_DIFFERENCE, signed)


def student_teacher_kl_divergence(
    teacher_probabilities: ArrayLike,
    student_probabilities: ArrayLike,
    signed: bool = False,
) -> NDArray[np.float64]:
    """Compute KL(student || teacher) for each sample.

    Args:
        teacher_probabilities: Teacher class probabilities with shape
            ``(n_samples, n_classes)``.
        student_probabilities: Student class probabilities with shape
            ``(n_samples, n_classes)``.
        signed: If True, apply this score's sign so higher values are more
            ID-like.

    Returns:
        One raw or signed divergence value per sample.

    Raises:
        ValueError: If inputs have different shapes, contain invalid values, or
            teacher probabilities are zero where student probabilities are
            positive.
    """

    teacher, student = _probability_matrices(
        teacher_probabilities,
        student_probabilities,
    )
    scores = _kl(teacher, student)
    return _maybe_signed(scores, STUDENT_TEACHER_KL_DIVERGENCE, signed)
