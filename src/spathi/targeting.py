"""Auditable target-estimability policies for SPATHI inference.

Target eligibility is deliberately independent of transcription-factor eligibility.
An ineligible gene is not fitted as a response, but it remains available as a predictor
whenever it occurs in the supplied TF list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spathi.config import TargetEligibilityMode

TARGET_ELIGIBILITY_CHUNK_SIZE = 256
TARGET_ELIGIBILITY_WORKING_MEMORY_BYTES = 64 * 1024**2


def _minimum_detected_count(n_cells: int, fraction: float) -> int:
    """Return the smallest count whose observed fraction reaches ``fraction``.

    Computing only ``ceil(fraction * n_cells)`` can overcount at an exact decimal
    boundary because the multiplication may round just above an integer (for
    example, ``0.07 * 100``).  The final comparison uses the same float semantics
    exposed by the API, so the reported count and the eligibility decision agree.
    """

    required = ceil(fraction * n_cells)
    while required > 0 and (required - 1) / n_cells >= fraction:
        required -= 1
    while required / n_cells < fraction:
        required += 1
    return required


def target_eligibility_chunk_size(n_cells: int, n_targets: int) -> int:
    """Return a bounded target width for one reusable boolean work matrix."""

    if n_cells < 1 or n_targets < 1:
        raise ValueError("target eligibility chunk dimensions must be positive")
    return min(
        TARGET_ELIGIBILITY_CHUNK_SIZE,
        n_targets,
        max(1, TARGET_ELIGIBILITY_WORKING_MEMORY_BYTES // n_cells),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetEligibilityRecord:
    """One global, data-derived target eligibility decision."""

    target: str
    mode: TargetEligibilityMode
    eligible: bool
    detected_cells: int | None
    detected_fraction: float | None
    expression_min: float | None
    expression_max: float | None
    required_detected_cells: int | None
    reason: str

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target must be non-empty")
        if self.mode not in {"all", "automatic"}:
            raise ValueError(f"unsupported target eligibility mode: {self.mode!r}")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be a boolean")
        if not self.reason:
            raise ValueError("target eligibility reason must be non-empty")
        for field_name in ("expression_min", "expression_max"):
            value = getattr(self, field_name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite or None")

    def to_dict(self) -> dict[str, str | bool | int | float | None]:
        return {
            "target": self.target,
            "mode": self.mode,
            "eligible": self.eligible,
            "detected_cells": self.detected_cells,
            "detected_fraction": self.detected_fraction,
            "expression_min": self.expression_min,
            "expression_max": self.expression_max,
            "required_detected_cells": self.required_detected_cells,
            "reason": self.reason,
        }


def _validate_assessment_inputs(
    expression: ArrayLike,
    target_names: Sequence[object],
    *,
    mode: TargetEligibilityMode,
    min_detected_cells: int,
    min_detected_fraction: float,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    try:
        values = np.asarray(expression, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("target expression must be numeric") from exc
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("target expression must be a non-empty cells-by-targets matrix")
    targets = tuple(str(target) for target in target_names)
    if len(targets) != values.shape[1] or any(not target for target in targets):
        raise ValueError("target names must match the target-expression columns")
    if len(set(targets)) != len(targets):
        raise ValueError("target names must be unique")
    if mode not in {"all", "automatic"}:
        raise ValueError(f"unsupported target eligibility mode: {mode!r}")
    if isinstance(min_detected_cells, bool) or not isinstance(min_detected_cells, Integral):
        raise TypeError("min_detected_cells must be an integer")
    if min_detected_cells < 1:
        raise ValueError("min_detected_cells must be positive")
    if isinstance(min_detected_fraction, bool) or not isinstance(min_detected_fraction, Real):
        raise TypeError("min_detected_fraction must be a number")
    fraction = float(min_detected_fraction)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("min_detected_fraction must be in (0, 1]")
    return values, targets


def assess_target_eligibility(
    expression: ArrayLike,
    target_names: Sequence[object],
    *,
    mode: TargetEligibilityMode,
    min_detected_cells: int,
    min_detected_fraction: float,
) -> tuple[TargetEligibilityRecord, ...]:
    """Assess targets without removing genes from PCA or the TF predictor space.

    Detection means an exact positive value in a finite, non-negative, zero-preserving
    matrix. In automatic mode a target must satisfy both the absolute and relative
    detected-cell requirements and have non-zero population variance. ``all`` is the
    unfiltered scientific reference and intentionally avoids scanning the target matrix
    a second time.
    """

    values, targets = _validate_assessment_inputs(
        expression,
        target_names,
        mode=mode,
        min_detected_cells=min_detected_cells,
        min_detected_fraction=min_detected_fraction,
    )
    if mode == "all":
        return unfiltered_target_eligibility(targets)

    n_cells = int(values.shape[0])
    required = max(
        int(min_detected_cells),
        _minimum_detected_count(n_cells, float(min_detected_fraction)),
    )
    records: list[TargetEligibilityRecord] = []
    chunk_size = target_eligibility_chunk_size(n_cells, len(targets))
    for start in range(0, len(targets), chunk_size):
        stop = min(len(targets), start + chunk_size)
        block = values[:, start:stop]
        working = np.empty(block.shape, dtype=np.bool_, order="F")
        np.isfinite(block, out=working)
        if not np.all(working):
            raise ValueError("target expression contains non-finite values")
        np.less(block, 0.0, out=working)
        if np.any(working):
            raise ValueError(
                "automatic target eligibility requires non-negative, zero-preserving "
                "target expression; use target_eligibility='all' for centered values"
            )
        np.not_equal(block, 0.0, out=working)
        detected = np.count_nonzero(working, axis=0)
        minima = np.min(block, axis=0)
        maxima = np.max(block, axis=0)
        variable = minima != maxima
        for offset, target in enumerate(targets[start:stop]):
            count = int(detected[offset])
            if count < required:
                eligible = False
                reason = "insufficient_detected_cells"
            elif not variable[offset]:
                eligible = False
                reason = "zero_variance"
            else:
                eligible = True
                reason = "eligible"
            records.append(
                TargetEligibilityRecord(
                    target=target,
                    mode=mode,
                    eligible=eligible,
                    detected_cells=count,
                    detected_fraction=count / n_cells,
                    expression_min=float(minima[offset]),
                    expression_max=float(maxima[offset]),
                    required_detected_cells=required,
                    reason=reason,
                )
            )
    return tuple(records)


def unfiltered_target_eligibility(
    target_names: Sequence[object],
) -> tuple[TargetEligibilityRecord, ...]:
    """Build the ``all`` policy without accessing an expression matrix."""

    targets = tuple(str(target) for target in target_names)
    if not targets or any(not target for target in targets):
        raise ValueError("target names must be non-empty")
    if len(set(targets)) != len(targets):
        raise ValueError("target names must be unique")
    return tuple(
        TargetEligibilityRecord(
            target=target,
            mode="all",
            eligible=True,
            detected_cells=None,
            detected_fraction=None,
            expression_min=None,
            expression_max=None,
            required_detected_cells=None,
            reason="unfiltered_reference",
        )
        for target in targets
    )


def weighted_detected_context_statistics(
    expression: NDArray[np.float64],
    target_indices: Sequence[int],
    weights: NDArray[np.float64],
    squared_weights: NDArray[np.float64],
    *,
    weight_sum: float,
) -> tuple[tuple[float, float], ...]:
    """Return weighted detection fractions and Kish ESS in bounded blocks.

    The cells-by-target detection mask is capped independently of the caller's
    inference batch. This avoids two Python-level reductions for every model and
    never creates a floating-point copy of the selected expression columns.
    """

    if expression.ndim != 2:
        raise ValueError("expression must be a cells-by-targets matrix")
    n_cells, n_targets = expression.shape
    if weights.shape != (n_cells,) or squared_weights.shape != (n_cells,):
        raise ValueError("weights must match the expression cell axis")
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("weight_sum must be positive and finite")
    indices = tuple(target_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or index < 0
        or index >= n_targets
        for index in indices
    ):
        raise ValueError("target indices must identify expression columns")

    values: list[tuple[float, float]] = []
    weight_column = weights[:, np.newaxis]
    squared_weight_column = squared_weights[:, np.newaxis]
    chunk_size = target_eligibility_chunk_size(n_cells, len(indices)) if indices else 1
    for start in range(0, len(indices), chunk_size):
        block_indices = indices[start : start + chunk_size]
        # Column-major storage preserves the exact scalar reduction order for
        # every target, independent of eligibility/inference batch width.
        detected = np.empty(
            (n_cells, len(block_indices)),
            dtype=np.bool_,
            order="F",
        )
        for position, target_index in enumerate(block_indices):
            np.not_equal(expression[:, target_index], 0.0, out=detected[:, position])
        detected_weight = np.sum(
            np.broadcast_to(weight_column, detected.shape),
            axis=0,
            where=detected,
            dtype=np.float64,
        )
        detected_squared_weight = np.sum(
            np.broadcast_to(squared_weight_column, detected.shape),
            axis=0,
            where=detected,
            dtype=np.float64,
        )
        block_ess = np.divide(
            detected_weight * detected_weight,
            detected_squared_weight,
            out=np.zeros_like(detected_weight),
            where=detected_squared_weight != 0.0,
        )
        detected_fractions = detected_weight / weight_sum
        values.extend(
            (float(fraction), float(ess))
            for fraction, ess in zip(detected_fractions, block_ess, strict=True)
        )
    return tuple(values)


__all__ = [
    "TargetEligibilityRecord",
    "TARGET_ELIGIBILITY_CHUNK_SIZE",
    "TARGET_ELIGIBILITY_WORKING_MEMORY_BYTES",
    "assess_target_eligibility",
    "unfiltered_target_eligibility",
    "target_eligibility_chunk_size",
    "weighted_detected_context_statistics",
]
