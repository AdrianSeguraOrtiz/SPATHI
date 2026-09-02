"""Weight-mass diagnostics and effective sample size for SPATHI."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .weighting import WeightResult


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightDiagnostics:
    """Diagnostics for the reusable weight vector of one target network."""

    target_group: str
    n_cells: int
    n_target_cells: int
    total_weight: float
    target_weight: float
    external_weight: float
    target_mass_percent: float
    external_mass_percent: float
    min_weight: float
    max_weight: float
    mean_weight: float
    median_weight: float
    positive_cell_count: int
    effective_sample_size: float
    group_weight_mass: Mapping[str, float]
    group_mass_percent: Mapping[str, float]
    warnings: tuple[str, ...]


def effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    r"""Calculate :math:`(\sum_i w_i)^2 / \sum_i w_i^2`.

    Invalid or negative input yields ``nan`` so the diagnostic layer can emit a
    warning without hiding the condition.  An all-zero vector has ESS zero.
    """

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        return float("nan")
    denominator = float(np.dot(values, values))
    if denominator == 0:
        return 0.0
    total = float(np.sum(values, dtype=np.float64))
    return (total * total) / denominator


def _validate_weights(weights: WeightResult) -> np.ndarray:
    values = np.asarray(weights.final_weight, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional vector")
    if values.shape != (len(weights.context.cells),):
        raise ValueError("weight vector length must match its weighting context")
    return values


def compute_weight_diagnostics(
    weights: WeightResult,
    *,
    low_ess_fraction: float = 0.1,
    dominant_external_fraction: float = 0.5,
    emit_warnings: bool = True,
) -> WeightDiagnostics:
    """Summarize weight quality and contributions from every source group.

    A low-ESS warning is emitted below ``max(2, low_ess_fraction * n_cells)``.
    An external group is considered dominant when it supplies more than
    ``dominant_external_fraction`` of total mass.  Both thresholds are explicit
    because the specification intentionally leaves their scientific tuning to
    future validation.
    """

    if not np.isfinite(low_ess_fraction) or not 0 <= low_ess_fraction <= 1:
        raise ValueError("low_ess_fraction must be between zero and one")
    if not np.isfinite(dominant_external_fraction) or not 0 < dominant_external_fraction <= 1:
        raise ValueError("dominant_external_fraction must be in (0, 1]")
    values = _validate_weights(weights)
    context = weights.context
    target = weights.target_group
    target_index = context.group_ids.index(target)
    messages: list[str] = []

    all_finite = bool(np.isfinite(values).all())
    all_nonnegative = bool(np.all(values >= 0)) if all_finite else False
    if not all_finite:
        messages.append("Weights contain non-finite values")
    if all_finite and not all_nonnegative:
        messages.append("Weights contain negative values")
    if all_finite and np.any(values > 1 + 1e-12):
        messages.append("Weights exceed the expected maximum of one")

    if all_finite:
        total = float(np.sum(values, dtype=np.float64))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        mean = float(np.mean(values))
        median = float(np.median(values))
    else:
        total = minimum = maximum = mean = median = float("nan")

    unique_groups = context.group_ids
    masses = np.bincount(
        context.group_codes,
        weights=values,
        minlength=len(unique_groups),
    )
    group_mass = {group: float(masses[index]) for index, group in enumerate(unique_groups)}
    target_weight = group_mass[target]
    external_weight = float(
        np.sum(masses[:target_index], dtype=np.float64)
        + np.sum(masses[target_index + 1 :], dtype=np.float64)
    )
    if not np.isfinite(total) or not np.isfinite(target_weight):
        external_weight = float("nan")
    if np.isfinite(total) and total > 0:
        group_percent = {group: (mass / total) * 100.0 for group, mass in group_mass.items()}
        target_percent = group_percent[target]
        external_percent = (external_weight / total) * 100.0
    else:
        group_percent = {group: float("nan") for group in unique_groups}
        target_percent = external_percent = float("nan")

    positive_count = int(np.count_nonzero(np.isfinite(values) & (values > 0)))
    ess = effective_sample_size(values)
    if not np.isfinite(total) or total <= 0 or positive_count == 0:
        messages.append("Weights are degenerate: no positive finite total mass")
    elif np.isfinite(ess) and ess < max(2.0, low_ess_fraction * values.size):
        messages.append(f"Effective sample size is very low ({ess:.3g} of {values.size} cells)")

    if np.isfinite(external_weight) and external_weight > target_weight:
        messages.append("External groups contribute more total weight than the target group")
    if np.isfinite(total) and total > 0:
        for group in unique_groups:
            if group == target:
                continue
            fraction = group_mass[group] / total
            if fraction > dominant_external_fraction:
                messages.append(
                    f"External group {group!r} contributes an excessive {fraction:.1%} of total weight"
                )

    if emit_warnings:
        for message in messages:
            warnings.warn(f"Target group {target!r}: {message}", RuntimeWarning, stacklevel=2)

    return WeightDiagnostics(
        target_group=target,
        n_cells=int(values.size),
        n_target_cells=int(context.group_counts[target_index]),
        total_weight=total,
        target_weight=target_weight,
        external_weight=external_weight,
        target_mass_percent=target_percent,
        external_mass_percent=external_percent,
        min_weight=minimum,
        max_weight=maximum,
        mean_weight=mean,
        median_weight=median,
        positive_cell_count=positive_count,
        effective_sample_size=ess,
        group_weight_mass=group_mass,
        group_mass_percent=group_percent,
        warnings=tuple(messages),
    )


__all__ = [
    "WeightDiagnostics",
    "compute_weight_diagnostics",
    "effective_sample_size",
]
