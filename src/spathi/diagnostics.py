"""Weight-mass diagnostics and effective sample size for SPATHI."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .weighting import WeightResult


@dataclass(frozen=True, slots=True)
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

    @property
    def ess(self) -> float:
        """Short alias for :attr:`effective_sample_size`."""

        return self.effective_sample_size

    @property
    def sum_weights(self) -> float:
        """Alias for :attr:`total_weight`."""

        return self.total_weight

    @property
    def target_weight_mass(self) -> float:
        """Alias for :attr:`target_weight`."""

        return self.target_weight

    @property
    def external_weight_mass(self) -> float:
        """Alias for :attr:`external_weight`."""

        return self.external_weight

    def to_summary_record(self) -> dict[str, str | int | float]:
        """Return a deterministic scalar record suitable for a TSV row."""

        return {
            "target_group": self.target_group,
            "n_cells": self.n_cells,
            "n_target_cells": self.n_target_cells,
            "total_weight": self.total_weight,
            "target_weight": self.target_weight,
            "external_weight": self.external_weight,
            "target_mass_percent": self.target_mass_percent,
            "external_mass_percent": self.external_mass_percent,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "mean_weight": self.mean_weight,
            "median_weight": self.median_weight,
            "positive_cell_count": self.positive_cell_count,
            "effective_sample_size": self.effective_sample_size,
            "warnings": " | ".join(self.warnings),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return one row per contributing group with summary metrics repeated.

        This long layout records every external group's mass without dynamic or
        JSON-encoded columns, while remaining straightforward to inspect in
        ``weight_diagnostics.tsv``.
        """

        return pd.DataFrame.from_records(self.iter_records())

    def iter_records(self) -> Iterator[dict[str, str | int | float | bool]]:
        """Yield one source-group row without retaining the complete long table."""

        summary = self.to_summary_record()
        for source_group, mass in self.group_weight_mass.items():
            yield {
                **summary,
                "source_group": source_group,
                "source_is_target": source_group == self.target_group,
                "source_weight": mass,
                "source_mass_percent": self.group_mass_percent[source_group],
            }


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


def _coerce_inputs(
    weights: WeightResult | Sequence[float] | np.ndarray,
    cell_groups: pd.Series | Sequence[Hashable] | None,
    target_group: Hashable | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if isinstance(weights, WeightResult):
        if cell_groups is not None or target_group is not None:
            raise ValueError(
                "cell_groups and target_group must be omitted when a WeightResult is supplied"
            )
        values = np.asarray(weights.final_weight, dtype=np.float64)
        groups = np.asarray(weights.cell_groups, dtype=str)
        target = weights.target_group
    else:
        if cell_groups is None or target_group is None:
            raise ValueError("cell_groups and target_group are required with a raw weight vector")
        values = np.asarray(weights, dtype=np.float64)
        if isinstance(cell_groups, pd.Series):
            raw_groups = cell_groups.astype("string").to_numpy()
        else:
            raw_groups = np.asarray(list(cell_groups), dtype=object)
        groups = np.asarray([str(group) for group in raw_groups], dtype=str)
        target = str(target_group)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional vector")
    if groups.shape != values.shape:
        raise ValueError("cell_groups length must match weights")
    if not np.any(groups == target):
        raise ValueError(f"target group {target!r} has no cells")
    return values, groups, target


def compute_weight_diagnostics(
    weights: WeightResult | Sequence[float] | np.ndarray,
    cell_groups: pd.Series | Sequence[Hashable] | None = None,
    target_group: Hashable | None = None,
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
    values, groups, target = _coerce_inputs(weights, cell_groups, target_group)
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

    # Factorize once and aggregate every source group in one linear pass.  The
    # previous per-group boolean mask scaled as O(n_cells * n_groups).
    group_codes, group_labels = pd.factorize(groups, sort=False)
    unique_groups = [str(group) for group in group_labels.tolist()]
    masses = np.bincount(group_codes, weights=values, minlength=len(unique_groups))
    group_counts = np.bincount(group_codes, minlength=len(unique_groups))
    group_mass = {group: float(masses[index]) for index, group in enumerate(unique_groups)}
    group_index = {group: index for index, group in enumerate(unique_groups)}
    target_weight = group_mass[target]
    external_weight = (
        total - target_weight if np.isfinite(total) and np.isfinite(target_weight) else float("nan")
    )
    if np.isfinite(total) and total > 0:
        group_percent = {group: (mass / total) * 100.0 for group, mass in group_mass.items()}
        target_percent = group_percent[target]
        external_percent = 100.0 - target_percent
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
        n_target_cells=int(group_counts[group_index[target]]),
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


def diagnostics_frame(
    diagnostics: Sequence[WeightDiagnostics] | Mapping[str, WeightDiagnostics],
) -> pd.DataFrame:
    """Combine diagnostics into deterministic target/source-group rows."""

    values = list(diagnostics.values()) if isinstance(diagnostics, Mapping) else list(diagnostics)
    if not values:
        return pd.DataFrame()
    frame = pd.concat([item.to_frame() for item in values], ignore_index=True)
    return frame.sort_values(["target_group", "source_group"], kind="stable", ignore_index=True)


compute_ess = effective_sample_size
summarize_weights = compute_weight_diagnostics


__all__ = [
    "WeightDiagnostics",
    "compute_ess",
    "compute_weight_diagnostics",
    "diagnostics_frame",
    "effective_sample_size",
    "summarize_weights",
]
