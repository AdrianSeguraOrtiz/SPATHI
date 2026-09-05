"""Distance kernels and reproducible bandwidth selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, Literal, cast

import numpy as np

_BANDWIDTH_IN_MEMORY_ELEMENTS = 2_000_000
_DISTANCE_SCAN_ELEMENTS = 1_000_000


def _flat_distance_view(values: np.ndarray) -> np.ndarray:
    """Return a one-dimensional view for either contiguous memory layout."""

    # ``order='K'`` preserves either C or Fortran storage.  In particular, the
    # core's column-contiguous disk-backed distance matrix remains a view
    # instead of being copied wholesale merely for validation.
    return values.ravel(order="K")


@dataclass(frozen=True, slots=True, kw_only=True)
class BandwidthSelection:
    """The effective bandwidth and the decision that produced it."""

    value: float
    requested: str | float
    method: Literal["explicit", "auto-median", "fallback"]
    positive_distance_count: int
    automatic_reference_value: float | None = None
    automatic_scale: float | None = None
    fallback_reason: str | None = None


def _validate_distances(distances: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    flattened = _flat_distance_view(values)
    for start in range(0, flattened.size, _DISTANCE_SCAN_ELEMENTS):
        block = flattened[start : start + _DISTANCE_SCAN_ELEMENTS]
        if not np.isfinite(block).all():
            raise ValueError("distances must all be finite")
        if np.any(block < 0):
            raise ValueError("distances must all be non-negative")
    return values


def _strictly_positive_count(values: np.ndarray) -> int:
    flattened = _flat_distance_view(values)
    positive_count = 0
    for start in range(0, flattened.size, _DISTANCE_SCAN_ELEMENTS):
        block = flattened[start : start + _DISTANCE_SCAN_ELEMENTS]
        positive_count += int(np.count_nonzero(block > 0))
    return positive_count


def _in_place_median(values: np.ndarray) -> float:
    """Return the exact median without overflowing the even-count midpoint."""

    midpoint = values.size // 2
    if values.size % 2:
        values.partition(midpoint)
        return float(values[midpoint])
    values.partition((midpoint - 1, midpoint))
    lower = float(values[midpoint - 1])
    upper = float(values[midpoint])
    # Both order statistics are finite and strictly positive.  Taking half of
    # each value avoids overflow at the upper end but can round two minimum
    # subnormals down to zero.  The ordered-difference form is safe at both
    # ends of the float64 range.
    return lower + (upper - lower) / 2.0


def _strictly_positive_median(
    values: np.ndarray,
    *,
    scratch_dir: Path | None = None,
) -> tuple[float | None, int]:
    """Find the exact positive median with bounded RAM for large arrays."""

    flattened = _flat_distance_view(values)
    positive_count = _strictly_positive_count(values)
    if positive_count == 0:
        return None, 0

    if flattened.size <= _BANDWIDTH_IN_MEMORY_ELEMENTS:
        return _in_place_median(flattened[flattened > 0]), positive_count

    with TemporaryFile(prefix="spathi-bandwidth-", dir=scratch_dir) as temporary:
        temporary.truncate(positive_count * np.dtype(np.float64).itemsize)
        positive = np.memmap(
            temporary,
            mode="r+",
            dtype=np.float64,
            shape=(positive_count,),
        )
        offset = 0
        try:
            for start in range(0, flattened.size, _DISTANCE_SCAN_ELEMENTS):
                block = flattened[start : start + _DISTANCE_SCAN_ELEMENTS]
                selected = block[block > 0]
                positive[offset : offset + selected.size] = selected
                offset += selected.size
            median = _in_place_median(positive)
        finally:
            try:
                positive.flush()
            finally:
                cast(Any, positive)._mmap.close()
        return median, positive_count


def _validate_bandwidth(bandwidth: float) -> float:
    if isinstance(bandwidth, bool):
        raise ValueError("bandwidth must be a positive finite number")
    try:
        value = float(bandwidth)
    except (TypeError, ValueError) as exc:
        raise ValueError("bandwidth must be 'auto' or a positive finite number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("bandwidth must be a positive finite number")
    return value


def _resolve_automatic_value(reference: float, scale: float) -> float:
    """Return one finite automatic bandwidth after applying its relative scale."""

    reference_value = _validate_bandwidth(reference)
    scale_value = _validate_bandwidth(scale)
    with np.errstate(over="ignore", invalid="ignore"):
        value = reference_value * scale_value
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            "automatic bandwidth reference multiplied by bandwidth_scale must be a "
            "positive finite number"
        )
    return float(value)


def _gaussian_values(values: np.ndarray, scale: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return np.exp(-0.5 * np.square(values / scale))


def _exponential_values(values: np.ndarray, scale: float) -> np.ndarray:
    # Extremely large, valid distance-to-bandwidth ratios map cleanly to zero.
    # Suppress only the expected overflow in the division and underflow in exp.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return np.exp(-(values / scale))


_KERNELS = {
    "gaussian": _gaussian_values,
    "exponential": _exponential_values,
}


def apply_kernel(
    distances: np.ndarray | list[float] | tuple[float, ...],
    bandwidth: float,
    *,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Transform distances into validated weights in the closed interval [0, 1]."""

    values = _validate_distances(distances)
    normalized_kernel = kernel.strip().casefold()
    try:
        function = _KERNELS[normalized_kernel]
    except KeyError:
        available = ", ".join(sorted(_KERNELS))
        raise ValueError(f"Unknown kernel {kernel!r}; available kernels: {available}") from None
    result = np.asarray(function(values, _validate_bandwidth(bandwidth)), dtype=np.float64)
    if result.shape != values.shape:
        raise ValueError(
            f"Kernel {kernel!r} returned shape {result.shape!r}; expected {values.shape!r}"
        )
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError(f"Kernel {kernel!r} produced weights outside the finite [0, 1] contract")
    return np.clip(result, 0.0, 1.0)


def resolve_bandwidth(
    distances: np.ndarray | list[float] | tuple[float, ...],
    bandwidth: str | float = "auto",
    *,
    bandwidth_scale: float = 1.0,
    fallback: float = 1.0,
    scratch_dir: Path | None = None,
) -> BandwidthSelection:
    """Resolve a global bandwidth from a supplied family of distances.

    ``auto`` uses the median of all strictly positive distances and multiplies it by
    ``bandwidth_scale``. If no positive distance exists, the same scale is applied to
    a positive deterministic fallback (1.0 by default), and the returned selection
    records why. Zeros are intentionally excluded so repeated cells or coincident
    group centroids cannot collapse the bandwidth. A numeric ``bandwidth`` is already
    the final bandwidth and therefore requires ``bandwidth_scale=1``.
    """

    values = _validate_distances(distances)
    is_auto = isinstance(bandwidth, str) and bandwidth.strip().casefold() == "auto"
    scale_value = _validate_bandwidth(bandwidth_scale)
    if not is_auto and scale_value != 1.0:
        raise ValueError(
            "bandwidth_scale applies only when bandwidth='auto'; an explicit numeric "
            "bandwidth is already the final bandwidth"
        )
    if is_auto:
        auto_median, positive_count = _strictly_positive_median(
            values,
            scratch_dir=scratch_dir,
        )
    else:
        auto_median = None
        positive_count = _strictly_positive_count(values)
    if isinstance(bandwidth, str) and not is_auto:
        try:
            explicit = _validate_bandwidth(float(bandwidth))
        except ValueError as exc:
            raise ValueError("bandwidth must be 'auto' or a positive finite number") from exc
        return BandwidthSelection(
            value=explicit,
            requested=bandwidth,
            method="explicit",
            positive_distance_count=positive_count,
        )
    if not is_auto:
        explicit = _validate_bandwidth(bandwidth)  # type: ignore[arg-type]
        return BandwidthSelection(
            value=explicit,
            requested=bandwidth,
            method="explicit",
            positive_distance_count=positive_count,
        )

    if auto_median is not None:
        value = auto_median
        # All entries are finite and strictly positive, so this is defensive
        # against unusual platform-level numeric behavior only.
        if np.isfinite(value) and value > 0:
            effective = _resolve_automatic_value(value, scale_value)
            return BandwidthSelection(
                value=effective,
                requested="auto",
                method="auto-median",
                positive_distance_count=positive_count,
                automatic_reference_value=value,
                automatic_scale=scale_value,
            )

    fallback_value = _validate_bandwidth(fallback)
    effective_fallback = _resolve_automatic_value(fallback_value, scale_value)
    reason = "No strictly positive distances were available for automatic bandwidth selection"
    return BandwidthSelection(
        value=effective_fallback,
        requested="auto",
        method="fallback",
        positive_distance_count=positive_count,
        automatic_reference_value=fallback_value,
        automatic_scale=scale_value,
        fallback_reason=reason,
    )


def resolve_bandwidth_for_mode(
    weight_mode: str,
    *,
    cell_to_centroid_distances: np.ndarray | None,
    centroid_distances: np.ndarray | None,
    bandwidth: str | float = "auto",
    bandwidth_scale: float = 1.0,
    fallback: float = 1.0,
    scratch_dir: Path | None = None,
) -> BandwidthSelection:
    """Select the distance family mandated by a SPATHI weighting mode."""

    if weight_mode in {"cell-distance", "cell-distance-group-anchored"}:
        if cell_to_centroid_distances is None:
            raise ValueError(f"{weight_mode!r} requires cell-to-centroid distances")
        family = cell_to_centroid_distances
    elif weight_mode == "group-distance":
        if centroid_distances is None:
            raise ValueError("'group-distance' requires centroid-to-centroid distances")
        family = centroid_distances
    else:
        raise ValueError(
            "weight_mode must be 'cell-distance', 'cell-distance-group-anchored', or 'group-distance'"
        )
    return resolve_bandwidth(
        family,
        bandwidth,
        bandwidth_scale=bandwidth_scale,
        fallback=fallback,
        scratch_dir=scratch_dir,
    )


__all__ = [
    "BandwidthSelection",
    "apply_kernel",
    "resolve_bandwidth",
    "resolve_bandwidth_for_mode",
]
