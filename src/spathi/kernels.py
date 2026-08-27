"""Extensible distance kernels and reproducible bandwidth selection."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import Literal

import numpy as np

KernelFunction = Callable[[np.ndarray, float], np.ndarray]
KernelName = Literal["gaussian", "exponential"]
_BANDWIDTH_IN_MEMORY_ELEMENTS = 2_000_000
_DISTANCE_SCAN_ELEMENTS = 1_000_000


def _flat_distance_view(values: np.ndarray) -> np.ndarray:
    """Return a one-dimensional view for either contiguous memory layout."""

    # ``order='K'`` preserves either C or Fortran storage.  In particular, the
    # pipeline's column-contiguous disk-backed distance matrix remains a view
    # instead of being copied wholesale merely for validation.
    return values.ravel(order="K")


@dataclass(frozen=True, slots=True)
class BandwidthSelection:
    """The effective bandwidth and the decision that produced it."""

    value: float
    requested: str | float
    method: Literal["explicit", "auto-median", "fallback"]
    positive_distance_count: int
    fallback_reason: str | None = None

    @property
    def bandwidth(self) -> float:
        """Alias useful when serializing effective run parameters."""

        return self.value

    def __float__(self) -> float:
        return self.value


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


def _strictly_positive_median(values: np.ndarray) -> tuple[float | None, int]:
    """Find the exact positive median with bounded RAM for large arrays."""

    flattened = _flat_distance_view(values)
    positive_count = _strictly_positive_count(values)
    if positive_count == 0:
        return None, 0

    if flattened.size <= _BANDWIDTH_IN_MEMORY_ELEMENTS:
        return float(np.median(flattened[flattened > 0])), positive_count

    with TemporaryFile(prefix="spathi-bandwidth-") as temporary:
        temporary.truncate(positive_count * np.dtype(np.float64).itemsize)
        positive = np.memmap(
            temporary,
            mode="r+",
            dtype=np.float64,
            shape=(positive_count,),
        )
        offset = 0
        for start in range(0, flattened.size, _DISTANCE_SCAN_ELEMENTS):
            block = flattened[start : start + _DISTANCE_SCAN_ELEMENTS]
            selected = block[block > 0]
            positive[offset : offset + selected.size] = selected
            offset += selected.size
        midpoint = positive_count // 2
        if positive_count % 2:
            positive.partition(midpoint)
            median = float(positive[midpoint])
        else:
            positive.partition((midpoint - 1, midpoint))
            median = float((positive[midpoint - 1] + positive[midpoint]) / 2.0)
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


def gaussian_kernel(
    distances: np.ndarray | list[float] | tuple[float, ...], bandwidth: float
) -> np.ndarray:
    r"""Apply :math:`\exp(-d^2/(2h^2))` elementwise."""

    values = _validate_distances(distances)
    scale = _validate_bandwidth(bandwidth)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result = np.exp(-0.5 * np.square(values / scale))
    return np.clip(result, 0.0, 1.0)


def exponential_kernel(
    distances: np.ndarray | list[float] | tuple[float, ...], bandwidth: float
) -> np.ndarray:
    r"""Apply :math:`\exp(-d/h)` elementwise."""

    values = _validate_distances(distances)
    scale = _validate_bandwidth(bandwidth)
    # Extremely large, valid distance-to-bandwidth ratios map cleanly to zero.
    # Suppress only the expected overflow in the division and underflow in exp.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result = np.exp(-(values / scale))
    return np.clip(result, 0.0, 1.0)


_KERNELS: dict[str, KernelFunction] = {
    "gaussian": gaussian_kernel,
    "exponential": exponential_kernel,
}


def register_kernel(name: str, function: KernelFunction, *, replace: bool = False) -> None:
    """Register a kernel without coupling weighting code to its implementation."""

    normalized = name.strip().casefold()
    if not normalized:
        raise ValueError("kernel name may not be empty")
    if normalized in _KERNELS and not replace:
        raise ValueError(f"kernel {normalized!r} is already registered")
    if not callable(function):
        raise TypeError("kernel function must be callable")
    _KERNELS[normalized] = function


def available_kernels() -> tuple[str, ...]:
    """Return registered kernel names in deterministic order."""

    return tuple(sorted(_KERNELS))


def get_kernel(name: str) -> KernelFunction:
    """Resolve a registered kernel by name."""

    normalized = name.strip().casefold()
    try:
        return _KERNELS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown kernel {name!r}; available kernels: {', '.join(available_kernels())}"
        ) from exc


def apply_kernel(
    distances: np.ndarray | list[float] | tuple[float, ...],
    bandwidth: float,
    *,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Transform distances into validated weights in the closed interval [0, 1]."""

    values = _validate_distances(distances)
    result = np.asarray(
        get_kernel(kernel)(values, _validate_bandwidth(bandwidth)), dtype=np.float64
    )
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
    fallback: float = 1.0,
) -> BandwidthSelection:
    """Resolve a global bandwidth from a supplied family of distances.

    ``auto`` uses the median of all strictly positive distances.  If none
    exist, a positive deterministic fallback (1.0 by default) is used and a
    runtime warning records why.  Zeros are intentionally excluded so repeated
    cells or coincident group centroids cannot collapse the bandwidth.
    """

    values = _validate_distances(distances)
    is_auto = isinstance(bandwidth, str) and bandwidth.strip().casefold() == "auto"
    if is_auto:
        auto_median, positive_count = _strictly_positive_median(values)
    else:
        auto_median = None
        positive_count = _strictly_positive_count(values)
    if isinstance(bandwidth, str) and not is_auto:
        try:
            explicit = _validate_bandwidth(float(bandwidth))
        except ValueError as exc:
            raise ValueError("bandwidth must be 'auto' or a positive finite number") from exc
        return BandwidthSelection(explicit, bandwidth, "explicit", positive_count)
    if not is_auto:
        explicit = _validate_bandwidth(bandwidth)  # type: ignore[arg-type]
        return BandwidthSelection(explicit, bandwidth, "explicit", positive_count)

    if auto_median is not None:
        value = auto_median
        # All entries are finite and strictly positive, so this is defensive
        # against unusual platform-level numeric behavior only.
        if np.isfinite(value) and value > 0:
            return BandwidthSelection(value, "auto", "auto-median", positive_count)

    fallback_value = _validate_bandwidth(fallback)
    reason = "No strictly positive distances were available for automatic bandwidth selection"
    warnings.warn(
        f"{reason}; using deterministic fallback bandwidth {fallback_value:g}",
        RuntimeWarning,
        stacklevel=2,
    )
    return BandwidthSelection(
        fallback_value,
        "auto",
        "fallback",
        positive_count,
        fallback_reason=reason,
    )


def resolve_bandwidth_for_mode(
    weight_mode: str,
    *,
    cell_to_centroid_distances: np.ndarray | None,
    centroid_distances: np.ndarray | None,
    bandwidth: str | float = "auto",
    fallback: float = 1.0,
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
    return resolve_bandwidth(family, bandwidth, fallback=fallback)


def estimate_auto_bandwidth(
    distances: np.ndarray | list[float] | tuple[float, ...], *, fallback: float = 1.0
) -> float:
    """Return only the numeric automatic bandwidth for compact callers."""

    return resolve_bandwidth(distances, "auto", fallback=fallback).value


select_bandwidth = estimate_auto_bandwidth
kernel_weights = apply_kernel


__all__ = [
    "BandwidthSelection",
    "KernelFunction",
    "KernelName",
    "apply_kernel",
    "available_kernels",
    "estimate_auto_bandwidth",
    "exponential_kernel",
    "gaussian_kernel",
    "get_kernel",
    "kernel_weights",
    "register_kernel",
    "resolve_bandwidth",
    "resolve_bandwidth_for_mode",
    "select_bandwidth",
]
