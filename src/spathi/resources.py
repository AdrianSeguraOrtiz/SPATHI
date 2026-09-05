"""Portable resource estimates used to keep model parallelism memory-safe."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from numbers import Real
from pathlib import Path, PurePosixPath

_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_PROC_SELF_MOUNTINFO = Path("/proc/self/mountinfo")
_PROC_MEMINFO = Path("/proc/meminfo")
_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
_CGROUP_V1_MEMORY_ROOT = Path("/sys/fs/cgroup/memory")
_CGROUP_V1_UNLIMITED_MINIMUM = 1 << 60
_BYTES_PER_TREE_NODE_ESTIMATE = 80

_TextReader = Callable[[Path], str | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryPlan:
    """Memory-derived concurrency decision for independent model fits."""

    available_bytes: int | None
    usable_bytes: int | None
    reserved_bytes: int
    estimated_bytes_per_model: int
    usable_fraction: float
    max_concurrent_models: int | None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None


def _read_nonnegative_integer(path: Path, *, read_text: _TextReader = _read_text) -> int | None:
    text = read_text(path)
    if text is None:
        return None
    text = text.strip()
    if not text.isdigit():
        return None
    return int(text)


def _unescape_mountinfo_path(value: str) -> str:
    for escaped, character in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(escaped, character)
    return value


def _cgroup_memberships(cgroup_text: str) -> tuple[str | None, str | None]:
    v2_path: str | None = None
    v1_memory_path: str | None = None
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers_text, cgroup_path = fields
        if not cgroup_path.startswith("/"):
            continue
        controllers = set(controllers_text.split(",")) if controllers_text else set()
        if hierarchy == "0" and not controllers:
            v2_path = cgroup_path
        elif "memory" in controllers:
            v1_memory_path = cgroup_path
    return v2_path, v1_memory_path


def _resolved_cgroup_directories(
    cgroup_text: str, mountinfo_text: str
) -> list[tuple[Path, Path, int]]:
    v2_path, v1_memory_path = _cgroup_memberships(cgroup_text)
    resolved: list[tuple[Path, Path, int]] = []
    for line in mountinfo_text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        mount_fields = left.split()
        filesystem_fields = right.split()
        if len(mount_fields) < 5 or len(filesystem_fields) < 3:
            continue
        filesystem_type, mount_source, super_options = filesystem_fields[:3]
        version: int
        membership_path: str | None
        if filesystem_type == "cgroup2":
            version = 2
            membership_path = v2_path
        elif filesystem_type == "cgroup" and (
            "memory" in super_options.split(",") or mount_source == "memory"
        ):
            version = 1
            membership_path = v1_memory_path
        else:
            continue
        if membership_path is None:
            continue

        mount_root = PurePosixPath(_unescape_mountinfo_path(mount_fields[3]))
        mount_point = Path(_unescape_mountinfo_path(mount_fields[4]))
        member = PurePosixPath(membership_path)
        try:
            relative = member.relative_to(mount_root)
        except ValueError:
            # A cgroup namespace reports its own root as "/", while mountinfo can
            # retain the host-side cgroup root. This common case maps to the mount.
            if member != PurePosixPath("/"):
                continue
            relative = PurePosixPath(".")
        if ".." in relative.parts or not mount_point.is_absolute():
            continue
        current = mount_point.joinpath(*relative.parts)
        resolved.append((current, mount_point, version))
    return resolved


def _cgroup_ancestor_directories(current: Path, mount_point: Path) -> list[Path]:
    if current != mount_point and mount_point not in current.parents:
        return []
    directories = [current]
    while directories[-1] != mount_point:
        directories.append(directories[-1].parent)
    return directories


def _cgroup_headroom(
    directory: Path, version: int, *, read_text: _TextReader = _read_text
) -> int | None:
    if version == 2:
        limit_path = directory / "memory.max"
        usage_path = directory / "memory.current"
    else:
        limit_path = directory / "memory.limit_in_bytes"
        usage_path = directory / "memory.usage_in_bytes"
    limit = _read_nonnegative_integer(limit_path, read_text=read_text)
    usage = _read_nonnegative_integer(usage_path, read_text=read_text)
    if limit is None or usage is None:
        return None
    if version == 1 and limit >= _CGROUP_V1_UNLIMITED_MINIMUM:
        return None
    return max(0, limit - usage)


def _cgroup_available_bytes(
    *,
    cgroup_text: str | None = None,
    mountinfo_text: str | None = None,
    read_text: _TextReader = _read_text,
) -> int | None:
    if cgroup_text is None:
        cgroup_text = read_text(_PROC_SELF_CGROUP)
    if mountinfo_text is None:
        mountinfo_text = read_text(_PROC_SELF_MOUNTINFO)

    headrooms: list[int] = []
    if cgroup_text is not None and mountinfo_text is not None:
        for current, mount_point, version in _resolved_cgroup_directories(
            cgroup_text, mountinfo_text
        ):
            for directory in _cgroup_ancestor_directories(current, mount_point):
                headroom = _cgroup_headroom(directory, version, read_text=read_text)
                if headroom is not None:
                    headrooms.append(headroom)
    if headrooms:
        return min(headrooms)

    # Retain support for conventional layouts if procfs is unavailable, hidden,
    # or unusual enough that the mount cannot be resolved.
    for directory, version in (
        (_CGROUP_V2_ROOT, 2),
        (_CGROUP_V1_MEMORY_ROOT, 1),
    ):
        headroom = _cgroup_headroom(directory, version, read_text=read_text)
        if headroom is not None:
            headrooms.append(headroom)
    return min(headrooms) if headrooms else None


def _linux_available_bytes(*, read_text: _TextReader = _read_text) -> int | None:
    text = read_text(_PROC_MEMINFO)
    if text is None:
        return None
    for line in text.splitlines():
        field, separator, value = line.partition(":")
        if field != "MemAvailable" or not separator:
            continue
        parts = value.split()
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] != "kB":
            return None
        return int(parts[0]) * 1024
    return None


def _system_available_bytes(
    *,
    platform: str = sys.platform,
    read_text: _TextReader = _read_text,
) -> int | None:
    if platform.startswith("linux"):
        linux_available = _linux_available_bytes(read_text=read_text)
        if linux_available is not None:
            return linux_available
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    available = page_size * available_pages
    return available if available >= 0 else None


def available_memory_bytes() -> int | None:
    """Return the tightest visible system/cgroup memory headroom, if detectable."""

    candidates = [
        value
        for value in (_system_available_bytes(), _cgroup_available_bytes())
        if value is not None
    ]
    return min(candidates) if candidates else None


def estimate_model_memory_bytes(
    *,
    n_cells: int,
    n_transcription_factors: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
) -> int:
    """Estimate a conservative peak for one fitted tree ensemble.

    The bound covers the fitted tree arrays plus the largest possible temporary
    self-exclusion predictor copy. It is intentionally independent of a particular
    scikit-learn private layout.
    """

    for field_name, value in (
        ("n_cells", n_cells),
        ("n_transcription_factors", n_transcription_factors),
        ("n_estimators", n_estimators),
        ("min_samples_leaf", min_samples_leaf),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
    ):
        raise ValueError("max_depth must be a positive integer or None")

    maximum_leaves = max(1, ceil(n_cells / min_samples_leaf))
    maximum_nodes = 2 * maximum_leaves - 1
    if max_depth is not None and max_depth + 1 < maximum_nodes.bit_length():
        depth_limited_nodes = 2 ** (max_depth + 1) - 1
        maximum_nodes = min(maximum_nodes, depth_limited_nodes)
    tree_bytes = n_estimators * maximum_nodes * _BYTES_PER_TREE_NODE_ESTIMATE
    self_exclusion_bytes = n_cells * max(0, n_transcription_factors - 1) * 4
    return max(1, tree_bytes + self_exclusion_bytes)


def plan_model_memory(
    *,
    estimated_bytes_per_model: int,
    available_bytes: int | None = None,
    usable_fraction: float = 0.7,
    reserved_bytes: int = 0,
) -> MemoryPlan:
    """Resolve a model cap after reserving upcoming non-model allocations."""

    if (
        isinstance(estimated_bytes_per_model, bool)
        or not isinstance(estimated_bytes_per_model, int)
        or estimated_bytes_per_model < 1
    ):
        raise ValueError("estimated_bytes_per_model must be a positive integer")
    if (
        isinstance(usable_fraction, bool)
        or not isinstance(usable_fraction, Real)
        or not 0.0 < float(usable_fraction) <= 1.0
    ):
        raise ValueError("usable_fraction must be in (0, 1]")
    if (
        isinstance(reserved_bytes, bool)
        or not isinstance(reserved_bytes, int)
        or reserved_bytes < 0
    ):
        raise ValueError("reserved_bytes must be a non-negative integer")
    detected = available_memory_bytes() if available_bytes is None else available_bytes
    if detected is not None and (
        isinstance(detected, bool) or not isinstance(detected, int) or detected < 0
    ):
        raise ValueError("available_bytes must be a non-negative integer or None")
    usable = None if detected is None else max(0, int(detected * float(usable_fraction)))
    model_budget = None if usable is None else max(0, usable - reserved_bytes)
    maximum = None if model_budget is None else model_budget // estimated_bytes_per_model
    return MemoryPlan(
        available_bytes=detected,
        usable_bytes=usable,
        reserved_bytes=reserved_bytes,
        estimated_bytes_per_model=estimated_bytes_per_model,
        usable_fraction=float(usable_fraction),
        max_concurrent_models=maximum,
    )


__all__ = [
    "MemoryPlan",
    "available_memory_bytes",
    "estimate_model_memory_bytes",
    "plan_model_memory",
]
