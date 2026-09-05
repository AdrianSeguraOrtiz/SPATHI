#!/usr/bin/env python3
"""Run validated SPATHI scaling profiles and aggregate their results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spathi.config import MAX_RANDOM_SEED
from spathi.parallel import available_cpu_count

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = BENCHMARK_DIR / "benchmark_scaling.py"
PACKAGE_SOURCE_DIR = BENCHMARK_DIR.parent / "src" / "spathi"
BUILTIN_PROFILE_DIR = BENCHMARK_DIR / "profiles" / "v1"
PROFILE_SCHEMA_VERSION = 1
SUITE_SCHEMA_VERSION = 1

_CASE_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_DEFAULT_FIELDS = {
    "n_estimators",
    "n_components",
    "targets",
    "threads",
    "warmups",
    "repeats",
    "seed",
    "checkpoint",
    "report",
    "resource_sample_ms",
    "case_timeout_seconds",
    "case_process_timeout_seconds",
}
_PROFILE_FIELDS = {
    "schema_version",
    "name",
    "description",
    "limitations",
    "source_snapshot",
    "defaults",
    "cases",
}
_CASE_FIELDS = {
    "id",
    "description",
    "tags",
    "cells",
    "genes",
    "tfs",
    "groups",
    *_DEFAULT_FIELDS,
}
_SUITE_CSV_FIELDS = (
    "suite_profile",
    "suite_case_id",
    "suite_case_description",
    "suite_case_tags",
    "suite_profile_sha256",
)
_REQUIRED_BENCHMARK_FIELDS = {
    "run_index",
    "run_type",
    "round",
    "position",
    "threads",
    "wall_seconds",
    "peak_rss_bytes",
    "sampled_cpu_user_seconds",
    "sampled_cpu_system_seconds",
    "status",
    "cells",
    "genes",
    "targets",
    "target_list",
    "tfs",
    "groups",
    "n_estimators",
    "n_components",
    "checkpoint",
    "report",
    "show_spathi_output",
    "seed",
    "resource_sample_ms",
    "case_timeout_seconds",
    "expression_sha256",
    "tf_list_sha256",
    "groups_sha256",
    "target_list_sha256",
    "peak_run_logical_bytes",
    "published_output_logical_bytes",
    "retained_run_logical_bytes",
    "implementation_sha256",
    "benchmark_sha256",
}


@contextmanager
def _sigterm_as_keyboard_interrupt() -> Iterator[None]:
    """Route SIGTERM through the same child-process cleanup as Ctrl-C."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt(_signal_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


class ProfileError(ValueError):
    """Raised when a scaling profile violates the current profile contract."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalingCase:
    """One dataset shape and one internally reusable target/thread experiment matrix."""

    id: str
    description: str
    tags: tuple[str, ...]
    cells: int
    genes: int
    tfs: int
    groups: int
    n_estimators: int
    n_components: int
    targets: tuple[int, ...] | None
    threads: tuple[int, ...]
    warmups: int
    repeats: int
    seed: int
    checkpoint: bool
    report: bool
    resource_sample_ms: float
    case_timeout_seconds: float
    case_process_timeout_seconds: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileSourceSnapshot:
    """Pinned observations from which a data-shaped profile was derived."""

    relative_path: str
    source_path: Path
    source_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalingProfile:
    """A validated scaling suite loaded from an immutable JSON definition."""

    name: str
    description: str
    limitations: tuple[str, ...]
    cases: tuple[ScalingCase, ...]
    source_snapshot: ProfileSourceSnapshot | None
    source_path: Path
    source_bytes: bytes
    sha256: str


class CaseProcessTimeoutError(TimeoutError):
    """Raised when a complete benchmark case exceeds its outer wall-time budget."""


def _utc_now() -> str:
    """Return a compact, timezone-explicit UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    """Hash one regular file without loading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProfileError(f"{location} must be a JSON object with string keys")
    return value


def _check_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    location: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ProfileError(f"{location} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ProfileError(f"{location} is missing fields: {', '.join(missing)}")


def _string(value: Any, *, location: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ProfileError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, *, location: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ProfileError(f"{location} must be at least {minimum}")
    return value


def _positive_number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ProfileError(f"{location} must be positive and finite")
    return result


def _boolean(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileError(f"{location} must be a boolean")
    return value


def _string_list(value: Any, *, location: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a JSON array" if allow_empty else "a non-empty JSON array"
        raise ProfileError(f"{location} must be {qualifier} of strings")
    result = tuple(
        _string(item, location=f"{location}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ProfileError(f"{location} must not contain duplicates")
    return result


def _integer_list(
    value: Any,
    *,
    location: str,
    item_validator: Any,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileError(f"{location} must be a non-empty JSON array")
    result = tuple(item_validator(item, index) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ProfileError(f"{location} must not contain duplicates")
    return result


def _resolve_profile_path(profile: str | Path) -> Path:
    supplied = Path(profile).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    if supplied.parent != Path(".") or supplied.suffix:
        raise ProfileError(f"profile file does not exist: {supplied}")
    builtin = BUILTIN_PROFILE_DIR / f"{supplied.name}.json"
    if not builtin.is_file():
        available = ", ".join(sorted(path.stem for path in BUILTIN_PROFILE_DIR.glob("*.json")))
        raise ProfileError(f"unknown built-in profile {supplied.name!r}; available: {available}")
    return builtin.resolve()


def _parse_source_snapshot(
    value: Any,
    *,
    profile_path: Path,
) -> ProfileSourceSnapshot | None:
    if value is None:
        return None
    source = _strict_object(value, location="source_snapshot")
    _check_fields(
        source,
        allowed={"path", "sha256"},
        required={"path", "sha256"},
        location="source_snapshot",
    )
    relative_path = Path(_string(source["path"], location="source_snapshot.path"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ProfileError("source_snapshot.path must stay within the profile directory")
    expected_sha256 = _string(source["sha256"], location="source_snapshot.sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ProfileError("source_snapshot.sha256 must be a lowercase SHA-256 digest")
    source_path = (profile_path.parent / relative_path).resolve()
    try:
        source_path.relative_to(profile_path.parent.resolve())
        source_bytes = source_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise ProfileError(f"cannot read source snapshot {relative_path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ProfileError(
            f"source snapshot hash mismatch for {relative_path}: expected "
            f"{expected_sha256}, observed {actual_sha256}"
        )
    return ProfileSourceSnapshot(
        relative_path=str(relative_path),
        source_path=source_path,
        source_bytes=source_bytes,
        sha256=actual_sha256,
    )


def _parse_case(
    raw_case: Any,
    *,
    defaults: Mapping[str, Any],
    case_index: int,
) -> ScalingCase:
    location = f"cases[{case_index}]"
    case_object = _strict_object(raw_case, location=location)
    _check_fields(
        case_object,
        allowed=_CASE_FIELDS,
        required={"id", "description", "tags", "cells", "genes", "tfs", "groups"},
        location=location,
    )
    merged = {**defaults, **case_object}

    case_id = _string(merged["id"], location=f"{location}.id")
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        raise ProfileError(
            f"{location}.id must use lowercase letters, digits, and internal hyphens"
        )
    description = _string(merged["description"], location=f"{location}.description")
    tags = _string_list(merged["tags"], location=f"{location}.tags", allow_empty=True)
    cells = _integer(merged["cells"], location=f"{location}.cells", minimum=1)
    genes = _integer(merged["genes"], location=f"{location}.genes", minimum=2)
    tfs = _integer(merged["tfs"], location=f"{location}.tfs", minimum=1)
    groups = _integer(merged["groups"], location=f"{location}.groups", minimum=1)
    n_estimators = _integer(merged["n_estimators"], location=f"{location}.n_estimators", minimum=1)
    n_components = _integer(merged["n_components"], location=f"{location}.n_components", minimum=1)
    warmups = _integer(merged["warmups"], location=f"{location}.warmups", minimum=0)
    repeats = _integer(merged["repeats"], location=f"{location}.repeats", minimum=1)
    seed = _integer(merged["seed"], location=f"{location}.seed")
    if not 0 <= seed <= MAX_RANDOM_SEED:
        raise ProfileError(f"{location}.seed must be between 0 and {MAX_RANDOM_SEED}")
    checkpoint = _boolean(merged["checkpoint"], location=f"{location}.checkpoint")
    report = _boolean(merged["report"], location=f"{location}.report")
    resource_sample_ms = _positive_number(
        merged["resource_sample_ms"], location=f"{location}.resource_sample_ms"
    )
    case_timeout_seconds = _positive_number(
        merged["case_timeout_seconds"], location=f"{location}.case_timeout_seconds"
    )
    case_process_timeout_seconds = _positive_number(
        merged["case_process_timeout_seconds"],
        location=f"{location}.case_process_timeout_seconds",
    )

    targets = (
        None
        if merged["targets"] is None
        else _integer_list(
            merged["targets"],
            location=f"{location}.targets",
            item_validator=lambda item, index: _integer(
                item, location=f"{location}.targets[{index}]", minimum=1
            ),
        )
    )
    threads = _integer_list(
        merged["threads"],
        location=f"{location}.threads",
        item_validator=lambda item, index: _integer(item, location=f"{location}.threads[{index}]"),
    )

    if cells < groups:
        raise ProfileError(f"{location}.cells must be at least groups")
    if tfs >= genes:
        raise ProfileError(f"{location}.tfs must be smaller than genes")
    if targets is not None and any(target > genes for target in targets):
        raise ProfileError(f"{location}.targets values must not exceed genes")
    if any(thread < 1 for thread in threads):
        raise ProfileError(f"{location}.threads values must be positive integers")
    scheduled_runs = (warmups + repeats) * (1 if targets is None else len(targets)) * len(threads)
    if case_process_timeout_seconds <= case_timeout_seconds * scheduled_runs:
        raise ProfileError(
            f"{location}.case_process_timeout_seconds must exceed the complete schedule's "
            f"{scheduled_runs} per-run timeout budgets"
        )

    return ScalingCase(
        id=case_id,
        description=description,
        tags=tags,
        cells=cells,
        genes=genes,
        tfs=tfs,
        groups=groups,
        n_estimators=n_estimators,
        n_components=n_components,
        targets=targets,
        threads=threads,
        warmups=warmups,
        repeats=repeats,
        seed=seed,
        checkpoint=checkpoint,
        report=report,
        resource_sample_ms=resource_sample_ms,
        case_timeout_seconds=case_timeout_seconds,
        case_process_timeout_seconds=case_process_timeout_seconds,
    )


def load_profile(profile: str | Path) -> ScalingProfile:
    """Resolve, parse, and strictly validate a built-in or explicit profile."""

    source_path = _resolve_profile_path(profile)
    try:
        raw_bytes = source_path.read_bytes()
        document = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read profile {source_path}: {exc}") from exc
    profile_object = _strict_object(document, location="profile")
    _check_fields(
        profile_object,
        allowed=_PROFILE_FIELDS,
        required=_PROFILE_FIELDS,
        location="profile",
    )
    schema_version = _integer(profile_object["schema_version"], location="schema_version")
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise ProfileError(
            f"profile schema_version must be {PROFILE_SCHEMA_VERSION}, got {schema_version}"
        )
    name = _string(profile_object["name"], location="name")
    if not _CASE_ID_PATTERN.fullmatch(name):
        raise ProfileError("profile name must use lowercase letters, digits, and internal hyphens")
    description = _string(profile_object["description"], location="description")
    limitations = _string_list(
        profile_object["limitations"], location="limitations", allow_empty=True
    )
    source_snapshot = _parse_source_snapshot(
        profile_object["source_snapshot"],
        profile_path=source_path,
    )
    defaults = _strict_object(profile_object["defaults"], location="defaults")
    _check_fields(
        defaults,
        allowed=_DEFAULT_FIELDS,
        required=_DEFAULT_FIELDS,
        location="defaults",
    )
    raw_cases = profile_object["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProfileError("cases must be a non-empty JSON array")
    cases = tuple(
        _parse_case(raw_case, defaults=defaults, case_index=index)
        for index, raw_case in enumerate(raw_cases)
    )
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ProfileError("case ids must be unique")
    signatures: dict[tuple[Any, ...], str] = {}
    for case in cases:
        values = asdict(case)
        signature = tuple(
            value for key, value in values.items() if key not in {"id", "description", "tags"}
        )
        if signature in signatures:
            raise ProfileError(
                f"cases {signatures[signature]!r} and {case.id!r} are computational duplicates"
            )
        signatures[signature] = case.id
    return ScalingProfile(
        name=name,
        description=description,
        limitations=limitations,
        cases=cases,
        source_snapshot=source_snapshot,
        source_path=source_path,
        source_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def build_case_command(
    case: ScalingCase,
    *,
    case_directory: Path,
    benchmark_script: Path = BENCHMARK_SCRIPT,
    retain_workspace: bool = False,
) -> list[str]:
    """Build a fully explicit benchmark command for one isolated profile case."""

    command = [
        sys.executable,
        str(benchmark_script),
        "--cells",
        str(case.cells),
        "--genes",
        str(case.genes),
        "--tfs",
        str(case.tfs),
        "--groups",
        str(case.groups),
        "--n-estimators",
        str(case.n_estimators),
        "--n-components",
        str(case.n_components),
    ]
    if case.targets is not None:
        command.extend(("--targets", *(str(target) for target in case.targets)))
    command.extend(
        (
            "--threads",
            *(str(thread) for thread in case.threads),
            "--warmups",
            str(case.warmups),
            "--repeats",
            str(case.repeats),
            "--seed",
            str(case.seed),
            "--checkpoint" if case.checkpoint else "--no-checkpoint",
            "--report" if case.report else "--no-report",
            "--resource-sample-ms",
            str(case.resource_sample_ms),
            "--case-timeout-seconds",
            str(case.case_timeout_seconds),
            "--work-dir",
            str(case_directory / "work"),
        )
    )
    if retain_workspace:
        command.append("--keep-work-dir")
    return command


def _relative(path: Path, *, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Publish immutable input bytes without rereading a mutable source path."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _snapshot_package(source: Path, destination: Path) -> str:
    """Copy the importable package from bytes and return its implementation digest."""

    destination.mkdir()
    python_files = sorted(source.glob("*.py"), key=lambda path: path.name)
    if not python_files:
        raise RuntimeError(f"SPATHI package snapshot has no Python modules: {source}")
    implementation_digest = hashlib.sha256()
    for source_path in python_files:
        content = source_path.read_bytes()
        _atomic_write_bytes(destination / source_path.name, content)
        implementation_digest.update(source_path.name.encode("utf-8"))
        implementation_digest.update(b"\0")
        implementation_digest.update(content)
        implementation_digest.update(b"\0")
    typed_marker = source / "py.typed"
    if typed_marker.is_file():
        _atomic_write_bytes(destination / typed_marker.name, typed_marker.read_bytes())
    return implementation_digest.hexdigest()


def _estimated_case_peak_bytes(case: ScalingCase) -> int:
    """Return a conservative disk estimate for one generated benchmark case."""

    dense_expression = case.cells * case.genes * 16
    target_counts = (case.genes,) if case.targets is None else case.targets
    target_sidecar_entries = 0 if case.targets is None else sum(case.targets)
    input_sidecars = (case.cells + case.genes + case.tfs + target_sidecar_entries) * 64
    scheduled_runs = (case.warmups + case.repeats) * len(target_counts) * len(case.threads)
    maximum_edges = case.groups * max(target_counts) * case.tfs
    outputs = scheduled_runs * maximum_edges * 96
    optional_reports = scheduled_runs * 32 * 1024 * 1024 if case.report else 0
    operational_margin = 512 * 1024 * 1024
    return dense_expression + input_sidecars + outputs + optional_reports + operational_margin


def _disk_preflight(case: ScalingCase, *, destination: Path) -> dict[str, int]:
    """Reject a case before generation when its conservative disk budget cannot fit."""

    estimate = _estimated_case_peak_bytes(case)
    free = shutil.disk_usage(destination).free
    if free < estimate:
        raise RuntimeError(
            f"case {case.id!r} needs an estimated {estimate} free bytes but only "
            f"{free} bytes are available at {destination}"
        )
    return {"estimated_peak_bytes": estimate, "free_bytes_before_case": free}


def _write_aggregate(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fields = list(_SUITE_CSV_FIELDS)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_case_rows(
    path: Path,
    *,
    profile: ScalingProfile,
    case: ScalingCase,
    expected_benchmark_sha256: str,
    expected_implementation_sha256: str,
) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("missing CSV header")
            missing_fields = sorted(_REQUIRED_BENCHMARK_FIELDS.difference(reader.fieldnames))
            if missing_fields:
                raise ValueError("missing required columns: " + ", ".join(missing_fields))
            benchmark_rows = list(reader)
    except (OSError, csv.Error, ValueError) as exc:
        raise RuntimeError(f"invalid benchmark CSV {path}: {exc}") from exc
    if not benchmark_rows:
        raise RuntimeError(f"benchmark CSV has no result rows: {path}")

    target_counts = (case.genes,) if case.targets is None else case.targets
    expected_keys = {
        (run_type, str(round_index), str(thread), str(target))
        for run_type, round_count in (
            ("warmup", case.warmups),
            ("measurement", case.repeats),
        )
        for round_index in range(1, round_count + 1)
        for target in target_counts
        for thread in case.threads
    }
    actual_keys = [
        (row["run_type"], row["round"], row["threads"], row["targets"]) for row in benchmark_rows
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise RuntimeError(f"benchmark CSV contains duplicate case identities: {path}")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(set(actual_keys).difference(expected_keys))
        raise RuntimeError(
            f"benchmark CSV has an incomplete or unexpected schedule: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    expected_constants = {
        "cells": str(case.cells),
        "genes": str(case.genes),
        "tfs": str(case.tfs),
        "groups": str(case.groups),
        "n_estimators": str(case.n_estimators),
        "n_components": str(case.n_components),
        "checkpoint": str(case.checkpoint),
        "report": str(case.report),
        "show_spathi_output": "False",
        "seed": str(case.seed),
        "resource_sample_ms": str(case.resource_sample_ms),
        "case_timeout_seconds": str(case.case_timeout_seconds),
        "target_list": str(case.targets is not None),
        "benchmark_sha256": expected_benchmark_sha256,
        "implementation_sha256": expected_implementation_sha256,
    }
    for row_index, row in enumerate(benchmark_rows, start=2):
        mismatches = {
            field: (row[field], expected)
            for field, expected in expected_constants.items()
            if row[field] != expected
        }
        if mismatches:
            raise RuntimeError(
                f"benchmark CSV row {row_index} does not match its profile case: {mismatches}"
            )
        if not row["implementation_sha256"]:
            raise RuntimeError(f"benchmark CSV row {row_index} lacks implementation provenance")
        if not row["status"]:
            raise RuntimeError(f"benchmark CSV row {row_index} lacks an execution status")
    implementation_hashes = {row["implementation_sha256"] for row in benchmark_rows}
    if len(implementation_hashes) != 1:
        raise RuntimeError(f"benchmark CSV mixes multiple SPATHI implementations: {path}")
    for field in ("expression_sha256", "tf_list_sha256", "groups_sha256"):
        observed = {row[field] for row in benchmark_rows}
        if len(observed) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(observed))) is None:
            raise RuntimeError(f"benchmark CSV has invalid or inconsistent {field}: {path}")
    target_hashes: dict[str, set[str]] = {}
    for row in benchmark_rows:
        target_hashes.setdefault(row["targets"], set()).add(row["target_list_sha256"])
    if case.targets is None:
        if target_hashes != {str(case.genes): {""}}:
            raise RuntimeError(f"full-target benchmark CSV unexpectedly uses a target list: {path}")
    elif any(
        len(hashes) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(hashes))) is None
        for hashes in target_hashes.values()
    ):
        raise RuntimeError(f"benchmark CSV has invalid or inconsistent target-list hashes: {path}")

    suite_values = {
        "suite_profile": profile.name,
        "suite_case_id": case.id,
        "suite_case_description": case.description,
        "suite_case_tags": ";".join(case.tags),
        "suite_profile_sha256": profile.sha256,
    }
    return [{**suite_values, **row} for row in benchmark_rows]


def _interrupt_process(process: subprocess.Popen[str]) -> None:
    """Interrupt a benchmark and reap descendants even across nested sessions."""

    import psutil

    members: list[psutil.Process] = []
    try:
        root = psutil.Process(process.pid)
        members = [*root.children(recursive=True), root]
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        pass

    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGINT)
            else:  # pragma: no cover - Windows-specific fallback
                process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass

    alive: list[psutil.Process] = []
    for member in reversed(members):
        if os.name == "posix":
            try:
                if member.is_running() and member.status() != psutil.STATUS_ZOMBIE:
                    member.terminate()
                    alive.append(member)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        else:  # pragma: no cover - platform-independent psutil path
            try:
                member.terminate()
                alive.append(member)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
    _, alive = psutil.wait_procs(alive, timeout=5)
    for member in alive:
        try:
            member.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    psutil.wait_procs(alive, timeout=5)

    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - kernel-level failure
        pass


def _drain_case_stderr(
    stream: Any,
    destination: Any,
    *,
    label: str,
    errors: list[BaseException],
) -> None:
    """Persist benchmark diagnostics while leaving process waiting unblocked."""

    try:
        for line in iter(stream.readline, ""):
            destination.write(line)
            destination.flush()
            print(f"[{label}] {line}", file=sys.stderr, end="")
    except BaseException as exc:  # pragma: no cover - defensive I/O path
        errors.append(exc)
    finally:
        try:
            stream.close()
        except OSError:  # pragma: no cover - defensive I/O path
            pass


def _execute_case(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    label: str,
    timeout_seconds: float,
    python_path: Path | None = None,
) -> int:
    started = time.monotonic()
    with (
        stdout_path.open("x", encoding="utf-8", newline="") as stdout_stream,
        stderr_path.open("x", encoding="utf-8") as stderr_stream,
    ):
        environment = os.environ.copy()
        # The package snapshot is provenance, not a cross-case bytecode cache.
        # Keeping it source-only also gives every child the same import path state.
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if python_path is not None:
            existing_python_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(python_path)
                if not existing_python_path
                else os.pathsep.join((str(python_path), existing_python_path))
            )
        process = subprocess.Popen(
            command,
            stdout=stdout_stream,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=os.name == "posix",
            env=environment,
        )
        assert process.stderr is not None
        drain_errors: list[BaseException] = []
        drain_thread = threading.Thread(
            target=_drain_case_stderr,
            args=(process.stderr, stderr_stream),
            kwargs={"label": label, "errors": drain_errors},
            name=f"spathi-suite-stderr-{label}",
        )
        drain_thread.start()
        try:
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _interrupt_process(process)
                elapsed = time.monotonic() - started
                raise CaseProcessTimeoutError(
                    f"benchmark case {label!r} exceeded its outer timeout of "
                    f"{timeout_seconds:g} seconds (elapsed {elapsed:.3f} seconds)"
                ) from exc
        except BaseException:
            _interrupt_process(process)
            raise
        finally:
            drain_thread.join(timeout=5)
            if drain_thread.is_alive():  # pragma: no cover - defensive pipe failure
                _interrupt_process(process)
                drain_thread.join(timeout=5)
        if drain_errors:
            raise RuntimeError(
                "failed while capturing benchmark diagnostics: "
                + "; ".join(str(error) for error in drain_errors)
            )
        return exit_code


def _profile_snapshot(profile: ScalingProfile) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "name": profile.name,
        "description": profile.description,
        "limitations": list(profile.limitations),
        "source_snapshot": (
            None
            if profile.source_snapshot is None
            else {
                "path": profile.source_snapshot.relative_path,
                "sha256": profile.source_snapshot.sha256,
            }
        ),
        "source_path": str(profile.source_path),
        "sha256": profile.sha256,
        "cases": [asdict(case) for case in profile.cases],
    }


def _initial_manifest(
    profile: ScalingProfile,
    *,
    suite_root: Path,
    benchmark_script: Path,
    benchmark_sha256: str,
    implementation_sha256: str,
    retain_workspaces: bool,
) -> dict[str, Any]:
    started_at = _utc_now()
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "profile": _profile_snapshot(profile),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "benchmark": {
            "source_path": str(BENCHMARK_SCRIPT),
            "snapshot_path": _relative(benchmark_script, root=suite_root),
            "sha256": benchmark_sha256,
        },
        "implementation": {
            "source_path": str(PACKAGE_SOURCE_DIR),
            "snapshot_path": "spathi",
            "sha256": implementation_sha256,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "artifacts": {
            "aggregate_csv": "results.csv",
            "profile_snapshot": "profile.json",
            "benchmark_snapshot": _relative(benchmark_script, root=suite_root),
            "implementation_snapshot": "spathi",
            "source_snapshot": (
                None if profile.source_snapshot is None else profile.source_snapshot.relative_path
            ),
        },
        "retain_workspaces": retain_workspaces,
        "aggregate_case_process_budget_seconds": sum(
            case.case_process_timeout_seconds for case in profile.cases
        ),
        "cases": [
            {
                "id": case.id,
                "status": "pending",
                "started_at_utc": None,
                "completed_at_utc": None,
                "exit_code": None,
                "result_rows": 0,
                "command": build_case_command(
                    case,
                    case_directory=suite_root / "cases" / case.id,
                    benchmark_script=benchmark_script,
                    retain_workspace=retain_workspaces,
                ),
                "disk_preflight": {},
                "artifacts": {},
                "error": "",
            }
            for case in profile.cases
        ],
    }


def _dry_run_document(
    profile: ScalingProfile,
    *,
    output_dir: Path,
    retain_workspaces: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "mode": "dry-run",
        "output_dir": str(output_dir),
        "profile": _profile_snapshot(profile),
        "retain_workspaces": retain_workspaces,
        "aggregate_case_process_budget_seconds": sum(
            case.case_process_timeout_seconds for case in profile.cases
        ),
        "cases": [
            {
                "id": case.id,
                "command": build_case_command(
                    case,
                    case_directory=output_dir / "cases" / case.id,
                    retain_workspace=retain_workspaces,
                ),
                "case_process_timeout_seconds": case.case_process_timeout_seconds,
                "estimated_peak_disk_bytes": _estimated_case_peak_bytes(case),
                "generated_input_reuse": {
                    "targets": None if case.targets is None else list(case.targets),
                    "threads": list(case.threads),
                    "note": (
                        "benchmark_scaling.py generates the dataset once for this case and "
                        "reuses it across every target/thread combination"
                    ),
                },
            }
            for case in profile.cases
        ],
    }


def run_suite(
    profile: ScalingProfile,
    *,
    output_dir: Path,
    retain_workspaces: bool = False,
) -> int:
    """Run all cases, persisting an interruption-safe audit trail after every case."""

    required_cpus = max(
        (thread for case in profile.cases for thread in case.threads if thread > 0),
        default=1,
    )
    available_cpus = available_cpu_count()
    if required_cpus > available_cpus:
        raise RuntimeError(
            f"profile {profile.name!r} requires at least {required_cpus} logical CPUs "
            f"to preserve distinct thread budgets; this host exposes {available_cpus}"
        )

    benchmark_bytes = BENCHMARK_SCRIPT.read_bytes()
    benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cases").mkdir()
    benchmark_snapshot = output_dir / "benchmark_scaling.py"
    package_snapshot = output_dir / "spathi"
    _atomic_write_bytes(output_dir / "profile.json", profile.source_bytes)
    if profile.source_snapshot is not None:
        source_snapshot_destination = output_dir / profile.source_snapshot.relative_path
        source_snapshot_destination.parent.mkdir(parents=True)
        _atomic_write_bytes(
            source_snapshot_destination,
            profile.source_snapshot.source_bytes,
        )
    _atomic_write_bytes(benchmark_snapshot, benchmark_bytes)
    implementation_sha256 = _snapshot_package(PACKAGE_SOURCE_DIR, package_snapshot)
    manifest_path = output_dir / "manifest.json"
    aggregate_path = output_dir / "results.csv"
    manifest = _initial_manifest(
        profile,
        suite_root=output_dir,
        benchmark_script=benchmark_snapshot,
        benchmark_sha256=benchmark_sha256,
        implementation_sha256=implementation_sha256,
        retain_workspaces=retain_workspaces,
    )
    aggregate_rows: list[dict[str, str]] = []
    _write_aggregate(aggregate_path, aggregate_rows)
    _atomic_write_json(manifest_path, manifest)

    had_failures = False
    try:
        for index, case in enumerate(profile.cases):
            case_manifest = manifest["cases"][index]
            case_directory = output_dir / "cases" / case.id
            case_directory.mkdir()
            stdout_path = case_directory / "benchmark-results.csv"
            stderr_path = case_directory / "benchmark.stderr.log"
            command = build_case_command(
                case,
                case_directory=case_directory,
                benchmark_script=benchmark_snapshot,
                retain_workspace=retain_workspaces,
            )
            case_manifest["status"] = "running"
            case_manifest["started_at_utc"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)
            print(
                f"Running scaling case {index + 1}/{len(profile.cases)}: {case.id}", file=sys.stderr
            )

            try:
                case_manifest["disk_preflight"] = _disk_preflight(
                    case,
                    destination=case_directory,
                )
                exit_code = _execute_case(
                    command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    label=case.id,
                    timeout_seconds=case.case_process_timeout_seconds,
                    python_path=output_dir,
                )
                case_manifest["exit_code"] = exit_code
                case_rows = _load_case_rows(
                    stdout_path,
                    profile=profile,
                    case=case,
                    expected_benchmark_sha256=manifest["benchmark"]["sha256"],
                    expected_implementation_sha256=manifest["implementation"]["sha256"],
                )
                aggregate_rows.extend(case_rows)
                row_failures = [row for row in case_rows if row.get("status") != "success"]
                successful = exit_code == 0 and not row_failures
                case_manifest["status"] = "success" if successful else "failed"
                case_manifest["result_rows"] = len(case_rows)
                if row_failures:
                    case_manifest["error"] = (
                        f"{len(row_failures)} benchmark result row(s) did not succeed"
                    )
                elif exit_code != 0:
                    case_manifest["error"] = f"benchmark exited with status {exit_code}"
                had_failures = had_failures or not successful
            except CaseProcessTimeoutError as exc:
                case_manifest["status"] = "timeout"
                case_manifest["exit_code"] = 124
                case_manifest["error"] = str(exc)
                had_failures = True
            except (OSError, RuntimeError, csv.Error) as exc:
                case_manifest["status"] = "failed"
                case_manifest["error"] = str(exc)
                had_failures = True
            except BaseException as exc:
                case_manifest["status"] = "interrupted"
                case_manifest["exit_code"] = 130 if isinstance(exc, KeyboardInterrupt) else None
                case_manifest["error"] = type(exc).__name__
                raise
            finally:
                workspaces = sorted((case_directory / "work").glob("spathi-benchmark-*"))
                case_manifest["completed_at_utc"] = _utc_now()
                case_manifest["artifacts"] = {
                    "results_csv": _relative(stdout_path, root=output_dir),
                    "stderr_log": _relative(stderr_path, root=output_dir),
                    "benchmark_workspaces": [
                        _relative(path, root=output_dir) for path in workspaces
                    ],
                }
                _write_aggregate(aggregate_path, aggregate_rows)
                _atomic_write_json(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["completed_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete_with_failures" if had_failures else "complete"
    manifest["completed_at_utc"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "profile": profile.name,
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "results": str(aggregate_path),
            },
            indent=2,
        )
    )
    return int(had_failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="smoke",
        help=(
            "built-in profile name (smoke, progressive, large-scale) or an explicit "
            "JSON profile path"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new suite directory; defaults to benchmarks/results/suites/<profile>-<UTC>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the profile and print every command without creating files",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help=(
            "retain generated dense inputs and individual run directories after successful "
            "cases; failures are always retained"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ProfileError as exc:
        parser.error(str(exc))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (BENCHMARK_DIR / "results" / "suites" / f"{profile.name}-{timestamp}").resolve()
    )
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_document(
                    profile,
                    output_dir=output_dir,
                    retain_workspaces=args.keep_workspaces,
                ),
                indent=2,
            )
        )
        return 0
    if output_dir.exists():
        parser.error(f"--output-dir already exists: {output_dir}")
    with _sigterm_as_keyboard_interrupt():
        try:
            return run_suite(
                profile,
                output_dir=output_dir,
                retain_workspaces=args.keep_workspaces,
            )
        except KeyboardInterrupt:
            print(f"Interrupted; partial results retained in {output_dir}", file=sys.stderr)
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
