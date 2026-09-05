"""Transactional model checkpoints used to resume interrupted SPATHI runs.

Checkpoint files are operational state, not scientific output.  Each completed
``(target group, target gene)`` model is committed independently to a private
SQLite database.  The final ANDREA-compatible artifacts are rebuilt in canonical
order and the checkpoint directory is removed only after atomic publication.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from spathi.inference import (
    EdgeRecord,
    ModelResult,
    ModelStat,
    ModelStatus,
    SkippedTargetRecord,
    SkipReason,
)

_DATABASE_NAME = "checkpoint.sqlite3"
_LOCK_DATABASE_NAME = "run-lock.sqlite3"
_SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm", "-journal")
_MODEL_PAYLOAD_MAGIC = b"SPTHMODL"
_MODEL_PAYLOAD_HEADER = struct.Struct("<8sBBBIQQIIdIIIddQdddIdIQQQQQ")
_MAX_SQLITE_ID = (1 << 63) - 1
_MAX_UINT32 = (1 << 32) - 1
_MODEL_FLAG_TARGET_DETECTED_CELLS = 1 << 0
_MODEL_FLAG_TARGET_DETECTED_FRACTION = 1 << 1
_MODEL_FLAG_TARGET_WEIGHTED_DETECTED_ESS = 1 << 2
_MODEL_FLAG_CONVERGENCE_DELTA = 1 << 3
_MODEL_FLAG_ADAPTIVE_CONVERGED = 1 << 4
_MODEL_FLAG_TARGET_WEIGHTED_DETECTED_FRACTION = 1 << 5
_MODEL_KNOWN_FLAGS = (
    _MODEL_FLAG_TARGET_DETECTED_CELLS
    | _MODEL_FLAG_TARGET_DETECTED_FRACTION
    | _MODEL_FLAG_TARGET_WEIGHTED_DETECTED_ESS
    | _MODEL_FLAG_CONVERGENCE_DELTA
    | _MODEL_FLAG_ADAPTIVE_CONVERGED
    | _MODEL_FLAG_TARGET_WEIGHTED_DETECTED_FRACTION
)
_ID_DTYPES: Mapping[int, np.dtype[Any]] = {
    1: np.dtype("u1"),
    2: np.dtype("<u2"),
    4: np.dtype("<u4"),
    8: np.dtype("<u8"),
}
_STATUS_TO_CODE: Mapping[ModelStatus, int] = {
    "trained": 0,
    "trained_no_positive_importance": 1,
    "insufficient_positive_weight_samples": 2,
    "constant_target": 3,
    "no_predictors_after_self_exclusion": 4,
    "no_variable_predictors": 5,
    "model_fit_failed": 6,
    "invalid_feature_importances": 7,
    "target_not_estimable": 8,
}
_CODE_TO_STATUS = {code: status for status, code in _STATUS_TO_CODE.items()}
CHECKPOINT_OWNED_FILENAMES = frozenset(
    f"{database_name}{suffix}"
    for database_name in (_DATABASE_NAME, _LOCK_DATABASE_NAME)
    for suffix in _SQLITE_SIDECAR_SUFFIXES
)
_SCIENTIFIC_IMPLEMENTATION_FILES = (
    "_workflow.py",
    "centroids.py",
    "checkpoint.py",
    "config.py",
    "diagnostics.py",
    "distances.py",
    "inference.py",
    "io.py",
    "kernels.py",
    "parallel.py",
    "representation.py",
    "resources.py",
    "targeting.py",
    "weighting.py",
)


def _unowned_or_unsafe_entries(directory: Path) -> list[str]:
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.name not in CHECKPOINT_OWNED_FILENAMES or path.is_symlink() or not path.is_file()
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _model_payload_sha256(target_group: str, target: str, payload: bytes) -> str:
    """Bind a model payload checksum to the SQLite primary-key identity."""

    digest = hashlib.sha256()
    for value in (target_group.encode("utf-8"), target.encode("utf-8"), payload):
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
    return digest.hexdigest()


def implementation_fingerprint() -> str:
    """Hash every installed SPATHI Python module for benchmark provenance."""

    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def scientific_implementation_fingerprint() -> str:
    """Hash modules capable of changing models, scientific inputs, or checkpoint payloads."""

    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in _SCIENTIFIC_IMPLEMENTATION_FILES:
        path = package_dir / filename
        if not path.is_file():
            raise RuntimeError(f"scientific implementation module is missing: {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_checkpoint_identity(
    *,
    input_fingerprints: Mapping[str, Mapping[str, str | int]],
    scientific_parameters: Mapping[str, Any],
    target_names: Sequence[str],
    group_names: Sequence[str],
    dependency_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact, path-independent identity required for safe reuse."""

    inputs = {
        name: {
            "size_bytes": int(fingerprint["size_bytes"]),
            "sha256": str(fingerprint["sha256"]),
        }
        for name, fingerprint in sorted(input_fingerprints.items())
    }
    return {
        "scientific_implementation_sha256": scientific_implementation_fingerprint(),
        "dependency_versions": dict(sorted(dependency_versions.items())),
        "inputs": inputs,
        "scientific_parameters": dict(sorted(scientific_parameters.items())),
        "target_names": list(target_names),
        "group_names": list(group_names),
    }


def _ordered_edge_data(
    result: ModelResult,
) -> tuple[tuple[EdgeRecord, ...], tuple[str, str, str] | None]:
    """Extract the compact per-model edge columns from the public record view."""

    ordered_edges = tuple(
        sorted(
            result.edges,
            key=lambda item: (item.context, item.target, item.source),
        )
    )
    if ordered_edges:
        first = ordered_edges[0]
        if any(
            (edge.context, edge.target, edge.sign, edge.evidence)
            != (first.context, first.target, first.sign, first.evidence)
            for edge in ordered_edges
        ):
            raise ValueError("one model checkpoint cannot contain heterogeneous edge metadata")
        metadata = (first.context, first.evidence, first.sign)
    else:
        metadata = None
    return ordered_edges, metadata


def _payload_symbol_values(
    result: ModelResult,
    *,
    edge_data: tuple[tuple[EdgeRecord, ...], tuple[str, str, str] | None] | None = None,
) -> frozenset[str]:
    """Return strings interned once in the checkpoint-wide symbol table."""

    ordered_edges, edge_metadata = _ordered_edge_data(result) if edge_data is None else edge_data
    values: list[str] = [
        result.stat.message,
        *result.stat.discarded_predictors,
        *result.stat.constant_predictors,
        *(edge.source for edge in ordered_edges),
    ]
    if edge_metadata is not None:
        values.extend(edge_metadata)
    if result.skipped is not None:
        values.append(result.skipped.detail)
    if any(type(value) is not str for value in values):
        raise TypeError("checkpoint result strings must be str instances")
    return frozenset(values)


def _validated_uint(value: int, maximum: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    validated = int(value)
    if validated < 0 or validated > maximum:
        raise ValueError(f"{label} exceeds its unsigned integer range")
    return validated


def _finite_float(value: float, *, label: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _symbol_width(symbol_ids: Sequence[int]) -> int:
    maximum = max(symbol_ids, default=0)
    if maximum <= np.iinfo(np.uint8).max:
        return 1
    if maximum <= np.iinfo(np.uint16).max:
        return 2
    if maximum <= np.iinfo(np.uint32).max:
        return 4
    if maximum <= _MAX_SQLITE_ID:
        return 8
    raise ValueError("checkpoint symbol ID exceeds SQLite's positive integer range")


def _symbol_array_bytes(values: Sequence[int], *, width: int) -> bytes:
    dtype = _ID_DTYPES[width]
    return np.asarray(list(values), dtype=dtype).tobytes(order="C")


def _symbol_value_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result_payload(
    result: ModelResult,
    symbol_id: Callable[[str], int],
    *,
    edge_data: tuple[tuple[EdgeRecord, ...], tuple[str, str, str] | None] | None = None,
    symbol_values: frozenset[str] | None = None,
) -> bytes:
    """Encode one model as a compact binary payload with shared string IDs."""

    prepared_edge_data = _ordered_edge_data(result) if edge_data is None else edge_data
    ordered_edges, edge_metadata = prepared_edge_data
    symbols = (
        _payload_symbol_values(result, edge_data=prepared_edge_data)
        if symbol_values is None
        else symbol_values
    )
    symbol_ids: dict[str, int] = {}
    values_by_id: dict[int, str] = {}
    for value in symbols:
        identifier = symbol_id(value)
        identifier = _validated_uint(identifier, _MAX_SQLITE_ID, label="symbol ID")
        if identifier < 1:
            raise ValueError("checkpoint symbol IDs must be positive")
        if identifier in values_by_id and values_by_id[identifier] != value:
            raise ValueError("checkpoint symbol IDs must identify exactly one string")
        symbol_ids[value] = identifier
        values_by_id[identifier] = value

    stat = result.stat
    try:
        status_code = _STATUS_TO_CODE[stat.status]
    except KeyError as exc:  # pragma: no cover - ModelStat enforces this first
        raise ValueError(f"unsupported checkpoint model status: {stat.status!r}") from exc
    random_seed = _validated_uint(stat.random_seed, _MAX_UINT32, label="random_seed")
    n_samples = _validated_uint(stat.n_samples, _MAX_SQLITE_ID, label="n_samples")
    n_positive = _validated_uint(
        stat.n_positive_weight_samples,
        _MAX_SQLITE_ID,
        label="n_positive_weight_samples",
    )
    n_predictors_input = _validated_uint(
        stat.n_predictors_input,
        _MAX_UINT32,
        label="n_predictors_input",
    )
    n_predictors_used = _validated_uint(
        stat.n_predictors_used,
        _MAX_UINT32,
        label="n_predictors_used",
    )
    n_estimators_fitted = _validated_uint(
        stat.n_estimators_fitted,
        _MAX_UINT32,
        label="n_estimators_fitted",
    )
    convergence_checks = _validated_uint(
        stat.convergence_checks,
        _MAX_UINT32,
        label="convergence_checks",
    )
    flags = 0
    if stat.target_detected_cells is None:
        target_detected_cells = 0
    else:
        flags |= _MODEL_FLAG_TARGET_DETECTED_CELLS
        target_detected_cells = _validated_uint(
            stat.target_detected_cells,
            _MAX_SQLITE_ID,
            label="target_detected_cells",
        )
    optional_floats = (
        (
            stat.target_detected_fraction,
            _MODEL_FLAG_TARGET_DETECTED_FRACTION,
            "target_detected_fraction",
        ),
        (
            stat.target_weighted_detected_ess,
            _MODEL_FLAG_TARGET_WEIGHTED_DETECTED_ESS,
            "target_weighted_detected_ess",
        ),
        (
            stat.target_weighted_detected_fraction,
            _MODEL_FLAG_TARGET_WEIGHTED_DETECTED_FRACTION,
            "target_weighted_detected_fraction",
        ),
        (stat.convergence_delta, _MODEL_FLAG_CONVERGENCE_DELTA, "convergence_delta"),
    )
    encoded_optional_floats: list[float] = []
    for numeric_value, flag, label in optional_floats:
        if numeric_value is None:
            encoded_optional_floats.append(0.0)
        else:
            flags |= flag
            encoded_optional_floats.append(_finite_float(numeric_value, label=label))
    (
        target_detected_fraction,
        target_weighted_detected_ess,
        target_weighted_detected_fraction,
        convergence_delta,
    ) = encoded_optional_floats
    if stat.adaptive_converged:
        flags |= _MODEL_FLAG_ADAPTIVE_CONVERGED
    discarded_ids = tuple(symbol_ids[value] for value in stat.discarded_predictors)
    constant_ids = tuple(symbol_ids[value] for value in stat.constant_predictors)
    source_ids = tuple(symbol_ids[edge.source] for edge in ordered_edges)
    width = _symbol_width((*symbol_ids.values(),))
    skipped_detail_id = 0 if result.skipped is None else symbol_ids[result.skipped.detail]
    if edge_metadata is None:
        edge_context_id = edge_evidence_id = edge_sign_id = 0
    else:
        edge_context_id, edge_evidence_id, edge_sign_id = (
            symbol_ids[value] for value in edge_metadata
        )
    header = _MODEL_PAYLOAD_HEADER.pack(
        _MODEL_PAYLOAD_MAGIC,
        status_code,
        width,
        flags,
        random_seed,
        n_samples,
        n_positive,
        n_predictors_input,
        n_predictors_used,
        _finite_float(stat.weight_sum, label="weight_sum"),
        len(discarded_ids),
        len(constant_ids),
        len(ordered_edges),
        _finite_float(stat.importance_sum, label="importance_sum"),
        _finite_float(stat.fit_seconds, label="fit_seconds"),
        target_detected_cells,
        target_detected_fraction,
        target_weighted_detected_ess,
        target_weighted_detected_fraction,
        n_estimators_fitted,
        convergence_delta,
        convergence_checks,
        symbol_ids[stat.message],
        skipped_detail_id,
        edge_context_id,
        edge_evidence_id,
        edge_sign_id,
    )
    scores = np.fromiter(
        (_finite_float(edge.score, label="edge score") for edge in ordered_edges),
        dtype="<f8",
        count=len(ordered_edges),
    )
    body = b"".join(
        (
            _symbol_array_bytes(discarded_ids, width=width),
            _symbol_array_bytes(constant_ids, width=width),
            _symbol_array_bytes(source_ids, width=width),
            scores.tobytes(order="C"),
        )
    )
    return header + zlib.compress(body, level=1)


def _result_from_payload(
    payload: bytes,
    *,
    target_group: str,
    target: str,
    symbols: Mapping[int, str],
) -> ModelResult:
    """Decode and fully validate one binary model payload."""

    try:
        if len(payload) < _MODEL_PAYLOAD_HEADER.size:
            raise ValueError("model payload header is truncated")
        (
            magic,
            status_code,
            width,
            flags,
            random_seed,
            n_samples,
            n_positive_weight_samples,
            n_predictors_input,
            n_predictors_used,
            weight_sum,
            n_discarded,
            n_constant,
            n_edges,
            importance_sum,
            fit_seconds,
            encoded_target_detected_cells,
            encoded_target_detected_fraction,
            encoded_target_weighted_detected_ess,
            encoded_target_weighted_detected_fraction,
            n_estimators_fitted,
            encoded_convergence_delta,
            convergence_checks,
            message_id,
            skipped_detail_id,
            edge_context_id,
            edge_evidence_id,
            edge_sign_id,
        ) = _MODEL_PAYLOAD_HEADER.unpack_from(payload)
        if magic != _MODEL_PAYLOAD_MAGIC:
            raise ValueError("model payload magic is invalid")
        if flags & ~_MODEL_KNOWN_FLAGS:
            raise ValueError("model payload contains unknown flags")
        try:
            id_dtype = _ID_DTYPES[width]
        except KeyError as exc:
            raise ValueError(f"unsupported model payload symbol width: {width}") from exc
        try:
            status = _CODE_TO_STATUS[status_code]
        except KeyError as exc:
            raise ValueError(f"unknown model status code: {status_code}") from exc
        for label, value in (
            ("weight_sum", weight_sum),
            ("importance_sum", importance_sum),
            ("fit_seconds", fit_seconds),
            ("target_detected_fraction", encoded_target_detected_fraction),
            ("target_weighted_detected_ess", encoded_target_weighted_detected_ess),
            (
                "target_weighted_detected_fraction",
                encoded_target_weighted_detected_fraction,
            ),
            ("convergence_delta", encoded_convergence_delta),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if n_samples > _MAX_SQLITE_ID or n_positive_weight_samples > _MAX_SQLITE_ID:
            raise ValueError("model sample count exceeds the codec range")
        if encoded_target_detected_cells > _MAX_SQLITE_ID:
            raise ValueError("target_detected_cells exceeds the codec range")

        def optional_float(encoded: float, flag: int, *, label: str) -> float | None:
            if flags & flag:
                return encoded
            if encoded != 0.0:
                raise ValueError(f"absent {label} must use the canonical zero placeholder")
            return None

        if flags & _MODEL_FLAG_TARGET_DETECTED_CELLS:
            target_detected_cells: int | None = encoded_target_detected_cells
        else:
            if encoded_target_detected_cells != 0:
                raise ValueError(
                    "absent target_detected_cells must use the canonical zero placeholder"
                )
            target_detected_cells = None
        target_detected_fraction = optional_float(
            encoded_target_detected_fraction,
            _MODEL_FLAG_TARGET_DETECTED_FRACTION,
            label="target_detected_fraction",
        )
        target_weighted_detected_ess = optional_float(
            encoded_target_weighted_detected_ess,
            _MODEL_FLAG_TARGET_WEIGHTED_DETECTED_ESS,
            label="target_weighted_detected_ess",
        )
        target_weighted_detected_fraction = optional_float(
            encoded_target_weighted_detected_fraction,
            _MODEL_FLAG_TARGET_WEIGHTED_DETECTED_FRACTION,
            label="target_weighted_detected_fraction",
        )
        convergence_delta = optional_float(
            encoded_convergence_delta,
            _MODEL_FLAG_CONVERGENCE_DELTA,
            label="convergence_delta",
        )
        adaptive_converged = bool(flags & _MODEL_FLAG_ADAPTIVE_CONVERGED)

        trained = status in {"trained", "trained_no_positive_importance"}
        if status == "trained":
            skipped_reason: SkipReason | None = None
        elif status == "trained_no_positive_importance":
            skipped_reason = "no_positive_feature_importance"
        else:
            skipped_reason = cast(SkipReason, status)
        if (skipped_reason is None) != (skipped_detail_id == 0):
            raise ValueError("model status and skipped-detail presence disagree")
        edge_metadata_ids = (edge_context_id, edge_evidence_id, edge_sign_id)
        if n_edges:
            if any(identifier == 0 for identifier in edge_metadata_ids):
                raise ValueError("model edges require complete common metadata")
        elif any(identifier != 0 for identifier in edge_metadata_ids):
            raise ValueError("an edgeless model cannot contain edge metadata")
        if message_id == 0:
            raise ValueError("model message must reference a symbol")

        identifier_count = n_discarded + n_constant + n_edges
        expected_body_size = identifier_count * width + n_edges * np.dtype("<f8").itemsize
        decompressor = zlib.decompressobj()
        body = decompressor.decompress(
            payload[_MODEL_PAYLOAD_HEADER.size :],
            expected_body_size + 1,
        )
        if (
            len(body) != expected_body_size
            or not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
        ):
            raise ValueError("decompressed model payload size or stream does not match its counts")
        offset = 0

        def read_ids(count: int) -> tuple[int, ...]:
            nonlocal offset
            values = np.frombuffer(body, dtype=id_dtype, count=count, offset=offset)
            offset += count * width
            return tuple(map(int, values))

        discarded_ids = read_ids(n_discarded)
        constant_ids = read_ids(n_constant)
        edge_source_ids = read_ids(n_edges)
        scores = np.frombuffer(body, dtype="<f8", count=n_edges, offset=offset)
        if not np.isfinite(scores).all() or np.any(scores <= 0.0):
            raise ValueError("model edge scores must be positive and finite")
        edge_values = tuple(zip(edge_source_ids, map(float, scores), strict=True))

        referenced_ids = {
            *discarded_ids,
            *constant_ids,
            message_id,
            *(identifier for identifier, _score in edge_values),
        }
        if skipped_detail_id:
            referenced_ids.add(skipped_detail_id)
        for identifier in edge_metadata_ids:
            if identifier:
                referenced_ids.add(identifier)
        if any(identifier < 1 for identifier in referenced_ids):
            raise ValueError("model payload contains a non-positive symbol ID")
        try:
            bindings = {identifier: symbols[identifier] for identifier in referenced_ids}
        except KeyError as exc:
            raise ValueError(f"model payload references unknown symbol ID {exc.args[0]}") from exc

        discarded = tuple(bindings[identifier] for identifier in discarded_ids)
        constant = tuple(bindings[identifier] for identifier in constant_ids)
        message = bindings[message_id]
        if skipped_reason is None:
            skipped = None
        else:
            skipped = SkippedTargetRecord(
                target_group=target_group,
                target=target,
                reason=skipped_reason,
                detail=bindings[skipped_detail_id],
            )
        if edge_values:
            context = bindings[edge_context_id]
            evidence = bindings[edge_evidence_id]
            sign = bindings[edge_sign_id]
            edges = tuple(
                EdgeRecord(
                    source=bindings[source_id],
                    target=target,
                    score=score,
                    sign=sign,
                    evidence=evidence,
                    context=context,
                )
                for source_id, score in edge_values
            )
            if tuple(edge.source for edge in edges) != tuple(sorted(edge.source for edge in edges)):
                raise ValueError("model payload edges are not in canonical source order")
        else:
            edges = ()
        stat = ModelStat(
            target_group=target_group,
            target=target,
            status=status,
            random_seed=random_seed,
            n_samples=n_samples,
            n_positive_weight_samples=n_positive_weight_samples,
            weight_sum=weight_sum,
            n_predictors_input=n_predictors_input,
            n_predictors_used=n_predictors_used,
            discarded_predictors=discarded,
            constant_predictors=constant,
            n_edges=n_edges,
            importance_sum=importance_sum,
            fit_seconds=fit_seconds,
            target_detected_cells=target_detected_cells,
            target_detected_fraction=target_detected_fraction,
            target_weighted_detected_ess=target_weighted_detected_ess,
            target_weighted_detected_fraction=target_weighted_detected_fraction,
            n_estimators_fitted=n_estimators_fitted,
            adaptive_converged=adaptive_converged,
            convergence_delta=convergence_delta,
            convergence_checks=convergence_checks,
            message=message,
        )
        result = ModelResult(edges=edges, skipped=skipped, stat=stat, trained=trained)
    except (KeyError, TypeError, ValueError, struct.error, UnicodeError, zlib.error) as exc:
        raise RuntimeError("checkpoint contains an invalid model-result payload") from exc
    return result


def _weight_fingerprint(weights: ArrayLike) -> str:
    values = np.asarray(weights, dtype="<f8")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("checkpoint weights must be a finite one-dimensional vector")
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape[0]).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


class ModelCheckpoint:
    """One validated SQLite checkpoint with per-model atomic commits."""

    def __init__(
        self,
        directory: Path,
        *,
        identity: Mapping[str, Any],
        resume: bool,
    ) -> None:
        self.directory = Path(directory)
        self.database_path = self.directory / _DATABASE_NAME
        lock_database_path = self.directory / _LOCK_DATABASE_NAME
        if self.directory.is_symlink():
            raise RuntimeError(f"checkpoint directory may not be a symbolic link: {self.directory}")
        if not self.directory.is_dir():
            raise RuntimeError(f"checkpoint directory does not exist: {self.directory}")
        if self.database_path.is_symlink() or lock_database_path.is_symlink():
            raise RuntimeError("checkpoint database files may not be symbolic links")
        if not resume and any(self.directory.iterdir()):
            raise RuntimeError(f"new checkpoint directory must be empty: {self.directory}")
        unexpected_entries = _unowned_or_unsafe_entries(self.directory)
        if unexpected_entries:
            raise RuntimeError(
                "checkpoint directory contains unowned files and was preserved: "
                + ", ".join(unexpected_entries[:5])
            )
        if resume and not self.database_path.is_file():
            raise RuntimeError(f"resume requires an existing {_DATABASE_NAME}: {self.directory}")

        self._lock_connection: sqlite3.Connection | None = None
        try:
            # A separate SQLite database provides a portable advisory run lock.
            # The exclusive transaction is released automatically if the
            # process dies, unlike a create-once PID file that can become stale.
            self._lock_connection = sqlite3.connect(
                lock_database_path,
                timeout=0.0,
                isolation_level=None,
            )
            self._lock_connection.execute("BEGIN EXCLUSIVE")
            unexpected_after_lock = _unowned_or_unsafe_entries(self.directory)
            if unexpected_after_lock:
                raise RuntimeError(
                    "checkpoint directory changed while being opened and was preserved: "
                    + ", ".join(unexpected_after_lock[:5])
                )
            database_existed = self.database_path.is_file()
            if resume and not database_existed:
                raise RuntimeError(
                    f"resume requires an existing {_DATABASE_NAME}: {self.directory}"
                )
            if not resume and database_existed:
                raise FileExistsError(
                    f"checkpoint already exists; pass resume=True or remove it: {self.directory}"
                )
            self._connection = sqlite3.connect(self.database_path, timeout=30.0)
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema()
            if resume and database_existed:
                if not self._has_identity():
                    raise RuntimeError("existing checkpoint database has no run identity")
                self._validate_identity(identity)
            else:
                self._write_identity(identity)
            self._set_expected_identity(identity)
            self._load_symbols()
        except BaseException as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            if self._lock_connection is not None:
                self._lock_connection.close()
                self._lock_connection = None
            if isinstance(exc, RuntimeError):
                raise
            if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
                raise RuntimeError(f"checkpoint is already in use: {self.directory}") from exc
            if not isinstance(exc, (OSError, sqlite3.DatabaseError)):
                raise
            raise RuntimeError(f"could not open checkpoint database: {self.database_path}") from exc

    def _set_expected_identity(self, identity: Mapping[str, Any]) -> None:
        try:
            groups = tuple(str(value) for value in identity["group_names"])
            targets = tuple(str(value) for value in identity["target_names"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "checkpoint identity must define group_names and target_names"
            ) from exc
        if (
            not groups
            or not targets
            or len(set(groups)) != len(groups)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("checkpoint group and target identities must be non-empty and unique")
        self._expected_groups = frozenset(groups)
        self._expected_targets = frozenset(targets)
        self._expected_model_count = len(groups) * len(targets)

    def _is_expected_key(self, target_group: str, target: str) -> bool:
        return target_group in self._expected_groups and target in self._expected_targets

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id INTEGER PRIMARY KEY,
                    value TEXT NOT NULL UNIQUE,
                    value_sha256 TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_results (
                    target_group TEXT NOT NULL,
                    target TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY (target_group, target)
                ) WITHOUT ROWID
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS group_weights (
                    target_group TEXT PRIMARY KEY,
                    weight_sha256 TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )

    def _load_symbols(self) -> None:
        """Load and validate the checkpoint-wide string dictionary."""

        try:
            rows = self._connection.execute(
                "SELECT symbol_id, value, value_sha256 FROM symbols ORDER BY symbol_id"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not read checkpoint symbol table") from exc
        by_id: dict[int, str] = {}
        by_value: dict[str, int] = {}
        for raw_identifier, raw_value, raw_sha256 in rows:
            if (
                isinstance(raw_identifier, bool)
                or not isinstance(raw_identifier, int)
                or raw_identifier < 1
                or raw_identifier > _MAX_SQLITE_ID
                or not isinstance(raw_value, str)
                or not isinstance(raw_sha256, str)
                or raw_sha256 != _symbol_value_sha256(raw_value)
                or raw_identifier in by_id
                or raw_value in by_value
            ):
                raise RuntimeError("checkpoint symbol table is corrupted")
            by_id[raw_identifier] = raw_value
            by_value[raw_value] = raw_identifier
        self._symbols_by_id = by_id
        self._symbol_ids_by_value = by_value

    def _has_identity(self) -> bool:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'identity'"
        ).fetchone()
        return row is not None

    def _write_identity(self, identity: Mapping[str, Any]) -> None:
        serialized = _canonical_json(identity)
        signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('identity', ?)",
                (serialized,),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('identity_sha256', ?)",
                (signature,),
            )

    def _validate_identity(self, identity: Mapping[str, Any]) -> None:
        expected = _canonical_json(identity)
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'identity'"
        ).fetchone()
        signature_row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'identity_sha256'"
        ).fetchone()
        if row is None or signature_row is None:
            raise RuntimeError("checkpoint metadata is incomplete")
        observed = str(row[0])
        observed_signature = str(signature_row[0])
        if hashlib.sha256(observed.encode("utf-8")).hexdigest() != observed_signature:
            raise RuntimeError("checkpoint identity metadata is corrupted")
        if observed != expected:
            raise RuntimeError(
                "checkpoint identity does not match the current inputs, parameters, "
                "dependencies, or SPATHI implementation"
            )

    def completion_counts_by_group(self) -> dict[str, int]:
        """Stream and validate committed identities using only group-sized memory."""

        counts = {group: 0 for group in self._expected_groups}
        try:
            cursor = self._connection.execute(
                "SELECT target_group, target FROM model_results ORDER BY target_group, target"
            )
            for raw_group, raw_target in cursor:
                group = str(raw_group)
                target = str(raw_target)
                if not self._is_expected_key(group, target):
                    raise RuntimeError(
                        "checkpoint contains a model identity outside its manifest: "
                        f"({group!r}, {target!r})"
                    )
                counts[group] += 1
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not read completed models from checkpoint") from exc
        return counts

    def completed_keys_for_groups(
        self,
        target_groups: Sequence[str],
    ) -> frozenset[tuple[str, str]]:
        """Load committed identities only for the active bounded group batch."""

        groups = tuple(str(group) for group in target_groups)
        if not groups or any(not group for group in groups) or len(set(groups)) != len(groups):
            raise ValueError("target_groups must contain unique non-empty identifiers")
        unknown = set(groups).difference(self._expected_groups)
        if unknown:
            raise ValueError(f"unknown checkpoint target groups: {sorted(unknown)!r}")
        keys: set[tuple[str, str]] = set()
        try:
            for group in groups:
                cursor = self._connection.execute(
                    "SELECT target FROM model_results WHERE target_group = ? ORDER BY target",
                    (group,),
                )
                for (raw_target,) in cursor:
                    target = str(raw_target)
                    if target not in self._expected_targets:
                        raise RuntimeError(
                            "checkpoint contains a model identity outside its manifest: "
                            f"({group!r}, {target!r})"
                        )
                    keys.add((group, target))
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not read completed models from checkpoint") from exc
        return frozenset(keys)

    @property
    def has_completed_models(self) -> bool:
        """Return whether at least one model result has been committed."""

        try:
            return (
                self._connection.execute("SELECT 1 FROM model_results LIMIT 1").fetchone()
                is not None
            )
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not inspect completed models in checkpoint") from exc

    def validate_or_record_weights(self, target_group: str, weights: ArrayLike) -> None:
        """Bind one group's committed models to the exact recalculated weights."""

        group = str(target_group)
        if group not in self._expected_groups:
            raise ValueError(f"unknown checkpoint target group: {group!r}")
        observed = _weight_fingerprint(weights)
        try:
            row = self._connection.execute(
                "SELECT weight_sha256 FROM group_weights WHERE target_group = ?",
                (group,),
            ).fetchone()
            if row is None:
                model_count = int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM model_results WHERE target_group = ?",
                        (group,),
                    ).fetchone()[0]
                )
                if model_count:
                    raise RuntimeError(
                        f"checkpoint has models for group {group!r} but no weight fingerprint"
                    )
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO group_weights(target_group, weight_sha256) VALUES(?, ?)",
                        (group, observed),
                    )
            elif str(row[0]) != observed:
                raise RuntimeError(
                    f"recalculated weights do not match checkpointed group {group!r}"
                )
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(
                f"could not validate checkpoint weights for group {group!r}"
            ) from exc

    def record_result(self, result: ModelResult) -> None:
        """Commit one complete result or reject an accidental duplicate."""

        group = result.stat.target_group
        target = result.stat.target
        key = (group, target)
        if not self._is_expected_key(group, target):
            raise ValueError(f"model result is outside the checkpoint identity: {key!r}")
        try:
            if (
                self._connection.execute(
                    "SELECT 1 FROM model_results WHERE target_group = ? AND target = ?",
                    (group, target),
                ).fetchone()
                is not None
            ):
                raise RuntimeError(f"checkpoint already contains model ({group!r}, {target!r})")

            edge_data = _ordered_edge_data(result)
            symbol_values = _payload_symbol_values(result, edge_data=edge_data)
            pending_symbols: dict[str, int] = {}
            with self._connection:
                for value in sorted(symbol_values.difference(self._symbol_ids_by_value)):
                    cursor = self._connection.execute(
                        "INSERT INTO symbols(value, value_sha256) VALUES(?, ?)",
                        (value, _symbol_value_sha256(value)),
                    )
                    raw_identifier = cursor.lastrowid
                    if raw_identifier is None:
                        raise RuntimeError("checkpoint symbol insert returned no identifier")
                    identifier = int(raw_identifier)
                    if identifier < 1 or identifier > _MAX_SQLITE_ID:
                        raise RuntimeError("checkpoint symbol ID exceeds the codec range")
                    pending_symbols[value] = identifier

                def symbol_id(value: str) -> int:
                    if value in pending_symbols:
                        return pending_symbols[value]
                    return self._symbol_ids_by_value[value]

                payload = _result_payload(
                    result,
                    symbol_id,
                    edge_data=edge_data,
                    symbol_values=symbol_values,
                )
                payload_sha256 = _model_payload_sha256(group, target, payload)
                self._connection.execute(
                    """
                    INSERT INTO model_results(
                        target_group, target, payload, payload_sha256
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (group, target, payload, payload_sha256),
                )
            for value, identifier in pending_symbols.items():
                self._symbol_ids_by_value[value] = identifier
                self._symbols_by_id[identifier] = value
        except RuntimeError:
            raise
        except (KeyError, sqlite3.IntegrityError, sqlite3.DatabaseError) as exc:
            raise RuntimeError(
                f"could not commit model ({group!r}, {target!r}) to checkpoint"
            ) from exc

    def iter_results(self) -> Iterator[ModelResult]:
        """Yield every committed result in final deterministic model order."""

        try:
            cursor = self._connection.execute(
                """
                SELECT target_group, target, payload, payload_sha256
                FROM model_results
                ORDER BY target_group, target
                """
            )
            for target_group, target, payload, payload_sha256 in cursor:
                payload_bytes = bytes(payload)
                if _model_payload_sha256(str(target_group), str(target), payload_bytes) != str(
                    payload_sha256
                ):
                    raise RuntimeError(
                        f"checkpoint payload checksum failed for ({target_group!r}, {target!r})"
                    )
                result = _result_from_payload(
                    payload_bytes,
                    target_group=str(target_group),
                    target=str(target),
                    symbols=self._symbols_by_id,
                )
                yield result
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not read model results from checkpoint") from exc

    def validate_complete(self) -> None:
        """Verify that every expected model and group weight is committed."""

        try:
            completed_count = 0
            cursor = self._connection.execute("SELECT target_group, target FROM model_results")
            for target_group, target in cursor:
                completed_count += 1
                if not self._is_expected_key(str(target_group), str(target)):
                    raise RuntimeError(
                        "checkpoint contains a model identity outside its manifest: "
                        f"({target_group!r}, {target!r})"
                    )
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not validate completed checkpoint models") from exc
        if completed_count != self._expected_model_count:
            raise RuntimeError(
                "cannot complete checkpoint before every expected model is committed; "
                f"completed={completed_count}, expected={self._expected_model_count}"
            )
        try:
            weight_groups = frozenset(
                str(row[0])
                for row in self._connection.execute(
                    "SELECT target_group FROM group_weights"
                ).fetchall()
            )
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not validate checkpoint weight groups") from exc
        if weight_groups != self._expected_groups:
            missing_groups = sorted(self._expected_groups.difference(weight_groups))
            raise RuntimeError(
                "cannot complete checkpoint before every group weight is validated; "
                f"missing={missing_groups}"
            )

    def close(self) -> None:
        self._connection.close()
        if self._lock_connection is not None:
            self._lock_connection.close()
            self._lock_connection = None

    def __enter__(self) -> ModelCheckpoint:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "CHECKPOINT_OWNED_FILENAMES",
    "ModelCheckpoint",
    "build_checkpoint_identity",
    "implementation_fingerprint",
    "scientific_implementation_fingerprint",
]
