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
import zlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from spathi.inference import (
    EdgeRecord,
    ModelResult,
    ModelStat,
    SkippedTargetRecord,
)

CHECKPOINT_SCHEMA_VERSION = 1
_DATABASE_NAME = "checkpoint.sqlite3"
_LOCK_DATABASE_NAME = "run-lock.sqlite3"
_SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm", "-journal")
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
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scientific_implementation_sha256": scientific_implementation_fingerprint(),
        "dependency_versions": dict(sorted(dependency_versions.items())),
        "inputs": inputs,
        "scientific_parameters": dict(sorted(scientific_parameters.items())),
        "target_names": list(target_names),
        "group_names": list(group_names),
    }


def _result_payload(result: ModelResult) -> bytes:
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
        edge_payload: dict[str, Any] = {
            "context": first.context,
            "evidence": first.evidence,
            "sign": first.sign,
            "values": [[edge.source, edge.score] for edge in ordered_edges],
        }
    else:
        edge_payload = {"context": None, "evidence": None, "sign": None, "values": []}
    payload = {
        "edges": edge_payload,
        "skipped": None if result.skipped is None else result.skipped.to_dict(),
        "stat": result.stat.to_dict(),
        "trained": result.trained,
    }
    return zlib.compress(_canonical_json(payload).encode("utf-8"), level=1)


def _result_from_payload(payload: bytes) -> ModelResult:
    try:
        raw = json.loads(zlib.decompress(payload).decode("utf-8"))
        edge_payload = raw["edges"]
        stat_target = raw["stat"]["target"]
        edge_context = edge_payload["context"]
        edge_evidence = edge_payload["evidence"]
        edge_sign = edge_payload["sign"]
        edges = tuple(
            sorted(
                (
                    EdgeRecord(
                        source=str(source),
                        target=str(stat_target),
                        score=float(score),
                        sign=str(edge_sign),
                        evidence=str(edge_evidence),
                        context=str(edge_context),
                    )
                    for source, score in edge_payload["values"]
                ),
                key=lambda item: (item.context, item.target, item.source),
            )
        )
        skipped_raw = raw["skipped"]
        skipped = None if skipped_raw is None else SkippedTargetRecord(**skipped_raw)
        stat_raw = dict(raw["stat"])
        stat_raw["discarded_predictors"] = tuple(stat_raw["discarded_predictors"])
        stat_raw["constant_predictors"] = tuple(stat_raw["constant_predictors"])
        stat = ModelStat(**stat_raw)
        trained = raw["trained"]
        if not isinstance(trained, bool):
            raise TypeError("trained is not a boolean")
        result = ModelResult(edges=edges, skipped=skipped, stat=stat, trained=trained)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, zlib.error) as exc:
        raise RuntimeError("checkpoint contains an invalid model-result payload") from exc
    return result


def _validate_result_identity(
    result: ModelResult,
    *,
    target_group: str,
    target: str,
) -> None:
    if (result.stat.target_group, result.stat.target) != (target_group, target):
        raise RuntimeError("checkpoint SQL key does not match its model-stat payload")


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

    @property
    def completed_keys(self) -> frozenset[tuple[str, str]]:
        """Return the exact model identities already committed."""

        try:
            rows = self._connection.execute(
                "SELECT target_group, target FROM model_results"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("could not read completed models from checkpoint") from exc
        keys = frozenset((str(group), str(target)) for group, target in rows)
        unexpected = {key for key in keys if not self._is_expected_key(*key)}
        if unexpected:
            raise RuntimeError(
                f"checkpoint contains model identities outside its manifest: {sorted(unexpected)[:5]}"
            )
        return keys

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
        _validate_result_identity(result, target_group=group, target=target)
        payload = _result_payload(result)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO model_results(
                        target_group, target, payload, payload_sha256
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (group, target, payload, payload_sha256),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"checkpoint already contains model ({group!r}, {target!r})"
            ) from exc
        except sqlite3.DatabaseError as exc:
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
                if hashlib.sha256(payload_bytes).hexdigest() != str(payload_sha256):
                    raise RuntimeError(
                        f"checkpoint payload checksum failed for ({target_group!r}, {target!r})"
                    )
                result = _result_from_payload(payload_bytes)
                _validate_result_identity(
                    result,
                    target_group=str(target_group),
                    target=str(target),
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
    "CHECKPOINT_SCHEMA_VERSION",
    "ModelCheckpoint",
    "build_checkpoint_identity",
    "implementation_fingerprint",
    "scientific_implementation_fingerprint",
]
