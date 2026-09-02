"""Deterministic writers for SPATHI run artifacts."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

import numpy as np
import pandas as pd

from spathi.diagnostics import WeightDiagnostics
from spathi.inference import EdgeRecord, ModelStat, SkippedTargetRecord
from spathi.weighting import WeightResult

NETWORK_COLUMNS = ("source", "target", "score", "sign", "evidence", "context")
CELL_WEIGHT_COLUMNS = (
    "target_group",
    "cell",
    "cell_group",
    "distance",
    "base_weight",
    "group_size_factor",
    "final_weight",
)
SKIPPED_COLUMNS = ("target_group", "target", "reason", "detail")
MODEL_DIAGNOSTIC_COLUMNS = (
    "target_group",
    "target",
    "status",
    "random_seed",
    "n_samples",
    "n_positive_weight_samples",
    "weight_sum",
    "n_predictors_input",
    "n_predictors_used",
    "discarded_predictors_json",
    "constant_predictors_json",
    "n_edges",
    "importance_sum",
    "fit_seconds",
    "message",
)
WEIGHT_DIAGNOSTIC_COLUMNS = (
    "target_group",
    "n_cells",
    "n_target_cells",
    "total_weight",
    "target_weight",
    "external_weight",
    "target_mass_percent",
    "external_mass_percent",
    "min_weight",
    "max_weight",
    "mean_weight",
    "median_weight",
    "positive_cell_count",
    "effective_sample_size",
    "warnings_json",
    "source_group",
    "source_is_target",
    "source_weight",
    "source_mass_percent",
)


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _open_reproducible_gzip_text(path: Path) -> TextIO:
    """Open deterministic gzip text output without a wall-clock timestamp."""

    binary = gzip.GzipFile(
        filename=path,
        mode="xb",
        compresslevel=6,
        mtime=0,
    )
    return io.TextIOWrapper(binary, encoding="utf-8", newline="")


class _RowWriter(Protocol):
    def writerow(self, row: Iterable[object], /) -> object: ...


class IncrementalRunWriter:
    """Stream canonical run records directly to deterministic artifacts.

    Callers must supply records in global lexical order. Enforcing that contract
    here keeps the writer single-pass and prevents output-sized sort/copy buffers.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._network_handle: TextIO | None = None
        self._weights_handle: TextIO | None = None
        self._skipped_handle: TextIO | None = None
        self._model_diagnostics_handle: TextIO | None = None
        self._weight_diagnostics_handle: TextIO | None = None
        self._network_writer: _RowWriter | None = None
        self._weights_writer: _RowWriter | None = None
        self._skipped_writer: _RowWriter | None = None
        self._model_diagnostics_writer: _RowWriter | None = None
        self._weight_diagnostics_writer: _RowWriter | None = None
        self._last_network_key: tuple[str, str, str] | None = None
        self._last_model_key: tuple[str, str] | None = None
        self._last_skipped_key: tuple[str, str] | None = None
        self._last_weight_group: str | None = None
        self._stack: ExitStack | None = None

    def __enter__(self) -> IncrementalRunWriter:
        if self._stack is not None:
            raise RuntimeError("IncrementalRunWriter is already open")
        self._last_network_key = None
        self._last_model_key = None
        self._last_skipped_key = None
        self._last_weight_group = None
        stack = ExitStack()
        try:
            self._network_handle = stack.enter_context(
                (self.output_dir / "network.csv").open("x", encoding="utf-8", newline="")
            )
            self._weights_handle = stack.enter_context(
                _open_reproducible_gzip_text(self.output_dir / "cell_weights.tsv.gz")
            )
            self._skipped_handle = stack.enter_context(
                (self.output_dir / "skipped_targets.tsv").open("x", encoding="utf-8", newline="")
            )
            self._model_diagnostics_handle = stack.enter_context(
                _open_reproducible_gzip_text(self.output_dir / "model_diagnostics.tsv.gz")
            )
            self._weight_diagnostics_handle = stack.enter_context(
                (self.output_dir / "weight_diagnostics.tsv").open("x", encoding="utf-8", newline="")
            )
            self._network_writer = csv.writer(self._network_handle, lineterminator="\n")
            self._weights_writer = csv.writer(
                self._weights_handle, delimiter="\t", lineterminator="\n"
            )
            self._skipped_writer = csv.writer(
                self._skipped_handle, delimiter="\t", lineterminator="\n"
            )
            self._model_diagnostics_writer = csv.writer(
                self._model_diagnostics_handle, delimiter="\t", lineterminator="\n"
            )
            self._weight_diagnostics_writer = csv.writer(
                self._weight_diagnostics_handle, delimiter="\t", lineterminator="\n"
            )
            self._network_writer.writerow(NETWORK_COLUMNS)
            self._weights_writer.writerow(CELL_WEIGHT_COLUMNS)
            self._skipped_writer.writerow(SKIPPED_COLUMNS)
            self._model_diagnostics_writer.writerow(MODEL_DIAGNOSTIC_COLUMNS)
            self._weight_diagnostics_writer.writerow(WEIGHT_DIAGNOSTIC_COLUMNS)
        except BaseException:
            stack.close()
            self._clear_open_state()
            raise
        self._stack = stack
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        stack = self._stack
        if stack is None:
            return
        try:
            stack.__exit__(exc_type, exc, traceback)
        finally:
            self._clear_open_state()

    def _clear_open_state(self) -> None:
        """Drop all handles and writers after closing or failed initialization."""

        self._stack = None
        self._network_handle = None
        self._weights_handle = None
        self._skipped_handle = None
        self._model_diagnostics_handle = None
        self._weight_diagnostics_handle = None
        self._network_writer = None
        self._weights_writer = None
        self._skipped_writer = None
        self._model_diagnostics_writer = None
        self._weight_diagnostics_writer = None

    def _accept_network_key(self, key: tuple[str, str, str]) -> None:
        if self._last_network_key is not None and key <= self._last_network_key:
            raise ValueError("network records are duplicated or not in canonical global order")
        self._last_network_key = key

    def write_edges(self, records: Iterable[EdgeRecord]) -> int:
        """Append positive edges in canonical ``context, target, source`` order."""

        if self._network_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        count = 0
        for record in records:
            score = float(record.score)
            if not np.isfinite(score) or score <= 0.0:
                raise ValueError("network edge scores must be positive and finite")
            self._accept_network_key((record.context, record.target, record.source))
            self._network_writer.writerow(
                (
                    record.source,
                    record.target,
                    score,
                    record.sign,
                    record.evidence,
                    record.context,
                )
            )
            count += 1
        return count

    def write_model_diagnostics(self, records: Iterable[ModelStat]) -> int:
        """Append per-model audit data, including excluded predictors and seeds."""

        if self._model_diagnostics_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        count = 0
        for record in records:
            key = (record.target_group, record.target)
            if self._last_model_key is not None and key <= self._last_model_key:
                raise ValueError("model diagnostics are duplicated or not in canonical order")
            self._last_model_key = key
            self._model_diagnostics_writer.writerow(
                (
                    record.target_group,
                    record.target,
                    record.status,
                    record.random_seed,
                    record.n_samples,
                    record.n_positive_weight_samples,
                    record.weight_sum,
                    record.n_predictors_input,
                    record.n_predictors_used,
                    json.dumps(
                        record.discarded_predictors,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        record.constant_predictors,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    record.n_edges,
                    record.importance_sum,
                    record.fit_seconds,
                    record.message,
                )
            )
            count += 1
        return count

    def write_weights(self, weights: WeightResult) -> int:
        """Append one target group's cell weights without an intermediate table."""

        if self._weights_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        count = 0
        for cell, cell_group, distance, base, factor, final in zip(
            weights.cells,
            weights.cell_groups,
            weights.distance,
            weights.base_weight,
            weights.group_size_factor,
            weights.final_weight,
            strict=True,
        ):
            self._weights_writer.writerow(
                (
                    weights.target_group,
                    cell,
                    cell_group,
                    _clean_scalar(distance),
                    _clean_scalar(base),
                    _clean_scalar(factor),
                    _clean_scalar(final),
                )
            )
            count += 1
        return count

    def write_weight_diagnostics(self, diagnostics: WeightDiagnostics) -> int:
        """Append one target group's diagnostics in canonical source-group order."""

        if self._weight_diagnostics_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        if (
            self._last_weight_group is not None
            and diagnostics.target_group <= self._last_weight_group
        ):
            raise ValueError("weight diagnostics are duplicated or not in canonical order")
        self._last_weight_group = diagnostics.target_group
        warning_text = json.dumps(
            diagnostics.warnings,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        count = 0
        for source_group in sorted(diagnostics.group_weight_mass):
            self._weight_diagnostics_writer.writerow(
                (
                    diagnostics.target_group,
                    diagnostics.n_cells,
                    diagnostics.n_target_cells,
                    diagnostics.total_weight,
                    diagnostics.target_weight,
                    diagnostics.external_weight,
                    diagnostics.target_mass_percent,
                    diagnostics.external_mass_percent,
                    diagnostics.min_weight,
                    diagnostics.max_weight,
                    diagnostics.mean_weight,
                    diagnostics.median_weight,
                    diagnostics.positive_cell_count,
                    diagnostics.effective_sample_size,
                    warning_text,
                    source_group,
                    source_group == diagnostics.target_group,
                    diagnostics.group_weight_mass[source_group],
                    diagnostics.group_mass_percent[source_group],
                )
            )
            count += 1
        return count

    def write_skipped_targets(self, records: Iterable[SkippedTargetRecord]) -> int:
        """Append skipped-target diagnostics in deterministic order."""

        if self._skipped_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        count = 0
        for record in records:
            key = (record.target_group, record.target)
            if self._last_skipped_key is not None and key <= self._last_skipped_key:
                raise ValueError("skipped targets are duplicated or not in canonical order")
            self._last_skipped_key = key
            self._skipped_writer.writerow(
                (record.target_group, record.target, record.reason, record.detail)
            )
            count += 1
        return count


def write_tsv(frame: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    """Write a table with stable column and row formatting."""

    if columns is not None:
        frame = frame.loc[:, list(columns)]
    frame.to_csv(path, mode="x", sep="\t", index=False, lineterminator="\n")


def write_tsv_gzip(
    frame: pd.DataFrame,
    path: Path,
    columns: Sequence[str] | None = None,
) -> None:
    """Write a table as deterministic gzip-compressed TSV."""

    if columns is not None:
        frame = frame.loc[:, list(columns)]
    with _open_reproducible_gzip_text(path) as handle:
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")


def write_tsv_records(
    records: Iterable[Mapping[str, object]],
    path: Path,
    columns: Sequence[str],
) -> int:
    """Stream canonical mapping records to a deterministic TSV."""

    count = 0
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for record in records:
            writer.writerow(_clean_scalar(record[column]) for column in columns)
            count += 1
    return count


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(cast(Any, value)))
    return value


def write_json(data: Mapping[str, Any], path: Path) -> None:
    """Write human-readable JSON with deterministic key order."""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            _json_compatible(data),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
