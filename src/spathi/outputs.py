"""Deterministic writers for SPATHI run artifacts."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import pandas as pd

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
    "discarded_predictors",
    "constant_predictors",
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
    "warnings",
    "source_group",
    "source_is_target",
    "source_weight",
    "source_mass_percent",
)


def create_output_directory(output_dir: Path) -> Path:
    """Create a new output directory, refusing every pre-existing path."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output path already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _record_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(cast(Any, record))
    if hasattr(record, "_asdict"):
        return record._asdict()
    if hasattr(record, "__dict__"):
        return vars(record)
    raise TypeError(f"Cannot serialize record of type {type(record).__name__}")


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
        mode="wb",
        compresslevel=6,
        mtime=0,
    )
    return io.TextIOWrapper(binary, encoding="utf-8", newline="")


def _edge_row(record: Any) -> dict[str, Any] | None:
    """Normalize one positive edge record, returning ``None`` for zero scores."""

    row = dict(_record_mapping(record))
    if float(row["score"]) <= 0:
        return None
    row.setdefault("sign", "?")
    return row


def _edge_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["context"]), str(row["target"]), str(row["source"]))


class IncrementalRunWriter:
    """Write large network and cell-weight tables one target group at a time."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._network_handle: TextIO | None = None
        self._weights_handle: TextIO | None = None
        self._skipped_handle: TextIO | None = None
        self._model_diagnostics_handle: TextIO | None = None
        self._weight_diagnostics_handle: TextIO | None = None
        self._network_writer: csv.DictWriter | None = None
        self._weights_writer: csv.DictWriter | None = None
        self._weights_row_writer: Any = None
        self._skipped_writer: csv.DictWriter | None = None
        self._model_diagnostics_writer: csv.DictWriter | None = None
        self._weight_diagnostics_writer: csv.DictWriter | None = None

    def __enter__(self) -> IncrementalRunWriter:
        self._network_handle = (self.output_dir / "network.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._weights_handle = _open_reproducible_gzip_text(self.output_dir / "cell_weights.tsv.gz")
        self._skipped_handle = (self.output_dir / "skipped_targets.tsv").open(
            "w", encoding="utf-8", newline=""
        )
        self._model_diagnostics_handle = _open_reproducible_gzip_text(
            self.output_dir / "model_diagnostics.tsv.gz"
        )
        self._weight_diagnostics_handle = (self.output_dir / "weight_diagnostics.tsv").open(
            "w", encoding="utf-8", newline=""
        )
        assert self._network_handle is not None
        assert self._weights_handle is not None
        assert self._skipped_handle is not None
        assert self._model_diagnostics_handle is not None
        assert self._weight_diagnostics_handle is not None
        self._network_writer = csv.DictWriter(
            self._network_handle,
            fieldnames=NETWORK_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._weights_writer = csv.DictWriter(
            self._weights_handle,
            fieldnames=CELL_WEIGHT_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._weights_row_writer = csv.writer(
            self._weights_handle,
            delimiter="\t",
            lineterminator="\n",
        )
        self._skipped_writer = csv.DictWriter(
            self._skipped_handle,
            fieldnames=SKIPPED_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._model_diagnostics_writer = csv.DictWriter(
            self._model_diagnostics_handle,
            fieldnames=MODEL_DIAGNOSTIC_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._weight_diagnostics_writer = csv.DictWriter(
            self._weight_diagnostics_handle,
            fieldnames=WEIGHT_DIAGNOSTIC_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._network_writer.writeheader()
        self._weights_writer.writeheader()
        self._skipped_writer.writeheader()
        self._model_diagnostics_writer.writeheader()
        self._weight_diagnostics_writer.writeheader()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in (
            self._network_handle,
            self._weights_handle,
            self._skipped_handle,
            self._model_diagnostics_handle,
            self._weight_diagnostics_handle,
        ):
            if handle is not None:
                handle.close()

    def write_edges(self, records: Iterable[Any]) -> int:
        """Append positive-score edges in deterministic target/source order."""

        if self._network_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")

        # InferenceResult exposes an already sorted tuple.  Verify and stream
        # sequences directly so a large network is not duplicated merely for a
        # redundant sort.  Arbitrary iterables retain the safe sorting path.
        if isinstance(records, Sequence):
            already_ordered = True
            positive_count = 0
            previous_key: tuple[str, str, str] | None = None
            for record in records:
                row = _edge_row(record)
                if row is None:
                    continue
                key = _edge_sort_key(row)
                if previous_key is not None and key < previous_key:
                    already_ordered = False
                    break
                previous_key = key
                positive_count += 1
            if already_ordered:
                for record in records:
                    row = _edge_row(record)
                    if row is not None:
                        self._network_writer.writerow(
                            {column: _clean_scalar(row[column]) for column in NETWORK_COLUMNS}
                        )
                return positive_count

        rows: list[dict[str, Any]] = []
        for record in records:
            row = _edge_row(record)
            if row is None:
                continue
            rows.append(row)
        rows.sort(key=_edge_sort_key)
        for row in rows:
            self._network_writer.writerow(
                {column: _clean_scalar(row[column]) for column in NETWORK_COLUMNS}
            )
        return len(rows)

    def write_model_diagnostics(self, records: Iterable[Any]) -> int:
        """Append per-model audit data, including excluded predictors and seeds."""

        if self._model_diagnostics_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        rows = [dict(_record_mapping(record)) for record in records]
        rows.sort(key=lambda row: (str(row["target_group"]), str(row["target"])))
        for row in rows:
            for column in ("discarded_predictors", "constant_predictors"):
                value = row.get(column, ())
                if isinstance(value, (list, tuple)):
                    row[column] = ";".join(map(str, value))
            self._model_diagnostics_writer.writerow(
                {column: _clean_scalar(row.get(column, "")) for column in MODEL_DIAGNOSTIC_COLUMNS}
            )
        return len(rows)

    def write_cell_weights(self, records: Iterable[Any]) -> int:
        """Append already ordered long-form cell weights."""

        if self._weights_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        count = 0
        for record in records:
            row = _record_mapping(record)
            self._weights_writer.writerow(
                {column: _clean_scalar(row[column]) for column in CELL_WEIGHT_COLUMNS}
            )
            count += 1
        return count

    def write_weight_result(self, weights: Any) -> int:
        """Append one WeightResult directly without an intermediate DataFrame."""

        if self._weights_row_writer is None:
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
            self._weights_row_writer.writerow(
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

    def write_weight_diagnostics(self, records: Iterable[Any]) -> int:
        """Append one target group's diagnostics in deterministic source order."""

        if self._weight_diagnostics_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        rows = [dict(_record_mapping(record)) for record in records]
        rows.sort(key=lambda row: (str(row["target_group"]), str(row["source_group"])))
        for row in rows:
            self._weight_diagnostics_writer.writerow(
                {column: _clean_scalar(row.get(column, "")) for column in WEIGHT_DIAGNOSTIC_COLUMNS}
            )
        return len(rows)

    def write_skipped(self, records: Iterable[Any]) -> int:
        """Append skipped-target diagnostics in deterministic order."""

        if self._skipped_writer is None:
            raise RuntimeError("IncrementalRunWriter is not open")
        rows = [dict(_record_mapping(record)) for record in records]
        rows.sort(key=lambda row: (str(row["target_group"]), str(row["target"])))
        for row in rows:
            row.setdefault("detail", "")
            self._skipped_writer.writerow(
                {column: _clean_scalar(row.get(column, "")) for column in SKIPPED_COLUMNS}
            )
        return len(rows)


def write_tsv(frame: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    """Write a table with stable column and row formatting."""

    if columns is not None:
        frame = frame.loc[:, list(columns)]
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_tsv_records(records: Iterable[Any], path: Path, columns: Sequence[str]) -> int:
    """Stream mapping-like records to a deterministic TSV with one header."""

    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(columns),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            row = _record_mapping(record)
            writer.writerow({column: _clean_scalar(row[column]) for column in columns})
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

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_compatible(data),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
