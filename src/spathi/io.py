"""Strict readers for the three ANDREA-compatible SPATHI inputs.

The expression matrix is deliberately kept in its on-disk orientation (genes by
cells).  Downstream code can therefore distinguish the inference matrix from the
cell-by-feature representation used only to calculate distances.
"""

from __future__ import annotations

import csv
import io
import os
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

import numpy as np
import pandas as pd

Pathish = str | PathLike[str]
InputFingerprint: TypeAlias = dict[str, str | int]


class InputValidationError(ValueError):
    """Raised when an input file does not satisfy the SPATHI input contract."""


@dataclass(frozen=True, slots=True)
class InputData:
    """Validated SPATHI input data.

    Attributes
    ----------
    expression:
        Numeric expression matrix with genes in rows and cells in columns.
    transcription_factors:
        TF identifiers, in the same order as the TF-list file.
    groups:
        Cell-to-group mapping, reordered to match the expression columns.
    input_fingerprints:
        Resolved paths, byte sizes, and SHA-256 digests of the exact parsed files.
    """

    expression: pd.DataFrame
    transcription_factors: tuple[str, ...]
    groups: pd.Series
    input_fingerprints: Mapping[str, InputFingerprint] = field(default_factory=dict)

    @property
    def tf_list(self) -> tuple[str, ...]:
        """Alias retained for callers that use the input-file terminology."""

        return self.transcription_factors


_EXPRESSION_FIRST_COLUMN_DISALLOWED_NAMES = {
    "cell",
    "cells",
    "sample",
    "samples",
    "timepoint",
    "timepoints",
    "perturbation",
    "perturbations",
    "cluster",
    "pseudotime",
}


def _normalise_header_role(value: str) -> str:
    """Apply the same first-header normalization as ANDREA's validator."""

    return value.strip().lower()


def _format_items(values: Sequence[str], *, limit: int = 8) -> str:
    shown = [repr(value) for value in values[:limit]]
    if len(values) > limit:
        shown.append(f"... ({len(values) - limit} more)")
    return ", ".join(shown)


def _validated_input_path(path: Pathish, *, description: str) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise InputValidationError(
            f"{description} file does not exist or is not a file: {file_path}"
        )
    return file_path.resolve(strict=True)


class _HashingRawReader(io.RawIOBase):
    """Raw adapter that hashes exactly the bytes consumed by a text reader."""

    def __init__(self, handle: BinaryIO, digest: Any) -> None:
        super().__init__()
        self._handle = handle
        self._digest = digest

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._handle.read(len(buffer))
        if not data:
            return 0
        count = len(data)
        memoryview(buffer)[:count] = data
        self._digest.update(data)
        return count


def _hashed_utf8_lines(handle: BinaryIO, digest: Any) -> Iterator[str]:
    """Yield universal-newline UTF-8 text while hashing exact source bytes."""

    raw = _HashingRawReader(handle, digest)
    buffered = io.BufferedReader(raw)
    text = io.TextIOWrapper(buffered, encoding="utf-8-sig", newline=None)
    try:
        yield from text
    finally:
        # Do not let the adapter close the file descriptor owned by the caller;
        # it is still needed for the before/after fstat consistency check.
        if not text.closed:
            detached_buffer = text.detach()
            detached_buffer.detach()


def _finish_fingerprint(
    path: Path,
    handle: BinaryIO,
    digest: Any,
    initial_stat: os.stat_result,
    *,
    description: str,
) -> InputFingerprint:
    final_stat = os.fstat(handle.fileno())
    identity_before = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
    )
    identity_after = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise InputValidationError(
            f"{description} changed while it was being read; retry with stable input files"
        )
    return {
        "path": str(path),
        "size_bytes": int(final_stat.st_size),
        "sha256": digest.hexdigest(),
    }


def _validate_tsv_header(header: list[str], *, description: str) -> None:
    if len(header) < 2:
        hint = (
            " It appears to be comma-separated; only TSV input is accepted."
            if header and "," in header[0]
            else ""
        )
        raise InputValidationError(
            f"{description} must be tab-separated and contain at least two columns.{hint}"
        )
    if any(value == "" for value in header):
        raise InputValidationError(f"{description} contains an empty column name")
    if any(value != value.strip() for value in header):
        raise InputValidationError(
            f"{description} column names may not have surrounding whitespace"
        )
    header_counts = Counter(header)
    if len(header_counts) != len(header):
        duplicates = sorted(value for value, count in header_counts.items() if count > 1)
        raise InputValidationError(
            f"{description} contains duplicate column names: {_format_items(duplicates)}"
        )


def _validate_tsv_row(row: list[str], *, line_number: int, width: int, description: str) -> None:
    if not row:
        raise InputValidationError(f"{description} contains an empty row at line {line_number}")
    if len(row) != width:
        raise InputValidationError(
            f"{description} line {line_number} has {len(row)} fields; expected {width}. "
            "The file must be a rectangular TSV."
        )


def _validate_identifier(value: str, *, kind: str, location: str) -> str:
    if value == "" or not value.strip():
        raise InputValidationError(f"Empty {kind} identifier at {location}")
    if value != value.strip():
        raise InputValidationError(
            f"{kind.capitalize()} identifier {value!r} at {location} has surrounding whitespace"
        )
    if "\x00" in value or "\r" in value or "\n" in value:
        raise InputValidationError(f"Invalid control character in {kind} identifier at {location}")
    return value


def _invalid_expression_value(supplied: str, *, gene: str, cell: str) -> InputValidationError:
    return InputValidationError(
        "Expression values must all be finite numbers; found "
        f"{supplied!r} for gene {gene!r}, cell {cell!r}"
    )


def _read_expression_matrix_with_fingerprint(
    path: Pathish,
) -> tuple[pd.DataFrame, InputFingerprint]:
    """Parse expression into one numeric array without a table-sized string copy."""

    description = "Expression matrix"
    file_path = _validated_input_path(path, description=description)

    # First pass validates the rectangular identifier layout and determines the
    # exact numeric allocation.  It retains only identifiers, never expression
    # strings.  The second pass fills that array one row at a time and verifies
    # an identical digest, preventing mixed provenance if an input changes.
    try:
        with file_path.open("rb") as handle:
            initial_stat = os.fstat(handle.fileno())
            digest = sha256()
            reader = csv.reader(_hashed_utf8_lines(handle, digest), delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise InputValidationError(
                    "Expression matrix is empty; a TSV header is required"
                ) from exc
            _validate_tsv_header(header, description=description)
            if _normalise_header_role(header[0]) in _EXPRESSION_FIRST_COLUMN_DISALLOWED_NAMES:
                raise InputValidationError(
                    f"Expression matrix first column header {header[0]!r} is not valid for "
                    "expression genes under the ANDREA contract and commonly indicates cells "
                    "in rows. SPATHI requires genes in rows and cells in columns."
                )
            cell_ids = [
                _validate_identifier(value, kind="cell", location=f"header column {index}")
                for index, value in enumerate(header[1:], start=2)
            ]
            duplicate_cells = sorted(
                value for value, count in Counter(cell_ids).items() if count > 1
            )
            if duplicate_cells:
                raise InputValidationError(
                    "Expression matrix contains duplicate cell identifiers: "
                    + _format_items(duplicate_cells)
                )

            gene_ids: list[str] = []
            seen_genes: set[str] = set()
            duplicate_genes: list[str] = []
            for row in reader:
                line_number = reader.line_num
                _validate_tsv_row(
                    row,
                    line_number=line_number,
                    width=len(header),
                    description=description,
                )
                gene = _validate_identifier(row[0], kind="gene", location=f"line {line_number}")
                if gene in seen_genes and gene not in duplicate_genes:
                    duplicate_genes.append(gene)
                seen_genes.add(gene)
                gene_ids.append(gene)
            fingerprint = _finish_fingerprint(
                file_path,
                handle,
                digest,
                initial_stat,
                description=description,
            )
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"Expression matrix must be UTF-8 text: {file_path}") from exc
    except csv.Error as exc:
        raise InputValidationError(f"Could not parse Expression matrix as TSV: {exc}") from exc

    if not gene_ids:
        raise InputValidationError("Expression matrix has a header but no gene rows")
    if duplicate_genes:
        raise InputValidationError(
            "Expression matrix contains duplicate gene identifiers: "
            + _format_items(sorted(duplicate_genes))
        )

    values = np.empty((len(gene_ids), len(cell_ids)), dtype=np.float64)
    try:
        with file_path.open("rb") as handle:
            initial_stat = os.fstat(handle.fileno())
            digest = sha256()
            reader = csv.reader(_hashed_utf8_lines(handle, digest), delimiter="\t", strict=True)
            second_header = next(reader, None)
            if second_header != header:
                raise InputValidationError(
                    "Expression matrix changed between validation and numeric parsing; retry"
                )
            parsed_rows = 0
            for row_index, row in enumerate(reader):
                line_number = reader.line_num
                _validate_tsv_row(
                    row,
                    line_number=line_number,
                    width=len(header),
                    description=description,
                )
                if row_index >= len(gene_ids) or row[0] != gene_ids[row_index]:
                    raise InputValidationError(
                        "Expression matrix changed between validation and numeric parsing; retry"
                    )
                try:
                    numeric_row = np.asarray(row[1:], dtype=np.float64)
                except (TypeError, ValueError):
                    for column_index, supplied in enumerate(row[1:]):
                        try:
                            parsed = float(supplied)
                        except (TypeError, ValueError):
                            raise _invalid_expression_value(
                                supplied,
                                gene=gene_ids[row_index],
                                cell=cell_ids[column_index],
                            ) from None
                        if not np.isfinite(parsed):
                            raise _invalid_expression_value(
                                supplied,
                                gene=gene_ids[row_index],
                                cell=cell_ids[column_index],
                            ) from None
                    raise AssertionError(
                        "numeric row conversion failed without an invalid value"
                    ) from None
                non_finite = np.flatnonzero(~np.isfinite(numeric_row))
                if non_finite.size:
                    column_index = int(non_finite[0])
                    raise _invalid_expression_value(
                        row[column_index + 1],
                        gene=gene_ids[row_index],
                        cell=cell_ids[column_index],
                    )
                values[row_index] = numeric_row
                parsed_rows += 1
            second_fingerprint = _finish_fingerprint(
                file_path,
                handle,
                digest,
                initial_stat,
                description=description,
            )
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"Expression matrix must be UTF-8 text: {file_path}") from exc
    except csv.Error as exc:
        raise InputValidationError(f"Could not parse Expression matrix as TSV: {exc}") from exc

    if parsed_rows != len(gene_ids) or second_fingerprint["sha256"] != fingerprint["sha256"]:
        raise InputValidationError(
            "Expression matrix changed between validation and numeric parsing; retry"
        )

    expression = pd.DataFrame(values, index=gene_ids, columns=cell_ids, copy=False)
    expression.index.name = header[0]
    expression.columns.name = None
    return expression, fingerprint


def read_expression_matrix(path: Pathish) -> pd.DataFrame:
    """Read and validate a genes-by-cells expression TSV as ``float64``.

    Parsing is two-pass and memory-bounded with respect to textual values: the
    first pass validates identifiers and shape, and the second fills one exact
    numeric array. Orientation is never inferred or corrected automatically.
    """

    expression, _ = _read_expression_matrix_with_fingerprint(path)
    return expression


def _validate_tf_membership(tf_ids: Sequence[str], expression_genes: Collection[str]) -> None:
    genes = set(expression_genes)
    missing = [tf for tf in tf_ids if tf not in genes]
    if missing:
        raise InputValidationError(
            "TF identifiers absent from the expression matrix: " + _format_items(missing)
        )


def _read_tf_list_with_fingerprint(
    path: Pathish,
    expression_genes: Collection[str] | None = None,
) -> tuple[list[str], InputFingerprint]:
    """Read a one-identifier-per-line TF list and validate exact membership.

    Blank lines (including whitespace-only lines), duplicate identifiers, and
    multi-field tab-separated lines are rejected.  When ``expression_genes`` is
    supplied, every TF must occur in that collection.
    """

    description = "TF list"
    file_path = _validated_input_path(path, description=description)
    tf_ids: list[str] = []
    try:
        with file_path.open("rb") as handle:
            initial_stat = os.fstat(handle.fileno())
            digest = sha256()
            for line_number, decoded_line in enumerate(_hashed_utf8_lines(handle, digest), start=1):
                value = decoded_line[:-1] if decoded_line.endswith("\n") else decoded_line
                if value.endswith("\r"):
                    value = value[:-1]
                if not value.strip():
                    raise InputValidationError(
                        f"TF list contains an empty line at line {line_number}"
                    )
                if "\t" in value:
                    raise InputValidationError(
                        f"TF list line {line_number} contains multiple tab-separated fields; "
                        "provide exactly one identifier per line"
                    )
                tf_ids.append(
                    _validate_identifier(value, kind="TF", location=f"line {line_number}")
                )
            fingerprint = _finish_fingerprint(
                file_path,
                handle,
                digest,
                initial_stat,
                description=description,
            )
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"TF list must be UTF-8 text: {file_path}") from exc

    duplicates = sorted(value for value, count in Counter(tf_ids).items() if count > 1)
    if duplicates:
        raise InputValidationError(
            f"TF list contains duplicate identifiers: {_format_items(duplicates)}"
        )
    if not tf_ids:
        raise InputValidationError("TF list is empty after validation")
    if expression_genes is not None:
        _validate_tf_membership(tf_ids, expression_genes)
    return tf_ids, fingerprint


def read_tf_list(
    path: Pathish,
    expression_genes: Collection[str] | None = None,
) -> list[str]:
    """Read a one-identifier-per-line TF list and validate exact membership."""

    tf_ids, _ = _read_tf_list_with_fingerprint(path, expression_genes)
    return tf_ids


def _validate_group_coverage(groups: pd.Series, expression_cells: Sequence[str]) -> pd.Series:
    expected = list(expression_cells)
    expected_set = set(expected)
    supplied_set = set(groups.index)
    missing = [cell for cell in expected if cell not in supplied_set]
    unexpected = [str(cell) for cell in groups.index if cell not in expected_set]
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"expression cells without a group: {_format_items(missing)}")
        if unexpected:
            details.append(f"group rows absent from expression: {_format_items(unexpected)}")
        raise InputValidationError(
            "groups.tsv must define every expression cell exactly once; " + "; ".join(details)
        )
    return groups.loc[expected].copy()


def _read_groups_with_fingerprint(
    path: Pathish,
    expression_cells: Sequence[str] | None = None,
) -> tuple[pd.Series, InputFingerprint]:
    """Read an ANDREA-compatible groups TSV.

    The first column is interpreted as the cell identifier regardless of its
    name.  A separate, literally named ``cluster`` column is mandatory.  If
    expression cells are supplied, exact one-to-one coverage is enforced and
    the returned series is ordered like the expression columns.
    """

    description = "Groups table"
    file_path = _validated_input_path(path, description=description)
    cells: list[str] = []
    clusters: list[str] = []
    try:
        with file_path.open("rb") as handle:
            initial_stat = os.fstat(handle.fileno())
            digest = sha256()
            reader = csv.reader(_hashed_utf8_lines(handle, digest), delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise InputValidationError(
                    "Groups table is empty; a TSV header is required"
                ) from exc
            _validate_tsv_header(header, description=description)
            if "cluster" not in header:
                raise InputValidationError(
                    "Groups table must contain a column named exactly 'cluster'"
                )
            if header[0] == "cluster":
                raise InputValidationError(
                    "The first groups-table column must contain cell identifiers and must be "
                    "distinct from 'cluster'"
                )
            cluster_index = header.index("cluster")
            for row in reader:
                line_number = reader.line_num
                _validate_tsv_row(
                    row,
                    line_number=line_number,
                    width=len(header),
                    description=description,
                )
                cell = _validate_identifier(
                    row[0], kind="cell", location=f"groups line {line_number}"
                )
                cluster = _validate_identifier(
                    row[cluster_index],
                    kind="cluster",
                    location=f"groups line {line_number}",
                )
                cells.append(cell)
                clusters.append(cluster)
            fingerprint = _finish_fingerprint(
                file_path,
                handle,
                digest,
                initial_stat,
                description=description,
            )
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"Groups table must be UTF-8 text: {file_path}") from exc
    except csv.Error as exc:
        raise InputValidationError(f"Could not parse Groups table as TSV: {exc}") from exc

    if not cells:
        raise InputValidationError("Groups table has no cell rows")
    duplicate_cells = sorted(value for value, count in Counter(cells).items() if count > 1)
    if duplicate_cells:
        raise InputValidationError(
            "Groups table contains repeated cells: " + _format_items(duplicate_cells)
        )

    groups = pd.Series(
        clusters, index=pd.Index(cells, name=header[0]), name="cluster", dtype="string"
    )
    if groups.empty or groups.nunique(dropna=False) == 0:
        raise InputValidationError("Groups table contains no non-empty groups")
    if expression_cells is not None:
        groups = _validate_group_coverage(groups, expression_cells)
    return groups, fingerprint


def read_groups(
    path: Pathish,
    expression_cells: Sequence[str] | None = None,
) -> pd.Series:
    """Read an ANDREA-compatible groups TSV with exact cell coverage."""

    groups, _ = _read_groups_with_fingerprint(path, expression_cells)
    return groups


def _check_joint_orientation(
    expression: pd.DataFrame,
    transcription_factors: Sequence[str],
    groups: pd.Series,
) -> None:
    expression_cells = set(map(str, expression.columns))
    expression_genes = set(map(str, expression.index))
    group_cells = set(map(str, groups.index))
    tf_set = set(transcription_factors)

    groups_match_rows = bool(group_cells) and group_cells == expression_genes
    tfs_match_columns = (
        bool(tf_set) and tf_set.issubset(expression_cells) and not tf_set.issubset(expression_genes)
    )
    if group_cells != expression_cells and (groups_match_rows or tfs_match_columns):
        raise InputValidationError(
            "Expression matrix appears to be inversely oriented: group cell identifiers and/or TF identifiers "
            "match the opposite axes. SPATHI accepts only genes in rows and cells in columns."
        )


def load_inputs(expression: Pathish, tf_list: Pathish, groups: Pathish) -> InputData:
    """Load all inputs and enforce their joint identifier contracts.

    Joint validation provides a stronger rejection of inverse expression
    orientation than any heuristic based on the expression table alone.
    """

    expression_frame, expression_fingerprint = _read_expression_matrix_with_fingerprint(expression)
    tf_ids, tf_fingerprint = _read_tf_list_with_fingerprint(tf_list)
    group_series, groups_fingerprint = _read_groups_with_fingerprint(groups)
    _check_joint_orientation(expression_frame, tf_ids, group_series)
    _validate_tf_membership(tf_ids, expression_frame.index)
    group_series = _validate_group_coverage(group_series, list(expression_frame.columns))
    return InputData(
        expression=expression_frame,
        transcription_factors=tuple(tf_ids),
        groups=group_series,
        input_fingerprints={
            "expression": expression_fingerprint,
            "tf_list": tf_fingerprint,
            "groups": groups_fingerprint,
        },
    )


# Compact aliases that read naturally in both notebooks and pipeline code.
read_expression = read_expression_matrix
load_and_validate_inputs = load_inputs


__all__ = [
    "InputData",
    "InputValidationError",
    "load_and_validate_inputs",
    "load_inputs",
    "read_expression",
    "read_expression_matrix",
    "read_groups",
    "read_tf_list",
]
