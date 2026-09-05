"""Generic, reproducible preparation of 10x data for SPATHI inference.

This module deliberately knows nothing about study-specific annotation systems,
diseases, or cell taxonomies. Study-specific adapters produce the canonical annotation table;
``prepare`` turns that table and a 10x feature-barcode matrix into one strict input
directory per analysis unit.  Its expression, groups, and TF-list files form the
ANDREA-compatible core; optional centroid weights remain SPATHI-specific.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np
from scipy import sparse

from spathi._publication import (
    path_is_occupied,
    preflight_atomic_publication,
    publish_directory_no_replace,
)
from spathi._version import __version__
from spathi.config import PrepareConfig
from spathi.io import InputValidationError, read_centroid_weights_with_fingerprint
from spathi.outputs import write_json

LOGGER = logging.getLogger(__name__)

_ANNOTATION_REQUIRED_COLUMNS = ("cell", "analysis_unit", "cluster")
_GENE_EXPRESSION_FEATURE_TYPE = "Gene Expression"
_MANIFEST_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 8 * 1024**2


class PreparationInputError(ValueError):
    """Raised when a preparation input violates its documented contract."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PrepareResult:
    """Compact result returned after an atomic preparation run."""

    output_dir: Path
    manifest_path: Path
    prepared_analysis_units: tuple[str, ...]
    excluded_analysis_units: tuple[str, ...]
    analysis_unit_directories: tuple[Path, ...]
    input_cells: int
    annotated_cells: int
    excluded_unannotated_cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path) or not isinstance(self.manifest_path, Path):
            raise TypeError("output_dir and manifest_path must be pathlib.Path values")
        if not isinstance(self.analysis_unit_directories, tuple) or not all(
            isinstance(path, Path) for path in self.analysis_unit_directories
        ):
            raise TypeError("analysis_unit_directories must be a tuple of pathlib.Path values")
        for name in ("prepared_analysis_units", "excluded_analysis_units"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise TypeError(f"{name} must be a tuple of non-empty strings")
        for name in ("input_cells", "annotated_cells", "excluded_unannotated_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.annotated_cells + self.excluded_unannotated_cells != self.input_cells:
            raise ValueError("annotated and excluded cell counts must partition input_cells")
        if len(self.prepared_analysis_units) != len(self.analysis_unit_directories):
            raise ValueError("each prepared analysis unit must have one output directory")


@dataclass(frozen=True, slots=True)
class _Annotation:
    cell: str
    analysis_unit: str
    cluster: str


@dataclass(frozen=True, slots=True)
class _AnnotationTable:
    records: tuple[_Annotation, ...]
    fingerprint: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class _CentroidWeightTable:
    by_cell: Mapping[str, float]
    fingerprint: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class _TfList:
    identifiers: tuple[str, ...]
    fingerprint: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class _TenXData:
    matrix: sparse.csc_matrix
    barcodes: tuple[str, ...]
    genes: tuple[str, ...]
    input_shape: tuple[int, int]
    gene_expression_features: int
    collapsed_duplicate_features: int
    duplicate_gene_identifiers: tuple[str, ...]
    fingerprint: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class _PreparationSummary:
    manifest: Mapping[str, Any]
    prepared_names: tuple[str, ...]
    excluded_names: tuple[str, ...]
    prepared_directories: tuple[Path, ...]


def _validated_file(path: Path, *, description: str) -> Path:
    if not path.is_file():
        raise PreparationInputError(f"{description} does not exist or is not a file: {path}")
    return path.resolve(strict=True)


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _read_stable_bytes(path: Path, *, description: str) -> tuple[bytes, os.stat_result]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after):
        raise PreparationInputError(f"{description} changed while it was being read; retry")
    return payload, after


def _fingerprint(path: Path) -> dict[str, str | int]:
    digest = sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after):
        raise PreparationInputError(f"Input file changed while it was being hashed: {path}")
    return {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _read_fingerprinted_text(path: Path, *, description: str) -> tuple[str, dict[str, str | int]]:
    try:
        payload, stat = _read_stable_bytes(path, description=description)
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PreparationInputError(f"{description} must be UTF-8 text: {path}") from exc
    return text, {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": sha256(payload).hexdigest(),
    }


def _validate_identifier(value: str, *, kind: str, location: str) -> str:
    if not value:
        raise PreparationInputError(f"Empty {kind} at {location}")
    if value != value.strip():
        raise PreparationInputError(
            f"{kind.capitalize()} contains surrounding whitespace at {location}"
        )
    if any(character in value for character in ("\t", "\r", "\n", "\x00")):
        raise PreparationInputError(
            f"{kind.capitalize()} contains a control character at {location}"
        )
    return value


def _read_annotations(path: Path) -> _AnnotationTable:
    resolved = _validated_file(path, description="Annotations table")
    text, fingerprint = _read_fingerprinted_text(resolved, description="Annotations table")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise PreparationInputError("Annotations table is empty; a TSV header is required") from exc
    if not header or any(not value for value in header):
        raise PreparationInputError("Annotations table contains an empty column name")
    duplicate_columns = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicate_columns:
        raise PreparationInputError(
            "Annotations table contains duplicate columns: " + ", ".join(duplicate_columns)
        )
    missing_columns = [name for name in _ANNOTATION_REQUIRED_COLUMNS if name not in header]
    unknown_columns = [name for name in header if name not in _ANNOTATION_REQUIRED_COLUMNS]
    if missing_columns or unknown_columns:
        details: list[str] = []
        if missing_columns:
            details.append("missing required columns: " + ", ".join(missing_columns))
        if unknown_columns:
            details.append("unsupported columns: " + ", ".join(unknown_columns))
        raise PreparationInputError("Invalid annotations header; " + "; ".join(details))

    positions = {name: header.index(name) for name in header}
    records: list[_Annotation] = []
    seen_cells: set[str] = set()
    try:
        for row in reader:
            line_number = reader.line_num
            if len(row) != len(header):
                raise PreparationInputError(
                    f"Annotations line {line_number} has {len(row)} fields; expected {len(header)}"
                )
            cell = _validate_identifier(
                row[positions["cell"]], kind="cell", location=f"annotations line {line_number}"
            )
            analysis_unit = _validate_identifier(
                row[positions["analysis_unit"]],
                kind="analysis unit",
                location=f"annotations line {line_number}",
            )
            cluster = _validate_identifier(
                row[positions["cluster"]],
                kind="cluster",
                location=f"annotations line {line_number}",
            )
            if cell in seen_cells:
                raise PreparationInputError(
                    f"Annotations table contains repeated cell identifier: {cell!r}"
                )
            seen_cells.add(cell)
            records.append(
                _Annotation(
                    cell=cell,
                    analysis_unit=analysis_unit,
                    cluster=cluster,
                )
            )
    except csv.Error as exc:
        raise PreparationInputError(f"Could not parse annotations as TSV: {exc}") from exc
    if not records:
        raise PreparationInputError("Annotations table contains no cell rows")
    return _AnnotationTable(tuple(records), fingerprint)


def _read_tf_list(path: Path) -> _TfList:
    resolved = _validated_file(path, description="TF list")
    text, fingerprint = _read_fingerprinted_text(resolved, description="TF list")
    lines = text.splitlines()
    if not lines:
        raise PreparationInputError("TF list is empty")
    identifiers: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        identifier = _validate_identifier(
            line, kind="TF identifier", location=f"line {line_number}"
        )
        if identifier in seen:
            raise PreparationInputError(f"TF list contains duplicate identifier: {identifier!r}")
        seen.add(identifier)
        identifiers.append(identifier)
    return _TfList(tuple(identifiers), fingerprint)


def _decode_hdf5_strings(values: np.ndarray, *, description: str) -> tuple[str, ...]:
    decoded: list[str] = []
    for index, raw in enumerate(values):
        try:
            if isinstance(raw, (bytes, np.bytes_)):
                value = bytes(raw).decode("utf-8", errors="strict")
            elif isinstance(raw, (str, np.str_)):
                value = str(raw)
            else:
                raise TypeError(type(raw).__name__)
        except (TypeError, UnicodeDecodeError) as exc:
            raise PreparationInputError(
                f"{description} must contain UTF-8 strings; invalid value at index {index}"
            ) from exc
        decoded.append(_validate_identifier(value, kind=description, location=f"index {index}"))
    return tuple(decoded)


def _require_dataset(group: h5py.Group, name: str, *, description: str) -> h5py.Dataset:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise PreparationInputError(f"10x H5 is missing {description}: {group.name}/{name}")
    return group[name]


def _require_vector_dataset(
    group: h5py.Group,
    name: str,
    *,
    description: str,
    allowed_dtype_kinds: frozenset[str] | None = None,
) -> h5py.Dataset:
    dataset = _require_dataset(group, name, description=description)
    if dataset.ndim != 1:
        raise PreparationInputError(f"10x H5 {description} must be a one-dimensional dataset")
    if allowed_dtype_kinds is not None and dataset.dtype.kind not in allowed_dtype_kinds:
        expected = ", ".join(sorted(allowed_dtype_kinds))
        raise PreparationInputError(
            f"10x H5 {description} has invalid dtype kind {dataset.dtype.kind!r}; "
            f"expected one of {expected}"
        )
    return dataset


def _load_tenx_h5(
    path: Path,
    *,
    gene_identifier: str,
    duplicate_gene_policy: str,
) -> _TenXData:
    resolved = _validated_file(path, description="10x H5 matrix")
    initial_identity = _stat_identity(resolved.stat())
    fingerprint = _fingerprint(resolved)
    try:
        with h5py.File(resolved, "r") as handle:
            if "matrix" not in handle or not isinstance(handle["matrix"], h5py.Group):
                raise PreparationInputError("10x H5 must contain a /matrix group")
            matrix_group = handle["matrix"]
            if "features" not in matrix_group or not isinstance(
                matrix_group["features"], h5py.Group
            ):
                raise PreparationInputError("10x H5 must contain a /matrix/features group")
            features_group = matrix_group["features"]
            shape_dataset = _require_vector_dataset(
                matrix_group,
                "shape",
                description="matrix shape",
                allowed_dtype_kinds=frozenset({"i", "u"}),
            )
            shape_values = np.asarray(shape_dataset[:], dtype=np.int64)
            if shape_values.shape != (2,) or np.any(shape_values <= 0):
                raise PreparationInputError(
                    "10x H5 matrix shape must contain two positive integers"
                )
            n_features, n_barcodes = map(int, shape_values)
            barcodes = _decode_hdf5_strings(
                np.asarray(
                    _require_vector_dataset(
                        matrix_group,
                        "barcodes",
                        description="barcodes",
                        allowed_dtype_kinds=frozenset({"O", "S", "U"}),
                    )[:]
                ),
                description="cell identifier",
            )
            feature_types = _decode_hdf5_strings(
                np.asarray(
                    _require_vector_dataset(
                        features_group,
                        "feature_type",
                        description="feature types",
                        allowed_dtype_kinds=frozenset({"O", "S", "U"}),
                    )[:]
                ),
                description="feature type",
            )
            feature_key = "name" if gene_identifier == "name" else "id"
            feature_identifiers = _decode_hdf5_strings(
                np.asarray(
                    _require_vector_dataset(
                        features_group,
                        feature_key,
                        description=f"feature {feature_key} values",
                        allowed_dtype_kinds=frozenset({"O", "S", "U"}),
                    )[:]
                ),
                description=f"feature {feature_key}",
            )
            if len(barcodes) != n_barcodes:
                raise PreparationInputError("10x H5 barcode count does not match matrix shape")
            if len(feature_types) != n_features or len(feature_identifiers) != n_features:
                raise PreparationInputError("10x H5 feature metadata does not match matrix shape")
            duplicate_barcodes = [value for value, count in Counter(barcodes).items() if count > 1]
            if duplicate_barcodes:
                raise PreparationInputError(
                    "10x H5 contains duplicate barcodes: "
                    + ", ".join(map(repr, duplicate_barcodes[:8]))
                )
            data = np.asarray(
                _require_vector_dataset(
                    matrix_group,
                    "data",
                    description="matrix data",
                    allowed_dtype_kinds=frozenset({"b", "f", "i", "u"}),
                )[:],
                dtype=np.float64,
            )
            indices = np.asarray(
                _require_vector_dataset(
                    matrix_group,
                    "indices",
                    description="matrix indices",
                    allowed_dtype_kinds=frozenset({"i", "u"}),
                )[:],
                dtype=np.int64,
            )
            indptr = np.asarray(
                _require_vector_dataset(
                    matrix_group,
                    "indptr",
                    description="matrix indptr",
                    allowed_dtype_kinds=frozenset({"i", "u"}),
                )[:],
                dtype=np.int64,
            )
    except OSError as exc:
        raise PreparationInputError(f"Could not read 10x H5 matrix {resolved}: {exc}") from exc
    if _stat_identity(resolved.stat()) != initial_identity:
        raise PreparationInputError("10x H5 matrix changed during preparation input loading; retry")

    if data.size != indices.size:
        raise PreparationInputError("10x H5 sparse data and indices must be aligned vectors")
    if indptr.shape != (n_barcodes + 1,) or indptr[0] != 0 or indptr[-1] != data.size:
        raise PreparationInputError("10x H5 indptr is inconsistent with its CSC matrix shape")
    if np.any(np.diff(indptr) < 0):
        raise PreparationInputError("10x H5 indptr must be non-decreasing")
    if indices.size and (np.any(indices < 0) or np.any(indices >= n_features)):
        raise PreparationInputError("10x H5 matrix indices fall outside the feature axis")
    if not np.isfinite(data).all() or np.any(data < 0):
        raise PreparationInputError("10x H5 matrix counts must be non-negative and finite")

    matrix = sparse.csc_matrix((data, indices, indptr), shape=(n_features, n_barcodes))
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise PreparationInputError(
            "10x H5 matrix counts overflowed while canonicalizing duplicate sparse entries"
        )
    gene_expression_positions = np.flatnonzero(
        np.asarray(feature_types, dtype=object) == _GENE_EXPRESSION_FEATURE_TYPE
    )
    if gene_expression_positions.size == 0:
        raise PreparationInputError("10x H5 contains no 'Gene Expression' features")
    source_genes = tuple(
        feature_identifiers[int(position)] for position in gene_expression_positions
    )
    duplicate_genes = tuple(value for value, count in Counter(source_genes).items() if count > 1)
    if duplicate_genes and duplicate_gene_policy == "error":
        raise PreparationInputError(
            f"10x H5 has duplicate Gene Expression {gene_identifier} values: "
            + ", ".join(map(repr, duplicate_genes[:8]))
        )
    source_expression_matrix = matrix[gene_expression_positions, :].tocsc()
    del matrix
    if duplicate_genes:
        genes = tuple(dict.fromkeys(source_genes))
        gene_positions = {gene: index for index, gene in enumerate(genes)}
        row_codes = np.fromiter(
            (gene_positions[gene] for gene in source_genes),
            dtype=np.int64,
            count=len(source_genes),
        )
        aggregation = sparse.csr_matrix(
            (
                np.ones(len(source_genes), dtype=np.int8),
                (row_codes, np.arange(len(source_genes), dtype=np.int64)),
            ),
            shape=(len(genes), len(source_genes)),
        )
        expression_matrix = (aggregation @ source_expression_matrix).tocsc()
        expression_matrix.sum_duplicates()
        expression_matrix.sort_indices()
        expression_matrix.eliminate_zeros()
        if not np.isfinite(expression_matrix.data).all() or np.any(expression_matrix.data < 0):
            raise PreparationInputError(
                "10x H5 matrix counts overflowed while collapsing duplicate gene identifiers"
            )
    else:
        genes = source_genes
        expression_matrix = source_expression_matrix
    return _TenXData(
        matrix=expression_matrix,
        barcodes=barcodes,
        genes=genes,
        input_shape=(n_features, n_barcodes),
        gene_expression_features=len(source_genes),
        collapsed_duplicate_features=len(source_genes) - len(genes),
        duplicate_gene_identifiers=duplicate_genes,
        fingerprint=fingerprint,
    )


def _normalize_library_size_log1p(
    matrix: sparse.csc_matrix,
    *,
    target_sum: float,
    cells: Sequence[str],
) -> tuple[sparse.csc_matrix, np.ndarray]:
    # The 10x loader has already promoted counts to float64 so canonical sparse
    # summation cannot wrap an integer dtype. The selected matrix is private to
    # this transformation and can therefore be normalized in place.
    normalized = matrix.astype(np.float64, copy=False).tocsc(copy=False)
    library_sizes = np.asarray(normalized.sum(axis=0), dtype=np.float64).reshape(-1)
    invalid = np.flatnonzero(~np.isfinite(library_sizes) | (library_sizes <= 0))
    if invalid.size:
        shown = ", ".join(repr(cells[int(index)]) for index in invalid[:8])
        raise PreparationInputError(
            "Annotated cells must have a positive finite Gene Expression library size; "
            f"invalid cells: {shown}"
        )
    factors = target_sum / library_sizes
    for column in range(normalized.shape[1]):
        start = int(normalized.indptr[column])
        stop = int(normalized.indptr[column + 1])
        normalized.data[start:stop] *= factors[column]
    np.log1p(normalized.data, out=normalized.data)
    if not np.isfinite(normalized.data).all():
        raise PreparationInputError("Normalization produced non-finite expression values")
    normalized.eliminate_zeros()
    return normalized, library_sizes


def _analysis_unit_directory_name(index: int, label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", normalized).strip("-").lower()
    if not slug:
        slug = "analysis"
    return f"unit-{index:03d}-{slug[:64]}"


def _write_groups(path: Path, records: Sequence[_Annotation]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "cluster"))
        writer.writerows((record.cell, record.cluster) for record in records)


def _write_centroid_weights(
    path: Path,
    records: Sequence[_Annotation],
    weights_by_cell: Mapping[str, float],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "centroid_weight"))
        for record in records:
            try:
                weight = weights_by_cell[record.cell]
            except KeyError as exc:  # pragma: no cover - validated coverage invariant
                raise RuntimeError("centroid weight alignment lost an annotated cell") from exc
            writer.writerow((record.cell, repr(weight)))


def _write_tf_list(path: Path, identifiers: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for identifier in identifiers:
            handle.write(identifier)
            handle.write("\n")


def _dense_sparse_rows(matrix: sparse.csr_matrix) -> Iterator[np.ndarray]:
    row = np.zeros(matrix.shape[1], dtype=np.float64)
    for row_index in range(matrix.shape[0]):
        row.fill(0.0)
        start = int(matrix.indptr[row_index])
        stop = int(matrix.indptr[row_index + 1])
        row[matrix.indices[start:stop]] = matrix.data[start:stop]
        yield row


def _write_expression(
    path: Path,
    matrix: sparse.csr_matrix,
    genes: Sequence[str],
    cells: Sequence[str],
) -> None:
    if matrix.shape != (len(genes), len(cells)):
        raise RuntimeError("expression output axes are inconsistent")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("gene", *cells))
        for gene, values in zip(genes, _dense_sparse_rows(matrix), strict=True):
            writer.writerow((gene, *(repr(float(value)) for value in values)))


def _relative_fingerprint(path: Path, root: Path) -> dict[str, str | int]:
    fingerprint = _fingerprint(path)
    fingerprint["path"] = path.relative_to(root).as_posix()
    return fingerprint


def _prepare_into(config: PrepareConfig, output_dir: Path) -> _PreparationSummary:
    LOGGER.info("Loading canonical annotations, optional centroid weights, and TF identifiers")
    annotations = _read_annotations(config.annotations)
    centroid_weights: _CentroidWeightTable | None = None
    if config.centroid_weights is not None:
        try:
            weights, fingerprint = read_centroid_weights_with_fingerprint(
                config.centroid_weights,
                [record.cell for record in annotations.records],
                expected_cell_source="annotation",
            )
        except InputValidationError as exc:
            raise PreparationInputError(str(exc)) from exc
        centroid_weights = _CentroidWeightTable(
            by_cell=weights.to_dict(),
            fingerprint=fingerprint,
        )
    tf_list = _read_tf_list(config.tf_list)
    LOGGER.info("Loading sparse 10x Gene Expression matrix")
    tenx = _load_tenx_h5(
        config.tenx_h5,
        gene_identifier=config.gene_identifier,
        duplicate_gene_policy=config.duplicate_gene_policy,
    )

    tenx_matrix = tenx.matrix
    tenx_barcodes = tenx.barcodes
    tenx_genes = tenx.genes
    tenx_input_shape = tenx.input_shape
    tenx_gene_expression_features = tenx.gene_expression_features
    tenx_collapsed_duplicate_features = tenx.collapsed_duplicate_features
    tenx_duplicate_gene_identifiers = tenx.duplicate_gene_identifiers
    tenx_fingerprint = tenx.fingerprint

    annotations_by_cell = {record.cell: record for record in annotations.records}
    barcode_set = set(tenx_barcodes)
    unexpected = [record.cell for record in annotations.records if record.cell not in barcode_set]
    if unexpected:
        raise PreparationInputError(
            "Annotations contain cells absent from the 10x H5 matrix: "
            + ", ".join(map(repr, unexpected[:8]))
        )
    selected_h5_positions = np.fromiter(
        (index for index, cell in enumerate(tenx_barcodes) if cell in annotations_by_cell),
        dtype=np.int64,
    )
    selected_cells = tuple(tenx_barcodes[int(index)] for index in selected_h5_positions)
    selected_records = tuple(annotations_by_cell[cell] for cell in selected_cells)
    if len(selected_records) != len(annotations.records):  # pragma: no cover - set invariant
        raise RuntimeError("annotation-to-H5 alignment lost cells")

    selected_matrix = tenx_matrix[:, selected_h5_positions].tocsc()
    del tenx_matrix, tenx
    LOGGER.info(
        "Normalizing %s annotated cells with %s",
        len(selected_cells),
        config.normalization,
    )
    normalized, library_sizes = _normalize_library_size_log1p(
        selected_matrix,
        target_sum=config.target_sum,
        cells=selected_cells,
    )
    del selected_matrix

    genes_set = set(tenx_genes)
    retained_source_tfs = tuple(
        identifier for identifier in tf_list.identifiers if identifier in genes_set
    )
    missing_source_tfs = tuple(
        identifier for identifier in tf_list.identifiers if identifier not in genes_set
    )
    if not retained_source_tfs:
        raise PreparationInputError(
            f"None of the {len(tf_list.identifiers)} supplied TF identifiers match 10x "
            f"Gene Expression {config.gene_identifier} values"
        )

    positions_by_unit: dict[str, list[int]] = {}
    for position, record in enumerate(selected_records):
        positions_by_unit.setdefault(record.analysis_unit, []).append(position)
    all_units = sorted(positions_by_unit)
    unit_records: list[dict[str, Any]] = []
    prepared_names: list[str] = []
    prepared_directories: list[Path] = []
    excluded_names: list[str] = []
    units_root = output_dir / "analysis_units"
    units_root.mkdir()

    for unit_index, unit_name in enumerate(all_units, start=1):
        aligned_positions = np.asarray(positions_by_unit[unit_name], dtype=np.int64)
        records = tuple(selected_records[int(index)] for index in aligned_positions)
        group_sizes = dict(sorted(Counter(record.cluster for record in records).items()))
        directory_name = _analysis_unit_directory_name(unit_index, unit_name)
        relative_directory = Path("analysis_units") / directory_name
        base_record: dict[str, Any] = {
            "analysis_unit": unit_name,
            "n_cells": len(records),
            "n_groups": len(group_sizes),
            "group_sizes": group_sizes,
        }
        if len(records) < config.min_cells:
            base_record.update(
                status="excluded",
                exclusion_reason="below-min-cells",
                n_genes=None,
                n_transcription_factors=None,
            )
            excluded_names.append(unit_name)
            unit_records.append(base_record)
            LOGGER.warning(
                "Excluding analysis unit %s: %s cells < min_cells=%s",
                unit_name,
                len(records),
                config.min_cells,
            )
            continue

        unit_matrix_all_genes = normalized[:, aligned_positions].tocsc()
        detection_counts = np.asarray(unit_matrix_all_genes.getnnz(axis=1)).reshape(-1)
        retained_gene_positions = np.flatnonzero(detection_counts >= config.min_gene_cells)
        if retained_gene_positions.size == 0:
            base_record.update(
                status="excluded",
                exclusion_reason="no-genes-after-filtering",
                n_genes=0,
                n_transcription_factors=0,
            )
            excluded_names.append(unit_name)
            unit_records.append(base_record)
            LOGGER.warning("Excluding analysis unit %s: no genes passed filtering", unit_name)
            continue

        unit_genes = tuple(tenx_genes[int(position)] for position in retained_gene_positions)
        unit_gene_set = set(unit_genes)
        unit_tfs = tuple(
            identifier for identifier in retained_source_tfs if identifier in unit_gene_set
        )
        if not unit_tfs:
            base_record.update(
                status="excluded",
                exclusion_reason="no-transcription-factors-after-filtering",
                n_genes=len(unit_genes),
                n_transcription_factors=0,
            )
            excluded_names.append(unit_name)
            unit_records.append(base_record)
            LOGGER.warning(
                "Excluding analysis unit %s: no supplied TF passed gene filtering", unit_name
            )
            continue

        unit_dir = output_dir / relative_directory
        unit_dir.mkdir()
        expression_path = unit_dir / "expression.tsv"
        groups_path = unit_dir / "groups.tsv"
        tf_path = unit_dir / "tf_list.txt"
        centroid_weights_path = (
            unit_dir / "centroid_weights.tsv" if centroid_weights is not None else None
        )
        unit_matrix = unit_matrix_all_genes[retained_gene_positions, :].tocsr()
        unit_matrix.sort_indices()
        LOGGER.info(
            "Writing %s | cells=%s genes=%s TFs=%s groups=%s",
            unit_name,
            len(records),
            len(unit_genes),
            len(unit_tfs),
            len(group_sizes),
        )
        _write_expression(
            expression_path, unit_matrix, unit_genes, [record.cell for record in records]
        )
        _write_groups(groups_path, records)
        _write_tf_list(tf_path, unit_tfs)
        if centroid_weights_path is not None:
            if centroid_weights is None:  # pragma: no cover - path construction invariant
                raise RuntimeError("centroid weight output requested without input weights")
            _write_centroid_weights(centroid_weights_path, records, centroid_weights.by_cell)

        outputs = {
            "expression": _relative_fingerprint(expression_path, output_dir),
            "groups": _relative_fingerprint(groups_path, output_dir),
            "tf_list": _relative_fingerprint(tf_path, output_dir),
        }
        if centroid_weights_path is not None:
            outputs["centroid_weights"] = _relative_fingerprint(centroid_weights_path, output_dir)
        base_record.update(
            status="prepared",
            exclusion_reason=None,
            directory=relative_directory.as_posix(),
            n_genes=len(unit_genes),
            n_transcription_factors=len(unit_tfs),
            outputs=outputs,
        )
        unit_records.append(base_record)
        prepared_names.append(unit_name)
        prepared_directories.append(relative_directory)

    if not prepared_names:
        reasons = Counter(str(record["exclusion_reason"]) for record in unit_records)
        formatted_reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items())
        )
        raise PreparationInputError(
            "Preparation produced no eligible analysis unit; " + formatted_reasons
        )

    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "software": {"spathi": __version__},
        "inputs": {
            "tenx_h5": dict(tenx_fingerprint),
            "annotations": dict(annotations.fingerprint),
            "tf_list": dict(tf_list.fingerprint),
        },
        "parameters": {
            "gene_identifier": config.gene_identifier,
            "duplicate_gene_policy": config.duplicate_gene_policy,
            "min_cells": config.min_cells,
            "min_gene_cells": config.min_gene_cells,
            "normalization": config.normalization,
            "target_sum": config.target_sum,
        },
        "matrix": {
            "input_features": tenx_input_shape[0],
            "input_cells": tenx_input_shape[1],
            "gene_expression_features": tenx_gene_expression_features,
            "prepared_gene_identifiers": len(tenx_genes),
            "collapsed_duplicate_features": tenx_collapsed_duplicate_features,
            "duplicate_gene_identifiers": list(tenx_duplicate_gene_identifiers),
            "annotated_cells": len(selected_cells),
            "excluded_unannotated_cells": len(tenx_barcodes) - len(selected_cells),
            "library_size_before_normalization": {
                "minimum": float(np.min(library_sizes)),
                "median": float(np.median(library_sizes)),
                "maximum": float(np.max(library_sizes)),
            },
        },
        "annotations": {
            "cells_are_exclusive_across_analysis_units": True,
        },
        "centroid_weights": {
            "provided": centroid_weights is not None,
            "input_cells": (len(centroid_weights.by_cell) if centroid_weights is not None else 0),
        },
        "transcription_factors": {
            "supplied": len(tf_list.identifiers),
            "present_in_gene_expression_features": len(retained_source_tfs),
            "absent_from_gene_expression_features": list(missing_source_tfs),
        },
        "summary": {
            "prepared_analysis_units": len(prepared_names),
            "excluded_analysis_units": len(excluded_names),
        },
        "analysis_units": unit_records,
    }
    if centroid_weights is not None:
        manifest["inputs"]["centroid_weights"] = dict(centroid_weights.fingerprint)
    write_json(manifest, output_dir / "prepare_manifest.json")
    return _PreparationSummary(
        manifest=manifest,
        prepared_names=tuple(prepared_names),
        excluded_names=tuple(excluded_names),
        prepared_directories=tuple(prepared_directories),
    )


def prepare(config: PrepareConfig) -> PrepareResult:
    """Prepare 10x counts in private storage, then atomically publish outputs.

    The source H5 remains sparse throughout preparation.  Only one dense row of
    one eligible analysis unit is materialized at a time while writing the TSV
    required by the strict inference contract.
    """

    if not isinstance(config, PrepareConfig):
        raise TypeError("config must be a PrepareConfig")
    final_output = config.output_dir
    if path_is_occupied(final_output):
        raise FileExistsError(
            f"Output path already exists and will not be overwritten: {final_output}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    preflight_atomic_publication(final_output.parent)
    prefix = f".{final_output.name or 'spathi-prepare'}.staging-"
    with TemporaryDirectory(prefix=prefix, dir=final_output.parent) as staging_name:
        staging = Path(staging_name)
        summary = _prepare_into(config, staging)
        if path_is_occupied(final_output):
            raise FileExistsError(
                f"Output path appeared during preparation and will not be overwritten: {final_output}"
            )
        publish_directory_no_replace(staging, final_output)

    directories = tuple(final_output / relative for relative in summary.prepared_directories)
    return PrepareResult(
        output_dir=final_output,
        manifest_path=final_output / "prepare_manifest.json",
        prepared_analysis_units=summary.prepared_names,
        excluded_analysis_units=summary.excluded_names,
        analysis_unit_directories=directories,
        input_cells=int(summary.manifest["matrix"]["input_cells"]),
        annotated_cells=int(summary.manifest["matrix"]["annotated_cells"]),
        excluded_unannotated_cells=int(summary.manifest["matrix"]["excluded_unannotated_cells"]),
    )


__all__ = [
    "PreparationInputError",
    "PrepareResult",
    "prepare",
]
