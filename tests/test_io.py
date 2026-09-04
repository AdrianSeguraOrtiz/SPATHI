from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spathi.io as io_module
from spathi.io import (
    InputData,
    InputValidationError,
    load_inputs,
)


def read_expression_matrix(path: Path) -> pd.DataFrame:
    return io_module._read_expression_matrix_with_fingerprint(path)[0]


def read_tf_list(path: Path, expression_genes: list[str]) -> list[str]:
    return io_module._read_tf_list_with_fingerprint(path, expression_genes)[0]


def read_target_list(path: Path, expression_genes: list[str]) -> list[str]:
    return io_module._read_target_list_with_fingerprint(path, expression_genes)[0]


def read_groups(path: Path, expression_cells: list[str] | None = None) -> pd.Series:
    return io_module._read_groups_with_fingerprint(path, expression_cells)[0]


def test_input_data_requires_explicit_keyword_fields_and_resolved_targets() -> None:
    frame = pd.DataFrame([[1.0]], index=["G"], columns=["C"])
    groups = pd.Series(["A"], index=["C"])
    fingerprints = {"expression": {"path": "x", "size_bytes": 1, "sha256": "a"}}

    inputs = InputData(
        expression=frame,
        transcription_factors=("G",),
        targets=("G",),
        groups=groups,
        centroid_weights=None,
        input_fingerprints=fingerprints,
    )

    assert inputs.groups is groups
    assert inputs.input_fingerprints is fingerprints
    assert inputs.targets == ("G",)
    assert inputs.centroid_weights is None
    with pytest.raises(TypeError):
        InputData(frame, ("G",), ("G",), groups)  # type: ignore[misc]


def test_expression_tsv_is_read_as_genes_by_cells(input_files: dict[str, Path]) -> None:
    expression = read_expression_matrix(input_files["expression"])
    assert expression.shape == (4, 4)
    assert expression.index.tolist() == ["TF1", "TF2", "G3", "CONST"]
    assert expression.columns.tolist() == ["cell_1", "cell_2", "cell_3", "cell_4"]
    assert expression.to_numpy().dtype == np.float64


def test_loaded_inputs_record_hashes_of_the_exact_parsed_files(
    input_files: dict[str, Path],
) -> None:
    inputs = load_inputs(input_files["expression"], input_files["tf_list"], input_files["groups"])
    assert set(inputs.input_fingerprints) == {"expression", "tf_list", "groups"}
    assert inputs.targets == ("TF1", "TF2", "G3", "CONST")
    for name, path in input_files.items():
        fingerprint = inputs.input_fingerprints[name]
        assert fingerprint["path"] == str(path.resolve())
        assert fingerprint["size_bytes"] == path.stat().st_size
        assert fingerprint["sha256"] == sha256(path.read_bytes()).hexdigest()


def test_explicit_target_list_preserves_order_and_records_fingerprint(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    target_list = tmp_path / "targets.txt"
    target_list.write_text("G3\nTF1\n", encoding="utf-8")

    inputs = load_inputs(
        input_files["expression"],
        input_files["tf_list"],
        input_files["groups"],
        target_list,
    )

    assert inputs.targets == ("G3", "TF1")
    fingerprint = inputs.input_fingerprints["target_list"]
    assert fingerprint["path"] == str(target_list.resolve())
    assert fingerprint["sha256"] == sha256(target_list.read_bytes()).hexdigest()


def test_explicit_centroid_weights_are_aligned_and_fingerprinted(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    path = tmp_path / "centroid_weights.tsv"
    path.write_text(
        "cell\tcentroid_weight\ncell_3\t0.7\ncell_1\t0.9\ncell_4\t0.6\ncell_2\t0.8\n",
        encoding="utf-8",
    )

    inputs = load_inputs(
        input_files["expression"],
        input_files["tf_list"],
        input_files["groups"],
        centroid_weights=path,
    )

    assert inputs.centroid_weights is not None
    assert inputs.centroid_weights.index.tolist() == [
        "cell_1",
        "cell_2",
        "cell_3",
        "cell_4",
    ]
    assert inputs.centroid_weights.tolist() == [0.9, 0.8, 0.7, 0.6]
    fingerprint = inputs.input_fingerprints["centroid_weights"]
    assert fingerprint["path"] == str(path.resolve())
    assert fingerprint["sha256"] == sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("cell\tweight\ncell_1\t1\n", "exactly two columns"),
        (
            "centroid_weight\tcell\n1\tcell_1\n",
            "exactly two columns",
        ),
        (
            "cell\tcentroid_weight\textra\ncell_1\t1\tx\n",
            "exactly two columns",
        ),
        (
            "cell\tcentroid_weight\ncell_1\t1\ncell_1\t2\ncell_2\t1\ncell_3\t1\ncell_4\t1\n",
            "repeated",
        ),
        (
            "cell\tcentroid_weight\ncell_1\t0\ncell_2\t1\ncell_3\t1\ncell_4\t1\n",
            "positive finite",
        ),
        (
            "cell\tcentroid_weight\ncell_1\t-1\ncell_2\t1\ncell_3\t1\ncell_4\t1\n",
            "positive finite",
        ),
        (
            "cell\tcentroid_weight\ncell_1\tNaN\ncell_2\t1\ncell_3\t1\ncell_4\t1\n",
            "positive finite",
        ),
        (
            "cell\tcentroid_weight\ncell_1\tvalue\ncell_2\t1\ncell_3\t1\ncell_4\t1\n",
            "numeric",
        ),
        (
            "cell\tcentroid_weight\ncell_1\t1\ncell_2\t1\ncell_3\t1\nunexpected\t1\n",
            "cell_4.*unexpected",
        ),
    ],
)
def test_centroid_weight_contract(
    tmp_path: Path,
    input_files: dict[str, Path],
    content: str,
    message: str,
) -> None:
    path = tmp_path / "invalid_centroid_weights.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        load_inputs(
            input_files["expression"],
            input_files["tf_list"],
            input_files["groups"],
            centroid_weights=path,
        )


def test_finite_centroid_weights_are_valid_even_when_their_raw_sum_exceeds_float64(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    path = tmp_path / "overflowing_centroid_weights.tsv"
    path.write_text(
        "cell\tcentroid_weight\n"
        "cell_1\t1.7976931348623157e308\n"
        "cell_2\t1.7976931348623157e308\n"
        "cell_3\t1\n"
        "cell_4\t1\n",
        encoding="utf-8",
    )
    inputs = load_inputs(
        input_files["expression"],
        input_files["tf_list"],
        input_files["groups"],
        centroid_weights=path,
    )

    assert inputs.centroid_weights is not None
    assert np.isfinite(inputs.centroid_weights).all()


def test_csv_expression_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "expression.csv"
    path.write_text("gene,cell_1\nTF1,1\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="tab-separated|TSV"):
        read_expression_matrix(path)


def test_explicit_inverse_orientation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "inverse.tsv"
    path.write_text("cell\tTF1\tG1\ncell_1\t1\t2\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="cells in rows"):
        read_expression_matrix(path)


@pytest.mark.parametrize(
    "first_header",
    [
        "cell",
        "cells",
        "sample",
        "samples",
        "timepoint",
        "timepoints",
        "perturbation",
        "perturbations",
        "cluster",
        "PSEUDOTIME",
    ],
)
def test_expression_rejects_exact_andrea_first_column_names(
    tmp_path: Path, first_header: str
) -> None:
    path = tmp_path / "invalid_header.tsv"
    path.write_text(f"{first_header}\tcell_1\nG1\t1\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="ANDREA contract"):
        read_expression_matrix(path)


@pytest.mark.parametrize("first_header", ["barcode", "cellid", "cell_id", "cell-id"])
def test_expression_does_not_add_non_andrea_header_heuristics(
    tmp_path: Path, first_header: str
) -> None:
    path = tmp_path / "valid_header.tsv"
    path.write_text(f"{first_header}\tcell_1\nG1\t1\n", encoding="utf-8")
    expression = read_expression_matrix(path)
    assert expression.index.name == first_header


def test_joint_identifiers_reject_inverse_orientation(tmp_path: Path) -> None:
    expression = tmp_path / "inverse.tsv"
    expression.write_text(
        "row_id\tTF1\tTF2\tG3\ncell_1\t1\t0\t2\ncell_2\t2\t1\t3\n",
        encoding="utf-8",
    )
    tfs = tmp_path / "tfs.txt"
    tfs.write_text("TF1\nTF2\n", encoding="utf-8")
    groups = tmp_path / "groups.tsv"
    groups.write_text("column\tcluster\ncell_1\tA\ncell_2\tB\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="inversely oriented"):
        load_inputs(expression, tfs, groups)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("TF1\n\nTF2\n", "empty line"),
        ("TF1\nTF1\n", "duplicate"),
        ("MISSING\n", "absent"),
        ("", "empty"),
        (" TF1\n", "whitespace"),
    ],
)
def test_tf_list_contract(
    tmp_path: Path, content: str, message: str, input_files: dict[str, Path]
) -> None:
    path = tmp_path / "invalid_tfs.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        read_tf_list(path, ["TF1", "TF2", "G3"])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("G3\n\nTF1\n", "empty line"),
        ("G3\nG3\n", "duplicate"),
        ("MISSING\n", "absent"),
        ("", "empty"),
        (" G3\n", "whitespace"),
        ("G3\tTF1\n", "exactly one identifier"),
    ],
)
def test_target_list_contract(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "invalid_targets.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        read_target_list(path, ["TF1", "TF2", "G3"])


def test_groups_are_reordered_to_expression_cells(input_files: dict[str, Path]) -> None:
    path = input_files["groups"]
    path.write_text(
        "anything\tcluster\textra\ncell_3\tB\tx\ncell_1\tA\tx\ncell_4\tB\tx\ncell_2\tA\tx\n",
        encoding="utf-8",
    )
    groups = read_groups(path, ["cell_1", "cell_2", "cell_3", "cell_4"])
    assert groups.index.tolist() == ["cell_1", "cell_2", "cell_3", "cell_4"]
    assert groups.tolist() == ["A", "A", "B", "B"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("column\tlabel\ncell_1\tA\n", "cluster"),
        ("column\tcluster\ncell_1\tA\ncell_1\tB\n", "repeated"),
        ("column\tcluster\ncell_1\t\n", "Empty cluster"),
    ],
)
def test_groups_contract(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "invalid_groups.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        read_groups(path)


def test_groups_require_exact_cell_coverage(input_files: dict[str, Path]) -> None:
    input_files["groups"].write_text(
        "column\tcluster\ncell_1\tA\ncell_2\tA\ncell_3\tB\nunexpected\tB\n",
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError) as error:
        load_inputs(input_files["expression"], input_files["tf_list"], input_files["groups"])
    assert "cell_4" in str(error.value)
    assert "unexpected" in str(error.value)


@pytest.mark.parametrize("newline", [b"\r\n", b"\r"])
def test_all_inputs_accept_universal_newlines(tmp_path: Path, newline: bytes) -> None:
    expression = tmp_path / "expression.tsv"
    expression.write_bytes(newline.join((b"gene\tc1\tc2", b"TF1\t1\t2", b"G\t3\t4")) + newline)
    tf_list = tmp_path / "tf_list.txt"
    tf_list.write_bytes(b"TF1" + newline)
    groups = tmp_path / "groups.tsv"
    groups.write_bytes(newline.join((b"column\tcluster", b"c1\tNA", b"c2\tB")) + newline)

    loaded = load_inputs(expression, tf_list, groups)
    assert loaded.expression.shape == (2, 2)
    assert loaded.transcription_factors == ("TF1",)
    assert loaded.groups.tolist() == ["NA", "B"]


@pytest.mark.parametrize("invalid_value", ["NaN", "Inf", "-Inf", "not-a-number"])
def test_expression_rejects_non_finite_or_non_numeric_values(
    tmp_path: Path, invalid_value: str
) -> None:
    path = tmp_path / "invalid.tsv"
    path.write_text(f"gene\tc1\nG1\t{invalid_value}\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="finite numbers.*G1.*c1"):
        read_expression_matrix(path)


def test_expression_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.tsv"
    path.write_bytes(b"\xef\xbb\xbfgene\tc1\nG1\t1\n")
    assert read_expression_matrix(path).index.name == "gene"


def test_invalid_utf8_is_rejected_actionably(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.tsv"
    path.write_bytes(b"gene\tc1\nG1\t1\xff\n")
    with pytest.raises(InputValidationError, match="UTF-8"):
        read_expression_matrix(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("gene\tc1\tc1\nG1\t1\t2\n", "duplicate (?:cell|column)"),
        ("gene\tc1\nG1\t1\nG1\t2\n", "duplicate gene"),
        ("gene\tgene\nG1\t1\n", "duplicate column"),
        ("gene\tc1\n", "no gene rows"),
    ],
)
def test_expression_rejects_duplicate_identifiers_and_empty_body(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "invalid-layout.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError, match=message):
        read_expression_matrix(path)


def test_expression_detects_change_between_validation_and_numeric_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "changing.tsv"
    path.write_text("gene\tc1\nG1\t1\n", encoding="utf-8")
    original_finish = io_module._finish_fingerprint
    expression_finishes = 0

    def finish_then_mutate(*args: object, **kwargs: object) -> dict[str, str | int]:
        nonlocal expression_finishes
        fingerprint = original_finish(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("description") == "Expression matrix":
            expression_finishes += 1
            if expression_finishes == 1:
                path.write_text("gene\tc1\nG1\t9\n", encoding="utf-8")
        return fingerprint

    monkeypatch.setattr(io_module, "_finish_fingerprint", finish_then_mutate)
    with pytest.raises(InputValidationError, match="changed between validation"):
        read_expression_matrix(path)
