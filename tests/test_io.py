from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import spathi.io as io_module
from spathi.io import (
    InputValidationError,
    load_inputs,
    read_expression_matrix,
    read_groups,
    read_tf_list,
)


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
    for name, path in input_files.items():
        fingerprint = inputs.input_fingerprints[name]
        assert fingerprint["path"] == str(path.resolve())
        assert fingerprint["size_bytes"] == path.stat().st_size
        assert fingerprint["sha256"] == sha256(path.read_bytes()).hexdigest()


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
