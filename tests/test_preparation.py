from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import spathi.preparation as preparation_module
from spathi import SpathiConfig, infer
from spathi.cli import build_parser, main, prepare_config_from_args
from spathi.config import PrepareConfig
from spathi.io import load_inputs
from spathi.preparation import PreparationInputError, PrepareResult, prepare


def write_tenx_h5(
    path: Path,
    values: np.ndarray,
    *,
    barcodes: list[str],
    names: list[str],
    identifiers: list[str] | None = None,
    feature_types: list[str] | None = None,
) -> None:
    matrix = sparse.csc_matrix(values)
    identifiers = identifiers or [f"ENSG{index:04d}" for index in range(len(names))]
    feature_types = feature_types or ["Gene Expression"] * len(names)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        features = group.create_group("features")
        features.create_dataset("name", data=np.asarray(names, dtype="S"))
        features.create_dataset("id", data=np.asarray(identifiers, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(feature_types, dtype="S"))


@pytest.fixture
def preparation_inputs(tmp_path: Path) -> dict[str, Path]:
    matrix = tmp_path / "matrix.h5"
    # Four features by five cells; the final feature is not gene expression.
    write_tenx_h5(
        matrix,
        np.asarray(
            [
                [1, 2, 0, 1, 4],
                [3, 0, 5, 1, 0],
                [0, 2, 1, 0, 0],
                [8, 9, 7, 5, 4],
            ],
            dtype=np.int32,
        ),
        barcodes=["c1", "c2", "c3", "c4", "c5"],
        names=["TF1", "G1", "G2", "ADT1"],
        feature_types=["Gene Expression", "Gene Expression", "Gene Expression", "Antibody Capture"],
    )
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text(
        "cell\tanalysis_unit\tcluster\tcentroid_weight\n"
        "c3\tB lineage\tmemory\t0.8\n"
        "c1\tB lineage\tnaive\t0.9\n"
        "c4\trare/type\tonly\t0.7\n"
        "c2\tB lineage\tnaive\t1.0\n",
        encoding="utf-8",
    )
    tf_list = tmp_path / "tf_list.txt"
    tf_list.write_text("MISSING\nTF1\n", encoding="utf-8")
    return {"matrix": matrix, "annotations": annotations, "tf_list": tf_list}


def config_for(
    inputs: dict[str, Path],
    output_dir: Path,
    **overrides: Any,
) -> PrepareConfig:
    values: dict[str, Any] = {
        "tenx_h5": inputs["matrix"],
        "annotations": inputs["annotations"],
        "tf_list": inputs["tf_list"],
        "output_dir": output_dir,
        "min_cells": 2,
    }
    values.update(overrides)
    return PrepareConfig(**values)


def test_prepare_config_has_explicit_reproducible_defaults() -> None:
    config = PrepareConfig(
        tenx_h5=Path("matrix.h5"),
        annotations=Path("annotations.tsv"),
        tf_list=Path("tf_list.txt"),
        output_dir=Path("prepared"),
    )
    assert config.min_cells == 300
    assert config.min_gene_cells == 1
    assert config.normalization == "library-size-log1p"
    assert config.target_sum == 10_000.0
    assert config.gene_identifier == "name"
    assert config.duplicate_gene_policy == "sum"
    assert config.to_dict()["tenx_h5"] == "matrix.h5"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"min_cells": 0}, "min_cells"),
        ({"min_gene_cells": True}, "min_gene_cells"),
        ({"target_sum": float("nan")}, "target_sum"),
        ({"gene_identifier": "symbol"}, "gene_identifier"),
        ({"duplicate_gene_policy": "first"}, "duplicate_gene_policy"),
    ],
)
def test_prepare_config_rejects_invalid_values(overrides: dict[str, Any], error: str) -> None:
    values: dict[str, Any] = {
        "tenx_h5": Path("matrix.h5"),
        "annotations": Path("annotations.tsv"),
        "tf_list": Path("tf_list.txt"),
        "output_dir": Path("prepared"),
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=error):
        PrepareConfig(**values)


def test_prepare_writes_atomic_andrea_inputs_and_manifest(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    output_dir = tmp_path / "prepared"
    result = prepare(config_for(preparation_inputs, output_dir))

    assert isinstance(result, PrepareResult)
    assert result.output_dir == output_dir
    assert result.prepared_analysis_units == ("B lineage",)
    assert result.excluded_analysis_units == ("rare/type",)
    assert result.input_cells == 5
    assert result.annotated_cells == 4
    assert result.excluded_unannotated_cells == 1
    unit_dir = result.analysis_unit_directories[0]
    assert unit_dir.name == "unit-001-b-lineage"
    assert {path.name for path in unit_dir.iterdir()} == {
        "centroid_weights.tsv",
        "expression.tsv",
        "groups.tsv",
        "tf_list.txt",
    }

    groups = (unit_dir / "groups.tsv").read_text(encoding="utf-8").splitlines()
    # H5 barcode order is canonical, regardless of annotation row order.
    assert groups == ["cell\tcluster", "c1\tnaive", "c2\tnaive", "c3\tmemory"]
    assert (unit_dir / "centroid_weights.tsv").read_text(encoding="utf-8").splitlines() == [
        "cell\tcentroid_weight",
        "c1\t0.9",
        "c2\t1.0",
        "c3\t0.8",
    ]
    assert (unit_dir / "tf_list.txt").read_text(encoding="utf-8") == "TF1\n"

    loaded = load_inputs(
        unit_dir / "expression.tsv",
        unit_dir / "tf_list.txt",
        unit_dir / "groups.tsv",
        centroid_weights=unit_dir / "centroid_weights.tsv",
    )
    assert loaded.expression.index.tolist() == ["TF1", "G1", "G2"]
    assert loaded.expression.columns.tolist() == ["c1", "c2", "c3"]
    assert loaded.centroid_weights is not None
    assert loaded.centroid_weights.tolist() == [0.9, 1.0, 0.8]
    np.testing.assert_allclose(
        loaded.expression.loc[:, "c1"].to_numpy(),
        np.log1p(np.asarray([1.0, 3.0, 0.0]) * 2_500.0),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["parameters"] == {
        "duplicate_gene_policy": "sum",
        "gene_identifier": "name",
        "min_cells": 2,
        "min_gene_cells": 1,
        "normalization": "library-size-log1p",
        "target_sum": 10_000.0,
    }
    assert manifest["matrix"]["gene_expression_features"] == 3
    assert manifest["transcription_factors"]["absent_from_gene_expression_features"] == ["MISSING"]
    assert [unit["status"] for unit in manifest["analysis_units"]] == [
        "prepared",
        "excluded",
    ]
    assert manifest["analysis_units"][1]["exclusion_reason"] == "below-min-cells"
    for output in manifest["analysis_units"][0]["outputs"].values():
        assert not Path(str(output["path"])).is_absolute()
        assert len(str(output["sha256"])) == 64


@pytest.mark.integration
def test_prepared_unit_runs_through_the_public_inference_api(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    prepared = prepare(config_for(preparation_inputs, tmp_path / "prepared"))
    unit = prepared.analysis_unit_directories[0]

    result = infer(
        SpathiConfig(
            expression=unit / "expression.tsv",
            tf_list=unit / "tf_list.txt",
            groups=unit / "groups.tsv",
            centroid_weights=unit / "centroid_weights.tsv",
            output_dir=tmp_path / "inference",
            distance_space="expression",
            distance_metric="euclidean",
            n_estimators=2,
            threads=1,
            report=False,
        ),
        checkpoint=False,
    )

    assert result.network_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["centroid_method"] == "weighted_mean"
    assert "centroid_weights" in metadata["inputs"]


def test_prepare_without_optional_weights_does_not_invent_them(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    preparation_inputs["annotations"].write_text(
        "cell\tanalysis_unit\tcluster\nc1\tB\tA\nc2\tB\tA\n",
        encoding="utf-8",
    )
    result = prepare(config_for(preparation_inputs, tmp_path / "prepared"))
    unit_dir = result.analysis_unit_directories[0]
    assert not (unit_dir / "centroid_weights.tsv").exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["annotations"]["centroid_weight_provided"] is False
    assert "centroid_weights" not in manifest["analysis_units"][0]["outputs"]


def test_prepare_never_densifies_a_complete_sparse_matrix(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_dense(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a sparse matrix was densified")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
    result = prepare(config_for(preparation_inputs, tmp_path / "prepared"))
    assert result.prepared_analysis_units == ("B lineage",)


def test_gene_filtering_is_per_unit_and_tf_intersection_follows_it(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    result = prepare(
        config_for(
            preparation_inputs,
            tmp_path / "prepared",
            min_gene_cells=2,
        )
    )
    expression = pd.read_csv(
        result.analysis_unit_directories[0] / "expression.tsv",
        sep="\t",
        index_col=0,
    )
    assert expression.index.tolist() == ["TF1", "G1", "G2"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("cell\tanalysis_unit\n c1\tB\n", "missing required columns"),
        ("cell\tanalysis_unit\tcluster\textra\nc1\tB\tA\tx\n", "unsupported columns"),
        ("cell\tanalysis_unit\tcluster\nc1\tB\tA\nc1\tB\tB\n", "repeated cell"),
        (
            "cell\tanalysis_unit\tcluster\tcentroid_weight\nc1\tB\tA\t0\n",
            "positive finite",
        ),
    ],
)
def test_annotations_contract_is_strict(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
    content: str,
    message: str,
) -> None:
    preparation_inputs["annotations"].write_text(content, encoding="utf-8")
    with pytest.raises(PreparationInputError, match=message):
        prepare(config_for(preparation_inputs, tmp_path / "prepared", min_cells=1))
    assert not (tmp_path / "prepared").exists()


def test_annotations_must_be_a_subset_of_h5_barcodes(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    preparation_inputs["annotations"].write_text(
        "cell\tanalysis_unit\tcluster\nmissing\tB\tA\n",
        encoding="utf-8",
    )
    with pytest.raises(PreparationInputError, match="absent from the 10x H5"):
        prepare(config_for(preparation_inputs, tmp_path / "prepared", min_cells=1))


@pytest.mark.parametrize(
    ("dataset_path", "replacement", "message"),
    [
        ("matrix/shape", np.asarray([4.0, 5.0]), "matrix shape has invalid dtype"),
        ("matrix/indices", np.asarray([0.0]), "matrix indices has invalid dtype"),
        (
            "matrix/data",
            np.asarray([1.0 + 0.0j]),
            "matrix data has invalid dtype",
        ),
        (
            "matrix/features/name",
            np.arange(4, dtype=np.int64),
            "feature name values has invalid dtype",
        ),
        (
            "matrix/barcodes",
            np.asarray([[b"c1"], [b"c2"], [b"c3"], [b"c4"], [b"c5"]]),
            "barcodes must be a one-dimensional dataset",
        ),
    ],
)
def test_tenx_h5_requires_exact_vector_shapes_and_dtype_families(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
    dataset_path: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    with h5py.File(preparation_inputs["matrix"], "a") as handle:
        parent_path, dataset_name = dataset_path.rsplit("/", maxsplit=1)
        parent = handle[parent_path]
        del parent[dataset_name]
        parent.create_dataset(dataset_name, data=replacement)

    with pytest.raises(PreparationInputError, match=message):
        prepare(config_for(preparation_inputs, tmp_path / "prepared", min_cells=1))
    assert not (tmp_path / "prepared").exists()


def test_duplicate_gene_names_can_be_rejected_explicitly(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix.h5"
    write_tenx_h5(
        matrix,
        np.asarray([[1], [2]], dtype=np.int32),
        barcodes=["c1"],
        names=["DUP", "DUP"],
        identifiers=["ID1", "ID2"],
    )
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text("cell\tanalysis_unit\tcluster\nc1\tA\tA\n", encoding="utf-8")
    tf_list = tmp_path / "tf_list.txt"
    tf_list.write_text("DUP\n", encoding="utf-8")
    config = PrepareConfig(
        tenx_h5=matrix,
        annotations=annotations,
        tf_list=tf_list,
        output_dir=tmp_path / "prepared",
        min_cells=1,
        duplicate_gene_policy="error",
    )
    with pytest.raises(PreparationInputError, match="duplicate Gene Expression name"):
        prepare(config)


def test_duplicate_gene_names_are_summed_sparsely_by_default(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.h5"
    write_tenx_h5(
        matrix,
        np.asarray([[1, 0], [2, 3], [1, 1]], dtype=np.int32),
        barcodes=["c1", "c2"],
        names=["TF", "TF", "G"],
        identifiers=["ID1", "ID2", "ID3"],
    )
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text("cell\tanalysis_unit\tcluster\nc1\tA\tA\nc2\tA\tA\n", encoding="utf-8")
    tf_list = tmp_path / "tf_list.txt"
    tf_list.write_text("TF\n", encoding="utf-8")
    result = prepare(
        PrepareConfig(
            tenx_h5=matrix,
            annotations=annotations,
            tf_list=tf_list,
            output_dir=tmp_path / "prepared",
            min_cells=1,
        )
    )
    expression = pd.read_csv(
        result.analysis_unit_directories[0] / "expression.tsv", sep="\t", index_col=0
    )
    assert expression.index.tolist() == ["TF", "G"]
    # c1: duplicate TF rows sum to 3 out of a total library of 4.
    assert expression.loc["TF", "c1"] == pytest.approx(np.log1p(7_500.0))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["matrix"]["gene_expression_features"] == 3
    assert manifest["matrix"]["prepared_gene_identifiers"] == 2
    assert manifest["matrix"]["collapsed_duplicate_features"] == 1
    assert manifest["matrix"]["duplicate_gene_identifiers"] == ["TF"]


def test_duplicate_sparse_entries_cannot_wrap_an_unsigned_count_dtype(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.h5"
    with h5py.File(matrix, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=np.asarray([250, 250, 1], dtype=np.uint8))
        group.create_dataset("indices", data=np.asarray([0, 0, 1], dtype=np.int32))
        group.create_dataset("indptr", data=np.asarray([0, 3], dtype=np.int32))
        group.create_dataset("shape", data=np.asarray([2, 1], dtype=np.int64))
        group.create_dataset("barcodes", data=np.asarray(["c1"], dtype="S"))
        features = group.create_group("features")
        features.create_dataset("name", data=np.asarray(["TF", "G"], dtype="S"))
        features.create_dataset("id", data=np.asarray(["ID1", "ID2"], dtype="S"))
        features.create_dataset(
            "feature_type",
            data=np.asarray(["Gene Expression", "Gene Expression"], dtype="S"),
        )
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text("cell\tanalysis_unit\tcluster\nc1\tA\tA\n", encoding="utf-8")
    tf_list = tmp_path / "tf_list.txt"
    tf_list.write_text("TF\n", encoding="utf-8")

    result = prepare(
        PrepareConfig(
            tenx_h5=matrix,
            annotations=annotations,
            tf_list=tf_list,
            output_dir=tmp_path / "prepared",
            min_cells=1,
        )
    )
    expression = pd.read_csv(
        result.analysis_unit_directories[0] / "expression.tsv",
        sep="\t",
        index_col=0,
    )

    assert expression.loc["TF", "c1"] == pytest.approx(np.log1p(10_000.0 * 500.0 / 501.0))


def test_all_units_below_threshold_fail_without_publishing(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    output_dir = tmp_path / "prepared"
    with pytest.raises(PreparationInputError, match="no eligible analysis unit"):
        prepare(config_for(preparation_inputs, output_dir, min_cells=10))
    assert not output_dir.exists()


def test_prepare_refuses_to_overwrite_an_existing_output(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
) -> None:
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    marker = output_dir / "owned.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        prepare(config_for(preparation_inputs, output_dir))
    assert marker.read_text(encoding="utf-8") == "keep"


def test_prepare_cli_builds_config_and_delegates_to_preparation_api(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "prepare",
        "--tenx-h5",
        "matrix.h5",
        "--annotations",
        "annotations.tsv",
        "--tf-list",
        "tf_list.txt",
        "--output-dir",
        "prepared",
        "--min-cells",
        "12",
        "--min-gene-cells",
        "3",
        "--target-sum",
        "5000",
        "--gene-identifier",
        "id",
        "--duplicate-gene-policy",
        "error",
    ]
    parsed = prepare_config_from_args(build_parser().parse_args(arguments))
    assert parsed.min_cells == 12
    assert parsed.min_gene_cells == 3
    assert parsed.target_sum == 5_000.0
    assert parsed.gene_identifier == "id"
    assert parsed.duplicate_gene_policy == "error"

    received: dict[str, PrepareConfig] = {}

    def fake_prepare(config: PrepareConfig) -> SimpleNamespace:
        received["config"] = config
        return SimpleNamespace(
            output_dir=config.output_dir,
            prepared_analysis_units=("A",),
            excluded_analysis_units=(),
        )

    module = ModuleType("spathi.preparation")
    module.__dict__["prepare"] = fake_prepare
    monkeypatch.setitem(sys.modules, "spathi.preparation", module)
    assert main(arguments) == 0
    assert received["config"] == parsed
    assert "Preparation complete" in capsys.readouterr().err


def test_prepare_preflights_publication_before_loading_inputs(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "unsupported-publication"
    annotations_loaded = False

    def reject_publication(parent: Path) -> None:
        assert parent == tmp_path
        raise RuntimeError("unsupported output filesystem")

    def observe_annotations(*_args: object, **_kwargs: object) -> None:
        nonlocal annotations_loaded
        annotations_loaded = True

    monkeypatch.setattr(preparation_module, "preflight_atomic_publication", reject_publication)
    monkeypatch.setattr(preparation_module, "_read_annotations", observe_annotations)

    with pytest.raises(RuntimeError, match="unsupported output filesystem"):
        prepare(config_for(preparation_inputs, output_dir))

    assert annotations_loaded is False
    assert not output_dir.exists()


def test_prepare_cannot_replace_output_created_during_publication_race(
    tmp_path: Path,
    preparation_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "publication-race"
    original_occupied = preparation_module.path_is_occupied
    output_checks = 0

    def create_destination_after_last_check(path: Path) -> bool:
        nonlocal output_checks
        if path == output_dir:
            output_checks += 1
            if output_checks == 2:
                output_dir.mkdir()
                return False
        return original_occupied(path)

    monkeypatch.setattr(
        preparation_module,
        "path_is_occupied",
        create_destination_after_last_check,
    )

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        prepare(config_for(preparation_inputs, output_dir))

    assert output_checks == 2
    assert list(output_dir.iterdir()) == []
