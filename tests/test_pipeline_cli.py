from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spathi.inference as inference_module
import spathi.pipeline as pipeline_module
import spathi.visualization as visualization_module
from spathi import SpathiConfig, infer_group_specific_grns

EXPECTED_ARTIFACTS = {
    "cell_embedding.tsv.gz",
    "cell_weights.tsv.gz",
    "centroids.tsv",
    "group_affinities.tsv",
    "group_distances.tsv",
    "model_diagnostics.tsv.gz",
    "network.csv",
    "parameters.json",
    "pca_explained_variance.tsv",
    "run_metadata.json",
    "skipped_targets.tsv",
    "weight_diagnostics.tsv",
    "visualizations",
}


def config_for(
    input_files: dict[str, Path],
    output_dir: Path,
    *,
    threads: int = 1,
    visualize: bool = False,
) -> SpathiConfig:
    return SpathiConfig(
        expression=input_files["expression"],
        tf_list=input_files["tf_list"],
        groups=input_files["groups"],
        output_dir=output_dir,
        distance_space="pca",
        n_components=50,
        weight_mode="cell-distance-group-anchored",
        n_estimators=12,
        random_seed=91,
        threads=threads,
        visualize=visualize,
    )


@pytest.mark.integration
def test_public_pipeline_writes_self_contained_deterministic_run(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    output_dir = tmp_path / "run"
    result = infer_group_specific_grns(config_for(input_files, output_dir, visualize=True))
    assert result.output_dir == output_dir
    assert {path.name for path in output_dir.iterdir()} == EXPECTED_ARTIFACTS
    assert result.total_models == 8

    network = pd.read_csv(output_dir / "network.csv", keep_default_na=False)
    assert network.columns.tolist() == [
        "source",
        "target",
        "score",
        "sign",
        "evidence",
        "context",
    ]
    assert (network["score"] > 0).all()
    assert (network["sign"] == "?").all()
    assert all(network["source"] != network["target"])
    assert network[["context", "target", "source"]].values.tolist() == sorted(
        network[["context", "target", "source"]].values.tolist()
    )

    with gzip.open(output_dir / "cell_weights.tsv.gz", "rt", encoding="utf-8") as handle:
        weights = pd.read_csv(handle, sep="\t")
    assert len(weights) == 8
    assert set(weights["target_group"]) == {"A", "B"}
    assert set(weights["cell"]) == {"cell_1", "cell_2", "cell_3", "cell_4"}

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["input_dimensions"] == {
        "cells": 4,
        "genes": 4,
        "groups": 2,
        "targets": 4,
        "transcription_factors": 2,
    }
    embedding = pd.read_csv(output_dir / "cell_embedding.tsv.gz", sep="\t")
    assert embedding.columns.tolist() == ["cell", "group", "PC1", "PC2", "PC3"]
    assert embedding["cell"].tolist() == ["cell_1", "cell_2", "cell_3", "cell_4"]
    assert embedding["group"].tolist() == ["A", "A", "B", "B"]

    explained_variance = pd.read_csv(output_dir / "pca_explained_variance.tsv", sep="\t")
    assert explained_variance.columns.tolist() == [
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    ]
    assert explained_variance["component"].tolist() == ["PC1", "PC2", "PC3"]
    assert explained_variance["cumulative_explained_variance_ratio"].is_monotonic_increasing

    assert metadata["effective_parameters"]["effective_n_components"] == 3
    assert metadata["effective_parameters"]["maximum_informative_n_components"] == 3
    assert metadata["effective_parameters"]["pca_svd_solver_requested"] == "auto"
    assert (
        metadata["effective_parameters"]["pca_svd_solver_resolution"] == "delegated-to-scikit-learn"
    )
    np.testing.assert_allclose(
        metadata["effective_parameters"]["pca_explained_variance_ratio"],
        explained_variance["explained_variance_ratio"],
    )
    np.testing.assert_allclose(
        metadata["effective_parameters"]["pca_cumulative_explained_variance_ratio"],
        explained_variance["cumulative_explained_variance_ratio"],
    )
    assert metadata["effective_parameters"]["tree_target_dtype"] == "float64"
    assert metadata["effective_parameters"]["tree_predictor_dtype"] == "float32"
    assert metadata["effective_parameters"]["bootstrap_requested"] is None
    assert metadata["effective_parameters"]["bootstrap_effective"] is False
    assert metadata["effective_parameters"]["targets_per_batch"] == 4
    assert metadata["memory_estimate_bytes"]["maximum_target_batch_edge_records"] == 8
    assert (
        metadata["memory_estimate_bytes"]["group_constant_filter_predictors_float32_upper_bound"]
        >= 0
    )
    assert metadata["memory_estimate_bytes"]["visualization_retained_rough_bytes"] > 0
    assert metadata["memory_estimate_bytes"]["visualization_panel_working_rough_bytes"] > 0
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_heap_float64"] == 64
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_mapped_float64"] == 0
    assert metadata["effective_parameters"]["cell_centroid_distances_computed"] is True
    assert (
        metadata["artifact_semantics"]["group_affinities.tsv"]["authoritative_model_weights"]
        == "cell_weights.tsv.gz:final_weight"
    )
    assert metadata["artifact_semantics"]["cell_embedding.tsv.gz"]["components"] == [
        "PC1",
        "PC2",
        "PC3",
    ]
    assert metadata["models"]["completed"] == 8
    assert metadata["parallelism"]["threads_requested"] == 1
    assert len(metadata["inputs"]["expression"]["sha256"]) == 64
    assert Path(metadata["inputs"]["expression"]["path"]).is_absolute()
    assert metadata["visualizations"]["requested"] is True
    assert metadata["visualizations"]["generated"] is True
    manifest_path = output_dir / metadata["visualizations"]["artifacts"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["projection"]["kind"] == "distance-pca"
    assert len(manifest["figures"]) == 3
    assert {figure["kind"] for figure in manifest["figures"]} == {
        "effective-mass-heatmap",
        "target-weight-panel",
    }
    for figure in manifest["figures"]:
        figure_path = output_dir / figure["relative_path"]
        assert figure_path.is_file()
        assert len(figure["sha256"]) == 64


@pytest.mark.integration
def test_pipeline_is_equivalent_with_more_than_one_thread(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    sequential_dir = tmp_path / "sequential"
    threaded_dir = tmp_path / "threaded"
    infer_group_specific_grns(config_for(input_files, sequential_dir, threads=1))
    infer_group_specific_grns(config_for(input_files, threaded_dir, threads=2))
    assert (sequential_dir / "network.csv").read_bytes() == (
        threaded_dir / "network.csv"
    ).read_bytes()
    assert (sequential_dir / "cell_weights.tsv.gz").read_bytes() == (
        threaded_dir / "cell_weights.tsv.gz"
    ).read_bytes()


@pytest.mark.integration
def test_pipeline_switches_to_disk_backed_distances_above_threshold(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "_DISTANCE_MEMMAP_THRESHOLD_BYTES", 1)
    original_compute = pipeline_module.compute_cell_to_centroid_distances
    output_layouts: list[tuple[bool, bool]] = []

    def capture_layout(*args: object, **kwargs: object) -> pd.DataFrame:
        output = kwargs.get("output")
        assert isinstance(output, np.memmap)
        output_layouts.append((output.flags.c_contiguous, output.flags.f_contiguous))
        return original_compute(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        pipeline_module,
        "compute_cell_to_centroid_distances",
        capture_layout,
    )
    output_dir = tmp_path / "disk-backed"
    infer_group_specific_grns(config_for(input_files, output_dir))
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["effective_parameters"]["cell_centroid_distance_storage"] == "temporary-memory-map"
    )
    assert output_layouts == [(False, True)]
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_heap_float64"] == 0
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_mapped_float64"] > 0


@pytest.mark.integration
def test_group_distance_does_not_compute_cell_to_centroid_distances(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("group-distance must not materialize the cell distance matrix")

    monkeypatch.setattr(
        pipeline_module,
        "compute_cell_to_centroid_distances",
        unexpected_materialization,
    )
    output_dir = tmp_path / "group-distance"
    config = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **config.to_dict(),
            "weight_mode": "group-distance",
        }
    )
    infer_group_specific_grns(config)
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["cell_centroid_distance_storage"] == "not-computed"
    assert metadata["effective_parameters"]["cell_centroid_distances_computed"] is False
    assert metadata["effective_parameters"]["distance_chunk_working_memory_mib"] is None
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_heap_float64"] == 0
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_mapped_float64"] == 0
    assert metadata["memory_estimate_bytes"]["distance_chunk_working_memory_upper_bound"] == 0


@pytest.mark.integration
def test_random_forest_uses_automatic_bootstrap_default(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    output_dir = tmp_path / "random-forest"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(**{**base.to_dict(), "tree_method": "random-forest"})
    infer_group_specific_grns(config)
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["bootstrap_requested"] is None
    assert metadata["effective_parameters"]["bootstrap_effective"] is True


@pytest.mark.integration
def test_prepublication_failure_leaves_requested_output_available_for_retry(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    output_dir = tmp_path / "retryable"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "weight_mode": "cell-distance",
            "group_size_correction": "none",
            "bandwidth": 1e-300,
        }
    )

    with pytest.raises(ValueError, match="All corrected weights are zero"):
        infer_group_specific_grns(config)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".retryable.staging-*"))


@pytest.mark.integration
def test_expression_visualization_is_auxiliary_and_respects_thread_budget(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_thread_counts: list[int] = []
    original_prepare = visualization_module.prepare_visualization_embedding

    def capture_thread_budget(*args: object, **kwargs: object):
        from threadpoolctl import threadpool_info

        observed_thread_counts.extend(
            int(pool["num_threads"])
            for pool in threadpool_info()
            if pool.get("num_threads") is not None
        )
        return original_prepare(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        visualization_module,
        "prepare_visualization_embedding",
        capture_thread_budget,
    )
    output_dir = tmp_path / "expression-visualized"
    base = config_for(input_files, output_dir, threads=1, visualize=True)
    config = SpathiConfig(**{**base.to_dict(), "distance_space": "expression"})

    infer_group_specific_grns(config)

    assert all(thread_count == 1 for thread_count in observed_thread_counts)
    embedding = pd.read_csv(output_dir / "cell_embedding.tsv.gz", sep="\t")
    assert embedding.columns.tolist() == [
        "cell",
        "group",
        "AuxiliaryPC1",
        "AuxiliaryPC2",
    ]
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    semantics = metadata["artifact_semantics"]["cell_embedding.tsv.gz"]
    assert semantics["projection_role"] == "visualization-only"
    assert metadata["memory_estimate_bytes"]["visualization_auxiliary_pca_working_rough_bytes"] > 0
    manifest = json.loads(
        (output_dir / metadata["visualizations"]["artifacts"]["manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["projection"]["kind"] == "auxiliary-pca"
    assert manifest["projection"]["distance_space"] == "expression"


@pytest.mark.integration
def test_visualization_failure_is_atomic_and_leaves_output_retryable(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_render(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated visualization failure")

    monkeypatch.setattr(visualization_module, "write_target_weight_panel", fail_render)
    output_dir = tmp_path / "visualization-retryable"

    with pytest.raises(RuntimeError, match="simulated visualization failure"):
        infer_group_specific_grns(config_for(input_files, output_dir, visualize=True))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".visualization-retryable.staging-*"))


@pytest.mark.integration
def test_completed_model_failure_is_published_with_diagnostics(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingEstimator:
        def fit(self, *args: object, **kwargs: object) -> None:
            raise ValueError("simulated fit failure")

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FailingEstimator(),
    )
    output_dir = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match=str(output_dir / "model_diagnostics.tsv.gz")):
        infer_group_specific_grns(config_for(input_files, output_dir))

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["models"]["fit_or_importance_failures"] > 0
    assert (output_dir / "model_diagnostics.tsv.gz").is_file()


@pytest.mark.integration
def test_cli_smoke_and_existing_output_failure(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    output_dir = tmp_path / "cli-run"
    command = [
        sys.executable,
        "-m",
        "spathi",
        "infer",
        "--expression",
        str(input_files["expression"]),
        "--tf-list",
        str(input_files["tf_list"]),
        "--groups",
        str(input_files["groups"]),
        "--output-dir",
        str(output_dir),
        "--distance-space",
        "expression",
        "--n-estimators",
        "8",
        "--threads",
        "2",
        "--no-visualize",
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert (output_dir / "network.csv").is_file()
    assert not (output_dir / "cell_embedding.tsv.gz").exists()
    assert not (output_dir / "pca_explained_variance.tsv").exists()
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["effective_n_components"] is None
    assert metadata["effective_parameters"]["pca_explained_variance_ratio"] is None
    assert metadata["artifact_semantics"]["cell_embedding.tsv.gz"] is None
    assert metadata["visualizations"] == {
        "artifacts": None,
        "generated": False,
        "requested": False,
    }
    assert metadata["memory_estimate_bytes"]["visualization_retained_rough_bytes"] == 0
    assert metadata["memory_estimate_bytes"]["visualization_panel_working_rough_bytes"] == 0
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode == 2
    assert "will not be overwritten" in second.stderr
