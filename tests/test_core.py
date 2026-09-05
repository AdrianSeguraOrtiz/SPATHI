from __future__ import annotations

import gzip
import json
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile as real_named_temporary_file
from threading import get_ident
from typing import Any

import numpy as np
import pandas as pd
import pytest

import spathi._report as report_module
import spathi._workflow as workflow_module
import spathi.checkpoint as checkpoint_module
import spathi.core as core_module
import spathi.inference as inference_module
import spathi.io as io_module
from spathi import SpathiConfig, infer
from spathi.progress import SpathiProgressEvent

EXPECTED_ARTIFACTS = {
    "cell_embedding.tsv.gz",
    "cell_weights.tsv.gz",
    "centroid_weight_diagnostics.tsv",
    "centroid_weights.tsv.gz",
    "centroids.tsv",
    "group_affinities.tsv",
    "group_distances.tsv",
    "model_diagnostics.tsv.gz",
    "network.csv",
    "parameters.json",
    "pca_explained_variance.tsv",
    "run_metadata.json",
    "skipped_targets.tsv",
    "target_eligibility.tsv.gz",
    "weight_diagnostics.tsv",
    "report.html",
}


def config_for(
    input_files: dict[str, Path],
    output_dir: Path,
    *,
    threads: int = 1,
    report: bool = False,
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
        report=report,
    )


def test_report_dependency_is_provenance_but_not_checkpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_module,
        "version",
        lambda distribution: {
            "numpy": "1",
            "pandas": "2",
            "scipy": "2.5",
            "scikit-learn": "3",
            "joblib": "4",
            "threadpoolctl": "5",
            "plotly": "6",
        }[distribution],
    )

    scientific = workflow_module.scientific_dependency_versions()
    complete = workflow_module.dependency_versions(include_report=True)

    assert "plotly" not in scientific
    assert complete == {**scientific, "plotly": "6"}


def test_report_parameters_exclude_local_paths(
    input_files: dict[str, Path], tmp_path: Path
) -> None:
    config = config_for(input_files, tmp_path / "private-output", report=True)

    parameters = workflow_module._report_run_parameters(config)

    assert parameters["weight_mode"] == config.weight_mode
    assert parameters["report"] is True
    assert parameters["target_selection"] == "all-expression-genes"
    assert parameters["centroid_method"] == "arithmetic_mean"
    assert parameters["centroid_weight_source"] == "uniform"
    assert parameters["centroid_weight_analysis_role"] == "primary"
    assert not {
        "expression",
        "tf_list",
        "groups",
        "target_list",
        "centroid_weights",
        "output_dir",
    }.intersection(parameters)


def test_centroid_weight_path_is_fingerprinted_not_a_scientific_parameter(
    input_files: dict[str, Path], tmp_path: Path
) -> None:
    first = config_for(input_files, tmp_path / "output")
    first = SpathiConfig(
        **{
            **first.to_dict(),
            "centroid_weights": tmp_path / "first.tsv",
        }
    )
    second = SpathiConfig(
        **{
            **first.to_dict(),
            "centroid_weights": tmp_path / "moved.tsv",
        }
    )

    assert core_module._checkpoint_scientific_parameters(first) == (
        core_module._checkpoint_scientific_parameters(second)
    )
    assert "centroid_weights" not in core_module._checkpoint_scientific_parameters(first)


def test_centroid_weight_diagnostics_are_stable_when_raw_sum_exceeds_float64() -> None:
    maximum = np.finfo(np.float64).max
    cells = ["c1", "c2", "c3"]
    groups = pd.Series(["A", "A", "B"], index=cells, dtype="string")
    supplied = pd.Series([maximum, maximum, 1.0], index=cells, dtype=np.float64)

    diagnostics = workflow_module._prepare_centroid_weight_data(
        groups,
        cell_ids=cells,
        group_ids=["A", "B"],
        supplied=supplied,
    )

    np.testing.assert_allclose(diagnostics.normalized, [0.5, 0.5, 1.0])
    assert diagnostics.summaries["A"]["weight_sum"] == "3.5953862697246314E+308"
    assert diagnostics.summaries["A"]["effective_sample_size"] == 2.0


def test_cell_distance_memory_plan_uses_live_headroom_below_size_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 2_000)

    plan = workflow_module._plan_cell_distance_memory(
        n_cells=100,
        n_groups=2,
        n_dimensions=3,
        compute_distances=True,
        distance_metric="euclidean",
    )

    assert plan.expected_output_bytes == 1_600
    assert plan.usable_bytes == 1_400
    assert plan.storage == "temporary-memory-map"
    assert plan.storage_reason == "available-memory"
    assert plan.working_memory_bytes == 1_400


def test_cell_distance_memory_plan_fails_if_one_chunk_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 10)

    with pytest.raises(MemoryError, match="one cell-to-centroid distance chunk"):
        workflow_module._plan_cell_distance_memory(
            n_cells=100,
            n_groups=2,
            n_dimensions=3,
            compute_distances=True,
            distance_metric="euclidean",
        )


def test_centroid_distance_memory_plan_reserves_dense_group_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 200_000)

    plan = workflow_module._plan_centroid_distance_memory(
        n_groups=100,
        n_dimensions=10,
    )

    assert plan.estimated_persistent_bytes == 96_000
    assert plan.usable_bytes == 140_000
    assert plan.working_memory_bytes == 44_000


def test_centroid_distance_memory_plan_fails_before_dense_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 100_000)

    with pytest.raises(MemoryError, match="centroids and group distances"):
        workflow_module._plan_centroid_distance_memory(
            n_groups=100,
            n_dimensions=10,
        )


def test_centroid_plan_rejects_budget_below_one_stable_euclidean_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # usable=floor(138*0.7)=96 and persistent=64 leave 32 bytes, while a
    # one-dimensional stable row for two groups needs 8D + 24G = 56 bytes.
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 138)

    with pytest.raises(MemoryError, match="centroids and group distances"):
        workflow_module._plan_centroid_distance_memory(
            n_groups=2,
            n_dimensions=1,
        )


@pytest.mark.integration
def test_cell_distance_headroom_is_measured_after_group_distances(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_compute = workflow_module.compute_centroid_distances
    group_distances_ready = False
    observed_snapshots: list[bool] = []

    def capture_group_distances(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal group_distances_ready
        result = original_compute(*args, **kwargs)  # type: ignore[arg-type]
        group_distances_ready = True
        return result

    def capture_headroom() -> None:
        observed_snapshots.append(group_distances_ready)
        return None

    monkeypatch.setattr(
        workflow_module,
        "compute_centroid_distances",
        capture_group_distances,
    )
    monkeypatch.setattr(workflow_module, "available_memory_bytes", capture_headroom)

    infer(config_for(input_files, tmp_path / "distance-plan-order"), checkpoint=False)

    assert observed_snapshots[:2] == [False, True]


def test_atomic_publication_is_preflighted_before_input_loading(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "unsupported-publication"
    loaded_inputs = False

    def reject_publication(parent: Path) -> None:
        assert parent == tmp_path
        raise RuntimeError("unsupported output filesystem")

    def observe_input_load(*args: object, **kwargs: object) -> None:
        nonlocal loaded_inputs
        loaded_inputs = True

    monkeypatch.setattr(core_module, "preflight_atomic_publication", reject_publication)
    monkeypatch.setattr(io_module, "load_inputs", observe_input_load)

    with pytest.raises(RuntimeError, match="unsupported output filesystem"):
        infer(config_for(input_files, output_dir))

    assert loaded_inputs is False
    assert not output_dir.exists()
    assert not (tmp_path / ".unsupported-publication.checkpoint").exists()


@pytest.mark.integration
def test_infer_cannot_replace_output_created_during_publication_race(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "publication-race"
    original_occupied = core_module.path_is_occupied
    output_checks = 0

    def create_destination_after_last_check(path: Path) -> bool:
        nonlocal output_checks
        if path == output_dir:
            output_checks += 1
            if output_checks == 2:
                output_dir.mkdir()
                return False
        return original_occupied(path)

    monkeypatch.setattr(core_module, "path_is_occupied", create_destination_after_last_check)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        infer(config_for(input_files, output_dir), checkpoint=False)

    assert output_checks == 2
    assert list(output_dir.iterdir()) == []


def test_batch_memory_plan_preserves_parallelism_then_reduces_group_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {
        "n_cells": 10,
        "n_groups": 4,
        "n_targets": 2,
        "n_transcription_factors": 2,
        "predictor_bytes": 1_000,
        "numeric_thread_limit": 8,
        "estimated_model_bytes": 100,
        "report_retained_bytes": 0,
        "report_auxiliary_bytes": 0,
        "report_render_bytes": 0,
        "checkpoint_enabled": True,
    }
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 1_000_000)
    unconstrained = workflow_module._plan_inference_batches(**parameters)
    assert unconstrained.desired_group_batch_size == 4
    assert unconstrained.group_batch_size == 4
    assert unconstrained.concurrent_fits == 8

    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 15_000)
    constrained = workflow_module._plan_inference_batches(**parameters)
    assert constrained.group_batch_size == 2
    assert constrained.concurrent_fits == 4
    assert constrained.model_plan.reserved_bytes == constrained.reserved_bytes


def test_batch_memory_plan_reduces_non_checkpointed_target_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 100_000)
    parameters = {
        "n_cells": 10,
        "n_groups": 4,
        "n_targets": 100,
        "n_transcription_factors": 10,
        "predictor_bytes": 1_000,
        "numeric_thread_limit": 8,
        "estimated_model_bytes": 100,
        "report_retained_bytes": 0,
        "report_auxiliary_bytes": 0,
        "report_render_bytes": 0,
    }

    streamed = workflow_module._plan_inference_batches(
        **parameters,
        checkpoint_enabled=True,
    )
    retained = workflow_module._plan_inference_batches(
        **parameters,
        checkpoint_enabled=False,
    )

    assert streamed.target_batch_size == 32
    assert retained.target_batch_size < streamed.target_batch_size
    assert retained.concurrent_fits == 8


def test_checkpoint_memory_plan_reserves_the_complete_rolling_result_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 10_000_000)

    plan = workflow_module._plan_inference_batches(
        n_cells=10,
        n_groups=1,
        n_targets=100,
        n_transcription_factors=10,
        predictor_bytes=1_000,
        numeric_thread_limit=4,
        estimated_model_bytes=100,
        report_retained_bytes=0,
        report_auxiliary_bytes=0,
        report_render_bytes=0,
        checkpoint_enabled=True,
    )

    pending_results = min(
        plan.models_per_inference_batch,
        4 * workflow_module.MAX_PENDING_TASKS_PER_WORKER,
    )
    result_bytes = pending_results * (10 * 256 + 768)
    retained_group_bytes = 10 * 8 + 10 + 1_000
    weight_working_bytes = 4 * 10 * 8
    assert pending_results == 8
    assert plan.reserved_bytes == result_bytes + retained_group_bytes + weight_working_bytes


def test_batch_memory_plan_reserves_automatic_target_eligibility_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 10_000_000_000)
    parameters = {
        "n_cells": 1_000,
        "n_groups": 3,
        "n_targets": 300,
        "n_transcription_factors": 20,
        "predictor_bytes": 80_000,
        "numeric_thread_limit": 8,
        "estimated_model_bytes": 100_000,
        "report_retained_bytes": 0,
        "report_auxiliary_bytes": 0,
        "report_render_bytes": 0,
        "checkpoint_enabled": True,
    }

    reference = workflow_module._plan_inference_batches(
        **parameters,
        automatic_target_eligibility=False,
    )
    automatic = workflow_module._plan_inference_batches(
        **parameters,
        automatic_target_eligibility=True,
    )

    expected_extra = 1_000 * np.dtype(np.float64).itemsize
    expected_extra += 1_000 * automatic.target_batch_size * np.dtype(np.bool_).itemsize
    assert automatic.group_batch_size == 1
    assert automatic.reserved_bytes == reference.reserved_bytes + expected_extra


def test_batch_memory_plan_fails_before_an_infeasible_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 1_000)

    with pytest.raises(MemoryError, match="Insufficient available memory"):
        workflow_module._plan_inference_batches(
            n_cells=10,
            n_groups=1,
            n_targets=1,
            n_transcription_factors=2,
            predictor_bytes=1_000,
            numeric_thread_limit=1,
            estimated_model_bytes=100,
            report_retained_bytes=0,
            report_auxiliary_bytes=0,
            report_render_bytes=0,
            checkpoint_enabled=True,
        )


def test_resume_target_eligibility_block_obeys_live_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 10_000)
    plan = workflow_module._plan_target_eligibility_memory(
        n_cells=100,
        n_targets=20,
    )
    assert plan.available_bytes == 10_000
    assert plan.usable_bytes == 7_000
    bytes_per_target = 100 * (np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize) + (
        np.dtype(np.intp).itemsize + 2 * np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize
    )
    assert plan.block_size == 7_000 // bytes_per_target
    assert plan.working_bytes == plan.block_size * bytes_per_target

    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 1_000)
    with pytest.raises(MemoryError, match="one target eligibility column"):
        workflow_module._plan_target_eligibility_memory(
            n_cells=100,
            n_targets=20,
        )


def test_inference_preparation_memory_is_preflighted_before_matrix_copies(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_predictors = False

    def reject_preparation(**_kwargs: object) -> None:
        raise MemoryError("preparation budget rejected")

    def observe_predictor_copy(*_args: object, **_kwargs: object) -> None:
        nonlocal copied_predictors
        copied_predictors = True

    monkeypatch.setattr(
        workflow_module,
        "_plan_inference_preparation_memory",
        reject_preparation,
    )
    monkeypatch.setattr(inference_module, "_extract_tf_predictors", observe_predictor_copy)

    with pytest.raises(MemoryError, match="preparation budget rejected"):
        infer(config_for(input_files, tmp_path / "preparation-preflight"), checkpoint=False)

    assert copied_predictors is False


def test_inference_preparation_memory_plan_accounts_for_persistent_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 200_000)

    plan = workflow_module._plan_inference_preparation_memory(
        n_cells=100,
        n_genes=1_000,
        n_transcription_factors=10,
        n_targets=20,
        target_subset=True,
        automatic_target_eligibility=True,
    )

    assert plan.predictor_bytes == 4_000
    assert plan.additional_target_bytes == 16_000
    assert plan.maximum_validation_working_bytes == 100_000
    assert plan.estimated_peak_additional_bytes == 100_000
    assert plan.usable_bytes == 140_000

    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 100_000)
    with pytest.raises(MemoryError, match="prepare SPATHI inference matrices"):
        workflow_module._plan_inference_preparation_memory(
            n_cells=100,
            n_genes=1_000,
            n_transcription_factors=10,
            n_targets=20,
            target_subset=True,
            automatic_target_eligibility=True,
        )


@pytest.mark.integration
def test_memory_limited_batch_spends_idle_cpu_budget_inside_one_model(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_planner = workflow_module._plan_inference_batches

    def restrict_concurrent_fits(**kwargs: object):
        plan = original_planner(**kwargs)  # type: ignore[arg-type]
        return replace(plan, concurrent_fits=2)

    monkeypatch.setattr(workflow_module, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(workflow_module, "_plan_inference_batches", restrict_concurrent_fits)
    output_dir = tmp_path / "memory-limited-workers"

    infer(config_for(input_files, output_dir, threads=4), checkpoint=False)

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parallelism"]["memory_concurrent_model_cap"] == 2
    assert metadata["parallelism"]["maximum_concurrent_model_fits"] == 1
    assert metadata["parallelism"]["threads_effective"] == 4
    assert metadata["parallelism"]["parallel_level"] == "estimator"


def test_single_group_pca_cosine_fails_before_scientific_preprocessing(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_files["groups"].write_text(
        "sample\tcluster\ncell_1\tA\ncell_2\tA\ncell_3\tA\ncell_4\tA\n",
        encoding="utf-8",
    )

    def unexpected_preprocessing(*args: object, **kwargs: object) -> None:
        raise AssertionError("representation must not be built for an invalid geometry")

    monkeypatch.setattr(
        workflow_module,
        "compute_distance_representation",
        unexpected_preprocessing,
    )
    output_dir = tmp_path / "single-group-pca-cosine"
    with pytest.raises(
        ValueError,
        match="single-group dataset must infer one individually cell-weighted network",
    ):
        infer(config_for(input_files, output_dir), checkpoint=False)

    assert not output_dir.exists()


def test_single_group_standardized_expression_cosine_fails_before_preprocessing(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_files["groups"].write_text(
        "sample\tcluster\ncell_1\tA\ncell_2\tA\ncell_3\tA\ncell_4\tA\n",
        encoding="utf-8",
    )

    def unexpected_preprocessing(*args: object, **kwargs: object) -> None:
        raise AssertionError("representation must not be built for an invalid geometry")

    monkeypatch.setattr(
        workflow_module,
        "compute_distance_representation",
        unexpected_preprocessing,
    )
    output_dir = tmp_path / "single-group-standardized-expression-cosine"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "distance_space": "expression",
            "distance_standardization": "standard",
            "distance_metric": "cosine",
        }
    )

    with pytest.raises(ValueError, match="centered distance spaces require"):
        infer(config, checkpoint=False)

    assert not output_dir.exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("distance_space", "distance_metric"),
    [("pca", "euclidean"), ("expression", "cosine")],
)
def test_single_group_remains_valid_outside_pca_cosine(
    tmp_path: Path,
    input_files: dict[str, Path],
    distance_space: str,
    distance_metric: str,
) -> None:
    input_files["groups"].write_text(
        "sample\tcluster\ncell_1\tA\ncell_2\tA\ncell_3\tA\ncell_4\tA\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / f"single-group-{distance_space}-{distance_metric}"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "weight_mode": "cell-distance",
            "group_size_correction": "none",
            "distance_space": distance_space,
            "distance_metric": distance_metric,
        }
    )

    result = infer(config, checkpoint=False)

    assert result.total_models == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"weight_mode": "cell-distance-group-anchored", "group_size_correction": "none"},
            "weight_mode must be 'cell-distance'",
        ),
        (
            {"weight_mode": "group-distance", "group_size_correction": "none"},
            "weight_mode must be 'cell-distance'",
        ),
        (
            {"weight_mode": "cell-distance", "group_size_correction": "cap-to-target"},
            "group_size_correction must be 'none'",
        ),
    ],
)
def test_single_group_rejects_semantically_incompatible_weighting_before_checkpoint(
    tmp_path: Path,
    input_files: dict[str, Path],
    overrides: dict[str, object],
    message: str,
) -> None:
    input_files["groups"].write_text(
        "sample\tcluster\ncell_1\tA\ncell_2\tA\ncell_3\tA\ncell_4\tA\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "invalid-single-group"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(**{**base.to_dict(), **overrides})

    with pytest.raises(ValueError, match=message):
        infer(config)

    assert not output_dir.exists()
    assert not (tmp_path / ".invalid-single-group.checkpoint").exists()


@pytest.mark.integration
def test_expression_gene_named_group_is_preserved_in_long_centroid_output(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    expression = pd.read_csv(input_files["expression"], sep="\t", index_col=0)
    expression = expression.rename(index={"G3": "group"})
    expression.to_csv(input_files["expression"], sep="\t", index_label="gene")
    output_dir = tmp_path / "gene-named-group"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "distance_space": "expression",
            "distance_metric": "euclidean",
        }
    )

    infer(config, checkpoint=False)

    centroids = pd.read_csv(output_dir / "centroids.tsv", sep="\t")
    assert centroids.columns.tolist() == ["group", "dimension", "centroid"]
    assert "group" in set(centroids["dimension"])


@pytest.mark.integration
def test_public_core_writes_self_contained_deterministic_run(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    output_dir = tmp_path / "run"
    config = replace(config_for(input_files, output_dir, report=True), bandwidth_scale=0.5)
    result = infer(config)
    assert result.output_dir == output_dir
    assert result.report_path == output_dir / "report.html"
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
    parameters = json.loads((output_dir / "parameters.json").read_text(encoding="utf-8"))
    assert parameters["bandwidth"] == "auto"
    assert parameters["bandwidth_scale"] == 0.5
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
    bandwidth = metadata["effective_parameters"]["bandwidth"]
    assert bandwidth["method"] == "auto-median"
    assert bandwidth["automatic_scale"] == 0.5
    assert bandwidth["value"] == pytest.approx(
        bandwidth["automatic_reference_value"] * bandwidth["automatic_scale"]
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
    assert metadata["effective_parameters"]["tree_budget"] == {
        "mode": "fixed",
        "schedule_active": False,
        "maximum_estimators": 12,
        "minimum_estimators": None,
        "estimator_step": None,
        "convergence_tolerance": None,
        "convergence_patience": None,
        "maximum_convergence_checks": None,
        "earliest_possible_stop_estimators": None,
    }
    assert metadata["effective_parameters"]["target_eligibility"] == {
        "mode": "all",
        "thresholds_active": False,
        "min_detected_cells": None,
        "min_detected_fraction": None,
        "min_weighted_detected_fraction": None,
        "min_weighted_detected_ess": None,
        "globally_eligible_targets": 4,
        "globally_ineligible_targets": 0,
        "predictor_space_changed": False,
    }
    assert metadata["effective_parameters"]["targets_per_batch"] == 4
    assert metadata["effective_parameters"]["target_selection"] == "all-expression-genes"
    assert metadata["effective_parameters"]["target_ids"] is None
    assert metadata["memory_estimate_bytes"]["maximum_target_batch_edge_records"] == 8
    assert (
        metadata["memory_estimate_bytes"]["group_constant_filter_predictors_float32_upper_bound"]
        >= 0
    )
    assert metadata["memory_estimate_bytes"]["report_retained_rough_bytes"] > 0
    assert metadata["memory_estimate_bytes"]["report_sampled_vectors_float64"] == 256
    assert metadata["memory_estimate_bytes"]["report_exact_summary_matrices"] == 256
    assert metadata["memory_estimate_bytes"]["report_render_working_rough_bytes"] > 0
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
    assert metadata["report"]["requested"] is True
    assert metadata["report"]["generated"] is True
    artifact = metadata["report"]["artifact"]
    assert artifact["path"] == "report.html"
    assert artifact["self_contained"] is True
    assert artifact["sampled_cells"] == artifact["total_cells"] == 4
    assert len(artifact["sha256"]) == 64
    report_text = (output_dir / artifact["path"]).read_text(encoding="utf-8")
    assert "SPATHI interactive report" in report_text
    assert "<script src=" not in report_text
    for local_path in (*input_files.values(), output_dir):
        assert str(local_path.resolve()) not in report_text
    assert "cell_1" in report_text


@pytest.mark.integration
def test_explicit_centroid_weights_change_only_centroid_construction_and_are_audited(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    centroid_weights = tmp_path / "centroid_weights.tsv"
    centroid_weights.write_text(
        "cell\tcentroid_weight\ncell_1\t1\ncell_2\t3\ncell_3\t2\ncell_4\t1\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "weighted-centroids"
    base = config_for(input_files, output_dir, report=True)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "centroid_weights": centroid_weights,
            "distance_space": "expression",
            "distance_metric": "euclidean",
        }
    )

    infer(config, checkpoint=False)

    centroids = pd.read_csv(output_dir / "centroids.tsv", sep="\t")
    tf1 = centroids.loc[centroids["dimension"] == "TF1"].set_index("group")["centroid"]
    assert tf1["A"] == pytest.approx(1.75)
    assert tf1["B"] == pytest.approx(13.0 / 3.0)

    with gzip.open(output_dir / "centroid_weights.tsv.gz", "rt", encoding="utf-8") as handle:
        effective = pd.read_csv(handle, sep="\t")
    assert effective.columns.tolist() == [
        "cell",
        "group",
        "centroid_weight",
        "normalized_centroid_weight",
    ]
    np.testing.assert_allclose(
        effective["normalized_centroid_weight"],
        [0.25, 0.75, 2.0 / 3.0, 1.0 / 3.0],
    )
    diagnostics = pd.read_csv(output_dir / "centroid_weight_diagnostics.tsv", sep="\t")
    assert diagnostics.set_index("group").loc["A", "weight_sum"] == 4.0
    assert diagnostics.set_index("group").loc["A", "effective_sample_size"] == pytest.approx(1.6)

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["centroid_method"] == "weighted_mean"
    assert metadata["effective_parameters"]["centroid_weight_source"] == "explicit"
    assert (
        metadata["effective_parameters"]["centroid_weight_analysis_role"]
        == "explicit-sensitivity-analysis"
    )
    assert "centroid_weights" in metadata["inputs"]
    assert metadata["artifact_semantics"]["centroid_weights.tsv.gz"]["not_a_model_weight"] is True
    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "explicit generic centroid weights" in report


@pytest.mark.integration
def test_default_centroid_weights_are_uniform_and_explicitly_audited(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    output_dir = tmp_path / "uniform-centroids"
    infer(config_for(input_files, output_dir), checkpoint=False)

    with gzip.open(output_dir / "centroid_weights.tsv.gz", "rt", encoding="utf-8") as handle:
        effective = pd.read_csv(handle, sep="\t")
    np.testing.assert_array_equal(effective["centroid_weight"], np.ones(4))
    np.testing.assert_allclose(effective["normalized_centroid_weight"], np.full(4, 0.5))
    diagnostics = pd.read_csv(output_dir / "centroid_weight_diagnostics.tsv", sep="\t")
    np.testing.assert_allclose(diagnostics["effective_sample_size"], [2.0, 2.0])
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["centroid_method"] == "arithmetic_mean"
    assert metadata["effective_parameters"]["centroid_weight_source"] == "uniform"
    assert metadata["effective_parameters"]["centroid_weight_analysis_role"] == "primary"
    assert "centroid_weights" not in metadata["inputs"]


@pytest.mark.integration
def test_groupwise_constant_centroid_weights_do_not_reweight_models_or_group_sizes(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    centroid_weights = tmp_path / "groupwise-constant-centroid-weights.tsv"
    centroid_weights.write_text(
        "cell\tcentroid_weight\ncell_1\t2\ncell_2\t2\ncell_3\t7\ncell_4\t7\n",
        encoding="utf-8",
    )
    uniform_dir = tmp_path / "uniform-model-weights"
    explicit_dir = tmp_path / "explicit-model-weights"
    uniform = config_for(input_files, uniform_dir)
    explicit = SpathiConfig(
        **{
            **config_for(input_files, explicit_dir).to_dict(),
            "centroid_weights": centroid_weights,
        }
    )

    infer(uniform, checkpoint=False)
    infer(explicit, checkpoint=False)

    uniform_weights = pd.read_csv(uniform_dir / "cell_weights.tsv.gz", sep="\t")
    explicit_weights = pd.read_csv(explicit_dir / "cell_weights.tsv.gz", sep="\t")
    pd.testing.assert_frame_equal(uniform_weights, explicit_weights)
    uniform_metadata = json.loads((uniform_dir / "run_metadata.json").read_text(encoding="utf-8"))
    explicit_metadata = json.loads((explicit_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert (
        uniform_metadata["group_sizes"]
        == explicit_metadata["group_sizes"]
        == {
            "A": 2,
            "B": 2,
        }
    )


@pytest.mark.integration
def test_core_uses_all_genes_for_distances_but_only_explicit_inference_targets(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_list = tmp_path / "targets.txt"
    target_list.write_text("G3\n", encoding="utf-8")
    output_dir = tmp_path / "targeted"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(**{**base.to_dict(), "target_list": target_list})
    original_representation = workflow_module.compute_distance_representation
    observed_distance_genes: list[list[str]] = []

    def capture_representation(expression: pd.DataFrame, **kwargs: object):
        observed_distance_genes.append(list(map(str, expression.index)))
        return original_representation(expression, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        workflow_module,
        "compute_distance_representation",
        capture_representation,
    )

    result = infer(config)

    assert observed_distance_genes == [["TF1", "TF2", "G3", "CONST"]]
    assert result.total_models == 2
    network = pd.read_csv(output_dir / "network.csv", keep_default_na=False)
    assert set(network["target"]) == {"G3"}
    diagnostics = pd.read_csv(output_dir / "model_diagnostics.tsv.gz", sep="\t")
    assert diagnostics["target"].tolist() == ["G3", "G3"]

    parameters = json.loads((output_dir / "parameters.json").read_text(encoding="utf-8"))
    assert parameters["target_list"] == str(target_list)
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["input_dimensions"] == {
        "cells": 4,
        "genes": 4,
        "groups": 2,
        "targets": 1,
        "transcription_factors": 2,
    }
    assert metadata["effective_parameters"]["target_selection"] == "explicit-list"
    assert metadata["effective_parameters"]["target_ids"] == ["G3"]
    assert "target_list" in metadata["inputs"]
    assert metadata["memory_estimate_bytes"]["tree_targets_logical_float64"] == 32
    assert metadata["memory_estimate_bytes"]["tree_targets_additional_float64"] == 32


@pytest.mark.integration
def test_target_subset_releases_complete_expression_before_model_fitting(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_list = tmp_path / "targets.txt"
    target_list.write_text("G3\n", encoding="utf-8")
    base = config_for(input_files, tmp_path / "released-expression")
    config = SpathiConfig(**{**base.to_dict(), "target_list": target_list})
    original_load = io_module.load_inputs
    original_fit = inference_module._fit_model_task
    expression_reference: list[weakref.ReferenceType[pd.DataFrame]] = []

    def capture_expression(*args: object, **kwargs: object):
        loaded = original_load(*args, **kwargs)
        expression_reference.append(weakref.ref(loaded.expression))
        return loaded

    def assert_released(task, context):
        assert expression_reference[0]() is None
        return original_fit(task, context)

    monkeypatch.setattr(io_module, "load_inputs", capture_expression)
    monkeypatch.setattr(inference_module, "_fit_model_task", assert_released)

    infer(config, checkpoint=False)


@pytest.mark.integration
def test_automatic_eligibility_and_adaptive_budget_are_fully_auditable(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    target_list = tmp_path / "targets.txt"
    target_list.write_text("G3\n", encoding="utf-8")
    output_dir = tmp_path / "auditable-optimizations"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "target_list": target_list,
            "n_estimators": 20,
            "adaptive_trees": True,
            "adaptive_min_estimators": 5,
            "adaptive_tree_step": 5,
            "adaptive_tolerance": 1.0,
            "adaptive_patience": 2,
            "target_eligibility": "automatic",
            "min_target_detected_cells": 2,
            "min_target_detected_fraction": 0.25,
            "min_target_weighted_detected_fraction": 0.01,
            "min_target_weighted_detected_ess": 1.0,
        }
    )

    infer(config)

    eligibility = pd.read_csv(output_dir / "target_eligibility.tsv.gz", sep="\t")
    assert eligibility.to_dict(orient="records") == [
        {
            "target": "G3",
            "mode": "automatic",
            "eligible": True,
            "detected_cells": 4,
            "detected_fraction": 1.0,
            "expression_min": 1.0,
            "expression_max": 6.0,
            "required_detected_cells": 2,
            "reason": "eligible",
        }
    ]
    diagnostics = pd.read_csv(output_dir / "model_diagnostics.tsv.gz", sep="\t")
    assert diagnostics["n_estimators_fitted"].tolist() == [15, 15]
    assert diagnostics["adaptive_converged"].tolist() == [True, True]
    assert diagnostics["target_weighted_detected_fraction"].between(0.0, 1.0).all()
    assert (diagnostics["target_weighted_detected_ess"] >= 1.0).all()

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["target_eligibility"] == {
        "mode": "automatic",
        "thresholds_active": True,
        "min_detected_cells": 2,
        "min_detected_fraction": 0.25,
        "min_weighted_detected_fraction": 0.01,
        "min_weighted_detected_ess": 1.0,
        "globally_eligible_targets": 1,
        "globally_ineligible_targets": 0,
        "predictor_space_changed": False,
    }
    assert metadata["effective_parameters"]["tree_budget"] == {
        "mode": "adaptive",
        "schedule_active": True,
        "maximum_estimators": 20,
        "minimum_estimators": 5,
        "estimator_step": 5,
        "convergence_tolerance": 1.0,
        "convergence_patience": 2,
        "maximum_convergence_checks": 3,
        "earliest_possible_stop_estimators": 15,
    }
    assert metadata["models"]["adaptive_converged"] == 2
    assert metadata["models"]["adaptive_early_stopped"] == 2
    assert metadata["models"]["fitted_estimators_total"] == 30
    assert metadata["memory_estimate_bytes"]["tree_importance_buffer_float64"] == 320
    assert metadata["memory_estimate_bytes"]["adaptive_convergence_history_float64"] == 48
    assert metadata["checkpoint"]["model_storage"] == ("sqlite-binary-columnar-zlib-per-model")


@pytest.mark.integration
def test_expression_distance_representation_is_released_before_model_fitting(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_list = tmp_path / "targets.txt"
    target_list.write_text("G3\n", encoding="utf-8")
    base = config_for(input_files, tmp_path / "released-representation")
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "target_list": target_list,
            "distance_space": "expression",
            "distance_metric": "euclidean",
        }
    )
    original_representation = workflow_module.compute_distance_representation
    original_fit = inference_module._fit_model_task
    representation_reference: list[weakref.ReferenceType[np.ndarray]] = []

    def capture_representation(*args: object, **kwargs: object):
        result = original_representation(*args, **kwargs)  # type: ignore[arg-type]
        representation_reference.append(weakref.ref(result.values))
        return result

    def assert_released(task, context):
        assert representation_reference[0]() is None
        return original_fit(task, context)

    monkeypatch.setattr(
        workflow_module,
        "compute_distance_representation",
        capture_representation,
    )
    monkeypatch.setattr(inference_module, "_fit_model_task", assert_released)

    infer(config, checkpoint=False)


@pytest.mark.integration
def test_core_is_equivalent_with_more_than_one_thread(
    tmp_path: Path, input_files: dict[str, Path]
) -> None:
    sequential_dir = tmp_path / "sequential"
    threaded_dir = tmp_path / "threaded"
    infer(config_for(input_files, sequential_dir, threads=1))
    infer(config_for(input_files, threaded_dir, threads=2))
    assert (sequential_dir / "network.csv").read_bytes() == (
        threaded_dir / "network.csv"
    ).read_bytes()
    assert (sequential_dir / "cell_weights.tsv.gz").read_bytes() == (
        threaded_dir / "cell_weights.tsv.gz"
    ).read_bytes()
    assert (sequential_dir / "centroid_weights.tsv.gz").read_bytes() == (
        threaded_dir / "centroid_weights.tsv.gz"
    ).read_bytes()


@pytest.mark.integration
def test_randomized_pca_scientific_artifacts_are_exact_across_thread_budgets(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    first_dir = tmp_path / "randomized-one"
    second_dir = tmp_path / "randomized-two"
    first = config_for(input_files, first_dir, threads=1)
    second = config_for(input_files, second_dir, threads=2)
    first = SpathiConfig(**{**first.to_dict(), "pca_svd_solver": "randomized"})
    second = SpathiConfig(**{**second.to_dict(), "pca_svd_solver": "randomized"})

    infer(first, checkpoint=False)
    infer(second, checkpoint=False)

    for name in (
        "network.csv",
        "cell_weights.tsv.gz",
        "centroid_weights.tsv.gz",
        "centroid_weight_diagnostics.tsv",
        "cell_embedding.tsv.gz",
        "group_distances.tsv",
    ):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


@pytest.mark.integration
def test_scientific_preprocessing_uses_one_native_thread(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threadpoolctl import threadpool_info

    observed: list[int] = []

    def wrap(function):
        def capture(*args: object, **kwargs: object):
            observed.extend(
                int(pool["num_threads"])
                for pool in threadpool_info()
                if pool.get("num_threads") is not None
            )
            return function(*args, **kwargs)

        return capture

    for name in (
        "compute_distance_representation",
        "compute_centroids",
        "compute_centroid_distances",
        "compute_cell_to_centroid_distances",
        "resolve_bandwidth_for_mode",
    ):
        monkeypatch.setattr(workflow_module, name, wrap(getattr(workflow_module, name)))

    output_dir = tmp_path / "single-thread-preprocessing"
    infer(
        config_for(input_files, output_dir, threads=2),
        checkpoint=False,
    )

    assert observed
    assert max(observed) == 1
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parallelism"]["preprocessing_thread_limit"] == 1


@pytest.mark.integration
def test_core_emits_global_model_progress_on_the_caller_thread(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    events: list[SpathiProgressEvent] = []
    callback_threads: list[int] = []
    caller_thread = get_ident()

    def capture(event: SpathiProgressEvent) -> None:
        events.append(event)
        callback_threads.append(get_ident())

    infer(
        config_for(input_files, tmp_path / "progress", threads=2, report=True),
        progress_callback=capture,
        checkpoint=False,
    )

    model_events = [event for event in events if event.phase == "model_inference"]
    assert [event.completed_models for event in model_events] == list(range(1, 9))
    assert all(event.total_models == 8 for event in model_events)
    assert events[0].phase == "validating_inputs"
    phases = [event.phase for event in events]
    assert phases.index("writing_outputs") < phases.index("building_report")
    assert phases.index("building_report") < phases.index("publishing")
    assert events[-1].phase == "complete"
    assert callback_threads == [caller_thread] * len(events)


@pytest.mark.integration
def test_final_progress_callback_failure_does_not_invalidate_published_output(
    tmp_path: Path,
    input_files: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_dir = tmp_path / "complete-callback"

    def fail_after_publication(event: SpathiProgressEvent) -> None:
        if event.phase == "complete":
            raise RuntimeError("notification sink unavailable")

    result = infer(
        config_for(input_files, output_dir),
        progress_callback=fail_after_publication,
    )

    assert result.output_dir == output_dir
    assert (output_dir / "run_metadata.json").is_file()
    assert "Progress callback failed after output publication" in caplog.text


@pytest.mark.integration
def test_checkpoint_resume_reuses_only_committed_models_and_matches_clean_run(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "resumed"
    base = config_for(input_files, output_dir, threads=1)
    optimization_parameters = {
        "adaptive_trees": True,
        "adaptive_min_estimators": 4,
        "adaptive_tree_step": 4,
        "adaptive_tolerance": 1.0,
        "adaptive_patience": 1,
        "target_eligibility": "automatic",
        "min_target_detected_cells": 1,
        "min_target_detected_fraction": 0.01,
        "min_target_weighted_detected_fraction": 0.01,
        "min_target_weighted_detected_ess": 1.0,
    }
    config = SpathiConfig(**{**base.to_dict(), **optimization_parameters})
    original_fit = inference_module._fit_model_task
    fitted_keys: list[tuple[str, str]] = []

    def count_fit(task, context):
        fitted_keys.append((task.group.name, task.target_name))
        return original_fit(task, context)

    monkeypatch.setattr(inference_module, "_fit_model_task", count_fit)

    def interrupt_after_third_model(event: SpathiProgressEvent) -> None:
        if event.phase == "model_inference" and event.completed_models == 3:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        infer(config, progress_callback=interrupt_after_third_model)

    checkpoint_dir = tmp_path / ".resumed.checkpoint"
    assert not output_dir.exists()
    assert checkpoint_dir.is_dir()
    assert len(fitted_keys) == 3

    fitted_keys.clear()
    resumed_events: list[SpathiProgressEvent] = []
    resumed_config = SpathiConfig(**{**config.to_dict(), "threads": 2})
    result = infer(
        resumed_config,
        resume=True,
        progress_callback=resumed_events.append,
    )
    assert result.resumed_models == 3
    assert len(fitted_keys) == 5
    assert not checkpoint_dir.exists()
    resume_model_events = [event for event in resumed_events if event.phase == "model_inference"]
    assert resume_model_events[0].completed_models == 4
    assert all(event.resumed_models == 3 for event in resume_model_events)

    clean_dir = tmp_path / "clean"
    clean_base = config_for(input_files, clean_dir, threads=2)
    infer(
        SpathiConfig(**{**clean_base.to_dict(), **optimization_parameters}),
        checkpoint=False,
    )
    for artifact in (
        "network.csv",
        "cell_weights.tsv.gz",
        "skipped_targets.tsv",
        "target_eligibility.tsv.gz",
    ):
        assert (output_dir / artifact).read_bytes() == (clean_dir / artifact).read_bytes()
    resumed_diagnostics = pd.read_csv(output_dir / "model_diagnostics.tsv.gz", sep="\t")
    clean_diagnostics = pd.read_csv(clean_dir / "model_diagnostics.tsv.gz", sep="\t")
    pd.testing.assert_frame_equal(
        resumed_diagnostics.drop(columns="fit_seconds"),
        clean_diagnostics.drop(columns="fit_seconds"),
    )
    assert set(resumed_diagnostics["target_detected_cells"]) == {3, 4}
    assert resumed_diagnostics["target_weighted_detected_fraction"].notna().sum() == 6
    assert set(
        resumed_diagnostics.loc[
            resumed_diagnostics["target_weighted_detected_fraction"].isna(), "status"
        ]
    ) == {"target_not_estimable"}
    assert resumed_diagnostics["n_estimators_fitted"].max() <= config.n_estimators
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint"]["models_reused"] == 3
    assert metadata["models"]["processed_this_attempt"] == 5


@pytest.mark.integration
def test_resume_rejects_changed_centroid_weight_input(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    centroid_weights = tmp_path / "centroid_weights.tsv"
    centroid_weights.write_text(
        "cell\tcentroid_weight\ncell_1\t1\ncell_2\t2\ncell_3\t3\ncell_4\t4\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "changed-centroid-input"
    base = config_for(input_files, output_dir)
    config = SpathiConfig(
        **{
            **base.to_dict(),
            "centroid_weights": centroid_weights,
        }
    )

    def interrupt_after_first_model(event: SpathiProgressEvent) -> None:
        if event.phase == "model_inference" and event.completed_models == 1:
            raise RuntimeError("interrupt weighted run")

    with pytest.raises(RuntimeError, match="interrupt weighted run"):
        infer(config, progress_callback=interrupt_after_first_model)

    checkpoint_dir = tmp_path / ".changed-centroid-input.checkpoint"
    assert checkpoint_dir.is_dir()
    centroid_weights.write_text(
        "cell\tcentroid_weight\ncell_1\t2\ncell_2\t1\ncell_3\t3\ncell_4\t4\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="checkpoint identity does not match"):
        infer(config, resume=True)

    assert checkpoint_dir.is_dir()
    assert not output_dir.exists()


@pytest.mark.integration
def test_fully_completed_checkpoint_resume_still_limits_numeric_threads(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threadpoolctl import threadpool_info

    output_dir = tmp_path / "fully-resumed"
    config = config_for(input_files, output_dir, threads=2)

    def interrupt_after_last_commit(event: SpathiProgressEvent) -> None:
        if event.phase == "model_inference" and event.completed_models == event.total_models:
            raise RuntimeError("interrupt after final model commit")

    with pytest.raises(RuntimeError, match="after final model commit"):
        infer(config, progress_callback=interrupt_after_last_commit)

    observed: list[int] = []
    original_diagnostics = workflow_module.compute_weight_diagnostics

    def capture_thread_limits(*args: object, **kwargs: object):
        observed.extend(
            int(pool["num_threads"])
            for pool in threadpool_info()
            if pool.get("num_threads") is not None
        )
        return original_diagnostics(*args, **kwargs)

    def unexpected_inference_preparation(*args: object, **kwargs: object) -> None:
        raise AssertionError("a complete checkpoint must not prepare inference matrices")

    def unexpected_inference_planning(*args: object, **kwargs: object) -> None:
        raise AssertionError("a complete checkpoint must not plan model memory")

    monkeypatch.setattr(workflow_module, "compute_weight_diagnostics", capture_thread_limits)
    monkeypatch.setattr(
        workflow_module,
        "prepare_inference",
        unexpected_inference_preparation,
    )
    monkeypatch.setattr(
        workflow_module,
        "_plan_inference_batches",
        unexpected_inference_planning,
    )
    result = infer(config, resume=True)

    assert result.resumed_models == result.total_models == 8
    assert observed
    assert max(observed) == 1
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_parameters"]["inference_preparation_performed"] is False
    assert metadata["effective_parameters"]["targets_per_batch"] is None
    assert metadata["parallelism"]["maximum_concurrent_model_fits"] == 0
    assert metadata["parallelism"]["memory_available_bytes_at_planning"] is None


@pytest.mark.integration
def test_fully_completed_resume_guards_report_memory_before_allocating_it(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "fully-resumed-report"
    base = config_for(input_files, output_dir, threads=1)

    def interrupt_after_last_commit(event: SpathiProgressEvent) -> None:
        if event.phase == "model_inference" and event.completed_models == event.total_models:
            raise RuntimeError("interrupt after complete checkpoint")

    with pytest.raises(RuntimeError, match="complete checkpoint"):
        infer(base, progress_callback=interrupt_after_last_commit)

    checkpoint_dir = tmp_path / ".fully-resumed-report.checkpoint"
    assert checkpoint_dir.is_dir()
    reporting = SpathiConfig(**{**base.to_dict(), "report": True})
    headroom_checks = iter((None, None, 1))
    monkeypatch.setattr(
        workflow_module,
        "available_memory_bytes",
        lambda: next(headroom_checks),
    )

    with pytest.raises(MemoryError, match="bounded interactive report"):
        infer(reporting, resume=True)

    assert checkpoint_dir.is_dir()
    assert not output_dir.exists()
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: None)
    result = infer(reporting, resume=True)
    assert result.resumed_models == result.total_models
    assert result.report_path == output_dir / "report.html"


@pytest.mark.integration
def test_resume_mode_is_recorded_even_when_checkpoint_has_no_models(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    output_dir = tmp_path / "empty-resume"
    config = config_for(input_files, output_dir)
    loaded = io_module.load_inputs(
        config.expression,
        config.tf_list,
        config.groups,
        config.target_list,
    )
    checkpoint_dir = tmp_path / ".empty-resume.checkpoint"
    checkpoint_dir.mkdir()
    identity = checkpoint_module.build_checkpoint_identity(
        input_fingerprints=loaded.input_fingerprints,
        scientific_parameters=core_module._checkpoint_scientific_parameters(config),
        target_names=loaded.targets,
        group_names=tuple(sorted(map(str, pd.unique(loaded.groups)))),
        dependency_versions=workflow_module.scientific_dependency_versions(),
    )
    with checkpoint_module.ModelCheckpoint(checkpoint_dir, identity=identity, resume=False):
        pass

    infer(config, resume=True)

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint"]["resumed"] is True
    assert metadata["checkpoint"]["models_reused"] == 0


@pytest.mark.integration
def test_checkpoint_initialization_failure_removes_only_new_owned_state(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "initialization-failure"
    checkpoint_dir = tmp_path / ".initialization-failure.checkpoint"

    def fail_initialization(directory: Path, **kwargs: object) -> None:
        (directory / "checkpoint.sqlite3").write_bytes(b"partial")
        raise RuntimeError("simulated checkpoint initialization failure")

    monkeypatch.setattr(checkpoint_module, "ModelCheckpoint", fail_initialization)

    with pytest.raises(RuntimeError, match="simulated checkpoint initialization failure"):
        infer(config_for(input_files, output_dir))

    assert not output_dir.exists()
    assert not checkpoint_dir.exists()


def test_checkpoint_cleanup_preserves_unowned_prefixed_files(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / ".run.checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.sqlite3").write_bytes(b"owned")
    unowned = checkpoint_dir / "checkpoint.sqlite3.user-data"
    unowned.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned files"):
        core_module._remove_completed_checkpoint(checkpoint_dir)

    assert unowned.read_text(encoding="utf-8") == "keep me"


def test_checkpoint_cleanup_never_recurses_into_owned_named_directories(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / ".run.checkpoint"
    checkpoint_dir.mkdir()
    disguised_directory = checkpoint_dir / "checkpoint.sqlite3-wal"
    disguised_directory.mkdir()
    marker = disguised_directory / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned files"):
        core_module._remove_completed_checkpoint(checkpoint_dir)

    assert marker.read_text(encoding="utf-8") == "keep me"


@pytest.mark.integration
def test_core_opens_one_persistent_executor_for_multiple_group_batches(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_executor = workflow_module.PersistentTaskExecutor
    entered = 0

    class CountingExecutor(real_executor):
        def __enter__(self):
            nonlocal entered
            entered += 1
            return super().__enter__()

    monkeypatch.setattr(workflow_module, "PersistentTaskExecutor", CountingExecutor)
    infer(
        config_for(input_files, tmp_path / "persistent-pool", threads=2),
        checkpoint=False,
    )
    assert entered == 1


@pytest.mark.integration
def test_core_switches_to_disk_backed_distances_above_threshold(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "_DISTANCE_MEMMAP_THRESHOLD_BYTES", 1)
    original_compute = workflow_module.compute_cell_to_centroid_distances
    output_layouts: list[tuple[bool, bool]] = []
    scratch_directories: list[Path] = []
    scratch_files: list[Path] = []

    def temporary_file(*args: object, **kwargs: object):
        directory = kwargs.get("dir")
        assert directory is not None
        scratch_directories.append(Path(str(directory)))
        temporary = real_named_temporary_file(*args, **kwargs)
        scratch_files.append(Path(temporary.name))
        return temporary

    def capture_layout(*args: object, **kwargs: object) -> pd.DataFrame:
        output = kwargs.get("output")
        assert isinstance(output, np.memmap)
        output_layouts.append((output.flags.c_contiguous, output.flags.f_contiguous))
        return original_compute(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        workflow_module,
        "compute_cell_to_centroid_distances",
        capture_layout,
    )
    monkeypatch.setattr(workflow_module, "TemporaryFile", temporary_file)
    output_dir = tmp_path / "disk-backed"
    infer(config_for(input_files, output_dir))
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["effective_parameters"]["cell_centroid_distance_storage"] == "temporary-memory-map"
    )
    assert output_layouts == [(False, True)]
    assert len(scratch_directories) == 1
    assert scratch_directories[0].parent == tmp_path
    assert scratch_directories[0].name.startswith(".disk-backed.staging-")
    assert len(scratch_files) == 1
    assert not scratch_files[0].exists()
    assert not list(output_dir.glob("spathi-cell-distances-*"))
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_heap_float64"] == 0
    assert metadata["memory_estimate_bytes"]["cell_centroid_distances_mapped_float64"] > 0


@pytest.mark.integration
def test_disk_backed_distance_is_closed_after_an_early_failure(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_module, "_DISTANCE_MEMMAP_THRESHOLD_BYTES", 1)
    temporary_files: list[Any] = []

    def temporary_file(*args: object, **kwargs: object):
        temporary = real_named_temporary_file(*args, **kwargs)
        temporary_files.append(temporary)
        return temporary

    def fail_after_mapping(*args: object, **kwargs: object) -> None:
        assert isinstance(kwargs.get("output"), np.memmap)
        raise RuntimeError("distance failure")

    monkeypatch.setattr(workflow_module, "TemporaryFile", temporary_file)
    monkeypatch.setattr(
        workflow_module,
        "compute_cell_to_centroid_distances",
        fail_after_mapping,
    )
    output_dir = tmp_path / "disk-backed-failure"

    with pytest.raises(RuntimeError, match="distance failure"):
        infer(config_for(input_files, output_dir))

    assert temporary_files
    assert all(temporary.closed for temporary in temporary_files)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".disk-backed-failure.staging-*"))


@pytest.mark.integration
def test_group_distance_does_not_compute_cell_to_centroid_distances(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("group-distance must not materialize the cell distance matrix")

    monkeypatch.setattr(
        workflow_module,
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
    infer(config)
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
    infer(config)
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
        infer(config)

    assert not output_dir.exists()
    assert not (tmp_path / ".retryable.checkpoint").exists()
    assert not list(tmp_path.glob(".retryable.staging-*"))


def test_broken_output_symlink_is_never_overwritten(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    output_dir = tmp_path / "broken-output-link"
    missing_target = tmp_path / "missing-output-target"
    output_dir.symlink_to(missing_target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        infer(config_for(input_files, output_dir), checkpoint=False)

    assert output_dir.is_symlink()
    assert output_dir.readlink() == missing_target


@pytest.mark.integration
def test_output_path_appearing_before_publication_is_never_overwritten(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "late-output-link"
    missing_target = tmp_path / "late-missing-target"
    run_workflow = workflow_module.run_workflow

    def occupy_output_before_publication(*args: object, **kwargs: object):
        summary = run_workflow(*args, **kwargs)
        output_dir.symlink_to(missing_target, target_is_directory=True)
        return summary

    monkeypatch.setattr(workflow_module, "run_workflow", occupy_output_before_publication)
    with pytest.raises(FileExistsError, match="appeared during the run"):
        infer(config_for(input_files, output_dir), checkpoint=False)

    assert output_dir.is_symlink()
    assert output_dir.readlink() == missing_target
    assert not list(tmp_path.glob(".late-output-link.staging-*"))


def test_non_checkpointed_run_rejects_stale_checkpoint_directory(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    output_dir = tmp_path / "stale-checkpoint"
    checkpoint_dir = tmp_path / ".stale-checkpoint.checkpoint"
    checkpoint_dir.mkdir()
    marker = checkpoint_dir / "keep.txt"
    marker.write_text("stale but owned by the user", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be resumed or removed"):
        infer(config_for(input_files, output_dir), checkpoint=False)

    assert marker.read_text(encoding="utf-8") == "stale but owned by the user"
    assert not output_dir.exists()


@pytest.mark.integration
def test_expression_report_is_auxiliary_and_respects_thread_budget(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_thread_counts: list[int] = []
    original_prepare = report_module.prepare_report_embedding

    def capture_thread_budget(*args: object, **kwargs: object):
        from threadpoolctl import threadpool_info

        observed_thread_counts.extend(
            int(pool["num_threads"])
            for pool in threadpool_info()
            if pool.get("num_threads") is not None
        )
        return original_prepare(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        report_module,
        "prepare_report_embedding",
        capture_thread_budget,
    )
    output_dir = tmp_path / "expression-report"
    base = config_for(input_files, output_dir, threads=2, report=True)
    config = SpathiConfig(**{**base.to_dict(), "distance_space": "expression"})

    infer(config)

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
    assert semantics["projection_role"] == "report-only"
    assert metadata["memory_estimate_bytes"]["report_auxiliary_pca_working_rough_bytes"] > 0
    report_text = (output_dir / "report.html").read_text(encoding="utf-8")
    assert '"kind":"auxiliary-pca"' in report_text
    assert '"distance_space":"expression"' in report_text


@pytest.mark.integration
def test_report_failure_is_atomic_and_leaves_output_resumable(
    tmp_path: Path,
    input_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = report_module.InteractiveReportBuilder.write
    attempts = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated report failure")
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(report_module.InteractiveReportBuilder, "write", fail_once)
    output_dir = tmp_path / "report-retryable"
    config = config_for(input_files, output_dir, report=True)

    with pytest.raises(RuntimeError, match="simulated report failure"):
        infer(config)

    assert not output_dir.exists()
    assert (tmp_path / ".report-retryable.checkpoint").is_dir()
    assert not list(tmp_path.glob(".report-retryable.staging-*"))

    result = infer(config, resume=True)
    assert result.report_path == output_dir / "report.html"


@pytest.mark.integration
def test_report_switch_does_not_change_scientific_artifacts(
    tmp_path: Path,
    input_files: dict[str, Path],
) -> None:
    without_report = tmp_path / "without-report"
    with_report = tmp_path / "with-report"
    infer(config_for(input_files, without_report, report=False), checkpoint=False)
    infer(config_for(input_files, with_report, report=True), checkpoint=False)

    for artifact in (
        "network.csv",
        "cell_weights.tsv.gz",
        "centroid_weights.tsv.gz",
        "centroid_weight_diagnostics.tsv",
        "centroids.tsv",
        "group_affinities.tsv",
        "group_distances.tsv",
        "skipped_targets.tsv",
        "weight_diagnostics.tsv",
    ):
        assert (without_report / artifact).read_bytes() == (with_report / artifact).read_bytes()
    assert not (without_report / "report.html").exists()
    assert (with_report / "report.html").is_file()


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
        infer(config_for(input_files, output_dir))

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
        "--no-report",
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
    assert metadata["report"] == {
        "artifact": None,
        "generated": False,
        "requested": False,
    }
    assert metadata["memory_estimate_bytes"]["report_retained_rough_bytes"] == 0
    assert metadata["memory_estimate_bytes"]["report_render_working_rough_bytes"] == 0
    assert "plotly" not in metadata["dependency_versions"]
    assert not (output_dir / "report.html").exists()
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode == 2
    assert "will not be overwritten" in second.stderr
