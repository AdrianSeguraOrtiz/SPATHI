from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import spathi._workflow as workflow_module
import spathi.convergence as convergence_module
from spathi import SpathiConfig, prepare_forest_prefixes
from spathi.inference import prepare_inference


def _without_runtime(result: object) -> tuple[object, ...]:
    return (
        result.edges,
        result.skipped_targets,
        tuple(replace(stat, fit_seconds=0.0) for stat in result.model_stats),
        result.group_order,
        result.tree_method,
        result.total_models,
        result.completed_models,
        result.trained_models,
        result.expression_dtype,
        result.expression_nbytes,
        result.predictor_nbytes,
    )


def _config(input_files: dict[str, Path], tmp_path: Path) -> SpathiConfig:
    return SpathiConfig(
        expression=input_files["expression"],
        tf_list=input_files["tf_list"],
        groups=input_files["groups"],
        output_dir=tmp_path / "unused-output",
        n_estimators=7,
        max_features=1.0,
        random_seed=73,
        threads=1,
        report=False,
    )


def test_prepared_forest_prefixes_match_independent_prepared_inference(
    input_files: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(input_files, tmp_path)
    study = prepare_forest_prefixes(config, estimator_counts=(3, 5, 7))
    batches = list(study.iter_batches(target_batch_size=100))

    assert len(batches) == len(study.group_order)
    assert tuple(item.n_estimators for item in batches[0]) == (3, 5, 7)
    expression = study.prepared_inference._expression
    genes = study.prepared_inference.gene_names
    tfs = study.prepared_inference.tf_names
    targets = study.prepared_inference.target_names
    for item in (item for batch in batches for item in batch):
        independent = prepare_inference(
            expression,
            genes,
            tfs,
            target_names=targets,
            n_estimators=item.n_estimators,
            max_features=config.max_features,
            random_seed=config.random_seed,
        )
        result = next(
            independent.iter_group_target_batches(
                {group: study.group_weights[group] for group in item.inference.group_order},
                group_order=item.inference.group_order,
                target_batch_size=100,
                threads=1,
            )
        )
        assert _without_runtime(item.inference) == _without_runtime(result)


def test_prefix_process_backend_preserves_exact_independent_results(
    input_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(convergence_module, "available_cpu_count", lambda: 2)
    monkeypatch.setattr(workflow_module, "available_memory_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(workflow_module, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    config = replace(_config(input_files, tmp_path), threads=2, parallel_backend="threads")
    reference = prepare_forest_prefixes(config, estimator_counts=(3, 5, 7))
    process = prepare_forest_prefixes(
        replace(config, parallel_backend="processes"), estimator_counts=(3, 5, 7)
    )
    reference_batches = list(reference.iter_batches(target_batch_size=100))
    process_batches = list(process.iter_batches(target_batch_size=100))
    assert len(reference_batches) == len(process_batches)
    for reference_batch, process_batch in zip(reference_batches, process_batches, strict=True):
        for expected, actual in zip(reference_batch, process_batch, strict=True):
            assert _without_runtime(expected.inference) == _without_runtime(actual.inference)
            assert actual.inference.parallel_plan.backend == "loky"


def test_prepare_forest_prefixes_rejects_non_prefix_semantics(
    input_files: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(input_files, tmp_path)
    with pytest.raises(ValueError, match="final estimator count"):
        prepare_forest_prefixes(config, estimator_counts=(3, 5))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"estimator_counts": ()}, "cannot be empty"),
        ({"estimator_counts": (3, 3, 7)}, "strictly increasing"),
        ({"group_order": (), "weight_results": ()}, "unique group identifiers"),
    ],
)
def test_prepared_forest_prefixes_constructor_enforces_public_invariants(
    input_files: dict[str, Path],
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    study = prepare_forest_prefixes(_config(input_files, tmp_path), estimator_counts=(3, 5, 7))
    with pytest.raises(ValueError, match=message):
        replace(study, **changes)


def test_single_group_prefixes_apply_the_ordinary_inference_boundary(
    input_files: dict[str, Path], tmp_path: Path
) -> None:
    rows: list[dict[str, str]] = []
    with input_files["groups"].open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle, delimiter="\t"))
    groups = tmp_path / "one-group.tsv"
    groups.write_text(
        "sample\tcluster\n" + "".join(f"{row['sample']}\tA\n" for row in rows),
        encoding="utf-8",
    )
    config = replace(_config(input_files, tmp_path), groups=groups)
    with pytest.raises(ValueError, match="single-group dataset"):
        prepare_forest_prefixes(config, estimator_counts=(3, 5, 7))
