"""Contracts for the opt-in local CLL reference/candidate benchmark."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from spathi import _workflow as workflow
from spathi.outputs import MODEL_DIAGNOSTIC_COLUMNS


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_cll_equivalence.py"
    spec = importlib.util.spec_from_file_location("spathi_cll_equivalence_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark() -> ModuleType:
    return _load_benchmark_module()


def test_available_cpu_count_uses_joblib_container_capacity(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_cpu_count(*, only_physical_cores: bool) -> int:
        calls.append(only_physical_cores)
        return 3

    monkeypatch.setattr(benchmark, "joblib_cpu_count", fake_cpu_count)
    monkeypatch.setattr(
        benchmark.os,
        "sched_getaffinity",
        lambda _pid: set(range(12)),
        raising=False,
    )

    assert benchmark.available_cpu_count() == 3
    assert calls == [False]


def test_available_cpu_count_falls_back_to_process_affinity(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_cpu_count(*, only_physical_cores: bool) -> int:
        assert only_physical_cores is False
        raise NotImplementedError

    monkeypatch.setattr(benchmark, "joblib_cpu_count", unavailable_cpu_count)
    monkeypatch.setattr(
        benchmark.os,
        "sched_getaffinity",
        lambda _pid: {2, 4},
        raising=False,
    )

    assert benchmark.available_cpu_count() == 2


def test_available_cpu_count_is_never_below_one(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "joblib_cpu_count",
        lambda *, only_physical_cores: 0,
    )

    assert benchmark.available_cpu_count() == 1


def test_benchmark_schema_tracks_current_optimization_diagnostics(
    benchmark: ModuleType,
) -> None:
    """Fail immediately if SPATHI evolves either optimization audit table."""

    assert benchmark._TABLE_COLUMNS["model_diagnostics.tsv.gz"] == MODEL_DIAGNOSTIC_COLUMNS
    assert (
        benchmark._TABLE_COLUMNS["target_eligibility.tsv.gz"]
        == workflow._TARGET_ELIGIBILITY_COLUMNS
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scaling_summary_never_reports_speedups_for_inequivalent_pairs(
    benchmark: ModuleType,
) -> None:
    row = {
        "run_type": "measurement",
        "profile": "test",
        "case_id": "case",
        "dataset_id": "dataset",
        "target_scope": "deterministic-slice",
        "target_count": 10,
        "n_estimators": 20,
        "threads": 2,
        "n_components": 3,
        "status": "compared",
        "equivalent": False,
        "reference_wall_seconds": 100.0,
        "candidate_wall_seconds": 1.0,
        "reference_model_seconds": 80.0,
        "candidate_model_seconds": 0.5,
    }
    datasets = {"dataset": SimpleNamespace(dimensions=SimpleNamespace(groups=2))}

    (summary,) = benchmark._scaling_rows([row], datasets)

    assert summary["measured_pairs"] == 1
    assert summary["equivalent_pairs"] == 0
    assert summary["reference_median_wall_seconds"] == ""
    assert summary["candidate_median_wall_seconds"] == ""
    assert summary["paired_wall_speedup_reference_over_candidate_median"] == ""
    assert summary["paired_model_speedup_reference_over_candidate_median"] == ""


def test_scaling_summary_uses_paired_ratios_and_descriptive_intervals(
    benchmark: ModuleType,
) -> None:
    base = {
        "run_type": "measurement",
        "profile": "test",
        "case_id": "case",
        "dataset_id": "dataset",
        "target_scope": "deterministic-slice",
        "target_count": 10,
        "n_estimators": 20,
        "threads": 2,
        "n_components": 3,
        "status": "compared",
        "equivalent": True,
        "performance_eligible": True,
        "reference_model_seconds": 8.0,
        "candidate_model_seconds": 4.0,
        "reference_peak_rss_bytes": 100.0,
        "candidate_peak_rss_bytes": 80.0,
        "reference_sampled_cpu_seconds": 7.0,
        "candidate_sampled_cpu_seconds": 6.0,
        "reference_peak_run_logical_bytes": 1000.0,
        "candidate_peak_run_logical_bytes": 900.0,
        "reference_peak_run_allocated_bytes": 1200.0,
        "candidate_peak_run_allocated_bytes": 1000.0,
        "reference_output_logical_bytes": 800.0,
        "candidate_output_logical_bytes": 800.0,
        "reference_output_allocated_bytes": 900.0,
        "candidate_output_allocated_bytes": 900.0,
        "model_speedup_reference_over_candidate": 2.0,
        "candidate_over_reference_peak_rss": 0.8,
        "candidate_over_reference_sampled_cpu": 6.0 / 7.0,
        "candidate_over_reference_peak_run_logical_bytes": 0.9,
        "candidate_over_reference_peak_run_allocated_bytes": 5.0 / 6.0,
    }
    rows = [
        {
            **base,
            "reference_wall_seconds": 10.0,
            "candidate_wall_seconds": 1.0,
            "wall_speedup_reference_over_candidate": 10.0,
        },
        {
            **base,
            "reference_wall_seconds": 100.0,
            "candidate_wall_seconds": 100.0,
            "wall_speedup_reference_over_candidate": 1.0,
        },
    ]
    datasets = {"dataset": SimpleNamespace(dimensions=SimpleNamespace(groups=2))}

    (summary,) = benchmark._scaling_rows(rows, datasets)

    assert summary["paired_wall_speedup_reference_over_candidate_median"] == 5.5
    assert summary["paired_wall_speedup_reference_over_candidate_q1"] == 3.25
    assert summary["paired_wall_speedup_reference_over_candidate_q3"] == 7.75
    assert summary["paired_wall_speedup_reference_over_candidate_bootstrap_ci95_low"] == 1.0
    assert summary["paired_wall_speedup_reference_over_candidate_bootstrap_ci95_high"] == 10.0
    assert summary["reference_median_wall_seconds"] == 55.0
    assert summary["candidate_median_wall_seconds"] == 50.5


@pytest.mark.parametrize(
    ("reference_attempt", "candidate_attempt", "candidate_eligible"),
    [(1, 2, True), (1, 1, False)],
)
def test_comparison_keeps_raw_measurements_but_blanks_ineligible_ratios(
    benchmark: ModuleType,
    tmp_path: Path,
    reference_attempt: int,
    candidate_attempt: int,
    candidate_eligible: bool,
) -> None:
    _, manifest = _create_local_manifest(benchmark, tmp_path)
    dataset = manifest.datasets[0]
    profile = benchmark.load_profile(_write_profile(tmp_path / "profile.json"))
    case = profile.cases[0]
    base = {
        "status": "success",
        "wall_seconds": 4.0,
        "peak_rss_bytes": 100.0,
        "sampled_cpu_user_seconds": 2.0,
        "sampled_cpu_system_seconds": 1.0,
        "peak_run_logical_bytes": 200.0,
        "peak_run_allocated_bytes": 300.0,
        "published_output_logical_bytes": 150.0,
        "published_output_allocated_bytes": 180.0,
        "phase_model_inference_seconds": 3.0,
        "error": "",
        "performance_eligible": True,
    }
    row = benchmark._comparison_row(
        comparison_id="pair",
        attempt=2,
        run_type="measurement",
        round_index=1,
        configuration_position=1,
        profile=profile,
        case=case,
        dataset=dataset,
        target_scope="deterministic-slice",
        target_count=4,
        n_estimators=3,
        threads=1,
        reference_row={**base, "run_id": "reference", "attempt": reference_attempt},
        candidate_row={
            **base,
            "run_id": "candidate",
            "attempt": candidate_attempt,
            "performance_eligible": candidate_eligible,
        },
        comparison=benchmark.OutputComparison(equivalent=True, artifacts=()),
        details_path=Path("comparison-details/pair.json"),
    )

    assert row["equivalent"] is True
    assert row["reference_wall_seconds"] == 4.0
    assert row["performance_eligible"] is False
    assert row["wall_speedup_reference_over_candidate"] == ""
    assert row["model_speedup_reference_over_candidate"] == ""


def test_balanced_configuration_orders_rotate_every_configuration_through_positions(
    benchmark: ModuleType,
) -> None:
    configurations = tuple(range(4))

    orders = benchmark._balanced_configuration_orders(configurations, 4, seed=11)

    assert all(set(order) == set(configurations) for order in orders)
    for configuration in configurations:
        assert {order.index(configuration) for order in orders} == set(range(4))

    role_orders = [
        benchmark._role_order(23, round_index=round_index, configuration_index=7)
        for round_index in range(1, 5)
    ]
    assert sum(order[0] == "reference" for order in role_orders) == 2
    assert sum(order[0] == "candidate" for order in role_orders) == 2


def _profile_document(*, targets: list[int] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "test-equivalence",
        "description": "Test profile.",
        "minimum_datasets": 1,
        "maximum_datasets": 1,
        "allow_identical_implementations": True,
        "limitations": ["Computational equivalence only."],
        "scientific_parameters": {
            "single_group_weight_mode": "cell-distance",
            "multi_group_weight_mode": "cell-distance-group-anchored",
            "distance_space": "pca",
            "distance_standardization": "none",
            "pca_svd_solver": "auto",
            "single_group_distance_metric": "euclidean",
            "multi_group_distance_metric": "cosine",
            "kernel": "gaussian",
            "bandwidth": "auto",
            "single_group_size_correction": "none",
            "multi_group_size_correction": "cap-to-target",
            "tree_method": "extra-trees",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
            "bootstrap": False,
            "adaptive_trees": False,
            "adaptive_min_estimators": 100,
            "adaptive_tree_step": 50,
            "adaptive_tolerance": 0.01,
            "adaptive_patience": 2,
            "target_eligibility": "all",
            "min_target_detected_cells": 20,
            "min_target_detected_fraction": 0.01,
            "min_target_weighted_detected_fraction": 0.01,
            "min_target_weighted_detected_ess": 10.0,
        },
        "defaults": {
            "target_counts": [4] if targets is None else targets,
            "n_estimators": [3],
            "threads": [1],
            "n_components": 3,
            "warmups": 0,
            "repeats": 1,
            "seed": 17,
            "checkpoint": False,
            "report": False,
            "resource_sample_ms": 20.0,
            "run_timeout_seconds": 60.0,
        },
        "comparison": {
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-10,
        },
        "cases": [
            {
                "id": "tiny-case",
                "description": "Tiny test case.",
                "dataset_ids": [],
            }
        ],
    }


def _write_profile(path: Path, *, targets: list[int] | None = None) -> Path:
    path.write_text(json.dumps(_profile_document(targets=targets)), encoding="utf-8")
    return path


def _write_prepared_unit(root: Path) -> Path:
    unit = root / "analysis_units" / "unit-001-b"
    unit.mkdir(parents=True)
    cells = [f"cell-{index}" for index in range(12)]
    expression = unit / "expression.tsv"
    with expression.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene", *cells])
        for gene_index in range(8):
            writer.writerow(
                [
                    f"G{gene_index}",
                    *[
                        0.2
                        + gene_index * 0.1
                        + cell_index * 0.03
                        + ((gene_index + cell_index) % 3) * 0.2
                        for cell_index in range(12)
                    ],
                ]
            )
    groups = unit / "groups.tsv"
    groups.write_text(
        "cell\tcluster\n"
        + "".join(f"{cell}\t{'A' if index < 6 else 'B'}\n" for index, cell in enumerate(cells)),
        encoding="utf-8",
    )
    tf_list = unit / "tf_list.txt"
    tf_list.write_text("G0\nG1\nG2\n", encoding="utf-8")

    def entry(path: Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(root)),
            "sha256": _hash(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = root / "prepare_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_units": [
                    {
                        "analysis_unit": "b_cell_maturation",
                        "status": "prepared",
                        "n_cells": 12,
                        "n_genes": 8,
                        "n_transcription_factors": 3,
                        "n_groups": 2,
                        "outputs": {
                            "expression": entry(expression),
                            "groups": entry(groups),
                            "tf_list": entry(tf_list),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _create_local_manifest(benchmark: ModuleType, tmp_path: Path) -> tuple[Path, object]:
    prepare_manifest = _write_prepared_unit(tmp_path / "prepared")
    output = tmp_path / "local-datasets.json"
    document = benchmark.create_dataset_manifest(
        [("patient", prepare_manifest)],
        output_path=output,
    )
    benchmark._atomic_json(output, document)
    return output, benchmark.load_dataset_manifest(output)


def test_builtin_equivalence_profiles_follow_the_current_contract(benchmark: ModuleType) -> None:
    smoke = benchmark.load_profile("cll-equivalence-smoke")
    progressive = benchmark.load_profile("cll-equivalence-progressive")
    production = benchmark.load_profile("cll-equivalence-production-envelope")

    assert smoke.cases[0].target_counts == (8,)
    assert smoke.cases[0].n_estimators == (5,)
    assert smoke.absolute_tolerance == 1e-12
    assert [case.id for case in progressive.cases] == [
        "targets-axis",
        "trees-axis",
        "threads-axis",
        "combined-slice",
    ]
    assert progressive.cases[2].threads == (1, 2, 4, 8)
    assert progressive.cases[0].warmups == 1
    assert progressive.cases[0].repeats == 4
    assert smoke.minimum_datasets == smoke.maximum_datasets == 1
    assert smoke.allow_identical_implementations
    assert production.minimum_datasets == production.maximum_datasets == 1
    assert not production.allow_identical_implementations
    assert production.cases[0].target_counts == ("all",)
    assert production.cases[0].n_estimators == (250,)


def test_serious_profiles_reject_identical_implementations_without_explicit_override(
    benchmark: ModuleType,
) -> None:
    profile = benchmark.load_profile("cll-equivalence-progressive")
    hashes = {"reference": "a" * 64, "candidate": "a" * 64}
    with pytest.raises(benchmark.ContractError, match="hashes are identical"):
        benchmark._validate_implementation_identity(
            profile,
            hashes,
            allow_identical_implementations=False,
        )
    assert benchmark._validate_implementation_identity(
        profile,
        hashes,
        allow_identical_implementations=True,
    )


def test_production_envelope_requires_a_dataset_and_omits_target_list(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    _, manifest = _create_local_manifest(benchmark, tmp_path)
    profile = benchmark.load_profile("cll-equivalence-production-envelope")
    dataset = manifest.datasets[0]

    with pytest.raises(benchmark.ContractError, match="explicit --dataset"):
        benchmark._select_datasets(profile, manifest, ())
    with pytest.raises(benchmark.ContractError, match="duplicates"):
        benchmark._select_datasets(profile, manifest, (dataset.id, dataset.id))

    dry_run = benchmark._dry_run_document(profile, manifest, (dataset,))
    assert dry_run["total_processes"] == 10
    assert dry_run["sequential_timeout_upper_bound_seconds"] == 10 * 172800
    assert dry_run["conservative_estimated_bytes"] > 0
    assert dry_run["matrix"][0]["target_scope"] == "all-expression-genes"
    assert dry_run["matrix"][0]["target_count"] == dataset.dimensions.genes

    command = benchmark.build_infer_command(
        SimpleNamespace(snapshot_parent=tmp_path / "snapshot"),
        dataset,
        target_list=None,
        output_dir=tmp_path / "output",
        scientific_parameters=profile.scientific_parameters,
        n_estimators=250,
        n_components=50,
        threads=8,
        seed=1729,
        checkpoint=True,
        resume=False,
        report=False,
    )
    assert "--target-list" not in command
    assert command[command.index("--n-estimators") + 1] == "250"
    assert command[command.index("--max-features") + 1] == "sqrt"
    assert "--no-bootstrap" in command
    assert "--no-adaptive-trees" in command
    assert command[command.index("--target-eligibility") + 1] == "all"
    resumed_command = benchmark.build_infer_command(
        SimpleNamespace(snapshot_parent=tmp_path / "snapshot"),
        dataset,
        target_list=None,
        output_dir=tmp_path / "output",
        scientific_parameters=profile.scientific_parameters,
        n_estimators=250,
        n_components=50,
        threads=8,
        seed=1729,
        checkpoint=True,
        resume=True,
        report=False,
    )
    assert "--resume" in resumed_command


def test_local_manifest_is_derived_from_and_pins_prepare_outputs(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    path, manifest = _create_local_manifest(benchmark, tmp_path)
    dataset = manifest.datasets[0]

    assert dataset.id == "patient-b-cell-maturation"
    assert dataset.dimensions.genes == 8
    assert dataset.expression.path.is_file()
    assert dataset.prepare_manifest.sha256 == _hash(tmp_path / "prepared/prepare_manifest.json")
    benchmark.verify_dataset(dataset)

    dataset.tf_list.path.write_text("G0\n", encoding="utf-8")
    with pytest.raises(benchmark.ContractError, match="size mismatch|hash mismatch"):
        benchmark.verify_dataset(dataset)
    assert path.is_file()


def test_dataset_manifest_creation_requires_the_current_prepare_contract(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    prepare_manifest = _write_prepared_unit(tmp_path / "prepared")
    document = json.loads(prepare_manifest.read_text(encoding="utf-8"))
    document["schema_version"] = 0
    prepare_manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(benchmark.ContractError, match="schema_version must be 1"):
        benchmark.create_dataset_manifest(
            [("patient", prepare_manifest)],
            output_path=tmp_path / "local-datasets.json",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.pop("execution_attempt"), "missing fields"),
        (lambda document: document.pop("keep_successful_outputs"), "missing fields"),
        (lambda document: document.pop("verify_inputs"), "missing fields"),
        (lambda document: document.update({"unexpected": True}), "unknown fields"),
    ],
)
def test_resume_requires_the_exact_current_suite_manifest(
    benchmark: ModuleType,
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    document = {field: None for field in benchmark._SUITE_MANIFEST_REQUIRED_FIELDS}
    document.update(
        {
            "schema_version": 1,
            "execution_attempt": 1,
            "keep_successful_outputs": False,
            "verify_inputs": True,
        }
    )
    mutation(document)
    (suite / "manifest.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(benchmark.ContractError, match=message):
        benchmark._load_resume_state(suite)


def test_target_manifest_requires_the_exact_current_top_level_contract(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    _, data_manifest = _create_local_manifest(benchmark, tmp_path)
    profile = benchmark.load_profile(_write_profile(tmp_path / "profile.json"))
    selected = data_manifest.datasets
    suite = tmp_path / "suite"
    (suite / "targets").mkdir(parents=True)
    benchmark._create_target_manifest(
        profile,
        data_manifest,
        selected,
        suite_root=suite,
    )
    target_manifest = suite / "targets-manifest.json"
    document = json.loads(target_manifest.read_text(encoding="utf-8"))
    document["unexpected"] = True
    benchmark._atomic_json(target_manifest, document)
    suite_manifest = {
        "target_manifest": {
            "snapshot_path": target_manifest.name,
            "sha256": _hash(target_manifest),
        }
    }

    with pytest.raises(benchmark.ContractError, match="unknown fields"):
        benchmark._load_target_manifest(
            suite,
            suite_manifest,
            profile,
            data_manifest,
            selected,
        )


def test_comparison_details_require_the_exact_current_top_level_contract(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "comparison.json"
    document = benchmark._comparison_details_document(
        benchmark.OutputComparison(equivalent=True, artifacts=())
    )
    document["unexpected"] = True
    benchmark._atomic_json(path, document)

    with pytest.raises(benchmark.ContractError, match="unknown fields"):
        benchmark._load_comparison_details(path)


def test_target_slices_are_nested_stable_and_expression_ordered(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    _, manifest = _create_local_manifest(benchmark, tmp_path)
    dataset = manifest.datasets[0]

    first = benchmark.create_target_lists(
        dataset,
        target_counts=[2, 5],
        seed=123,
        destination=tmp_path / "targets-first",
    )
    second = benchmark.create_target_lists(
        dataset,
        target_counts=[5, 2],
        seed=123,
        destination=tmp_path / "targets-second",
    )

    small = first[2].path.read_text(encoding="utf-8").splitlines()
    large = first[5].path.read_text(encoding="utf-8").splitlines()
    assert set(small) < set(large)
    assert first[2].sha256 == second[2].sha256
    assert first[5].sha256 == second[5].sha256
    assert large == sorted(large, key=lambda gene: int(gene[1:]))


def test_suite_resume_lock_rejects_concurrent_owner(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    with benchmark._suite_lock(suite):
        with pytest.raises(benchmark.ContractError, match="already locked"):
            with benchmark._suite_lock(suite):
                pass


def test_run_journal_restoration_accepts_seed_zero(benchmark: ModuleType) -> None:
    row = {field: "" for field in benchmark.RUN_FIELDS}
    row.update(
        {
            "attempt": "1",
            "round": "1",
            "configuration_position": "1",
            "execution_position": "1",
            "target_count": "1",
            "n_estimators": "1",
            "threads": "1",
            "n_components": "1",
            "seed": "0",
            "checkpoint": "False",
            "resumed_from_checkpoint": "False",
            "performance_eligible": "True",
            "report": "False",
            "run_type": "measurement",
            "status": "success",
            "wall_seconds": "0.1",
            "peak_rss_bytes": "1",
            "sampled_cpu_user_seconds": "0",
            "sampled_cpu_system_seconds": "0",
            "input_logical_bytes": "1",
            "input_allocated_bytes": "1",
            "input_file_count": "1",
            "peak_run_logical_bytes": "1",
            "peak_run_allocated_bytes": "1",
            "peak_run_file_count": "1",
            "published_output_logical_bytes": "1",
            "published_output_allocated_bytes": "1",
            "published_output_file_count": "1",
            "models_reused_from_checkpoint": "0",
            **{
                field: "a" * 64
                for field in (
                    "profile_sha256",
                    "dataset_manifest_sha256",
                    "implementation_sha256",
                    "expression_sha256",
                    "groups_sha256",
                    "tf_list_sha256",
                )
            },
        }
    )

    restored = benchmark._restore_run_row_types(row, location="runs[0]")

    assert restored["seed"] == 0


def test_canonical_table_comparison_tolerates_only_declared_numeric_delta_and_fit_time(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.tsv.gz"
    candidate = tmp_path / "candidate.tsv.gz"
    rule = next(
        rule for rule in benchmark._TABLE_RULES if rule.filename == "model_diagnostics.tsv.gz"
    )
    fields = list(benchmark._TABLE_COLUMNS[rule.filename])
    reference_row = {field: "" for field in fields}
    reference_row.update(
        {
            "target_group": "A",
            "target": "G",
            "status": "trained",
            "random_seed": "1",
            "n_samples": "12",
            "n_positive_weight_samples": "12",
            "weight_sum": "10",
            "n_predictors_input": "3",
            "n_predictors_used": "3",
            "discarded_predictors_json": "[]",
            "constant_predictors_json": "[]",
            "n_edges": "2",
            "importance_sum": "1.0",
            "fit_seconds": "0.2",
            "n_estimators_fitted": "5",
            "adaptive_converged": "False",
            "convergence_checks": "0",
        }
    )
    candidate_row = {**reference_row, "weight_sum": "10.00000000001", "fit_seconds": "9.8"}

    def write(path: Path, row: dict[str, str]) -> None:
        with benchmark.gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(row)

    write(reference, reference_row)
    write(candidate, candidate_row)

    equivalent = benchmark.compare_table(
        reference,
        candidate,
        rule=rule,
        absolute_tolerance=1e-10,
        relative_tolerance=0.0,
    )
    assert equivalent.equivalent
    assert equivalent.numeric_values_compared == 10

    candidate.write_bytes(reference.read_bytes())
    byte_identical = benchmark.compare_table(
        reference,
        candidate,
        rule=rule,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    assert byte_identical.equivalent
    assert byte_identical.comparison_mode == "byte-identical"
    assert byte_identical.rows_compared == 0

    candidate_row["status"] = "skipped"
    write(candidate, candidate_row)
    different = benchmark.compare_table(
        reference,
        candidate,
        rule=rule,
        absolute_tolerance=1e-10,
        relative_tolerance=0.0,
    )
    assert not different.equivalent
    assert "status" in different.first_mismatches[0]


def test_run_metadata_audit_pins_inputs_and_scientific_controls(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    _, manifest = _create_local_manifest(benchmark, tmp_path)
    dataset = manifest.datasets[0]
    targets = benchmark.create_target_lists(
        dataset,
        target_counts=[4],
        seed=4,
        destination=tmp_path / "targets",
    )[4]
    scientific = benchmark._parse_scientific_parameters(
        _profile_document()["scientific_parameters"]
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata = {
        "spathi_version": "test",
        "dependency_versions": {"numpy": "test"},
        "input_dimensions": {
            "cells": dataset.dimensions.cells,
            "genes": dataset.dimensions.genes,
            "transcription_factors": dataset.dimensions.transcription_factors,
            "groups": dataset.dimensions.groups,
            "targets": 4,
        },
        "inputs": {
            "expression": {"sha256": dataset.expression.sha256},
            "groups": {"sha256": dataset.groups.sha256},
            "tf_list": {"sha256": dataset.tf_list.sha256},
            "target_list": {"sha256": targets.sha256},
        },
        "requested_parameters": {
            "n_estimators": 3,
            "n_components": 3,
            "threads": 1,
            "random_seed": 17,
            "tree_method": "extra-trees",
            "weight_mode": "cell-distance-group-anchored",
            "distance_space": "pca",
            "distance_standardization": "none",
            "pca_svd_solver": "auto",
            "distance_metric": "cosine",
            "kernel": "gaussian",
            "bandwidth": "auto",
            "group_size_correction": "cap-to-target",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
            "bootstrap": False,
            "adaptive_trees": False,
            "adaptive_min_estimators": 100,
            "adaptive_tree_step": 50,
            "adaptive_tolerance": 0.01,
            "adaptive_patience": 2,
            "target_eligibility": "all",
            "min_target_detected_cells": 20,
            "min_target_detected_fraction": 0.01,
            "min_target_weighted_detected_fraction": 0.01,
            "min_target_weighted_detected_ess": 10.0,
            "report": False,
        },
        "checkpoint": {"enabled": False, "resumed": False},
        "report": {"requested": False},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    provenance, error = benchmark._audit_run_metadata(
        metadata_path,
        dataset=dataset,
        target_list=targets,
        target_count=4,
        n_estimators=3,
        n_components=3,
        threads=1,
        seed=17,
        checkpoint=False,
        resume=False,
        report=False,
        scientific_parameters=scientific,
    )
    assert not error
    assert provenance["spathi_version"] == "test"

    metadata["requested_parameters"]["max_features"] = "log2"  # type: ignore[index]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _, error = benchmark._audit_run_metadata(
        metadata_path,
        dataset=dataset,
        target_list=targets,
        target_count=4,
        n_estimators=3,
        n_components=3,
        threads=1,
        seed=17,
        checkpoint=False,
        resume=False,
        report=False,
        scientific_parameters=scientific,
    )
    assert "requested_parameters.max_features" in error

    metadata["requested_parameters"]["max_features"] = "sqrt"  # type: ignore[index]
    metadata["inputs"]["expression"]["sha256"] = "0" * 64  # type: ignore[index]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _, error = benchmark._audit_run_metadata(
        metadata_path,
        dataset=dataset,
        target_list=targets,
        target_count=4,
        n_estimators=3,
        n_components=3,
        threads=1,
        seed=17,
        checkpoint=False,
        resume=False,
        report=False,
        scientific_parameters=scientific,
    )
    assert "inputs.expression" in error

    metadata["inputs"]["expression"]["sha256"] = dataset.expression.sha256  # type: ignore[index]
    del metadata["inputs"]["target_list"]  # type: ignore[index]
    metadata["input_dimensions"]["targets"] = dataset.dimensions.genes  # type: ignore[index]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _, error = benchmark._audit_run_metadata(
        metadata_path,
        dataset=dataset,
        target_list=None,
        target_count=dataset.dimensions.genes,
        n_estimators=3,
        n_components=3,
        threads=1,
        seed=17,
        checkpoint=False,
        resume=False,
        report=False,
        scientific_parameters=scientific,
    )
    assert not error

    metadata["inputs"]["target_list"] = {"sha256": targets.sha256}  # type: ignore[index]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _, error = benchmark._audit_run_metadata(
        metadata_path,
        dataset=dataset,
        target_list=None,
        target_count=dataset.dimensions.genes,
        n_estimators=3,
        n_components=3,
        threads=1,
        seed=17,
        checkpoint=False,
        resume=False,
        report=False,
        scientific_parameters=scientific,
    )
    assert "inputs.target_list" in error


def test_dry_run_expands_pairs_without_reading_expression(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    manifest_path, _ = _create_local_manifest(benchmark, tmp_path)
    profile_path = _write_profile(tmp_path / "profile.json", targets=[2, 4])
    output = tmp_path / "unused"

    result = benchmark.main(
        [
            "run",
            "--profile",
            str(profile_path),
            "--dataset-manifest",
            str(manifest_path),
            "--reference-source",
            str(Path(__file__).parents[1]),
            "--output-dir",
            str(output),
            "--dataset",
            "patient-b-cell-maturation",
            "--dry-run",
        ]
    )

    assert result == 0
    assert not output.exists()


def test_tiny_reference_candidate_suite_is_equivalent_and_discards_successful_outputs(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, data_manifest = _create_local_manifest(benchmark, tmp_path)
    profile = benchmark.load_profile(_write_profile(tmp_path / "profile.json"))
    output = tmp_path / "suite"
    repository = Path(__file__).parents[1]

    def fake_measure(command: list[str], **_kwargs: object) -> object:
        output_path = Path(command[command.index("--output-dir") + 1])
        output_path.mkdir()
        for rule in benchmark._TABLE_RULES:
            path = output_path / rule.filename
            fields = list(benchmark._TABLE_COLUMNS[rule.filename])
            stream_context = (
                benchmark.gzip.open(path, "wt", encoding="utf-8", newline="")
                if path.suffix == ".gz"
                else path.open("w", encoding="utf-8", newline="")
            )
            with stream_context as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter=rule.delimiter)
                writer.writeheader()
                writer.writerow(
                    {
                        **{field: "row" for field in fields},
                        **{field: "1.0" for field in rule.numeric_columns},
                        **{field: "9.9" for field in rule.ignored_columns},
                    }
                )
        (output_path / "parameters.json").write_text(
            json.dumps({"output_dir": str(output_path), "seed": 17}), encoding="utf-8"
        )
        (output_path / "run_metadata.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            wall_seconds=1.0,
            peak_rss_bytes=1000,
            sampled_cpu_user_seconds=0.8,
            sampled_cpu_system_seconds=0.1,
            exit_code=0,
            status="success",
            peak_run_logical_bytes=2000,
            peak_run_allocated_bytes=4096,
            peak_run_file_count=14,
            error="",
        )

    metadata = {
        "run_metadata_status": "complete",
        "actual_cells": 12,
        "actual_genes": 8,
        "actual_targets": 4,
        "actual_tfs": 3,
        "actual_groups": 2,
        "models_requested": 8,
        "models_completed": 8,
        "models_trained": 8,
        "models_preflight_skipped": 0,
        "models_fit_or_importance_failures": 0,
        "models_trained_with_positive_edges": 8,
        "models_trained_without_positive_edges": 0,
        "positive_edges": 4,
        "models_reused_from_checkpoint": 0,
        "models_processed_this_attempt": 8,
        "threads_effective": 1,
        "threads_available": 8,
        "inference_thread_budget": 1,
        "maximum_concurrent_model_fits": 1,
        "memory_concurrent_model_cap": 3,
        "memory_available_bytes_at_planning": 8_000_000_000,
        "memory_usable_bytes_at_planning": 6_000_000_000,
        "memory_usable_fraction": 0.75,
        "memory_reserved_for_batch_bytes": 1_000_000,
        "parallel_backend": "sequential",
        "parallel_level": "none",
        "persistent_worker_pool": False,
        "effective_n_components": 3,
        "maximum_informative_n_components": 8,
        "pca_svd_solver_resolution": "full",
        "bandwidth_method": "auto-median",
        "bandwidth_value": 1.25,
        "bandwidth_positive_distance_count": 12,
        "bandwidth_fallback_reason": "",
        "tree_target_dtype": "float64",
        "tree_predictor_dtype": "float32",
        "bootstrap_effective": False,
        "targets_per_batch": 4,
        "targets_per_batch_without_memory_limit": 4,
        "target_groups_per_batch": 2,
        "target_groups_per_batch_without_memory_limit": 2,
        "cell_centroid_distance_storage": "memory",
        "cell_centroid_distances_computed": True,
        "distance_storage_reason": "fits-memory-plan",
        "centroid_distance_memory_available_bytes_at_planning": 8_000_000_000,
        "centroid_distance_memory_usable_bytes_at_planning": 6_000_000_000,
        "distance_memory_available_bytes_at_planning": 8_000_000_000,
        "distance_memory_usable_bytes_at_planning": 6_000_000_000,
        "phase_input_validation_seconds": 0.1,
        "phase_distance_representation_seconds": 0.1,
        "phase_centroids_and_distances_seconds": 0.1,
        "phase_bandwidth_selection_seconds": 0.05,
        "phase_inference_preparation_seconds": 0.1,
        "phase_weighting_and_diagnostics_seconds": 0.05,
        "phase_model_inference_seconds": 0.5,
        "phase_artifact_writing_seconds": 0.1,
        "phase_report_seconds": 0.0,
        "phase_total_seconds": 1.0,
    }
    monkeypatch.setattr(benchmark, "measure_command", fake_measure)
    monkeypatch.setattr(
        benchmark,
        "extract_run_metadata",
        lambda _path: SimpleNamespace(csv_fields=metadata, error=""),
    )
    monkeypatch.setattr(
        benchmark,
        "_audit_run_metadata",
        lambda *_args, **_kwargs: (
            {"spathi_version": "test", "dependency_versions_json": "{}"},
            "",
        ),
    )

    exit_code = benchmark.run_benchmark(
        profile,
        data_manifest,
        reference_source=repository,
        candidate_source=repository,
        output_dir=output,
        requested_datasets=("patient-b-cell-maturation",),
        keep_outputs=False,
    )

    assert exit_code == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["runs_completed"] == 2
    assert manifest["comparisons_completed"] == 1
    assert manifest["runner"]["sha256"] == _hash(output / manifest["runner"]["snapshot_path"])
    assert manifest["resource_measurement_helper"]["sha256"] == _hash(
        output / manifest["resource_measurement_helper"]["snapshot_path"]
    )
    assert manifest["target_manifest"]["sha256"] == _hash(
        output / manifest["target_manifest"]["snapshot_path"]
    )
    with (output / "runs.csv").open(encoding="utf-8", newline="") as stream:
        runs = list(csv.DictReader(stream))
    assert {row["implementation_role"] for row in runs} == {"reference", "candidate"}
    assert all(row["status"] == "success" for row in runs)
    assert all(row["memory_concurrent_model_cap"] == "3" for row in runs)
    assert all(row["parallel_backend"] == "sequential" for row in runs)
    assert all(row["phase_bandwidth_selection_seconds"] == "0.05" for row in runs)
    assert all(row["bandwidth_method"] == "auto-median" for row in runs)
    assert all(row["pca_svd_solver_resolution"] == "full" for row in runs)
    assert all(row["tree_predictor_dtype"] == "float32" for row in runs)
    assert all(row["memory_usable_bytes_at_planning"] == "6000000000" for row in runs)
    with (output / "comparisons.csv").open(encoding="utf-8", newline="") as stream:
        comparisons = list(csv.DictReader(stream))
    assert comparisons[0]["equivalent"] == "True"
    assert comparisons[0]["status"] == "compared"
    with (output / "scaling.csv").open(encoding="utf-8", newline="") as stream:
        scaling = list(csv.DictReader(stream))
    assert scaling[0]["expected_models"] == "8"
    assert not any((output / "runs").iterdir())
    assert manifest_path.is_file()

    rows_before = (output / "runs.csv").read_bytes()
    comparisons_before = (output / "comparisons.csv").read_bytes()
    monkeypatch.setattr(
        benchmark,
        "measure_command",
        lambda *_args, **_kwargs: pytest.fail("a complete suite must be an idempotent no-op"),
    )
    assert benchmark.resume_benchmark(output) == 0
    assert (output / "runs.csv").read_bytes() == rows_before
    assert (output / "comparisons.csv").read_bytes() == comparisons_before

    # Simulate interruption after the atomic details file but before the
    # authoritative comparison journal row. Resume must close the pair without
    # rerunning either scientific process.
    manifest["status"] = "interrupted"
    manifest["completed_at_utc"] = "2025-01-01T00:00:00Z"
    benchmark._atomic_json(output / "manifest.json", manifest)
    benchmark._write_csv(output / "comparisons.csv", benchmark.COMPARISON_FIELDS, [])
    benchmark._write_csv(output / "scaling.csv", benchmark.SCALING_FIELDS, [])
    assert benchmark.resume_benchmark(output) == 0
    recovered = list(
        csv.DictReader((output / "comparisons.csv").open(encoding="utf-8", newline=""))
    )
    assert len(recovered) == 1
    assert recovered[0]["attempt"] == "2"
    assert recovered[0]["performance_eligible"] == "True"

    # A pair split by interruption remains scientifically comparable, but is no
    # longer a paired timing observation.
    current_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    current_manifest["status"] = "interrupted"
    current_manifest["completed_at_utc"] = "2025-01-01T00:00:00Z"
    benchmark._atomic_json(output / "manifest.json", current_manifest)
    first_run = runs[0]
    benchmark._write_csv(output / "runs.csv", benchmark.RUN_FIELDS, [first_run])
    benchmark._write_csv(output / "comparisons.csv", benchmark.COMPARISON_FIELDS, [])
    benchmark._write_csv(output / "scaling.csv", benchmark.SCALING_FIELDS, [])
    details_path = next((output / "comparison-details").glob("*.json"))
    details_path.unlink()
    retained_output = output / "runs" / first_run["run_id"]
    fake_measure(["--output-dir", str(retained_output)])
    monkeypatch.setattr(benchmark, "measure_command", fake_measure)
    assert benchmark.resume_benchmark(output) == 0
    split_pair = next(
        csv.DictReader((output / "comparisons.csv").open(encoding="utf-8", newline=""))
    )
    assert split_pair["equivalent"] == "True"
    assert split_pair["performance_eligible"] == "False"
    assert split_pair["wall_speedup_reference_over_candidate"] == ""
    split_scaling = next(
        csv.DictReader((output / "scaling.csv").open(encoding="utf-8", newline=""))
    )
    assert split_scaling["equivalent_pairs"] == "1"
    assert split_scaling["performance_eligible_pairs"] == "0"
    assert split_scaling["reference_median_wall_seconds"] == ""

    current_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    current_manifest["status"] = "interrupted"
    current_manifest["completed_at_utc"] = "2025-01-01T00:00:00Z"
    benchmark._atomic_json(output / "manifest.json", current_manifest)
    completed_runs = list(csv.DictReader((output / "runs.csv").open(encoding="utf-8", newline="")))
    benchmark._write_csv(output / "runs.csv", benchmark.RUN_FIELDS, list(reversed(completed_runs)))
    with pytest.raises(benchmark.ContractError, match="exact schedule prefix"):
        benchmark.resume_benchmark(output)
    benchmark._write_csv(output / "runs.csv", benchmark.RUN_FIELDS, completed_runs)
    current_manifest["status"] = "complete"
    benchmark._atomic_json(output / "manifest.json", current_manifest)

    target_list = next((output / "targets").rglob("targets-*.txt"))
    target_list.write_text("CORRUPTED\n", encoding="utf-8")
    with pytest.raises(benchmark.ContractError, match="target-list snapshot mismatch"):
        benchmark._load_resume_state(output)
