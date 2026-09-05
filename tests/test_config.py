from pathlib import Path

import pytest

from spathi.config import SpathiConfig, adaptive_convergence_schedule


def make_config(**changes: object) -> SpathiConfig:
    values: dict[str, object] = {
        "expression": Path("expression.tsv"),
        "tf_list": Path("tf_list.txt"),
        "groups": Path("groups.tsv"),
        "output_dir": Path("results"),
    }
    values.update(changes)
    return SpathiConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_components", 0),
        ("bandwidth", 0.0),
        ("bandwidth", float("inf")),
        ("bandwidth", float("nan")),
        ("bandwidth", "median"),
        ("bandwidth", 10**400),
        ("bandwidth_scale", 0.0),
        ("bandwidth_scale", float("inf")),
        ("bandwidth_scale", float("nan")),
        ("n_estimators", 0),
        ("max_features", 0.0),
        ("max_features", 1.5),
        ("max_features", "all"),
        ("min_samples_leaf", 0),
        ("max_depth", 0),
        ("adaptive_min_estimators", 0),
        ("adaptive_tree_step", 0),
        ("adaptive_tolerance", 0.0),
        ("adaptive_tolerance", 1.1),
        ("adaptive_tolerance", float("nan")),
        ("adaptive_patience", 0),
        ("min_target_detected_cells", 0),
        ("min_target_detected_fraction", 0.0),
        ("min_target_detected_fraction", 1.1),
        ("min_target_weighted_detected_fraction", 0.0),
        ("min_target_weighted_detected_fraction", 1.1),
        ("min_target_weighted_detected_ess", 0.0),
        ("min_target_weighted_detected_ess", float("inf")),
        ("random_seed", -1),
        ("random_seed", 2**32),
        ("threads", 0),
        ("threads", -2),
    ],
)
def test_invalid_scalar_configuration_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expression", 1),
        ("target_list", 1),
        ("centroid_weights", 1),
        ("weight_mode", True),
        ("distance_space", None),
        ("n_components", 1.5),
        ("n_components", True),
        ("distance_standardization", False),
        ("pca_svd_solver", 1),
        ("distance_metric", ["euclidean"]),
        ("kernel", None),
        ("bandwidth", True),
        ("bandwidth", None),
        ("bandwidth_scale", True),
        ("bandwidth_scale", "1.0"),
        ("group_size_correction", 1),
        ("tree_method", False),
        ("n_estimators", 1.5),
        ("n_estimators", True),
        ("max_features", True),
        ("max_features", None),
        ("min_samples_leaf", "1"),
        ("min_samples_leaf", False),
        ("max_depth", 2.0),
        ("max_depth", False),
        ("bootstrap", "yes"),
        ("adaptive_trees", 1),
        ("adaptive_min_estimators", 1.0),
        ("adaptive_tree_step", True),
        ("adaptive_tolerance", "0.01"),
        ("adaptive_patience", False),
        ("min_target_detected_cells", True),
        ("min_target_detected_fraction", "0.01"),
        ("min_target_weighted_detected_fraction", "0.01"),
        ("min_target_weighted_detected_ess", "10"),
        ("random_seed", 1.0),
        ("random_seed", True),
        ("threads", 1.5),
        ("threads", False),
        ("report", "yes"),
        ("report", 1),
    ],
)
def test_wrong_runtime_types_are_rejected(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        make_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight_mode", "local"),
        ("distance_space", "umap"),
        ("distance_standardization", "center"),
        ("pca_svd_solver", "arpack"),
        ("distance_metric", "manhattan"),
        ("kernel", "linear"),
        ("group_size_correction", "balance"),
        ("tree_method", "gradient-boosting"),
        ("target_eligibility", "variance"),
    ],
)
def test_unknown_choice_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_config(**{field: value})


def test_valid_max_features_forms_remain_distinct() -> None:
    assert make_config(max_features=1).max_features == 1
    assert type(make_config(max_features=1).max_features) is int
    assert make_config(max_features=1.0).max_features == 1.0
    assert type(make_config(max_features=1.0).max_features) is float
    assert make_config(max_features="sqrt").max_features == "sqrt"


def test_configuration_serializes_paths_and_defaults() -> None:
    values = make_config().to_dict()
    assert values["expression"] == "expression.tsv"
    assert values["distance_space"] == "pca"
    assert values["distance_metric"] == "cosine"
    assert values["bandwidth"] == "auto"
    assert values["bandwidth_scale"] == 1.0
    assert values["weight_mode"] == "cell-distance-group-anchored"
    assert values["group_size_correction"] == "cap-to-target"
    assert values["bootstrap"] is None
    assert values["n_estimators"] == 250
    assert values["max_features"] == "sqrt"
    assert values["adaptive_trees"] is False
    assert values["adaptive_min_estimators"] == 100
    assert values["adaptive_tree_step"] == 50
    assert values["adaptive_tolerance"] == 0.01
    assert values["adaptive_patience"] == 2
    assert values["target_eligibility"] == "all"
    assert values["min_target_detected_cells"] == 20
    assert values["min_target_detected_fraction"] == 0.01
    assert values["min_target_weighted_detected_fraction"] == 0.01
    assert values["min_target_weighted_detected_ess"] == 10.0
    assert values["target_list"] is None
    assert values["centroid_weights"] is None
    assert values["report"] is True
    assert values["threads"] == "auto"


def test_configuration_serializes_optional_target_list_path() -> None:
    values = make_config(target_list=Path("targets.txt")).to_dict()
    assert values["target_list"] == "targets.txt"


def test_configuration_serializes_optional_centroid_weights_path() -> None:
    values = make_config(centroid_weights=Path("centroid_weights.tsv")).to_dict()
    assert values["centroid_weights"] == "centroid_weights.tsv"


def test_configuration_requires_explicit_keywords() -> None:
    with pytest.raises(TypeError):
        SpathiConfig(  # type: ignore[misc]
            Path("expression.tsv"),
            Path("tf_list.txt"),
            Path("groups.tsv"),
            Path("results"),
        )


def test_configuration_accepts_explicit_bootstrap_overrides() -> None:
    assert make_config(bootstrap=True).bootstrap is True
    assert make_config(bootstrap=False).bootstrap is False


def test_bandwidth_scale_is_only_valid_for_automatic_bandwidth() -> None:
    assert make_config(bandwidth="auto", bandwidth_scale=0.5).bandwidth_scale == 0.5
    with pytest.raises(ValueError, match="applies only when bandwidth='auto'"):
        make_config(bandwidth=2.5, bandwidth_scale=0.5)


def test_configuration_accepts_largest_sklearn_random_seed() -> None:
    assert make_config(random_seed=2**32 - 1).random_seed == 2**32 - 1


def test_adaptive_budget_requires_enough_comparisons_to_satisfy_patience() -> None:
    with pytest.raises(ValueError, match="cannot satisfy adaptive_patience"):
        make_config(
            n_estimators=100,
            adaptive_trees=True,
            adaptive_min_estimators=50,
            adaptive_tree_step=100,
            adaptive_patience=2,
        )


def test_adaptive_schedule_reports_check_capacity_and_earliest_stop() -> None:
    assert adaptive_convergence_schedule(
        n_estimators=250,
        minimum_estimators=100,
        estimator_step=50,
        patience=2,
    ) == (4, 150)


def test_adaptive_schedule_can_use_history_collected_before_the_minimum() -> None:
    assert adaptive_convergence_schedule(
        n_estimators=200,
        minimum_estimators=180,
        estimator_step=50,
        patience=2,
    ) == (1, 200)
    assert make_config(
        n_estimators=200,
        adaptive_trees=True,
        adaptive_min_estimators=180,
        adaptive_tree_step=50,
        adaptive_patience=2,
    ).adaptive_trees
