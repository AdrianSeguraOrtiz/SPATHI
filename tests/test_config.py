from pathlib import Path

import pytest

from spathi.config import SpathiConfig


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
        ("min_weight_fraction_leaf", -0.01),
        ("min_weight_fraction_leaf", 0.500_001),
        ("min_weight_fraction_leaf", float("inf")),
        ("min_weight_fraction_leaf", float("nan")),
        ("random_seed", -1),
        ("random_seed", 2**32),
        ("threads", 0),
        ("threads", -2),
        ("parallel_backend", "loky"),
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
        ("min_weight_fraction_leaf", "0.1"),
        ("min_weight_fraction_leaf", False),
        ("bootstrap", "yes"),
        ("random_seed", 1.0),
        ("random_seed", True),
        ("threads", 1.5),
        ("threads", False),
        ("parallel_backend", None),
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
    assert values["group_size_correction"] == "none"
    assert values["bootstrap"] is None
    assert values["n_estimators"] == 50
    assert values["max_features"] == 0.5
    assert values["min_samples_leaf"] == 2
    assert values["min_weight_fraction_leaf"] == 0.1
    assert values["target_list"] is None
    assert values["centroid_weights"] is None
    assert values["report"] is True
    assert values["threads"] == "auto"
    assert values["parallel_backend"] == "auto"


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


def test_configuration_canonicalizes_valid_minimum_leaf_weight_fraction() -> None:
    configured = make_config(min_weight_fraction_leaf=0.1)
    assert configured.min_weight_fraction_leaf == 0.1
    assert type(configured.min_weight_fraction_leaf) is float
    assert make_config(min_weight_fraction_leaf=0).min_weight_fraction_leaf == 0.0


def test_bandwidth_scale_is_only_valid_for_automatic_bandwidth() -> None:
    assert make_config(bandwidth="auto", bandwidth_scale=0.5).bandwidth_scale == 0.5
    with pytest.raises(ValueError, match="applies only when bandwidth='auto'"):
        make_config(bandwidth=2.5, bandwidth_scale=0.5)


def test_configuration_accepts_largest_sklearn_random_seed() -> None:
    assert make_config(random_seed=2**32 - 1).random_seed == 2**32 - 1
