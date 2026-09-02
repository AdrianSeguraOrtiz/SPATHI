"""Typed configuration for SPATHI inference runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias

WeightMode: TypeAlias = Literal["cell-distance", "cell-distance-group-anchored", "group-distance"]
DistanceSpace: TypeAlias = Literal["pca", "expression"]
DistanceStandardization: TypeAlias = Literal["none", "standard"]
PCASVDSolver: TypeAlias = Literal["auto", "randomized", "full"]
DistanceMetric: TypeAlias = Literal["euclidean", "cosine"]
KernelName: TypeAlias = Literal["gaussian", "exponential"]
GroupSizeCorrection: TypeAlias = Literal["none", "cap-to-target"]
TreeMethod: TypeAlias = Literal["extra-trees", "random-forest"]
Bandwidth: TypeAlias = Literal["auto"] | int | float
MaxFeatures: TypeAlias = Literal["sqrt", "log2"] | int | float

WEIGHT_MODES: tuple[WeightMode, ...] = (
    "cell-distance",
    "cell-distance-group-anchored",
    "group-distance",
)
DISTANCE_SPACES: tuple[DistanceSpace, ...] = ("pca", "expression")
DISTANCE_STANDARDIZATIONS: tuple[DistanceStandardization, ...] = ("none", "standard")
PCA_SVD_SOLVERS: tuple[PCASVDSolver, ...] = ("auto", "randomized", "full")
DISTANCE_METRICS: tuple[DistanceMetric, ...] = ("euclidean", "cosine")
KERNEL_NAMES: tuple[KernelName, ...] = ("gaussian", "exponential")
GROUP_SIZE_CORRECTIONS: tuple[GroupSizeCorrection, ...] = ("none", "cap-to-target")
TREE_METHODS: tuple[TreeMethod, ...] = ("extra-trees", "random-forest")
MAX_FEATURES_NAMES: tuple[Literal["sqrt", "log2"], ...] = ("sqrt", "log2")
MAX_RANDOM_SEED = 2**32 - 1
DEFAULT_DISTANCE_METRIC: DistanceMetric = "cosine"
DEFAULT_N_ESTIMATORS = 250
DEFAULT_MAX_FEATURES: MaxFeatures = "sqrt"


def _coerce_path(field_name: str, value: object) -> Path:
    """Return a text-backed path and reject unrelated objects early."""

    if isinstance(value, str):
        return Path(value)
    if isinstance(value, PathLike):
        file_system_value = value.__fspath__()
        if isinstance(file_system_value, str):
            return Path(file_system_value)
    raise TypeError(f"{field_name} must be a pathlib.Path or text path")


def _validate_choice(field_name: str, value: object, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value not in choices:
        formatted = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{field_name} must be one of: {formatted}")


def _validate_integer(
    field_name: str,
    value: object,
    *,
    minimum: int | None = None,
) -> None:
    """Validate a built-in integer without accepting booleans as 0/1."""

    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")


@dataclass(frozen=True, slots=True, kw_only=True)
class SpathiConfig:
    """Complete, immutable configuration for one SPATHI run.

    Input paths remain part of the configuration so a run can be reconstructed from
    ``parameters.json``. Validation that depends on file contents is intentionally
    performed by :mod:`spathi.io`.
    """

    expression: Path
    tf_list: Path
    groups: Path
    output_dir: Path
    weight_mode: WeightMode = "cell-distance-group-anchored"
    distance_space: DistanceSpace = "pca"
    n_components: int = 50
    distance_standardization: DistanceStandardization = "none"
    pca_svd_solver: PCASVDSolver = "auto"
    distance_metric: DistanceMetric = DEFAULT_DISTANCE_METRIC
    kernel: KernelName = "gaussian"
    bandwidth: Bandwidth = "auto"
    group_size_correction: GroupSizeCorrection = "cap-to-target"
    tree_method: TreeMethod = "extra-trees"
    n_estimators: int = DEFAULT_N_ESTIMATORS
    max_features: MaxFeatures = DEFAULT_MAX_FEATURES
    min_samples_leaf: int = 1
    max_depth: int | None = None
    bootstrap: bool | None = None
    random_seed: int = 123
    threads: int = -1
    report: bool = True
    target_list: Path | None = None

    def __post_init__(self) -> None:
        """Reject invalid scalar configuration before any input is read."""

        for field_name in ("expression", "tf_list", "groups", "output_dir"):
            object.__setattr__(
                self,
                field_name,
                _coerce_path(field_name, getattr(self, field_name)),
            )
        if self.target_list is not None:
            object.__setattr__(
                self,
                "target_list",
                _coerce_path("target_list", self.target_list),
            )

        _validate_choice("weight_mode", self.weight_mode, WEIGHT_MODES)
        _validate_choice("distance_space", self.distance_space, DISTANCE_SPACES)
        _validate_integer("n_components", self.n_components, minimum=1)
        _validate_choice(
            "distance_standardization",
            self.distance_standardization,
            DISTANCE_STANDARDIZATIONS,
        )
        _validate_choice("pca_svd_solver", self.pca_svd_solver, PCA_SVD_SOLVERS)
        _validate_choice("distance_metric", self.distance_metric, DISTANCE_METRICS)
        _validate_choice("kernel", self.kernel, KERNEL_NAMES)

        if isinstance(self.bandwidth, str):
            if self.bandwidth != "auto":
                raise ValueError("bandwidth must be 'auto' or a positive finite number")
        elif type(self.bandwidth) not in {int, float}:
            raise TypeError("bandwidth must be 'auto' or a number")
        else:
            try:
                bandwidth_value = float(self.bandwidth)
            except (OverflowError, ValueError) as exc:
                raise ValueError("bandwidth must be 'auto' or a positive finite number") from exc
            if not isfinite(bandwidth_value) or bandwidth_value <= 0:
                raise ValueError("bandwidth must be 'auto' or a positive finite number")

        _validate_choice(
            "group_size_correction",
            self.group_size_correction,
            GROUP_SIZE_CORRECTIONS,
        )
        _validate_choice("tree_method", self.tree_method, TREE_METHODS)
        _validate_integer("n_estimators", self.n_estimators, minimum=1)

        if isinstance(self.max_features, str):
            if self.max_features not in MAX_FEATURES_NAMES:
                raise ValueError("string max_features must be 'sqrt' or 'log2'")
        elif type(self.max_features) is int:
            if self.max_features < 1:
                raise ValueError("integer max_features must be at least 1")
        elif type(self.max_features) is float:
            if not isfinite(self.max_features) or not 0 < self.max_features <= 1:
                raise ValueError("float max_features must be in the interval (0, 1]")
        else:
            raise TypeError("max_features must be 'sqrt', 'log2', an integer, or a float")

        _validate_integer("min_samples_leaf", self.min_samples_leaf, minimum=1)
        if self.max_depth is not None:
            _validate_integer("max_depth", self.max_depth, minimum=1)
        if self.bootstrap is not None and type(self.bootstrap) is not bool:
            raise TypeError("bootstrap must be a boolean or None")
        _validate_integer("random_seed", self.random_seed, minimum=0)
        if self.random_seed > MAX_RANDOM_SEED:
            raise ValueError(f"random_seed must be at most {MAX_RANDOM_SEED}")
        _validate_integer("threads", self.threads)
        if self.threads == 0 or self.threads < -1:
            raise ValueError("threads must be -1 or a positive integer")
        if type(self.report) is not bool:
            raise TypeError("report must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this configuration."""

        values = asdict(self)
        for key in ("expression", "tf_list", "groups", "output_dir"):
            values[key] = str(values[key])
        if values["target_list"] is not None:
            values["target_list"] = str(values["target_list"])
        return values
