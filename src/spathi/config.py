"""Typed configurations for SPATHI preparation and inference runs."""

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
TargetEligibilityMode: TypeAlias = Literal["all", "automatic"]
ThreadBudget: TypeAlias = Literal["auto"] | int
Bandwidth: TypeAlias = Literal["auto"] | int | float
MaxFeatures: TypeAlias = Literal["sqrt", "log2"] | int | float
GeneIdentifier: TypeAlias = Literal["name", "id"]
DuplicateGenePolicy: TypeAlias = Literal["sum", "error"]
PreparationNormalization: TypeAlias = Literal["library-size-log1p"]

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
TARGET_ELIGIBILITY_MODES: tuple[TargetEligibilityMode, ...] = ("all", "automatic")
MAX_FEATURES_NAMES: tuple[Literal["sqrt", "log2"], ...] = ("sqrt", "log2")
MAX_RANDOM_SEED = 2**32 - 1
DEFAULT_DISTANCE_METRIC: DistanceMetric = "cosine"
DEFAULT_N_ESTIMATORS = 250
DEFAULT_MAX_FEATURES: MaxFeatures = "sqrt"
DEFAULT_TARGET_ELIGIBILITY: TargetEligibilityMode = "all"
DEFAULT_MIN_TARGET_DETECTED_CELLS = 20
DEFAULT_MIN_TARGET_DETECTED_FRACTION = 0.01
DEFAULT_MIN_TARGET_WEIGHTED_DETECTED_FRACTION = 0.01
DEFAULT_MIN_TARGET_WEIGHTED_DETECTED_ESS = 10.0
DEFAULT_ADAPTIVE_MIN_ESTIMATORS = 100
DEFAULT_ADAPTIVE_TREE_STEP = 50
DEFAULT_ADAPTIVE_TOLERANCE = 0.01
DEFAULT_ADAPTIVE_PATIENCE = 2
DEFAULT_THREADS: ThreadBudget = "auto"
GENE_IDENTIFIERS: tuple[GeneIdentifier, ...] = ("name", "id")
DUPLICATE_GENE_POLICIES: tuple[DuplicateGenePolicy, ...] = ("sum", "error")
PREPARATION_NORMALIZATIONS: tuple[PreparationNormalization, ...] = ("library-size-log1p",)
DEFAULT_PREPARE_MIN_CELLS = 300
DEFAULT_PREPARE_MIN_GENE_CELLS = 1
DEFAULT_PREPARE_TARGET_SUM = 10_000.0


def adaptive_convergence_schedule(
    *,
    n_estimators: int,
    minimum_estimators: int,
    estimator_step: int,
    patience: int,
) -> tuple[int, int | None]:
    """Return maximum eligible checks and the earliest possible adaptive stop.

    Ensembles start with one ``estimator_step`` block and grow cumulatively to
    ``n_estimators``. A convergence check requires a previous block and the configured
    minimum tree count; stopping requires ``patience`` complete prior checkpoints,
    including checkpoints collected before that minimum.
    """

    total_fits = (n_estimators + estimator_step - 1) // estimator_step
    first_check_fit = max(2, (minimum_estimators + estimator_step - 1) // estimator_step)
    maximum_checks = max(0, total_fits - first_check_fit + 1)
    first_stop_fit = max(first_check_fit, patience + 1)
    first_stop_estimators = (
        None if first_stop_fit > total_fits else min(n_estimators, first_stop_fit * estimator_step)
    )
    return maximum_checks, first_stop_estimators


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


def _coerce_positive_float(
    field_name: str,
    value: object,
    *,
    maximum: float | None = None,
) -> float:
    """Return a finite positive built-in number, optionally bounded above."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    requirement = (
        "a positive finite number" if maximum is None else f"in the interval (0, {maximum:g}]"
    )
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be {requirement}") from exc
    if not isfinite(numeric) or numeric <= 0 or (maximum is not None and numeric > maximum):
        raise ValueError(f"{field_name} must be {requirement}")
    return numeric


@dataclass(frozen=True, slots=True, kw_only=True)
class PrepareConfig:
    """Complete, immutable configuration for one generic preparation run.

    The input is a 10x Genomics feature-barcode HDF5 matrix plus a generic
    annotation table. Preparation selects annotated cells, splits them by
    analysis unit, normalizes counts, and writes strict inference inputs.
    """

    tenx_h5: Path
    annotations: Path
    tf_list: Path
    output_dir: Path
    centroid_weights: Path | None = None
    min_cells: int = DEFAULT_PREPARE_MIN_CELLS
    min_gene_cells: int = DEFAULT_PREPARE_MIN_GENE_CELLS
    normalization: PreparationNormalization = "library-size-log1p"
    target_sum: float = DEFAULT_PREPARE_TARGET_SUM
    gene_identifier: GeneIdentifier = "name"
    duplicate_gene_policy: DuplicateGenePolicy = "sum"

    def __post_init__(self) -> None:
        """Reject invalid scalar configuration before opening any input."""

        for field_name in ("tenx_h5", "annotations", "tf_list", "output_dir"):
            object.__setattr__(
                self,
                field_name,
                _coerce_path(field_name, getattr(self, field_name)),
            )
        if self.centroid_weights is not None:
            object.__setattr__(
                self,
                "centroid_weights",
                _coerce_path("centroid_weights", self.centroid_weights),
            )
        _validate_integer("min_cells", self.min_cells, minimum=1)
        _validate_integer("min_gene_cells", self.min_gene_cells, minimum=1)
        _validate_choice(
            "normalization",
            self.normalization,
            PREPARATION_NORMALIZATIONS,
        )
        _validate_choice("gene_identifier", self.gene_identifier, GENE_IDENTIFIERS)
        _validate_choice(
            "duplicate_gene_policy",
            self.duplicate_gene_policy,
            DUPLICATE_GENE_POLICIES,
        )
        object.__setattr__(
            self,
            "target_sum",
            _coerce_positive_float("target_sum", self.target_sum),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this configuration."""

        values = asdict(self)
        for key in ("tenx_h5", "annotations", "tf_list", "output_dir"):
            values[key] = str(values[key])
        if values["centroid_weights"] is not None:
            values["centroid_weights"] = str(values["centroid_weights"])
        return values


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
    adaptive_trees: bool = False
    adaptive_min_estimators: int = DEFAULT_ADAPTIVE_MIN_ESTIMATORS
    adaptive_tree_step: int = DEFAULT_ADAPTIVE_TREE_STEP
    adaptive_tolerance: float = DEFAULT_ADAPTIVE_TOLERANCE
    adaptive_patience: int = DEFAULT_ADAPTIVE_PATIENCE
    target_eligibility: TargetEligibilityMode = DEFAULT_TARGET_ELIGIBILITY
    min_target_detected_cells: int = DEFAULT_MIN_TARGET_DETECTED_CELLS
    min_target_detected_fraction: float = DEFAULT_MIN_TARGET_DETECTED_FRACTION
    min_target_weighted_detected_fraction: float = DEFAULT_MIN_TARGET_WEIGHTED_DETECTED_FRACTION
    min_target_weighted_detected_ess: float = DEFAULT_MIN_TARGET_WEIGHTED_DETECTED_ESS
    random_seed: int = 123
    threads: ThreadBudget = DEFAULT_THREADS
    report: bool = True
    target_list: Path | None = None
    centroid_weights: Path | None = None

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
        if self.centroid_weights is not None:
            object.__setattr__(
                self,
                "centroid_weights",
                _coerce_path("centroid_weights", self.centroid_weights),
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
        if type(self.adaptive_trees) is not bool:
            raise TypeError("adaptive_trees must be a boolean")
        _validate_integer(
            "adaptive_min_estimators",
            self.adaptive_min_estimators,
            minimum=1,
        )
        _validate_integer("adaptive_tree_step", self.adaptive_tree_step, minimum=1)
        _validate_integer("adaptive_patience", self.adaptive_patience, minimum=1)
        object.__setattr__(
            self,
            "adaptive_tolerance",
            _coerce_positive_float(
                "adaptive_tolerance",
                self.adaptive_tolerance,
                maximum=1.0,
            ),
        )
        if self.adaptive_trees and self.adaptive_min_estimators >= self.n_estimators:
            raise ValueError(
                "adaptive_min_estimators must be smaller than n_estimators when "
                "adaptive_trees is enabled"
            )
        if self.adaptive_trees:
            _maximum_checks, first_stop = adaptive_convergence_schedule(
                n_estimators=self.n_estimators,
                minimum_estimators=self.adaptive_min_estimators,
                estimator_step=self.adaptive_tree_step,
                patience=self.adaptive_patience,
            )
            if first_stop is None:
                raise ValueError(
                    "adaptive tree budget cannot satisfy adaptive_patience at or before "
                    "n_estimators; reduce adaptive_tree_step, adaptive_min_estimators, "
                    "or adaptive_patience"
                )

        _validate_choice(
            "target_eligibility",
            self.target_eligibility,
            TARGET_ELIGIBILITY_MODES,
        )
        _validate_integer(
            "min_target_detected_cells",
            self.min_target_detected_cells,
            minimum=1,
        )
        object.__setattr__(
            self,
            "min_target_detected_fraction",
            _coerce_positive_float(
                "min_target_detected_fraction",
                self.min_target_detected_fraction,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "min_target_weighted_detected_fraction",
            _coerce_positive_float(
                "min_target_weighted_detected_fraction",
                self.min_target_weighted_detected_fraction,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "min_target_weighted_detected_ess",
            _coerce_positive_float(
                "min_target_weighted_detected_ess",
                self.min_target_weighted_detected_ess,
            ),
        )
        _validate_integer("random_seed", self.random_seed, minimum=0)
        if self.random_seed > MAX_RANDOM_SEED:
            raise ValueError(f"random_seed must be at most {MAX_RANDOM_SEED}")
        if self.threads != "auto":
            _validate_integer("threads", self.threads, minimum=1)
        if type(self.report) is not bool:
            raise TypeError("report must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this configuration."""

        values = asdict(self)
        for key in ("expression", "tf_list", "groups", "output_dir"):
            values[key] = str(values[key])
        if values["target_list"] is not None:
            values["target_list"] = str(values["target_list"])
        if values["centroid_weights"] is not None:
            values["centroid_weights"] = str(values["centroid_weights"])
        return values
