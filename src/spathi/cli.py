"""Command-line interface for SPATHI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import MISSING, fields
from math import isfinite
from pathlib import Path
from typing import Any, cast

from rich_argparse import RichHelpFormatter

from spathi._version import __version__
from spathi.config import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_MAX_FEATURES,
    DEFAULT_N_ESTIMATORS,
    DISTANCE_METRICS,
    DISTANCE_SPACES,
    DISTANCE_STANDARDIZATIONS,
    DUPLICATE_GENE_POLICIES,
    GENE_IDENTIFIERS,
    GROUP_SIZE_CORRECTIONS,
    KERNEL_NAMES,
    MAX_FEATURES_NAMES,
    MAX_RANDOM_SEED,
    PCA_SVD_SOLVERS,
    PREPARATION_NORMALIZATIONS,
    TARGET_ELIGIBILITY_MODES,
    TREE_METHODS,
    WEIGHT_MODES,
    MaxFeatures,
    PrepareConfig,
    SpathiConfig,
)
from spathi.console import InferenceProgress, configure_logging, create_console, print_banner

_CONFIG_DEFAULTS = {
    field.name: field.default for field in fields(SpathiConfig) if field.default is not MISSING
}
_PREPARE_CONFIG_DEFAULTS = {
    field.name: field.default for field in fields(PrepareConfig) if field.default is not MISSING
}


def _config_default(field_name: str) -> Any:
    """Return a scientific default from the canonical typed configuration."""

    try:
        return _CONFIG_DEFAULTS[field_name]
    except KeyError as exc:  # pragma: no cover - import-time developer error
        raise RuntimeError(f"SpathiConfig has no default for {field_name!r}") from exc


def _prepare_config_default(field_name: str) -> Any:
    """Return a preparation default from the canonical typed configuration."""

    try:
        return _PREPARE_CONFIG_DEFAULTS[field_name]
    except KeyError as exc:  # pragma: no cover - import-time developer error
        raise RuntimeError(f"PrepareConfig has no default for {field_name!r}") from exc


class _HelpFormatter(RichHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Render colored terminal help while retaining explicit default values."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.required or action.default is None:
            return action.help or ""
        return super()._get_help_string(action) or ""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _threads(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def _random_seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_RANDOM_SEED:
        raise argparse.ArgumentTypeError(f"must be an integer between 0 and {MAX_RANDOM_SEED}")
    return parsed


def _bandwidth(value: str) -> str | float:
    if value == "auto":
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive number") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive finite number")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _unit_fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in (0, 1]") from exc
    if not isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in (0, 1]")
    return parsed


def _max_features(value: str) -> MaxFeatures:
    if value in MAX_FEATURES_NAMES:
        return cast(MaxFeatures, value)
    try:
        if any(marker in value.lower() for marker in (".", "e")):
            parsed_float = float(value)
            if not 0 < parsed_float <= 1:
                raise ValueError
            return parsed_float
        parsed_int = int(value)
        if parsed_int < 1:
            raise ValueError
        return parsed_int
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be 'sqrt', 'log2', a positive integer, or a float in (0, 1]; "
            "use '1' for one predictor and '1.0' for all predictors"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build and return the public argument parser."""

    parser = argparse.ArgumentParser(
        prog="spathi",
        description=(
            "Prepare single-cell inputs and infer group-specific gene-regulatory networks."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"SPATHI {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="prepare a sparse 10x matrix as strict per-analysis-unit inference inputs",
        description=(
            "Select annotated cells from a 10x feature-barcode H5 matrix, split them by "
            "analysis unit, normalize counts reproducibly, and write SPATHI inputs with "
            "an ANDREA-compatible core."
        ),
        formatter_class=_HelpFormatter,
    )
    prepare_inputs = prepare.add_argument_group("Inputs and outputs")
    preprocessing = prepare.add_argument_group("Preprocessing")
    prepare_execution = prepare.add_argument_group("Execution")
    prepare_inputs.add_argument(
        "--tenx-h5",
        required=True,
        type=Path,
        help="10x Genomics feature-barcode H5 matrix containing raw counts",
    )
    prepare_inputs.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="canonical TSV with exactly the cell, analysis_unit, and cluster columns",
    )
    prepare_inputs.add_argument(
        "--centroid-weights",
        type=Path,
        help=(
            "optional TSV with exactly cell and centroid_weight columns; values are "
            "aligned to annotations and split into per-unit sensitivity inputs"
        ),
    )
    prepare_inputs.add_argument(
        "--tf-list",
        required=True,
        type=Path,
        help="plain-text TF identifiers to intersect with each prepared expression matrix",
    )
    prepare_inputs.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for prepared units; existing paths are never overwritten",
    )
    preprocessing.add_argument(
        "--min-cells",
        type=_positive_int,
        default=_prepare_config_default("min_cells"),
        help="minimum total annotated cells required to emit an analysis unit",
    )
    preprocessing.add_argument(
        "--min-gene-cells",
        type=_positive_int,
        default=_prepare_config_default("min_gene_cells"),
        help="minimum cells with non-zero counts required to retain a gene within a unit",
    )
    preprocessing.add_argument(
        "--normalization",
        choices=PREPARATION_NORMALIZATIONS,
        default=_prepare_config_default("normalization"),
        help="explicit count normalization applied before splitting outputs",
    )
    preprocessing.add_argument(
        "--target-sum",
        type=_positive_float,
        default=_prepare_config_default("target_sum"),
        help="per-cell library size before log1p in library-size-log1p normalization",
    )
    preprocessing.add_argument(
        "--gene-identifier",
        choices=GENE_IDENTIFIERS,
        default=_prepare_config_default("gene_identifier"),
        help="10x feature field used for expression rows and TF matching",
    )
    preprocessing.add_argument(
        "--duplicate-gene-policy",
        choices=DUPLICATE_GENE_POLICIES,
        default=_prepare_config_default("duplicate_gene_policy"),
        help="explicit handling of repeated selected gene identifiers",
    )
    prepare_execution.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="minimum log severity emitted to stderr",
    )

    infer = subparsers.add_parser(
        "infer",
        help="infer one weighted regulatory network per cell group",
        description=(
            "Validate ANDREA-compatible inputs and infer one similarity-weighted "
            "regulatory network per cell group."
        ),
        formatter_class=_HelpFormatter,
    )
    inputs = infer.add_argument_group("Inputs and outputs")
    weighting = infer.add_argument_group("Weighting and distance")
    model = infer.add_argument_group("Regulatory model")
    execution = infer.add_argument_group("Execution")

    inputs.add_argument(
        "--expression",
        required=True,
        type=Path,
        help="genes-by-cells expression matrix in ANDREA-compatible TSV format",
    )
    inputs.add_argument(
        "--tf-list",
        required=True,
        type=Path,
        help="plain-text file containing one transcription-factor identifier per line",
    )
    inputs.add_argument(
        "--target-list",
        type=Path,
        help=(
            "optional plain-text file containing one target-gene identifier per line; "
            "all expression genes are targets when omitted"
        ),
    )
    inputs.add_argument(
        "--groups",
        required=True,
        type=Path,
        help="ANDREA-compatible TSV assigning every expression cell to a cluster",
    )
    inputs.add_argument(
        "--centroid-weights",
        type=Path,
        help=(
            "optional exact two-column TSV for an explicit centroid-only sensitivity "
            "analysis; omitting it keeps primary arithmetic centroids"
        ),
    )
    inputs.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for run artifacts; existing paths are never overwritten",
    )
    execution.add_argument(
        "--resume",
        action="store_true",
        help="resume exact completed models from the output directory's hidden checkpoint",
    )
    execution.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="transactionally checkpoint each completed model for interruption-safe resume",
    )
    weighting.add_argument(
        "--weight-mode",
        choices=WEIGHT_MODES,
        default=_config_default("weight_mode"),
        help="observation-weight construction used for each target group",
    )
    weighting.add_argument(
        "--distance-space",
        choices=DISTANCE_SPACES,
        default=_config_default("distance_space"),
        help="representation used only for centroids and distances",
    )
    weighting.add_argument(
        "--n-components",
        type=_positive_int,
        default=_config_default("n_components"),
        help="requested PCA components (capped to centered-data rank, except one-cell PC1)",
    )
    weighting.add_argument(
        "--distance-standardization",
        choices=DISTANCE_STANDARDIZATIONS,
        default=_config_default("distance_standardization"),
        help="whether to standardize genes before constructing the distance space",
    )
    weighting.add_argument(
        "--pca-svd-solver",
        choices=PCA_SVD_SOLVERS,
        default=_config_default("pca_svd_solver"),
        help="SVD solver requested from scikit-learn PCA",
    )
    weighting.add_argument(
        "--distance-metric",
        choices=DISTANCE_METRICS,
        default=DEFAULT_DISTANCE_METRIC,
        help="metric for cell-to-centroid and centroid-to-centroid distances",
    )
    weighting.add_argument(
        "--kernel",
        choices=KERNEL_NAMES,
        default=_config_default("kernel"),
        help="kernel that transforms distances into affinities",
    )
    weighting.add_argument(
        "--bandwidth",
        type=_bandwidth,
        default=_config_default("bandwidth"),
        help="global positive kernel bandwidth or median-based automatic selection",
    )
    weighting.add_argument(
        "--group-size-correction",
        choices=GROUP_SIZE_CORRECTIONS,
        default=_config_default("group_size_correction"),
        help="optional correction that caps external groups to the target-group size",
    )
    model.add_argument(
        "--tree-method",
        choices=TREE_METHODS,
        default=_config_default("tree_method"),
        help="weighted tree ensemble used for each target gene",
    )
    model.add_argument(
        "--n-estimators",
        type=_positive_int,
        default=DEFAULT_N_ESTIMATORS,
        help=(
            "trees in every fixed ensemble, or the strict per-model maximum when "
            "adaptive trees are enabled"
        ),
    )
    model.add_argument(
        "--max-features",
        type=_max_features,
        default=DEFAULT_MAX_FEATURES,
        metavar="{sqrt,log2,INT,FRACTION}",
        help=(
            "predictors considered per split: '1' means exactly one predictor, while "
            "'1.0' means 100%% of predictors"
        ),
    )
    model.add_argument(
        "--min-samples-leaf",
        type=_positive_int,
        default=_config_default("min_samples_leaf"),
        help="minimum number of samples required in each tree leaf",
    )
    model.add_argument(
        "--max-depth",
        type=_positive_int,
        default=_config_default("max_depth"),
        help="maximum tree depth; omit for unbounded depth",
    )
    model.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=_config_default("bootstrap"),
        help=(
            "override estimator bootstrap sampling; if omitted, Extra-Trees disables it "
            "and Random Forest enables it"
        ),
    )
    model.add_argument(
        "--random-seed",
        type=_random_seed,
        default=_config_default("random_seed"),
        help=(
            "global seed in scikit-learn's unsigned 32-bit range, used to derive "
            "deterministic model seeds"
        ),
    )
    model.add_argument(
        "--adaptive-trees",
        action=argparse.BooleanOptionalAction,
        default=_config_default("adaptive_trees"),
        help=(
            "grow deterministic tree blocks and stop a target model after stable "
            "importance estimates; n-estimators remains the strict maximum"
        ),
    )
    model.add_argument(
        "--adaptive-min-estimators",
        type=_positive_int,
        default=_config_default("adaptive_min_estimators"),
        help="minimum trees required before adaptive convergence can stop a model",
    )
    model.add_argument(
        "--adaptive-tree-step",
        type=_positive_int,
        default=_config_default("adaptive_tree_step"),
        help="trees added between adaptive convergence checks",
    )
    model.add_argument(
        "--adaptive-tolerance",
        type=_unit_fraction,
        default=_config_default("adaptive_tolerance"),
        help="maximum total-variation change accepted between importance estimates",
    )
    model.add_argument(
        "--adaptive-patience",
        type=_positive_int,
        default=_config_default("adaptive_patience"),
        help=(
            "number of preceding importance checkpoints that must all lie within "
            "the convergence tolerance before stopping a model"
        ),
    )
    model.add_argument(
        "--target-eligibility",
        choices=TARGET_ELIGIBILITY_MODES,
        default=_config_default("target_eligibility"),
        help=(
            "use every requested target or automatically skip targets lacking enough "
            "detectable expression; TF predictor eligibility is never changed"
        ),
    )
    model.add_argument(
        "--min-target-detected-cells",
        type=_positive_int,
        default=_config_default("min_target_detected_cells"),
        help="absolute detected-cell requirement used by automatic target eligibility",
    )
    model.add_argument(
        "--min-target-detected-fraction",
        type=_unit_fraction,
        default=_config_default("min_target_detected_fraction"),
        help="relative detected-cell requirement used by automatic target eligibility",
    )
    model.add_argument(
        "--min-target-weighted-detected-fraction",
        type=_unit_fraction,
        default=_config_default("min_target_weighted_detected_fraction"),
        help=(
            "minimum fraction of each target group's total model-weight mass carried "
            "by cells in which the target is detected"
        ),
    )
    model.add_argument(
        "--min-target-weighted-detected-ess",
        type=_positive_float,
        default=_config_default("min_target_weighted_detected_ess"),
        help=(
            "minimum effective sample size among detected cells under each target-group "
            "weight vector"
        ),
    )
    execution.add_argument(
        "--threads",
        type=_threads,
        default=_config_default("threads"),
        help="single parallelism budget; 'auto' uses all process-visible CPUs",
    )
    execution.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=_config_default("report"),
        help="write one self-contained interactive HTML report alongside tabular artifacts",
    )
    execution.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "show an interactive progress bar on terminals and periodic progress logs "
            "when redirected"
        ),
    )
    execution.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="minimum log severity emitted to stderr",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SpathiConfig:
    """Translate validated CLI arguments to an immutable configuration."""

    return SpathiConfig(
        expression=args.expression,
        tf_list=args.tf_list,
        groups=args.groups,
        output_dir=args.output_dir,
        target_list=args.target_list,
        centroid_weights=args.centroid_weights,
        weight_mode=args.weight_mode,
        distance_space=args.distance_space,
        n_components=args.n_components,
        distance_standardization=args.distance_standardization,
        pca_svd_solver=args.pca_svd_solver,
        distance_metric=args.distance_metric,
        kernel=args.kernel,
        bandwidth=args.bandwidth,
        group_size_correction=args.group_size_correction,
        tree_method=args.tree_method,
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        bootstrap=args.bootstrap,
        adaptive_trees=args.adaptive_trees,
        adaptive_min_estimators=args.adaptive_min_estimators,
        adaptive_tree_step=args.adaptive_tree_step,
        adaptive_tolerance=args.adaptive_tolerance,
        adaptive_patience=args.adaptive_patience,
        target_eligibility=args.target_eligibility,
        min_target_detected_cells=args.min_target_detected_cells,
        min_target_detected_fraction=args.min_target_detected_fraction,
        min_target_weighted_detected_fraction=args.min_target_weighted_detected_fraction,
        min_target_weighted_detected_ess=args.min_target_weighted_detected_ess,
        random_seed=args.random_seed,
        threads=args.threads,
        report=args.report,
    )


def prepare_config_from_args(args: argparse.Namespace) -> PrepareConfig:
    """Translate validated preparation CLI arguments to its immutable configuration."""

    return PrepareConfig(
        tenx_h5=args.tenx_h5,
        annotations=args.annotations,
        tf_list=args.tf_list,
        output_dir=args.output_dir,
        centroid_weights=args.centroid_weights,
        min_cells=args.min_cells,
        min_gene_cells=args.min_gene_cells,
        normalization=args.normalization,
        target_sum=args.target_sum,
        gene_identifier=args.gene_identifier,
        duplicate_gene_policy=args.duplicate_gene_policy,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SPATHI command line and return a process exit code."""

    cli_args = list(sys.argv[1:] if argv is None else argv)
    console = create_console()
    if "--version" not in cli_args:
        print_banner(console)
    parser = build_parser()
    args = parser.parse_args(cli_args)
    logger = configure_logging(console, args.log_level)

    if args.command == "prepare":
        try:
            from spathi.preparation import prepare

            prepare_result = prepare(prepare_config_from_args(args))
        except KeyboardInterrupt:
            logger.warning("Preparation interrupted by user; no output was published.")
            return 130
        except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Preparation failed | %s", exc)
            logger.debug("Preparation failure details", exc_info=True)
            return 2
        logger.info(
            "Preparation complete | output=%s | prepared_units=%s | excluded_units=%s",
            prepare_result.output_dir,
            len(prepare_result.prepared_analysis_units),
            len(prepare_result.excluded_analysis_units),
        )
        return 0

    # ``prepare`` returned above, and argparse accepts only the two registered
    # required subcommands, so the remaining execution path is inference.
    try:
        from spathi.core import infer

        with InferenceProgress(
            console=console,
            logger=logger,
            enabled=args.progress,
        ) as progress:
            inference_result = infer(
                config_from_args(args),
                progress_callback=progress.callback,
                resume=args.resume,
                checkpoint=args.checkpoint,
            )
    except KeyboardInterrupt:
        if args.checkpoint:
            logger.warning(
                "Interrupted by user; rerun with --resume if completed models were checkpointed."
            )
        else:
            logger.warning("Interrupted by user.")
        return 130
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Inference failed | %s", exc)
        logger.debug("Inference failure details", exc_info=True)
        return 2
    logger.info(
        "Run complete | output=%s | edges=%s",
        inference_result.output_dir,
        inference_result.n_edges,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
