"""Command-line interface for SPATHI."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import cast

from spathi._version import __version__
from spathi.config import (
    DISTANCE_METRICS,
    DISTANCE_SPACES,
    DISTANCE_STANDARDIZATIONS,
    GROUP_SIZE_CORRECTIONS,
    KERNEL_NAMES,
    MAX_FEATURES_NAMES,
    MAX_RANDOM_SEED,
    PCA_SVD_SOLVERS,
    TREE_METHODS,
    WEIGHT_MODES,
    MaxFeatures,
    SpathiConfig,
)

LOGGER = logging.getLogger("spathi")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _threads(value: str) -> int:
    parsed = int(value)
    if parsed == 0 or parsed < -1:
        raise argparse.ArgumentTypeError("must be -1 or a positive integer")
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
            "Infer group-specific gene-regulatory networks with transcriptomic similarity weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"SPATHI {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser(
        "infer",
        help="infer one weighted regulatory network per cell group",
        description=(
            "Validate ANDREA-compatible inputs and infer one similarity-weighted "
            "regulatory network per cell group."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    infer.add_argument(
        "--expression",
        required=True,
        type=Path,
        help="genes-by-cells expression matrix in ANDREA-compatible TSV format",
    )
    infer.add_argument(
        "--tf-list",
        required=True,
        type=Path,
        help="plain-text file containing one transcription-factor identifier per line",
    )
    infer.add_argument(
        "--groups",
        required=True,
        type=Path,
        help="ANDREA-compatible TSV assigning every expression cell to a cluster",
    )
    infer.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for run artifacts; existing paths are never overwritten",
    )
    infer.add_argument(
        "--weight-mode",
        choices=WEIGHT_MODES,
        default="cell-distance-group-anchored",
        help="observation-weight construction used for each target group",
    )
    infer.add_argument(
        "--distance-space",
        choices=DISTANCE_SPACES,
        default="pca",
        help="representation used only for centroids and distances",
    )
    infer.add_argument(
        "--n-components",
        type=_positive_int,
        default=50,
        help="requested PCA components (safely capped to the matrix dimensions)",
    )
    infer.add_argument(
        "--distance-standardization",
        choices=DISTANCE_STANDARDIZATIONS,
        default="none",
        help="whether to standardize genes before constructing the distance space",
    )
    infer.add_argument(
        "--pca-svd-solver",
        choices=PCA_SVD_SOLVERS,
        default="auto",
        help="SVD solver requested from scikit-learn PCA",
    )
    infer.add_argument(
        "--distance-metric",
        choices=DISTANCE_METRICS,
        default="euclidean",
        help="metric for cell-to-centroid and centroid-to-centroid distances",
    )
    infer.add_argument(
        "--kernel",
        choices=KERNEL_NAMES,
        default="gaussian",
        help="kernel that transforms distances into affinities",
    )
    infer.add_argument(
        "--bandwidth",
        type=_bandwidth,
        default="auto",
        help="global positive kernel bandwidth or median-based automatic selection",
    )
    infer.add_argument(
        "--group-size-correction",
        choices=GROUP_SIZE_CORRECTIONS,
        default="cap-to-target",
        help="optional correction that caps external groups to the target-group size",
    )
    infer.add_argument(
        "--tree-method",
        choices=TREE_METHODS,
        default="extra-trees",
        help="weighted tree ensemble used for each target gene",
    )
    infer.add_argument(
        "--n-estimators",
        type=_positive_int,
        default=500,
        help="number of trees in every fitted ensemble",
    )
    infer.add_argument(
        "--max-features",
        type=_max_features,
        default=1.0,
        metavar="{sqrt,log2,INT,FRACTION}",
        help=(
            "predictors considered per split: '1' means exactly one predictor, while "
            "'1.0' means 100%% of predictors"
        ),
    )
    infer.add_argument(
        "--min-samples-leaf",
        type=_positive_int,
        default=1,
        help="minimum number of samples required in each tree leaf",
    )
    infer.add_argument(
        "--max-depth",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help="maximum tree depth; omit for unbounded depth",
    )
    infer.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=(
            "override estimator bootstrap sampling; if omitted, Extra-Trees disables it "
            "and Random Forest enables it"
        ),
    )
    infer.add_argument(
        "--random-seed",
        type=_random_seed,
        default=123,
        help=(
            "global seed in scikit-learn's unsigned 32-bit range, used to derive "
            "deterministic model seeds"
        ),
    )
    infer.add_argument(
        "--threads",
        type=_threads,
        default=-1,
        help="single parallelism budget; -1 uses all available CPUs",
    )
    infer.add_argument(
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
        max_depth=getattr(args, "max_depth", None),
        bootstrap=getattr(args, "bootstrap", None),
        random_seed=args.random_seed,
        threads=args.threads,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SPATHI command line and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.command == "infer":
        try:
            from spathi.pipeline import infer_group_specific_grns

            result = infer_group_specific_grns(config_from_args(args))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            LOGGER.error("%s", exc)
            return 2
        LOGGER.info("Completed SPATHI run: %s", result.output_dir)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
