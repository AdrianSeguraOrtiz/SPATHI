#!/usr/bin/env python3
"""Measure SPATHI wall-clock scaling on a reproducible synthetic dataset.

This benchmark intentionally lives outside the test suite. Timings include CLI startup,
input validation, inference, and artifact writing, so they represent end-to-end runs.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=240, help="number of synthetic cells")
    parser.add_argument("--genes", type=int, default=80, help="number of target genes")
    parser.add_argument("--tfs", type=int, default=12, help="number of candidate TF genes")
    parser.add_argument("--groups", type=int, default=4, help="number of cell groups")
    parser.add_argument("--n-estimators", type=int, default=25, help="trees per target model")
    parser.add_argument("--n-components", type=int, default=20, help="requested PCA components")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, -1],
        help="SPATHI thread budgets to compare (default: 1 2 -1)",
    )
    parser.add_argument("--repeats", type=int, default=2, help="runs per thread budget")
    parser.add_argument("--seed", type=int, default=1729, help="synthetic and SPATHI seed")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="parent directory for temporary inputs and outputs",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="retain generated inputs and run directories for inspection",
    )
    parser.add_argument(
        "--show-spathi-output",
        action="store_true",
        help="stream SPATHI logging instead of capturing it",
    )
    args = parser.parse_args()

    if args.cells < args.groups:
        parser.error("--cells must be at least --groups")
    if args.genes < 2:
        parser.error("--genes must be at least 2")
    if not 1 <= args.tfs < args.genes:
        parser.error("--tfs must be positive and smaller than --genes")
    if args.groups < 2:
        parser.error("--groups must be at least 2")
    if args.n_estimators < 1 or args.n_components < 1 or args.repeats < 1:
        parser.error("--n-estimators, --n-components, and --repeats must be positive")
    if any(value == 0 or value < -1 for value in args.threads):
        parser.error("each --threads value must be -1 or a positive integer")
    return args


def create_dataset(
    destination: Path,
    *,
    n_cells: int,
    n_genes: int,
    n_tfs: int,
    n_groups: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    """Create a non-negative expression matrix with reproducible group structure."""

    rng = np.random.default_rng(seed)
    cell_names = [f"cell_{index:05d}" for index in range(n_cells)]
    group_index = np.arange(n_cells, dtype=np.int64) % n_groups
    rng.shuffle(group_index)
    group_names = [f"population_{index + 1}" for index in range(n_groups)]

    group_tf_means = rng.uniform(0.4, 2.4, size=(n_groups, n_tfs))
    tf_values = group_tf_means[group_index] + rng.normal(0.0, 0.15, size=(n_cells, n_tfs))
    tf_values = np.clip(tf_values, 0.0, None)

    n_non_tfs = n_genes - n_tfs
    coefficients = rng.uniform(0.0, 1.0, size=(n_non_tfs, n_tfs))
    coefficients /= coefficients.sum(axis=1, keepdims=True)
    group_offsets = rng.uniform(0.0, 0.6, size=(n_groups, n_non_tfs))
    target_values = tf_values @ coefficients.T
    target_values += group_offsets[group_index]
    target_values += rng.normal(0.0, 0.08, size=target_values.shape)
    target_values = np.clip(target_values, 0.0, None)

    gene_names = [f"TF_{index:03d}" for index in range(n_tfs)]
    gene_names.extend(f"GENE_{index:05d}" for index in range(n_non_tfs))
    cells_by_genes = np.concatenate((tf_values, target_values), axis=1)

    data_dir = destination / "data"
    data_dir.mkdir(parents=True)
    expression_path = data_dir / "expression.tsv"
    tf_path = data_dir / "tf_list.txt"
    groups_path = data_dir / "groups.tsv"

    pd.DataFrame(cells_by_genes.T, index=gene_names, columns=cell_names).to_csv(
        expression_path,
        sep="\t",
        index_label="gene",
        float_format="%.8g",
    )
    tf_path.write_text("".join(f"{name}\n" for name in gene_names[:n_tfs]), encoding="utf-8")
    pd.DataFrame(
        {
            "cell": cell_names,
            "cluster": [group_names[index] for index in group_index],
        }
    ).to_csv(groups_path, sep="\t", index=False)
    return expression_path, tf_path, groups_path


def run_once(
    *,
    expression: Path,
    tf_list: Path,
    groups: Path,
    output_dir: Path,
    threads: int,
    n_components: int,
    n_estimators: int,
    seed: int,
    show_output: bool,
) -> float:
    """Run one CLI process and return elapsed wall-clock seconds."""

    command = [
        sys.executable,
        "-m",
        "spathi",
        "infer",
        "--expression",
        str(expression),
        "--tf-list",
        str(tf_list),
        "--groups",
        str(groups),
        "--output-dir",
        str(output_dir),
        "--weight-mode",
        "cell-distance-group-anchored",
        "--distance-space",
        "pca",
        "--n-components",
        str(n_components),
        "--tree-method",
        "extra-trees",
        "--n-estimators",
        str(n_estimators),
        "--random-seed",
        str(seed),
        "--threads",
        str(threads),
    ]

    started = time.perf_counter()
    if show_output:
        completed = subprocess.run(command, check=False)
    else:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        if not show_output:
            if completed.stdout:
                print(completed.stdout, file=sys.stderr)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"SPATHI exited with status {completed.returncode}: {' '.join(command)}")
    return elapsed


def main() -> int:
    """Generate data, execute configurations, and emit machine-readable timings."""

    args = parse_args()
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = Path(tempfile.mkdtemp(prefix="spathi-benchmark-", dir=args.work_dir))
    print(f"Benchmark workspace: {benchmark_root}", file=sys.stderr)

    try:
        expression, tf_list, groups = create_dataset(
            benchmark_root,
            n_cells=args.cells,
            n_genes=args.genes,
            n_tfs=args.tfs,
            n_groups=args.groups,
            seed=args.seed,
        )
        rows: list[dict[str, str | int | float]] = []
        for thread_budget in args.threads:
            for repeat in range(1, args.repeats + 1):
                label = "all" if thread_budget == -1 else str(thread_budget)
                output_dir = benchmark_root / f"threads-{label}-repeat-{repeat}"
                elapsed = run_once(
                    expression=expression,
                    tf_list=tf_list,
                    groups=groups,
                    output_dir=output_dir,
                    threads=thread_budget,
                    n_components=args.n_components,
                    n_estimators=args.n_estimators,
                    seed=args.seed,
                    show_output=args.show_spathi_output,
                )
                rows.append(
                    {
                        "threads": thread_budget,
                        "repeat": repeat,
                        "elapsed_seconds": round(elapsed, 6),
                        "cells": args.cells,
                        "genes": args.genes,
                        "groups": args.groups,
                        "n_estimators": args.n_estimators,
                    }
                )
                print(
                    f"threads={thread_budget}, repeat={repeat}: {elapsed:.3f} s",
                    file=sys.stderr,
                )

        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.keep_work_dir:
            print(f"Retained benchmark workspace: {benchmark_root}", file=sys.stderr)
        else:
            shutil.rmtree(benchmark_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
