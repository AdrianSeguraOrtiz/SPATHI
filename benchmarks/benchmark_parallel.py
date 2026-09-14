#!/usr/bin/env python3
"""Bounded engineering comparison of SPATHI model threads and shared-memory processes.

This calls PreparedInference's real model preparation and fitting path. It does not
replace the published scientific benchmark. Backends run in separate processes;
their first invocation includes lazy pool startup, subsequent invocations reuse it.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from spathi.inference import _fit_model_task, prepare_inference
from spathi.parallel import PersistentTaskExecutor, available_cpu_count, resolve_thread_budget

CASES = {"smoke": (80, 12, 4, 10), "small": (600, 100, 16, 250), "large": (3000, 500, 4, 250)}
BACKENDS = ("sequential", "threading", "loky", "loky_bounded")


def source_identity(root: Path | None = None) -> dict[str, str]:
    root = root or Path(__file__).resolve().parents[1]
    paths = [
        root / "benchmarks" / name for name in ("benchmark_parallel.py", "benchmark_scaling.py")
    ]
    paths.extend((root / "src/spathi").rglob("*.py"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def require_source_identity(expected: dict[str, str], root: Path | None = None) -> None:
    observed = source_identity(root)
    changed = sorted(
        path
        for path in expected.keys() | observed.keys()
        if expected.get(path) != observed.get(path)
    )
    if changed:
        raise RuntimeError(
            "Benchmark source changed during measurement; timings cannot be "
            f"compared across implementations. Start a fresh run: {changed}"
        )


def prepare_case(name: str, seed: int) -> Any:
    """Generate entirely synthetic data and the real weighted SPATHI model tasks."""
    cells, tfs, targets, trees = CASES[name]
    random = np.random.default_rng(seed)
    latent = random.normal(size=(cells, 4))
    predictors = latent @ random.normal(size=(4, tfs)) + random.normal(size=(cells, tfs))
    responses = predictors[:, :targets] + random.normal(scale=0.7, size=(cells, targets))
    expression = np.log1p(np.maximum(np.column_stack((predictors, responses)) + 4.0, 0.0))
    genes = tuple(f"TF{i:04d}" for i in range(tfs)) + tuple(f"G{i:04d}" for i in range(targets))
    prepared = prepare_inference(
        expression,
        genes,
        genes[:tfs],
        target_names=genes[-targets:],
        tree_method="extra-trees",
        n_estimators=trees,
        max_features=0.5,
        min_samples_leaf=2,
        bootstrap=False,
        random_seed=seed,
    )
    weights = {
        "group_1": np.where(np.arange(cells) < cells // 2, 1.0, 0.25),
        "group_2": np.where(np.arange(cells) >= cells // 2, 1.0, 0.25),
    }
    batch = next(
        prepared._iter_batch_specs(
            weights,
            target_batch_size=targets,
            group_order=tuple(weights),
            completed_models=(),
            threads=1,
            executor=None,
        )
    )
    return prepared._prepare_model_execution(
        batch.groups,
        target_items=batch.targets,
        completed_models=(),
        threads=1,
        executor=None,
    )


def fit_task(task: Any, context: Any) -> tuple[Any, dict[str, Any]]:
    result = _fit_model_task(task, context)
    arrays = (task.group.variable_tf_expression, context.expression)
    return result, {
        "pid": os.getpid(),
        "predictors_memmapped": isinstance(arrays[0], np.memmap),
        "targets_memmapped": isinstance(arrays[1], np.memmap),
        "memmaps_readonly": all(
            not array.flags.writeable for array in arrays if isinstance(array, np.memmap)
        ),
    }


def result_fingerprint(outputs: list[tuple[Any, dict[str, Any]]]) -> str:
    """Require exact scores, seeds and diagnostics, excluding measured fit duration."""
    records = []
    for result, _evidence in outputs:
        record = asdict(result)
        record["stat"].pop("fit_seconds")
        records.append(record)
    records.sort(key=lambda item: (item["stat"]["target_group"], item["stat"]["target"]))
    return hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measured(call: Any) -> tuple[list[Any], dict[str, Any]]:
    from benchmark_scaling import ProcessTreeUsageSampler

    sampler = ProcessTreeUsageSampler()
    parent = psutil.Process()
    peak_rss = sampler.sample(parent)
    cpu_before = sampler.sampled_cpu_user_seconds + sampler.sampled_cpu_system_seconds
    stopped = threading.Event()

    def sample() -> None:
        nonlocal peak_rss
        while not stopped.wait(0.05):
            peak_rss = max(peak_rss, sampler.sample(parent))

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        result = call()
    finally:
        wall = time.perf_counter() - started
        stopped.set()
        monitor.join()
        peak_rss = max(peak_rss, sampler.sample(parent))
    cpu = sampler.sampled_cpu_user_seconds + sampler.sampled_cpu_system_seconds - cpu_before
    return result, {
        "wall_seconds": wall,
        "sampled_cpu_seconds": cpu,
        "sampled_peak_process_tree_rss_bytes": peak_rss,
        "average_cpu_cores": cpu / wall,
    }


def worker(args: argparse.Namespace) -> None:
    source_hashes = source_identity()
    execution = prepare_case(args.case, args.seed)
    workers = min(args.threads, available_cpu_count(), len(execution.tasks))
    records = []

    def repeat(call: Any) -> None:
        for repetition in range(args.repeats):
            require_source_identity(source_hashes)
            outputs, measurement = measured(call)
            require_source_identity(source_hashes)
            if any(not result.trained for result, _ in outputs):
                raise RuntimeError("Synthetic benchmark contains an unfitted model")
            records.append(
                {
                    "repetition": repetition + 1,
                    "pool_state": "first" if repetition == 0 else "reused",
                    **measurement,
                    "scientific_sha256": result_fingerprint(outputs),
                    "status_counts": dict(Counter(result.stat.status for result, _ in outputs)),
                    "fit_seconds_sum": sum(result.stat.fit_seconds for result, _ in outputs),
                    "worker_pids": sorted({item["pid"] for _, item in outputs}),
                    "predictors_memmapped": all(
                        item["predictors_memmapped"] for _, item in outputs
                    ),
                    "targets_memmapped": all(item["targets_memmapped"] for _, item in outputs),
                    "memmaps_readonly": all(item["memmaps_readonly"] for _, item in outputs),
                }
            )
            print(
                json.dumps({"case": args.case, "backend": args.worker, **records[-1]}), flush=True
            )

    with threadpool_limits(limits=1):
        if args.worker == "sequential":
            repeat(lambda: [fit_task(task, execution.context) for task in execution.tasks])
        elif args.worker in {"threading", "loky_bounded"}:
            plan = resolve_thread_budget(
                workers,
                len(execution.tasks),
                task_backend="loky" if args.worker == "loky_bounded" else "threading",
            )
            with PersistentTaskExecutor(plan) as executor:
                repeat(
                    lambda: executor.execute(
                        lambda task: fit_task(task, execution.context), execution.tasks
                    )
                )
        else:
            with tempfile.TemporaryDirectory(prefix="spathi-parallel-memmap-") as directory:
                with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
                    with Parallel(
                        max_nbytes="32K",
                        mmap_mode="r",
                        temp_folder=directory,
                        batch_size=1,
                        pre_dispatch=2 * workers,
                        return_as="generator_unordered",
                    ) as executor:
                        repeat(
                            lambda: list(
                                executor(
                                    delayed(fit_task)(task, execution.context)
                                    for task in execution.tasks
                                )
                            )
                        )
        if args.worker == "sequential" and args.case != "large":
            profiler = cProfile.Profile()
            profiler.runcall(_fit_model_task, execution.tasks[0], execution.context)
            profiler.dump_stats(str(args.output.with_suffix(".prof")))
            with args.output.with_suffix(".profile.txt").open("w") as handle:
                pstats.Stats(profiler, stream=handle).sort_stats("cumulative").print_stats(35)
    payload = {
        "case": args.case,
        "backend": args.worker,
        "seed": args.seed,
        "source_sha256": source_hashes,
        "threads": 1 if args.worker == "sequential" else workers,
        "dimensions": dict(
            zip(("cells", "tfs", "targets", "trees"), CASES[args.case], strict=True)
        ),
        "groups": 2,
        "models": len(execution.tasks),
        "measurements": records,
    }
    require_source_identity(source_hashes)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--threads", type=int, default=min(8, available_cpu_count()))
    parser.add_argument(
        "--repeats", type=int, help="Defaults to two for smoke/small and one for large."
    )
    parser.add_argument("--seed", type=int, default=739391)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300,
        help="Total deadline per case across backends, including profiling.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/parallel"))
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.threads < 1
        or (args.repeats is not None and args.repeats < 1)
        or args.timeout_seconds <= 0
        or not 0 <= args.seed < 2**32
    ):
        parser.error("threads, repeats and timeout must be positive; seed must be a uint32")
    if args.worker:
        args.repeats = args.repeats or (1 if args.case == "large" else 2)
        worker(args)
        return 0
    from benchmark_scaling import measure_command

    args.output.mkdir(parents=True, exist_ok=False)
    frozen_sources = source_identity()
    cases = ("small", "large") if args.case == "all" else (args.case,)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "claim_scope": "engineering_not_scientific_accuracy",
        "measurement_notes": [
            "Each backend has a fresh driver process; repeated calls reuse its pool.",
            "First call includes lazy worker startup and initial memmap publication.",
            "RSS sums parent and children and can double count shared/memmapped pages.",
            "CPU and peak RSS are sampled; short-lived process work may be missed.",
            "Scientific hashes include exact positive edge scores and all diagnostics except fit_seconds.",
            "cProfile identifies Python call costs, not GIL wait or a proof of its cause.",
        ],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": frozen_sources,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scikit-learn", "joblib", "psutil")},
        "available_cpu_count": available_cpu_count(),
        "runs": [],
    }
    for case in cases:
        deadline = time.monotonic() + args.timeout_seconds
        for backend in BACKENDS:
            require_source_identity(frozen_sources)
            output = args.output / f"{case}-{backend}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                "--case",
                case,
                "--threads",
                str(args.threads),
                "--repeats",
                str(args.repeats or (1 if case == "large" else 2)),
                "--seed",
                str(args.seed),
                "--output",
                str(output),
            ]
            measurement = measure_command(
                command,
                sample_interval_seconds=0.05,
                timeout_seconds=max(0.01, deadline - time.monotonic()),
                show_output=True,
                stdout_log_path=output.with_suffix(".stdout.log"),
                stderr_log_path=output.with_suffix(".stderr.log"),
            )
            require_source_identity(frozen_sources)
            run = {"case": case, "backend": backend, "process": asdict(measurement)}
            if measurement.status == "success" and output.is_file():
                run["result"] = json.loads(output.read_text())
                if run["result"]["source_sha256"] != frozen_sources:
                    raise RuntimeError(
                        "Benchmark worker used a different implementation; "
                        "timings cannot be compared. Start a fresh run."
                    )
            report["runs"].append(run)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["equivalence"] = {}
    for case in cases:
        runs = [run for run in report["runs"] if run["case"] == case]
        hashes = {
            item["scientific_sha256"]
            for run in runs
            for item in run.get("result", {}).get("measurements", [])
        }
        successful = len(runs) == len(BACKENDS) and all("result" in run for run in runs)
        report["equivalence"][case] = successful and len(hashes) == 1
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"summary": str(args.output / "summary.json"), "equivalence": report["equivalence"]}
        )
    )
    return 0 if all(report["equivalence"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
