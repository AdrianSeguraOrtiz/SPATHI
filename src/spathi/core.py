"""Public programmatic API for one atomic SPATHI inference run.

This module coordinates input validation, checkpoint lifecycle, and atomic publication.
The scientific workflow lives in :mod:`spathi._workflow`; the command-line
interface is only an adapter around :func:`infer`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import spathi._workflow as workflow
import spathi.checkpoint as checkpoint_module
import spathi.io as io_module
from spathi._publication import (
    path_is_occupied,
    preflight_atomic_publication,
    publish_directory_no_replace,
)
from spathi.config import SpathiConfig
from spathi.progress import (
    ProgressCallback,
    SpathiProgressEvent,
    emit_progress,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpathiRunResult:
    """Compact result returned after all run artifacts have been published."""

    output_dir: Path
    network_path: Path
    metadata_path: Path
    report_path: Path | None
    n_edges: int
    total_models: int
    trained_models: int
    skipped_target_records: int
    resumed_models: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("output_dir", "network_path", "metadata_path"):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be a pathlib.Path")
        if self.report_path is not None and not isinstance(self.report_path, Path):
            raise TypeError("report_path must be a pathlib.Path or None")
        for field_name in (
            "n_edges",
            "total_models",
            "trained_models",
            "skipped_target_records",
            "resumed_models",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("trained_models", "skipped_target_records", "resumed_models"):
            if getattr(self, field_name) > self.total_models:
                raise ValueError(f"{field_name} cannot exceed total_models")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(message, str) and message for message in self.warnings
        ):
            raise TypeError("warnings must be a tuple of non-empty strings")


def _checkpoint_directory(output_dir: Path) -> Path:
    name = output_dir.name or "spathi"
    return output_dir.parent / f".{name}.checkpoint"


def _remove_completed_checkpoint(directory: Path) -> None:
    """Remove a checkpoint only when every entry is an owned regular file."""

    if directory.is_symlink():
        raise RuntimeError(f"checkpoint path became a symbolic link: {directory}")
    if not directory.is_dir():
        raise RuntimeError(f"checkpoint path is not a directory: {directory}")
    entries = list(directory.iterdir())
    unexpected = sorted(
        path.name
        for path in entries
        if path.name not in checkpoint_module.CHECKPOINT_OWNED_FILENAMES
        or path.is_symlink()
        or not path.is_file()
    )
    if unexpected:
        raise RuntimeError(
            "checkpoint directory contains unowned files and was preserved: "
            + ", ".join(unexpected[:5])
        )
    for path in entries:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    # Avoid recursive deletion: if an entry appeared after validation, rmdir
    # fails and preserves it for explicit inspection.
    directory.rmdir()


def _checkpoint_scientific_parameters(config: SpathiConfig) -> dict[str, Any]:
    values = config.to_dict()
    for operational_or_fingerprinted in (
        "expression",
        "tf_list",
        "groups",
        "target_list",
        "centroid_weights",
        "output_dir",
        "threads",
        "report",
    ):
        values.pop(operational_or_fingerprinted, None)
    return values


def _validate_call(
    config: SpathiConfig,
    progress_callback: ProgressCallback | None,
    resume: bool,
    checkpoint: bool,
) -> None:
    if not isinstance(config, SpathiConfig):
        raise TypeError("config must be a SpathiConfig instance")
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    if type(checkpoint) is not bool:
        raise TypeError("checkpoint must be a boolean")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if resume and not checkpoint:
        raise ValueError("resume=True requires checkpoint=True")


def _validate_output_paths(
    final_output_dir: Path,
    checkpoint_dir: Path,
    *,
    checkpoint: bool,
    resume: bool,
) -> None:
    if path_is_occupied(final_output_dir):
        raise FileExistsError(
            f"Output path already exists and will not be overwritten: {final_output_dir}"
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.is_symlink():
        raise RuntimeError(f"checkpoint path may not be a symbolic link: {checkpoint_dir}")
    if not checkpoint:
        if checkpoint_dir.exists():
            raise FileExistsError(
                "checkpoint path already exists and must be resumed or removed before a "
                f"non-checkpointed run: {checkpoint_dir}"
            )
        return
    if resume and not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"no checkpoint is available to resume: {checkpoint_dir}")
    if not resume and checkpoint_dir.exists():
        raise FileExistsError(
            f"checkpoint path already exists; pass resume=True or remove it: {checkpoint_dir}"
        )


def _create_checkpoint_context(
    config: SpathiConfig,
    inputs: workflow.RunInputs,
    checkpoint_dir: Path,
    *,
    resume: bool,
) -> checkpoint_module.ModelCheckpoint:
    created_directory = False
    if not resume:
        checkpoint_dir.mkdir(parents=False, exist_ok=False)
        created_directory = True
    try:
        identity = checkpoint_module.build_checkpoint_identity(
            input_fingerprints=inputs.input_fingerprints,
            scientific_parameters=_checkpoint_scientific_parameters(config),
            target_names=inputs.targets,
            group_names=tuple(sorted(map(str, inputs.groups.unique()))),
            dependency_versions=workflow.scientific_dependency_versions(),
        )
        return checkpoint_module.ModelCheckpoint(checkpoint_dir, identity=identity, resume=resume)
    except BaseException:
        if created_directory and checkpoint_dir.exists():
            try:
                _remove_completed_checkpoint(checkpoint_dir)
            except (OSError, RuntimeError) as cleanup_error:
                LOGGER.warning(
                    "Could not clean newly initialized checkpoint %s: %s",
                    checkpoint_dir,
                    cleanup_error,
                )
        raise


def _remove_checkpoint_or_warn(directory: Path, *, description: str) -> None:
    try:
        _remove_completed_checkpoint(directory)
    except (OSError, RuntimeError) as error:
        LOGGER.warning("Could not %s %s: %s", description, directory, error)


def infer(
    config: SpathiConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
    checkpoint: bool = True,
) -> SpathiRunResult:
    """Infer a run privately, then atomically publish its complete output.

    By default, each completed model is committed to a hidden sibling checkpoint.
    Scientific or I/O failures remove only the private output staging directory;
    passing ``resume=True`` reuses exact committed models after validating inputs,
    parameters, dependencies, and the installed SPATHI implementation. A complete
    run is published atomically and its checkpoint is then removed.
    """

    _validate_call(config, progress_callback, resume, checkpoint)
    final_output_dir = config.output_dir
    checkpoint_dir = _checkpoint_directory(final_output_dir)
    _validate_output_paths(
        final_output_dir,
        checkpoint_dir,
        checkpoint=checkpoint,
        resume=resume,
    )
    preflight_atomic_publication(final_output_dir.parent)

    run_started_at = datetime.now(UTC)
    run_started = perf_counter()
    emit_progress(
        progress_callback,
        SpathiProgressEvent(
            phase="validating_inputs",
            message=(
                "Validating expression, TF-list, groups, and optional target-list and "
                "centroid-weight inputs"
            ),
        ),
    )
    validation_started = perf_counter()
    inputs = workflow.RunInputs.from_input_data(
        io_module.load_inputs(
            config.expression,
            config.tf_list,
            config.groups,
            config.target_list,
            config.centroid_weights,
        )
    )
    input_validation_seconds = perf_counter() - validation_started
    group_count = len(inputs.groups.unique())
    workflow.validate_group_configuration(config, group_count=group_count)

    checkpoint_context: Any = nullcontext(None)
    if checkpoint:
        checkpoint_context = _create_checkpoint_context(
            config,
            inputs,
            checkpoint_dir,
            resume=resume,
        )

    remove_empty_checkpoint = False
    prefix = f".{final_output_dir.name or 'spathi'}.staging-"
    try:
        with checkpoint_context as checkpoint_store:
            try:
                with TemporaryDirectory(prefix=prefix, dir=final_output_dir.parent) as staging_name:
                    staged_output_dir = Path(staging_name)
                    summary = workflow.run_workflow(
                        config,
                        inputs=inputs,
                        input_validation_seconds=input_validation_seconds,
                        run_started_at=run_started_at,
                        run_started=run_started,
                        output_dir=staged_output_dir,
                        checkpoint=checkpoint_store,
                        progress_callback=progress_callback,
                        resume_requested=resume,
                    )
                    if checkpoint_store is not None:
                        checkpoint_store.validate_complete()
                    emit_progress(
                        progress_callback,
                        SpathiProgressEvent(
                            phase="publishing",
                            message="Publishing the complete SPATHI output directory",
                            completed_models=summary.total_models,
                            total_models=summary.total_models,
                            completed_groups=group_count,
                            total_groups=group_count,
                            resumed_models=summary.resumed_models,
                        ),
                    )
                    if path_is_occupied(final_output_dir):
                        raise FileExistsError(
                            "Output path appeared during the run and will not be overwritten: "
                            f"{final_output_dir}"
                        )
                    publish_directory_no_replace(staged_output_dir, final_output_dir)
            except BaseException:
                if checkpoint_store is not None:
                    try:
                        remove_empty_checkpoint = not checkpoint_store.completed_keys
                    except Exception as inspection_error:
                        LOGGER.warning(
                            "Could not inspect failed checkpoint %s: %s",
                            checkpoint_dir,
                            inspection_error,
                        )
                raise
    except BaseException:
        if remove_empty_checkpoint and path_is_occupied(checkpoint_dir):
            _remove_checkpoint_or_warn(checkpoint_dir, description="remove empty checkpoint")
        raise

    if checkpoint and path_is_occupied(checkpoint_dir):
        _remove_checkpoint_or_warn(checkpoint_dir, description="remove completed checkpoint")

    if summary.failed_models:
        raise RuntimeError(
            f"SPATHI run failed because {summary.failed_models} model(s) could not be fitted "
            f"safely; inspect {final_output_dir / 'model_diagnostics.tsv.gz'}"
        )
    result = SpathiRunResult(
        output_dir=final_output_dir,
        network_path=final_output_dir / "network.csv",
        metadata_path=final_output_dir / "run_metadata.json",
        report_path=final_output_dir / "report.html" if config.report else None,
        n_edges=summary.n_edges,
        total_models=summary.total_models,
        trained_models=summary.trained_models,
        skipped_target_records=summary.skipped_target_records,
        resumed_models=summary.resumed_models,
        warnings=summary.warnings,
    )
    try:
        emit_progress(
            progress_callback,
            SpathiProgressEvent(
                phase="complete",
                message="SPATHI run completed and was published",
                completed_models=result.total_models,
                total_models=result.total_models,
                completed_groups=group_count,
                total_groups=group_count,
                resumed_models=result.resumed_models,
            ),
        )
    except Exception as error:
        # Publication cannot safely be rolled back. A notification failure must
        # not turn a valid, non-repeatable output into an apparent failed run.
        LOGGER.warning("Progress callback failed after output publication: %s", error)
    return result


__all__ = ["SpathiRunResult", "infer"]
