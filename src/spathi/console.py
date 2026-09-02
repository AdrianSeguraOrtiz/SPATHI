"""Terminal presentation for the SPATHI command-line interface."""

from __future__ import annotations

import logging
import os
from types import TracebackType
from typing import Literal

from rich.console import Console, Group
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from spathi.progress import ProgressCallback, SpathiProgressEvent

SPATHI_LOGO = r"""
 ____  ____   _  _____ _   _ ___
/ ___||  _ \ / \|_   _| | | |_ _|
\___ \| |_) / _ \ | | | |_| || |
 ___) |  __/ ___ \| | |  _  || |
|____/|_| /_/   \_\_| |_| |_|___|
""".strip("\n")

_PHASE_LABELS = {
    "validating_inputs": "Validating inputs",
    "building_representation": "Building representation",
    "computing_distances": "Computing distances",
    "preparing_inference": "Preparing inference",
    "preparing_group": "Preparing group",
    "model_inference": "Inferring networks",
    "writing_outputs": "Writing outputs",
    "building_report": "Building report",
    "publishing": "Publishing results",
    "complete": "Complete",
}


def create_console() -> Console:
    """Create the stderr console used by the CLI.

    Rich detects terminal capabilities itself, including ``NO_COLOR`` and
    redirected streams. No ANSI control sequences are forced into logs captured
    by workflow engines.
    """

    color_system: Literal["auto"] | None = None if "NO_COLOR" in os.environ else "auto"
    return Console(stderr=True, highlight=False, color_system=color_system)


def print_banner(console: Console) -> None:
    """Render the SPATHI identity on interactive terminals only."""

    if not console.is_terminal:
        return
    logo = Text(SPATHI_LOGO, style="bold bright_cyan", justify="center")
    tagline = Text(
        "SPATHI · Similarity-weighted gene-regulatory network inference",
        style="dim white",
        justify="center",
    )
    console.print(
        Panel.fit(
            Group(logo, tagline),
            border_style="bright_blue",
            padding=(0, 2),
        )
    )


def configure_logging(console: Console, level: str) -> logging.Logger:
    """Configure package logs once for a command-line invocation."""

    logger = logging.getLogger("spathi")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    handler = RichHandler(
        console=console,
        show_path=False,
        omit_repeated_times=False,
        markup=False,
        rich_tracebacks=False,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


class InferenceProgress:
    """Bridge structured core events to a terminal bar or plain progress logs."""

    def __init__(
        self,
        *,
        console: Console,
        logger: logging.Logger,
        enabled: bool,
    ) -> None:
        self._console = console
        self._logger = logger
        self._enabled = enabled
        self._interactive = enabled and console.is_terminal
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._last_phase: str | None = None
        self._last_group: str | None = None
        self._last_model_decile = -1

    @property
    def callback(self) -> ProgressCallback | None:
        """Return a callback for the core, or ``None`` when progress is disabled."""

        return self if self._enabled else None

    def __enter__(self) -> InferenceProgress:
        if self._interactive:
            self._progress = Progress(
                SpinnerColumn(style="bright_cyan", finished_text="[bold green]✓"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None, style="blue", complete_style="bright_cyan"),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
                expand=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("Starting SPATHI", total=None)
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()

    def __call__(self, event: SpathiProgressEvent) -> None:
        if not self._enabled:
            return
        if self._progress is not None and self._task_id is not None:
            total = event.total_models
            completed = event.completed_models if total is not None else 0
            self._progress.update(
                self._task_id,
                description=self._description(event),
                total=total,
                completed=completed,
            )
        if self._should_log(event):
            self._logger.info("Progress | %s", self._plain_status(event))
        self._last_phase = event.phase
        self._last_group = event.current_group

    def _should_log(self, event: SpathiProgressEvent) -> bool:
        phase_changed = event.phase != self._last_phase
        group_changed = (
            event.phase == "preparing_group"
            and event.current_group is not None
            and event.current_group != self._last_group
        )
        if self._interactive:
            return False
        model_milestone = False
        if event.phase == "model_inference" and event.total_models not in {None, 0}:
            decile = (10 * event.completed_models) // event.total_models
            if decile > self._last_model_decile:
                self._last_model_decile = decile
                model_milestone = True
        return phase_changed or group_changed or event.phase == "complete" or model_milestone

    @staticmethod
    def _description(event: SpathiProgressEvent) -> str:
        label = _PHASE_LABELS[event.phase]
        if event.current_group is not None:
            return f"{label}: {escape(event.current_group)}"
        return label

    @staticmethod
    def _plain_status(event: SpathiProgressEvent) -> str:
        fields = [f"phase={event.phase}"]
        if event.total_models is not None:
            fields.append(f"models={event.completed_models}/{event.total_models}")
        if event.total_groups is not None:
            fields.append(f"groups={event.completed_groups}/{event.total_groups}")
        if event.current_group is not None:
            fields.append(f"group={event.current_group}")
        fields.append(event.message)
        return " | ".join(fields)


__all__ = [
    "InferenceProgress",
    "SPATHI_LOGO",
    "configure_logging",
    "create_console",
    "print_banner",
]
