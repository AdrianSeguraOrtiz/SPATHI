from __future__ import annotations

import io
import logging

import pytest
from rich.console import Console
from rich.logging import RichHandler

from spathi.console import SPATHI_LOGO, InferenceProgress, create_console, print_banner
from spathi.progress import SpathiProgressEvent


def test_banner_is_colored_and_identifies_spathi_on_a_terminal() -> None:
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="standard",
        width=80,
    )
    print_banner(console)
    rendered = output.getvalue()
    assert SPATHI_LOGO.splitlines()[0].strip() in rendered
    assert "SPATHI · Similarity-weighted" in rendered
    assert "\x1b[" in rendered


def test_banner_is_silent_when_output_is_redirected() -> None:
    output = io.StringIO()
    print_banner(Console(file=output, force_terminal=False, color_system=None))
    assert output.getvalue() == ""


def test_console_disables_color_when_no_color_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "")
    assert create_console().color_system is None


def test_redirected_progress_uses_periodic_plain_logs_instead_of_a_bar() -> None:
    console_output = io.StringIO()
    log_output = io.StringIO()
    logger = logging.Logger("spathi-test-progress", level=logging.INFO)
    logger.addHandler(logging.StreamHandler(log_output))
    progress = InferenceProgress(
        console=Console(file=console_output, force_terminal=False, color_system=None),
        logger=logger,
        enabled=True,
    )

    with progress:
        assert progress.callback is not None
        progress.callback(
            SpathiProgressEvent(
                phase="validating_inputs",
                message="Validating inputs",
            )
        )
        progress.callback(
            SpathiProgressEvent(
                phase="model_inference",
                message="Fitted first model",
                completed_models=1,
                total_models=20,
                current_group="mature",
            )
        )
        progress.callback(
            SpathiProgressEvent(
                phase="model_inference",
                message="Same progress bucket",
                completed_models=1,
                total_models=20,
                current_group="mature",
            )
        )
        progress.callback(
            SpathiProgressEvent(
                phase="model_inference",
                message="Alternating worker group in the same progress bucket",
                completed_models=1,
                total_models=20,
                current_group="immature",
            )
        )

    rendered_logs = log_output.getvalue()
    assert console_output.getvalue() == ""
    assert "phase=validating_inputs" in rendered_logs
    assert "models=1/20" in rendered_logs
    assert "group=mature" in rendered_logs
    assert "Same progress bucket" not in rendered_logs
    assert "Alternating worker group" not in rendered_logs


def test_progress_callback_is_absent_when_disabled() -> None:
    progress = InferenceProgress(
        console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
        logger=logging.getLogger("spathi-test-disabled-progress"),
        enabled=False,
    )
    assert progress.callback is None


def test_terminal_progress_keeps_core_logs_visible() -> None:
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system=None,
        width=100,
    )
    logger = logging.Logger("spathi-test-interactive-progress", level=logging.INFO)
    logger.addHandler(
        RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=False,
        )
    )
    progress = InferenceProgress(console=console, logger=logger, enabled=True)
    report_event = SpathiProgressEvent(
        phase="building_report",
        message="Building report",
        completed_models=2,
        total_models=2,
    )
    assert progress._description(report_event) == "Building report"

    with progress:
        progress(
            SpathiProgressEvent(
                phase="model_inference",
                message="One model complete",
                completed_models=1,
                total_models=2,
                current_group="mature",
            )
        )
        logger.info("Core inference log")
        progress(report_event)
        progress(
            SpathiProgressEvent(
                phase="complete",
                message="Run complete",
                completed_models=2,
                total_models=2,
            )
        )

    rendered = output.getvalue()
    assert "Core inference log" in rendered
    assert "Complete" in rendered
    assert "2/2" in rendered
