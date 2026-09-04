import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from spathi.cli import build_parser, config_from_args, main
from spathi.config import SpathiConfig

BASE_ARGUMENTS = [
    "infer",
    "--expression",
    "expression.tsv",
    "--tf-list",
    "tf_list.txt",
    "--groups",
    "groups.tsv",
    "--output-dir",
    "results",
]


def test_cli_builds_typed_configuration() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            *BASE_ARGUMENTS,
            "--bandwidth",
            "2.5",
            "--max-features",
            "sqrt",
            "--threads",
            "2",
            "--bootstrap",
        ]
    )
    config = config_from_args(args)
    assert config.expression == Path("expression.tsv")
    assert config.bandwidth == 2.5
    assert config.max_features == "sqrt"
    assert config.threads == 2
    assert config.bootstrap is True


def test_cli_scientific_defaults_are_the_core_configuration_defaults() -> None:
    configured = config_from_args(build_parser().parse_args(BASE_ARGUMENTS))
    expected = SpathiConfig(
        expression=Path("expression.tsv"),
        tf_list=Path("tf_list.txt"),
        groups=Path("groups.tsv"),
        output_dir=Path("results"),
    )
    assert configured == expected


def test_cli_bootstrap_is_automatic_unless_explicitly_overridden() -> None:
    parser = build_parser()
    assert config_from_args(parser.parse_args(BASE_ARGUMENTS)).bootstrap is None
    assert (
        config_from_args(parser.parse_args([*BASE_ARGUMENTS, "--no-bootstrap"])).bootstrap is False
    )


def test_cli_target_list_is_optional_and_typed_as_a_path() -> None:
    parser = build_parser()
    assert config_from_args(parser.parse_args(BASE_ARGUMENTS)).target_list is None
    configured = config_from_args(
        parser.parse_args([*BASE_ARGUMENTS, "--target-list", "targets.txt"])
    )
    assert configured.target_list == Path("targets.txt")


def test_cli_centroid_weights_are_optional_and_typed_as_a_path() -> None:
    parser = build_parser()
    assert config_from_args(parser.parse_args(BASE_ARGUMENTS)).centroid_weights is None
    configured = config_from_args(
        parser.parse_args([*BASE_ARGUMENTS, "--centroid-weights", "centroid_weights.tsv"])
    )
    assert configured.centroid_weights == Path("centroid_weights.tsv")


def test_cli_report_is_enabled_unless_explicitly_disabled() -> None:
    parser = build_parser()
    assert config_from_args(parser.parse_args(BASE_ARGUMENTS)).report is True
    assert config_from_args(parser.parse_args([*BASE_ARGUMENTS, "--no-report"])).report is False


def test_cli_checkpoint_is_enabled_by_default_and_resume_is_explicit() -> None:
    parser = build_parser()
    defaults = parser.parse_args(BASE_ARGUMENTS)
    assert defaults.checkpoint is True
    assert defaults.resume is False
    disabled = parser.parse_args([*BASE_ARGUMENTS, "--no-checkpoint"])
    assert disabled.checkpoint is False
    resumed = parser.parse_args([*BASE_ARGUMENTS, "--resume"])
    assert resumed.resume is True


def test_cli_progress_is_enabled_by_default_and_can_be_disabled() -> None:
    parser = build_parser()
    assert parser.parse_args(BASE_ARGUMENTS).progress is True
    assert parser.parse_args([*BASE_ARGUMENTS, "--no-progress"]).progress is False


def test_cli_distinguishes_integer_and_fractional_max_features() -> None:
    parser = build_parser()
    integer = config_from_args(
        parser.parse_args([*BASE_ARGUMENTS, "--max-features", "1"])
    ).max_features
    fraction = config_from_args(
        parser.parse_args([*BASE_ARGUMENTS, "--max-features", "1.0"])
    ).max_features
    assert integer == 1 and type(integer) is int
    assert fraction == 1.0 and type(fraction) is float


@pytest.mark.parametrize(
    "arguments",
    [
        [*BASE_ARGUMENTS, "--threads", "0"],
        [*BASE_ARGUMENTS, "--bandwidth", "0"],
        [*BASE_ARGUMENTS, "--bandwidth", "nan"],
        [*BASE_ARGUMENTS, "--bandwidth", "inf"],
        [*BASE_ARGUMENTS, "--max-features", "1.5"],
        [*BASE_ARGUMENTS, "--max-features", "all"],
        [*BASE_ARGUMENTS, "--random-seed", "-1"],
        [*BASE_ARGUMENTS, "--random-seed", str(2**32)],
    ],
)
def test_cli_rejects_invalid_numeric_options(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_cli_help_explains_ambiguous_max_features_and_automatic_bootstrap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["infer", "--help"])
    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "'1' means exactly one predictor" in help_text
    assert "'1.0' means 100% of predictors" in help_text
    assert "Extra-Trees disables it" in help_text
    assert "Random Forest enables it" in help_text
    assert "(default: None)" not in help_text
    assert "\x1b[" not in help_text


def test_cli_infer_delegates_to_the_canonical_core(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def fake_infer(config: SpathiConfig, **options: Any) -> SimpleNamespace:
        received["config"] = config
        received.update(options)
        return SimpleNamespace(output_dir=config.output_dir, n_edges=17)

    core_module = ModuleType("spathi.core")
    core_module.__dict__["infer"] = fake_infer
    monkeypatch.setitem(sys.modules, "spathi.core", core_module)
    assert main([*BASE_ARGUMENTS, "--no-progress"]) == 0
    assert isinstance(received["config"], SpathiConfig)
    assert received["progress_callback"] is None
    assert received["resume"] is False
    assert received["checkpoint"] is True


@pytest.mark.integration
def test_cli_no_progress_suppresses_all_periodic_progress_output(
    tmp_path: Path,
    input_files: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "no-progress"
    arguments = [
        "infer",
        "--expression",
        str(input_files["expression"]),
        "--tf-list",
        str(input_files["tf_list"]),
        "--groups",
        str(input_files["groups"]),
        "--output-dir",
        str(output_dir),
        "--n-estimators",
        "2",
        "--threads",
        "1",
        "--no-checkpoint",
        "--no-report",
        "--no-progress",
    ]

    assert main(arguments) == 0

    error_output = capsys.readouterr().err
    assert "Progress |" not in error_output
    assert "Validating expression" not in error_output
    assert "Building the pca distance representation" not in error_output
    assert "Preparing target group" not in error_output
    assert "Run complete" in error_output


@pytest.mark.parametrize(
    "failure",
    [ValueError("invalid scientific input"), MemoryError("insufficient inference memory")],
)
def test_cli_reports_expected_core_failures_without_a_traceback(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise failure

    core_module = ModuleType("spathi.core")
    core_module.__dict__["infer"] = fail
    monkeypatch.setitem(sys.modules, "spathi.core", core_module)
    assert main([*BASE_ARGUMENTS, "--no-progress"]) == 2
    error_output = capsys.readouterr().err
    assert "Inference failed" in error_output
    assert str(failure) in error_output
    assert "Traceback" not in error_output
