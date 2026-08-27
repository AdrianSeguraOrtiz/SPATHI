from pathlib import Path

import pytest

from spathi.cli import build_parser, config_from_args

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


def test_cli_bootstrap_is_automatic_unless_explicitly_overridden() -> None:
    parser = build_parser()
    assert config_from_args(parser.parse_args(BASE_ARGUMENTS)).bootstrap is None
    assert (
        config_from_args(parser.parse_args([*BASE_ARGUMENTS, "--no-bootstrap"])).bootstrap is False
    )


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
