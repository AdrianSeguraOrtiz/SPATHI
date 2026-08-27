# Releasing SPATHI

This is a maintainer checklist for a future release. The MVP has not been published.
Do not upload to TestPyPI or PyPI, create tags, or push commits without explicit
maintainer authorization.

The distribution name and import/command names are all `spathi`. Confirm that the PyPI
project name is available and that release ownership has been agreed before the first
upload.

## 1. Choose and record the version

`src/spathi/_version.py` is the single source of truth. Replace its development version
with the intended PEP 440 release, for example:

```python
__version__ = "0.1.0"
```

Move the relevant entries in `CHANGELOG.md` from `Unreleased` into a heading containing
the same version and the release date. Review the README, compatibility notes, and
package metadata. Then verify the resolved version without importing the checkout by
building the metadata or, in an installed development environment, with:

```bash
python -c "import spathi; print(spathi.__version__)"
spathi --version
```

Commit version and changelog changes together. Tagging and pushing happen only after
the artifacts below pass review.

## 2. Run quality checks

Use Python 3.11 (the minimum supported version) in a clean environment for the static
type check, then rely on the CI matrix for newer supported interpreters:

```bash
python -m venv .venv-release
source .venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/spathi
pytest -m "not distribution"
```

The `distribution`-marked test builds a wheel itself and is intentionally excluded
here because steps 3–5 perform the authoritative build, metadata validation, clean
installation, and CLI smoke inference. Run plain `pytest` only when that redundant
local packaging assertion is useful.

Run the bundled minimal example and at least one multi-thread smoke run. Inspect its
weight diagnostics, skipped targets, metadata, and deterministic network ordering—not
just the exit code.

## 3. Build wheel and source distribution

Ensure `dist/` contains no artifacts from another version before building. Inspect or
archive existing files before removing them. From the repository root:

```bash
python -m build
```

The command must produce one `.tar.gz` source distribution and one `.whl`. Do not edit
either artifact after construction.

## 4. Validate artifacts and contents

```bash
python -m twine check --strict dist/*
python -m zipfile --list dist/*.whl
tar -tzf dist/*.tar.gz
```

Confirm that names and versions agree, metadata renders correctly, the wheel contains
the `spathi` package, and the sdist contains the README, license, changelog,
documentation, examples, tests, and `pyproject.toml`. It must not contain secrets,
local environments, results, caches, or unrelated data.

## 5. Test the wheel in a clean environment

Installation must be tested from the wheel, not from the working tree:

```bash
wheel_venv="$(mktemp -d)/venv"
python -m venv "$wheel_venv"
"$wheel_venv/bin/python" -m pip install --upgrade pip
"$wheel_venv/bin/python" -m pip install dist/*.whl
"$wheel_venv/bin/spathi" --version
"$wheel_venv/bin/spathi" --help
```

Run an inference from outside the repository so an accidental import of `src/spathi`
cannot hide a packaging error. Use absolute paths for the example inputs and an output
directory that does not already exist:

```bash
cd /tmp
"$wheel_venv/bin/spathi" infer \
  --expression /absolute/path/to/SPATHI/examples/minimal/expression.tsv \
  --tf-list /absolute/path/to/SPATHI/examples/minimal/tf_list.txt \
  --groups /absolute/path/to/SPATHI/examples/minimal/groups.tsv \
  --output-dir /tmp/spathi-release-smoke \
  --n-components 3 \
  --n-estimators 10 \
  --threads 1
```

Repeat with more than one thread and compare deterministically sorted network content
within the documented numeric tolerance. Deterministic gzip tables should also match
byte for byte because their gzip timestamp is fixed; model diagnostics contain fit
durations and are not expected to have identical content. Use a new output directory
for every run.

## 6. Future TestPyPI publication

Only after explicit authorization, upload the already validated artifacts. Configure
credentials outside the repository, preferably through a scoped token or trusted
publishing; never write tokens into project files or shell history.

```bash
python -m twine upload --repository testpypi dist/*
```

TestPyPI may not host all dependencies. Install the exact candidate in another clean
environment while obtaining dependencies from PyPI:

```bash
test_venv="$(mktemp -d)/venv"
python -m venv "$test_venv"
"$test_venv/bin/python" -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  spathi==<version>
"$test_venv/bin/spathi" --version
```

Repeat the CLI smoke inference and inspect metadata to confirm that the installed
version is recorded.

## 7. Future PyPI publication

After TestPyPI validation and a final maintainer review, upload the **same** wheel and
sdist to PyPI:

```bash
python -m twine upload dist/*
```

Verify the project page and install `spathi==<version>` from PyPI in a new environment.
Only then create the release tag and hosted release according to repository policy.
Tags and published filenames are immutable; corrections require a new version.

After release, restore a development version in `_version.py`, add a new `Unreleased`
section to the changelog, and commit those changes separately.

## ANDREA integration boundary

A future ANDREA container should pin a published artifact:

```bash
python -m pip install spathi==<version>
```

Do not copy SPATHI source into ANDREA or add ANDREA as a SPATHI dependency. Dockerfiles,
wrappers, ToolSpecs, and catalogue entries are intentionally outside this release
procedure and outside the MVP scope.
