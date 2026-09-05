# Releasing SPATHI

This is the maintainer procedure for SPATHI's first release. No version has been
published. Do not change versions, push a release tag, approve the `pypi` environment,
or publish anything without explicit maintainer authorization.

The distribution, import package, and command are all named `spathi`.

## 1. Configure publication once

Before preparing a release:

1. Make the repository and every URL declared in `pyproject.toml` publicly accessible.
2. Confirm that the normalized `spathi` project name can be created on PyPI and agree
   who owns the project.
3. Enable two-factor authentication for the maintainer's PyPI and GitHub accounts and
   store account recovery material securely.
4. Create a GitHub environment named exactly `pypi`. Protect it with a required
   reviewer so a tag cannot publish without an explicit approval.
5. Configure a PyPI Trusted Publisher for:

   - GitHub owner: `AdrianSeguraOrtiz`
   - repository: `SPATHI`
   - workflow: `publish.yml`
   - environment: `pypi`

For the first upload, use PyPI's pending-publisher flow if the project does not yet
exist. The release workflow authenticates through short-lived OpenID Connect
credentials; do not configure a password or long-lived PyPI token in GitHub.

The trusted identity is the workflow filename as well as the repository and
environment. Renaming `publish.yml` therefore requires updating the publisher
configuration before another release.

## 2. Prepare the versioned commit

Start from the intended release branch and a clean checkout. Confirm that no generated
results, credentials, local environments, or unrelated files are tracked:

```bash
git diff --exit-code
test -z "$(git status --porcelain)"
```

`src/spathi/_version.py` is the single source of truth. Replace the development
version with the intended normalized PEP 440 release, for example:

```python
__version__ = "0.1.0"
```

Move the completed entries in `CHANGELOG.md` from `Unreleased` into a heading with the
same version and release date. Review the README, package metadata, public API, license,
author attribution, dependency bounds, and public links. The README must describe the
actual release and must not retain development-version or installation placeholders.

Commit the version, changelog, and release-facing documentation together. Do not tag
the commit yet.

## 3. Validate the candidate locally

Use Python 3.11, the minimum supported version, in a clean environment:

```bash
release_dev_venv="$(mktemp -d)/venv"
python -m venv "$release_dev_venv"
source "$release_dev_venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
ruff check .
ruff format --check .
mypy src/spathi
pytest -m "not distribution"
```

Run the minimal example with one and multiple threads. Inspect its network, weights,
skipped targets, model diagnostics, metadata, and report rather than checking only the
exit status.

Build local review artifacts outside the repository so stale files in `dist/` cannot
be mistaken for the candidate:

```bash
release_artifacts="$(mktemp -d)"
python -m build --outdir "$release_artifacts"
python -m twine check --strict "$release_artifacts"/*
python -m zipfile --list "$release_artifacts"/*.whl
tar -tzf "$release_artifacts"/*.tar.gz
```

There must be exactly one wheel and one source distribution with the same version.
Confirm that the wheel contains the `spathi` package, `py.typed`, console entry point,
and license metadata. Confirm that the source distribution contains the README,
license, changelog, documentation, examples, tests, benchmark, source, and
`pyproject.toml`, with no secrets, caches, environments, or run outputs.

Test the built wheel from outside the checkout:

```bash
repository_root="$(pwd -P)"
release_venv="$(mktemp -d)/venv"
release_smoke="$(mktemp -d)/output"
python -m venv "$release_venv"
"$release_venv/bin/python" -m pip install --upgrade pip
"$release_venv/bin/python" -m pip install "$release_artifacts"/*.whl
"$release_venv/bin/python" -m pip check
cd /tmp
"$release_venv/bin/spathi" --version
"$release_venv/bin/spathi" --help
"$release_venv/bin/spathi" infer \
  --expression "$repository_root/examples/minimal/expression.tsv" \
  --tf-list "$repository_root/examples/minimal/tf_list.txt" \
  --groups "$repository_root/examples/minimal/groups.tsv" \
  --output-dir "$release_smoke" \
  --n-components 3 \
  --n-estimators 10 \
  --threads 1
```

Return to the checkout with `cd "$repository_root"` and verify that it is still clean.
All required CI jobs must also pass on the exact release commit: supported Python
versions, direct dependency
lower bounds, Linux quality checks, macOS and Windows tests, and distribution smoke.
The locally built files are review aids only; they are not uploaded.

## 4. Create the release tag

Only after the candidate and its CI run pass, create one annotated tag whose name is
the package version prefixed by `v`:

```bash
release_version="0.1.0"
git tag -a "v${release_version}" -m "SPATHI ${release_version}"
git push origin "v${release_version}"
```

Use a signed tag instead when the maintainer has an established signing identity.
Never move or reuse a release tag. A mistake requires a new version.

The tag starts `.github/workflows/publish.yml`. Its unprivileged build job:

1. verifies that the tag exactly matches `_version.py` and rejects development
   versions;
2. builds the wheel and source distribution from the tagged checkout;
3. validates metadata with Twine;
4. installs the wheel into a clean environment;
5. runs an installed CLI inference; and
6. uploads the validated artifacts to the workflow run.

That job has read-only repository access and cannot request an OIDC token.

## 5. Approve Trusted Publishing

Review the tagged commit, build logs, smoke result, artifact names, and version before
approving the protected `pypi` environment. The publish job downloads the exact
artifacts produced by the build job and has only the `id-token: write` permission.
It publishes with metadata verification, SHA-256 reporting, and PEP 740 attestations.

Do not rebuild, modify, or manually upload parallel artifacts. PyPI filenames and
release versions are immutable.

## 6. Verify the public release

After publication:

1. Check the project metadata, README rendering, files, hashes, and attestations on
   PyPI.
2. Install the exact release from PyPI into another clean environment:

   ```bash
   verification_venv="$(mktemp -d)/venv"
   python -m venv "$verification_venv"
   "$verification_venv/bin/python" -m pip install --upgrade pip
   "$verification_venv/bin/python" -m pip install "spathi==<version>"
   "$verification_venv/bin/python" -m pip check
   "$verification_venv/bin/spathi" --version
   ```

3. Repeat the installed minimal inference and inspect its metadata and HTML report.
4. Create the hosted GitHub release from the existing tag. If distribution files are
   attached, use the validated workflow artifacts without rebuilding them.

When development resumes, set the next development version in `_version.py`, create a
new `Unreleased` section in the changelog, and commit those two changes together.

## TestPyPI

Production and TestPyPI must not share credentials or an unreviewed deployment path.
If TestPyPI publication is needed later, configure a separate `testpypi` GitHub
environment and Trusted Publisher, and add a deliberately triggered job that reuses
the same build-and-validation boundary. Do not fall back to a repository token merely
to perform a dry run.

## ANDREA integration boundary

An ANDREA container should pin a published artifact:

```bash
python -m pip install "spathi==<version>"
```

Do not copy SPATHI source into ANDREA or add ANDREA as a SPATHI dependency. Dockerfiles,
wrappers, ToolSpecs, and catalogue entries remain outside the SPATHI distribution.
