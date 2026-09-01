# Changelog

All notable changes to SPATHI will be documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and released versions will
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial standalone Python package and `spathi infer` command.
- Strict ANDREA-compatible TSV and transcription-factor-list validation.
- PCA and expression distance spaces with reusable group centroids.
- Gaussian and exponential kernels with reproducible automatic bandwidth selection.
- Cell-distance, group-anchored cell-distance, and group-distance weighting modes.
- Optional external-group size correction and per-group weight diagnostics.
- Weighted Extra-Trees and Random-Forest inference with deterministic task seeds.
- Self-contained, deterministically ordered run artifacts and metadata.
- Cell-embedding and explained-variance artifacts for inspecting the fitted PCA
  distance representation or the explicitly auxiliary expression-space display PCA.
- Default-on target weight panels, an exact effective-weight-mass heatmap, and a
  visualization manifest with projection semantics and file hashes; `--no-visualize`
  disables their generation.
- Unit, integration, CLI, parallel reproducibility, and distribution tests.
- Developer documentation, minimal example, scaling benchmark, and CI workflow.

### Changed

- Preserve target responses in `float64` while storing only reusable TF predictors in
  the tree implementation's `float32` working representation.
- Resolve bootstrap sampling automatically by estimator: disabled for Extra-Trees and
  enabled for Random Forest unless explicitly overridden; record requested and
  effective values separately.
- Cap PCA components at the informative `n_cells - 1` limit (with a defined one-cell
  exception), delegate `auto` solver negotiation without private state, and record the
  solver policy and explained-variance diagnostics.
- Bound pairwise-distance working chunks to 64 MiB and avoid computing
  cell-to-centroid distances entirely in `group-distance` mode.
- Treat `group_affinities.tsv` as a centroid-level diagnostic and identify
  `cell_weights.tsv.gz` and `weight_diagnostics.tsv` as the authoritative effective
  model weights and group contributions.
- Exclude TF predictors that are constant among positive-weight cells while preserving
  predictor-name mappings and reporting used, discarded, and constant predictors.
- Process target genes in bounded sub-batches in addition to group batches.
- Parse large expression TSVs into one exact numeric allocation without retaining a
  table-sized string copy, and hash the exact bytes consumed during validation.
- Stream group-distance, group-affinity, and weight-diagnostic rows instead of
  accumulating redundant quadratic Python record collections.
- Use column-contiguous cell-distance storage in memory and on disk, and account
  separately for heap, mapped, scratch, edge-batch, and conservative tree memory.
- Publish completed run directories from private staging so early failures leave the
  requested path immediately reusable.
- Keep wheel construction out of the Python-version test matrix; the dedicated
  distribution CI job remains the authoritative build, clean-install, and smoke test.

### Fixed

- Reject undefined cosine distances for zero-norm cells or centroids and canonicalize
  round-off residues near exact zero before automatic bandwidth selection.
- Make gzip containers reproducible by fixing their header timestamp, enabling
  byte-identical compressed output for deterministic tables.
- Correct the README's minimal example output path so it does not collide with an
  existing example directory.
- Reject random seeds outside scikit-learn's supported unsigned 32-bit interval and
  convert oversized numeric bandwidths into actionable configuration errors.
- Accept universal LF, CRLF, and CR newlines while preserving exact input hashes and
  avoid imposing extra reserved cluster labels beyond the ANDREA contract.

No version represented here has been published to PyPI yet.
