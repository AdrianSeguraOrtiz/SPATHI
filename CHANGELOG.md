# Changelog

All notable changes to SPATHI will be documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and released versions will
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Generic `spathi prepare` API and CLI stage for sparse 10x feature-barcode H5
  matrices, strict study-independent annotations, explicit library-size-plus-log1p
  normalization, per-analysis-unit filtering, TF intersection, optional centroid
  weight preservation, complete provenance, and atomic ANDREA-compatible outputs.
- Optional strict per-cell centroid weights for explicit sensitivity analyses, with
  stable weighted centroids, checkpoint identity, cell-aligned raw and normalized
  audit output, per-group effective-sample-size summaries, and report provenance. The
  primary mode remains the uniform arithmetic centroid, and centroid weights are never
  applied directly as estimator sample weights or as group-size corrections.
- Initial standalone Python package, currently identified as `0.1.0.dev0`, with a
  CLI-independent inference core and the canonical
  `from spathi import SpathiConfig, SpathiProgressEvent, infer` API.
- Thin `spathi infer` adapter with a terminal-only SPATHI banner, coloured Rich help
  and logs, an interactive model-progress bar, plain redirected progress records,
  `--no-progress`, and `NO_COLOR` support.
- Strict ANDREA-format contracts for expression, transcription-factor, cell-group,
  and optional target-gene inputs, including exact fingerprints and actionable
  validation errors.
- Separate PCA or expression distance spaces, overflow-safe arithmetic group
  centroids, numerically stable Euclidean or cosine distances, Gaussian or exponential
  kernels, and reproducible global bandwidth selection.
- Cell-distance, group-anchored cell-distance, and group-distance weighting modes,
  with optional group-size correction and complete per-cell and per-group diagnostics.
- Weighted Extra-Trees and Random-Forest inference with deterministic task seeds,
  strict target subsets, constant-predictor filtering, and positive unsigned edge
  scores.
- Calibrated defaults shared by core and CLI: cosine distance, 250 trees, and
  `max_features=sqrt`.
- One process-wide thread budget, non-nested parallelism, a reusable worker pool,
  bounded model-result backpressure, deterministic single-thread preprocessing, and
  memory-aware group/target batching.
- `MemAvailable`- and cgroup-aware memory planning, live-headroom-sized distance
  chunks, exception-safe disk-backed cell-distance storage selected by size or memory
  pressure, streamed outputs, and early release of redundant expression allocations.
- Structured phase/model progress callbacks and exact SQLite checkpoints with input,
  parameter, dependency, implementation, and per-group weight identity validation.
- Private staging followed by atomic no-replace publication, an early target-filesystem
  capability probe, a direct Linux syscall fallback for libc versions without a
  `renameat2` wrapper, deterministic run artifacts, reproducible gzip files, and
  complete metadata for scientific and operational decisions.
- Default-on `report.html`: one self-contained offline Plotly report with target-level
  and aggregate interactive charts, exact full-data summaries, shared deterministic
  group-stratified sampling, cell embeddings, explained-variance output, provenance,
  accessible tab navigation, path-free shareable settings, an identifier-handling
  notice, and explicit memory and timing accounting. `--report`/`--no-report` and
  `report=True`/`report=False` control generation without changing inference, while
  `SpathiRunResult.report_path` exposes the published artifact.
- Unit, numerical, integration, CLI, checkpoint, concurrency, distribution, and
  interruption tests, plus developer documentation, a minimal example, CI, and a
  reproducible scaling benchmark with target subsets, balanced schedules, wall time,
  peak process-tree RSS, persistent logs, and safe child-process cancellation.

No SPATHI version has been published yet.
