# SPATHI

**Similarity-weighted Population-Aware Transcriptional Heterogeneity Inference**

SPATHI infers a gene-regulatory network for each cell group in a preprocessed
single-cell RNA-seq expression matrix. Instead of fitting each group in isolation, it
fits every target model with all cells and changes each observation's contribution
according to its transcriptomic proximity to the group of interest.

> [!IMPORTANT]
> SPATHI is an early-stage scientific tool. It produces hypotheses based on predictive
> feature importance. An inferred edge is not evidence of causality, and SPATHI does
> not infer whether regulation is activating or repressing.

The project is independent of ANDREA. Its input contracts and primary network
output follow the same strict formats so an ANDREA container can install a pinned
SPATHI release without coupling the two codebases.

## Why population-aware inference?

Inferring a separate network from each cluster discards observations from related cell
populations and can make estimates unstable for small groups. SPATHI's working
hypothesis is that transcriptomically close groups may share part of their regulatory
programs. Nearby cells can therefore provide context while receiving less influence
than cells central to the target population.

No temporal order, lineage, hierarchy, or pseudotime is assumed. Similarity is used to
weight observations, not to claim a biological direction between groups.

For each target group, SPATHI:

1. builds a cell representation for distance calculations;
2. computes one arithmetic or explicitly weighted centroid per group and reusable distances;
3. converts distances to base weights with a kernel;
4. optionally corrects external groups for multiplicity;
5. trains one weighted tree ensemble per target gene on the supplied expression;
6. writes positive transcription-factor importances as directed candidate edges; and
7. by default, writes one self-contained interactive report for exploring assigned
   weights and the effective contribution of each source group.

The PCA or expression representation in steps 1–3 is used **only to define weights**.
Models always use the expression values supplied by the user.

## Installation

SPATHI requires Python 3.11 or newer and supports Linux, macOS, and Windows. Before
loading a dataset, it verifies that the target filesystem provides the atomic
no-replace directory rename required by its never-overwrite output contract.

For development from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first release remains under development as `0.1.0.dev0` and has not been published
to PyPI. Once that release is published, installation will be:

```bash
python -m pip install spathi==<version>
```

The wheel installs the `spathi` library and command. Example datasets and benchmark
utilities belong to the repository and source distribution, not to the installed
wheel; commands below that reference `examples/` or `benchmarks/` therefore assume a
source checkout or an unpacked source distribution.

## Preparing a 10x matrix

`spathi prepare` is an explicit stage for raw 10x Genomics feature-barcode H5
matrices. It is deliberately separate from `spathi infer`: preparation makes
normalization and filtering choices visible and produces the same strict TSV and text
inputs accepted by SPATHI and ANDREA. `infer` does not read H5 files or silently alter
raw counts.

Preparation requires a study-independent annotation TSV with exactly these columns:

```text
cell    analysis_unit    cluster
cell_1  unit_alpha       state_early
cell_2  unit_alpha       state_late
```

The fields shown aligned above are tabs. No additional annotation columns are
accepted. Optional centroid sensitivity values use a separate strict TSV supplied
with `--centroid-weights`:

```text
cell    centroid_weight
cell_1  0.92
cell_2  0.81
```

This file must cover every annotation cell exactly once with a positive finite value.
Preparation aligns it by cell identifier and emits one `centroid_weights.tsv` per
eligible analysis unit. No dataset-specific label or score name is built into SPATHI.

Only cells listed in the annotation table are prepared. This makes exclusion of
unclassified cells explicit: an annotation cell absent from the H5 is an error, while
an H5 barcode absent from the annotations is counted as excluded in the manifest.
Every annotated cell belongs to exactly one analysis unit and one cluster.

For example:

```bash
spathi prepare \
  --tenx-h5 data/filtered_feature_bc_matrix.h5 \
  --annotations data/annotations.tsv \
  --tf-list data/tf_list.txt \
  --output-dir prepared/sample-01
```

For an explicitly weighted-centroid sensitivity run, add
`--centroid-weights data/centroid_weights.tsv`.

The default transformation is library-size normalization to 10,000 counts followed
by `log1p`. Library sizes use every `Gene Expression` feature before per-unit gene
filtering. Units with fewer than 300 annotated cells are recorded but not emitted;
override this study threshold explicitly with `--min-cells`. Genes detected in fewer
than `--min-gene-cells` cells of a unit are omitted, and the supplied TF list is
intersected with the retained genes while preserving its order.

Repeated feature names are common in current human 10x references. With the default
`--gene-identifier name --duplicate-gene-policy sum`, rows carrying the same symbol
are collapsed by sparse summation before normalization and the decision is recorded.
Use `--duplicate-gene-policy error` when aggregation is not appropriate.

Outputs are written under `analysis_units/`, one directory per eligible unit:

```text
prepared/sample-01/
├── prepare_manifest.json
└── analysis_units/
    └── unit-001-unit-alpha/
        ├── expression.tsv
        ├── groups.tsv
        ├── tf_list.txt
        └── centroid_weights.tsv  # only with --centroid-weights
```

The H5 matrix remains sparse throughout processing. Writing the required dense TSV
materializes only one gene row of one analysis unit at a time, never the complete
input matrix. See [the preparation contract](docs/preparation.md) for formulas,
validation rules, output provenance, and all options.

## Input contracts

Inputs are intentionally strict: no CSV input, automatic transposition, or H5AD/AnnData
input is supported. An optional plain-text target list can restrict inference without
changing the genes used to construct the distance representation.

### Expression matrix

`--expression` must point to a tab-separated file with a header, unique genes in the
first column, unique cells in the remaining column names, and finite numeric expression
values. Genes are rows and cells are columns:

```text
gene    cell_1    cell_2    cell_3
G1      0.12      0.20      0.08
G2      1.30      0.95      1.12
```

The fields shown aligned above are separated by tab characters in the actual file.
First-column names reserved by ANDREA for cell-oriented matrices are rejected, as are
joint expression/group identifiers that reveal an inverse cell-by-gene orientation.

The matrix must already be normalized and transformed appropriately for inference.
SPATHI does not silently perform a complete scRNA-seq normalization workflow. Every
gene is used as a target unless `--target-list` is supplied, and every expression gene
continues to participate in the configured distance representation either way.

### Transcription factors

`--tf-list` is plain text with exactly one expression-matrix gene identifier per line:

```text
SOX2
MYC
TP53
```

Blank lines, duplicates, missing genes, and an empty list are errors. When a target is
also a transcription factor, that TF is removed from its own predictor set so no
self-edge is emitted. For each target-group model, any other TF that is constant among
the cells with positive weight is also excluded before fitting. The exact exclusions
and number of predictors actually used are recorded in `model_diagnostics.tsv.gz`.

### Target genes

`--target-list` is optional and uses the same strict one-identifier-per-line format as
`--tf-list`. Every listed target must occur exactly once in the expression matrix;
blank lines, duplicate identifiers, missing genes, and an empty list are errors. When
the option is omitted, every expression gene is inferred as a target.

Restricting targets changes only which response models and corresponding network rows are
produced. PCA or expression-space distances still use the complete expression matrix,
and candidate predictors still come from the complete `--tf-list`. The exact target
file is fingerprinted in `run_metadata.json`.

### Optional automatic target eligibility

The primary, exact-scope policy is `--target-eligibility all`, which attempts every
requested target and is the default. For very large all-gene analyses,
`--target-eligibility automatic` can avoid fitting responses that have too little
observable support. A target must then:

- be non-zero in at least `--min-target-detected-cells` cells and the fraction set by
  `--min-target-detected-fraction` over the complete input;
- vary over the complete input; and
- for each target-group model, carry at least
  `--min-target-weighted-detected-fraction` of that model's total weight and reach
  `--min-target-weighted-detected-ess` Kish effective detected-cell sample size.

The delivered thresholds are 20 cells, 0.01 unweighted fraction, 0.01 weighted
fraction, and ESS 10. Detection means a value greater than zero, and automatic mode
therefore requires non-negative, zero-preserving target expression such as the output
of `spathi prepare`; centered or otherwise signed inputs must use `all`. Global
decisions are written to `target_eligibility.tsv.gz`; contextual values
and every skipped model are written to `model_diagnostics.tsv.gz` and
`skipped_targets.tsv`.

This policy changes only whether a gene is fitted as a response. It never removes an
eligible TF from another target's predictor set and never changes the genes used for
PCA or expression-space distances. Because it intentionally changes the inferred
target universe, it remains opt-in until its accuracy and coverage gates have been
validated on an independent benchmark.

### Groups

`--groups` is a TSV with a header. Its first column contains cell identifiers and it
must contain a column named exactly `cluster`:

```text
column    cluster
cell_1    state_A
cell_2    state_B
cell_3    state_A
```

The fields shown aligned above are separated by tab characters in the actual file.

Every expression cell must occur exactly once, and no extra cell is allowed. Cell IDs,
cluster values, and group membership must be non-empty and valid.

### Centroid weights

`--centroid-weights` is an optional UTF-8 TSV with exactly these two columns in this
order:

```text
cell    centroid_weight
cell_1  0.92
cell_2  0.81
cell_3  0.76
```

Every expression cell must occur exactly once, no extra cell is allowed, and every
weight must be numeric, finite, and strictly positive. Rows are aligned to expression
columns by cell identifier. The exact file is fingerprinted for provenance and
checkpoint identity.

Omitting the file is the primary analysis: every group centroid is its arithmetic
mean, equivalent to uniform centroid weights. Supplying the file changes only the
location of each group centroid to its within-group weighted mean. SPATHI never
multiplies these values directly into the tree ensemble's sample weights and does not
use them to redefine observed group sizes. Explicit centroid weighting is therefore a
sensitivity analysis, not a replacement for the uniform primary result. The generic
field intentionally carries no dataset-specific score interpretation.

See the
[`examples/minimal`](https://github.com/AdrianSeguraOrtiz/SPATHI/tree/main/examples/minimal)
directory for a complete small dataset.

## Weighting modes

The `--weight-mode` option provides exactly three definitions:

- `cell-distance`: every base weight is the kernel of that cell's distance to the
  target centroid. Group labels do not affect the base weight; after optional size
  correction, final weights are rescaled to a maximum of one.
- `cell-distance-group-anchored`: target-group cells have base and final weight one;
  external cells receive individual distance-derived weights.
- `group-distance`: target-group cells have weight one; all cells in the same external
  group share the kernel affinity between that group's centroid and the target
  centroid.

The conservative default is `cell-distance-group-anchored`, which fixes the target
population as the unit-weight core while keeping external-cell weights individual.

The provisional development baseline uses cosine distance, 250 trees per ensemble,
and `sqrt` predictors considered at each split. These values are encoded once in
`SpathiConfig` and shared by both the Python API and CLI. They remain provisional until
the full calibration study selects a first-release configuration; they are not a claim
of optimality for every biological dataset.

SPATHI stores distance, base weight, group-size factor, and final model weight
separately. With `--group-size-correction cap-to-target`, an external cell from group
`h` receives the factor `min(1, n_target / n_h)`. The target group is never corrected.
The correction changes population contribution, not biological distance. Use
`--group-size-correction none` to disable it.

The default Gaussian kernel is available alongside an exponential kernel. `--bandwidth`
accepts either a final positive numeric value or `auto`. Automatic selection multiplies
the median of the relevant strictly positive distances by the positive
`--bandwidth-scale` (default `1.0`); the same scale applies to the documented fallback.
An explicit numeric bandwidth is already final and therefore requires a scale of `1.0`.
The reference value, scale, effective value, and any fallback are recorded. See
[`docs/methodology.md`](https://github.com/AdrianSeguraOrtiz/SPATHI/blob/main/docs/methodology.md)
for definitions and equations.

Cosine distance requires every vector actually used by the selected mode to have a
non-zero norm: group centroids in all modes, and cell representations in the two
cell-distance modes. SPATHI rejects required zero vectors with their identifiers rather
than assigning an arbitrary similarity. Floating-point residues indistinguishable from
zero under the documented dot-product error bound are canonicalized before bandwidth
selection and weighting.

A single observed group has no population-to-population borrowing to estimate. SPATHI
therefore requires `cell-distance` with `--group-size-correction none`, preserving
individual cell-to-centroid variation without applying a meaningless multiplicity
correction. A centered distance space (PCA or standardized expression) additionally
requires Euclidean distance under this single-group contract. The primary uniform
centroid would be exactly zero there; supplying explicit centroid weights does not
silently relax the declared single-group geometry. These contracts are validated
before scientific preprocessing, and incompatible requested options are rejected.

For PCA, the centered-data informative rank is capped at
`min(n_genes, max(0, n_cells - 1))`; the one-cell case retains one structural
component with diagnosed zero explained variance. Requested and effective component
counts, the informative rank bound, the SVD policy, and explained-variance ratios are
recorded in `run_metadata.json`. With `auto`, concrete solver negotiation remains
delegated to the recorded scikit-learn version rather than copied from private state.
The solver policy is an operational numerical choice that stays fixed during scientific
calibration, not a hyperparameter to optimize.

## Command-line usage

The CLI is a presentation adapter over the inference core: `spathi infer` validates
command-line values, constructs `SpathiConfig`, and calls `spathi.core.infer`. It does
not contain a second inference implementation. On an interactive terminal it displays
a coloured SPATHI banner, structured Rich logs, and a progress bar that coexists with
log messages. Redirected output stays plain and emits periodic progress records;
`NO_COLOR` is respected.

From a source checkout or unpacked source distribution, run the bundled example with
the provisional baseline:

```bash
spathi infer \
  --expression examples/minimal/expression.tsv \
  --tf-list examples/minimal/tf_list.txt \
  --groups examples/minimal/groups.tsv \
  --output-dir results/minimal-example
```

For a quick development run with an explicitly smaller ensemble:

```bash
spathi infer \
  --expression examples/minimal/expression.tsv \
  --tf-list examples/minimal/tf_list.txt \
  --groups examples/minimal/groups.tsv \
  --output-dir results/minimal-development \
  --weight-mode group-distance \
  --distance-space pca \
  --n-components 3 \
  --distance-metric euclidean \
  --kernel gaussian \
  --bandwidth auto \
  --bandwidth-scale 1.0 \
  --group-size-correction cap-to-target \
  --tree-method extra-trees \
  --n-estimators 25 \
  --random-seed 123 \
  --threads 1
```

The interactive HTML report is generated by default. To run the same inference without
it, add `--no-report`:

```bash
spathi infer \
  --expression examples/minimal/expression.tsv \
  --tf-list examples/minimal/tf_list.txt \
  --groups examples/minimal/groups.tsv \
  --output-dir results/minimal-without-report \
  --no-report
```

Use `--no-report` for calibration loops or runtime benchmarks so report preparation and
rendering are not included in the measured cost. `--report` enables the report
explicitly and is the default. Either choice leaves numeric networks and weights
unchanged.

Progress reporting is enabled by default. Use `--no-progress` to suppress both the
interactive bar and redirected periodic progress messages, and `--log-level` to select
`DEBUG`, `INFO`, `WARNING`, or `ERROR` logging. These options affect presentation only;
they do not alter scientific results.

Inspect all options and their effective defaults with:

```bash
spathi infer --help
```

For a targeted production panel, add a strict target file without reducing the genes
used for PCA or expression-space distances:

```bash
spathi infer \
  --expression expression.tsv \
  --tf-list tf_list.txt \
  --target-list target_list.txt \
  --groups groups.tsv \
  --output-dir results/targeted-run
```

To run an explicit centroid-weight sensitivity analysis, repeat the otherwise
identical primary command with a different output directory and add the optional
input:

```bash
spathi infer \
  --expression expression.tsv \
  --tf-list tf_list.txt \
  --groups groups.tsv \
  --centroid-weights centroid_weights.tsv \
  --output-dir results/weighted-centroid-sensitivity
```

The output directory must not already exist; this prevents accidental replacement of a
previous run.

### Checkpoint and resume

Model checkpoints are enabled by default. SPATHI commits each completed
`(target group, target gene)` model to a hidden sibling directory such as
`.targeted-run.checkpoint/`. If the process is interrupted after at least one model has
committed, repeat the same command with `--resume`; SPATHI revalidates the exact input
fingerprints, scientific parameters, implementation, dependency versions, and
recalculated group weights before reusing any model. A failed attempt with no committed
model removes its empty checkpoint automatically, so the ordinary command remains
immediately retryable. Model payloads use a checksummed, compressed columnar codec, and
resume loads completed target identities only for the active bounded group batch rather
than retaining the full groups-by-targets key set in memory.

```bash
spathi infer \
  --expression expression.tsv \
  --tf-list tf_list.txt \
  --target-list target_list.txt \
  --groups groups.tsv \
  --output-dir results/targeted-run \
  --resume
```

The final output remains atomic and never contains the checkpoint database. A
successful publication removes the checkpoint. Use `--no-checkpoint` for short-lived
calibration jobs whose external orchestrator already provides durable retry semantics.

## Python API

The canonical public API calls the same CLI-independent core directly:

```python
from pathlib import Path

from spathi import SpathiConfig, SpathiProgressEvent, infer

config = SpathiConfig(
    expression=Path("examples/minimal/expression.tsv"),
    tf_list=Path("examples/minimal/tf_list.txt"),
    groups=Path("examples/minimal/groups.tsv"),
    output_dir=Path("results/minimal-api"),
    weight_mode="group-distance",
    n_components=3,
    n_estimators=25,
    threads=1,
    report=True,
)


def show_progress(event: SpathiProgressEvent) -> None:
    if event.total_models:
        print(f"{event.phase}: {event.completed_models}/{event.total_models}")


result = infer(config, progress_callback=show_progress)
if result.report_path is not None:
    print(result.report_path)
```

`SpathiConfig` is immutable and validates scalar configuration before the core
reads data. Content-level validation remains part of input loading. Set `report=False`
to omit `report.html` when using the Python API. `SpathiRunResult.report_path` is the
published report path when enabled and `None` otherwise. Progress callbacks run
synchronously on the orchestration thread, never in fitting workers. A model event is
delivered after its transaction commits when checkpointing is enabled, or immediately
after fitting when it is disabled. Callback exceptions before publication abort the
attempt; a checkpointed attempt can then be resumed. An exception from the final
notification is logged because the complete output has already been published
atomically.

## Outputs

Each run directory is self-contained. Its primary artifact is `network.csv` with these
columns in this exact order:

| Column | Meaning |
|---|---|
| `source` | transcription-factor predictor |
| `target` | modeled expression gene |
| `score` | non-negative weighted feature importance |
| `sign` | `?` because the estimator does not infer regulatory direction |
| `evidence` | `weighted_extra_trees_feature_importance` or `weighted_random_forest_feature_importance` |
| `context` | `group:<group_id>` |

Only scores strictly greater than zero are retained. Scores keep the estimator's
magnitude and are relative within a target model; they are not normalized across
targets or groups. Rows are deterministically ordered by context, target, and source.

Additional artifacts are:

| File | Purpose |
|---|---|
| `cell_weights.tsv.gz` | authoritative long-form distances, base weights, size factors, and final model weights |
| `centroid_weights.tsv.gz` | cell-aligned raw and within-group-normalized centroid weights; ones are written in primary uniform mode |
| `centroid_weight_diagnostics.tsv` | exact per-group raw-weight sum, minimum, median, maximum, cell count, and effective sample size |
| `group_distances.tsv` | pairwise centroid distances |
| `group_affinities.tsv` | group-level centroid affinities and per-cell size-correction factors |
| `centroids.tsv` | long-form `(group, dimension, centroid)` values for each reusable arithmetic or explicitly weighted centroid |
| `weight_diagnostics.tsv` | authoritative effective weight mass, ranges, sample size, and source-group contributions |
| `target_eligibility.tsv.gz` | global per-target automatic-eligibility decisions and their measured support |
| `skipped_targets.tsv` | constant, automatically ineligible, or otherwise non-trainable target models and reasons |
| `model_diagnostics.tsv.gz` | per-model seeds, predictor exclusions, target support, actual tree count, convergence, fit status, and timing |
| `cell_embedding.tsv.gz` | cells, groups, and report coordinates: retained PCs for PCA-distance runs, or auxiliary PCs for expression-distance runs when reporting is enabled |
| `pca_explained_variance.tsv` | per-PC and cumulative ratios for the fitted or auxiliary PCA; absent only for expression-distance runs with `--no-report` |
| `report.html` | single self-contained interactive report with the Plotly runtime and run data embedded; omitted with `--no-report` |
| `parameters.json` | requested configuration |
| `run_metadata.json` | effective settings, versions, timing, resources, and warnings |

`group_affinities.tsv` is a centroid-level diagnostic. Its `base_affinity` is an
actual per-cell base model weight only in `group-distance` mode; it must not be read as
the effective contribution of a group in either cell-distance mode. The exact weights
passed to the models are `cell_weights.tsv.gz:final_weight`, and their exact aggregate
contributions are in `weight_diagnostics.tsv`.

Likewise, `centroid_weights.tsv.gz:centroid_weight` is an input to centroid
construction only. Its `normalized_centroid_weight` sums to one within each observed
group and makes the fitted weighted mean auditable. Neither column is the estimator's
sample weight; that remains `cell_weights.tsv.gz:final_weight`.

Columns ending in `_json` contain compact JSON arrays. In particular, predictor names
and warning messages remain unambiguous even when identifiers contain punctuation such
as semicolons or commas.

`report.html` works offline: Plotly, styles, report data, and interaction logic are all
embedded in that one file. Its **Target explorer** provides a group selector, warnings,
summary cards, a PC1/PC2 cell cloud, distance-to-final-weight points, exact source-group
mass bars, and exact full-data weight summaries. Marker fill uses one fixed 0–1
monochromatic blue scale for `final_weight`; shape identifies the observed source group,
cell markers have no outline, and a star identifies the selected target centroid. Group
envelopes are intentionally omitted because a two-dimensional PCA projection does not
establish biological boundaries.
The cloud is not itself a plot of the configured metric: in particular, visual
Euclidean separation in PC1/PC2 must not be read as cosine distance. Exact configured
distances are shown in the adjacent distance-to-final-weight chart and in the tabular
artifacts.

The **Overview** contains the shared cell cloud and centroids, observed group sizes,
PCA explained variance, an exact target-by-source mass heatmap, and target-group mass
alongside effective sample size. **Method & provenance** records projection semantics,
interpretation limits, a run summary, and requested parameters. `run_metadata.json`
also records the report path, SHA-256 digest, byte size, cell and group counts, and
sampling contract.

The standalone report contains the input identifiers of sampled cells, group labels,
and derived coordinates, distances, weights, and summaries. It deliberately omits local
input/output paths and the expression matrix. Treat the HTML according to the applicable
data-governance rules before sharing it, especially when cell identifiers encode sample
or subject information. The complete local paths and input fingerprints remain available
in `run_metadata.json` rather than in the shareable report.

The cell-level plots share one deterministic, group-stratified sample across every
target group. The default budget aims to display at most 30,000 cells and about 300,000
sampled target-cell values while preserving at least one cell from every observed group.
All metric cards, source-group mass values, effective sample sizes, and weight summary
statistics are nevertheless computed from every cell, not from the display sample.
The Overview plots are created only when that tab is first opened. If WebGL is
unavailable, the browser further limits SVG scatter rendering to approximately 5,000
group-stratified points while retaining the bounded report sample for other data.
Scatter legends state the plotted and total cell counts for each group so rare-group
preservation cannot be mistaken for the observed population proportion.

For PCA-distance runs, PC1/PC2 are the first two components of the fitted distance
representation; the second coordinate is zero when only one component exists. Weights
still use every retained component, so separation or overlap in two dimensions can omit
relevant distance. For expression-distance runs, SPATHI fits a deterministic auxiliary
two-component PCA solely for the report. Inference distances, weights, and models remain
in the configured expression space.

Run metadata also stores the resolved path, byte size, and SHA-256 digest of every
input so the exact source bytes can be verified later. The expression parser validates
shape and identifiers in a first pass, fills one exact `float64` array row by row in a
second pass, and requires both passes to have the same digest; it never retains a
second table-sized matrix of expression strings.

Gzip tables are written with a fixed header timestamp. Tables whose rows contain only
deterministic values, such as `cell_weights.tsv.gz`, are therefore reproducible byte
for byte for the same inputs and effective settings. `model_diagnostics.tsv.gz`
contains measured fit durations, so its content is intentionally not byte-stable even
though its gzip container has no wall-clock timestamp.

## Models, parallelism, and reproducibility

SPATHI uses scikit-learn's `ExtraTreesRegressor` (default) or
`RandomForestRegressor`, passing the target group's final weight vector through
`sample_weight`. The resulting weighted impurity-based `feature_importances_` values
are the edge scores. Before each fit, SPATHI excludes the target itself and any
remaining TF predictor that is constant among positive-weight cells. Predictor names,
importances, and diagnostic counts retain the resulting filtered-column mapping.

Target responses remain `float64`, preserving valid expression differences smaller
than one `float32` unit. The reusable TF predictor matrix is copied directly into its
final C-contiguous `float32` allocation, without first materializing the complete
selection as a second `float64` matrix. An explicit target subset is likewise copied
directly into one final Fortran-contiguous response matrix. SPATHI preflights both
persistent allocations against live memory and bounds their finiteness masks to
64 MiB. Distance, weight, and `sample_weight` calculations remain `float64`.

When neither `--bootstrap` nor `--no-bootstrap` is supplied, SPATHI uses the
estimator-appropriate default: bootstrap sampling is disabled for Extra-Trees and
enabled for Random Forest. An explicit CLI flag overrides that choice.
`parameters.json` retains the requested value (`null` means automatic), while
`run_metadata.json` records the effective boolean used for training.

`--adaptive-trees` is an opt-in compute policy. It grows the same seeded ensemble in
deterministic blocks, up to the strict `--n-estimators` ceiling, and stops a model only
after its normalized TF-importance vector remains within the configured total-variation
`--adaptive-tolerance` over `--adaptive-patience` prior checkpoints. The first possible
stop is constrained by `--adaptive-min-estimators`; `--adaptive-tree-step` controls the
block size. Delivered controls are 100, 50, 0.01, and 2 respectively. The actual tree
count and convergence diagnostics are persisted per model and summarized in metadata.
Importance stability is a computational stopping rule, not proof that the result equals
the maximum-tree ensemble, so this option is disabled by default pending formal
non-inferiority validation.

`--threads` is the only public parallelism budget: `auto` uses all logical CPUs, `1` is
sequential, and a positive integer caps available workers. Independent
`(target group, target gene)` tasks use one persistent, bounded rolling thread queue.
At most two tasks per worker are pending, completed work is replenished immediately,
and results return to canonical order before serialization. Ensembles run with one
worker when outer parallelism is active, while threadpoolctl limits numerical-library
thread pools to avoid nested oversubscription. Per-task seeds depend on the global seed,
group ID, and target ID—not scheduler completion order.

Before fitting, SPATHI estimates the conservative memory cost of one ensemble and
detects available host or cgroup memory when the platform exposes it. On Linux it uses
`MemAvailable`, then applies the tightest visible cgroup headroom. The outer model
concurrency is capped to retain space for shared arrays; the estimate, detected
headroom, and selected cap are recorded in `run_metadata.json`. This is a planning
heuristic rather than a replacement for monitoring exceptionally large runs.

PCA, centroid distances, and the optional report-only auxiliary PCA use one numerical
thread deliberately. Different BLAS reduction orders can otherwise perturb distances
slightly and tree splits can amplify those differences. The configured `--threads`
budget remains fully available to the dominant model-inference phase.

Target groups and their genes are processed in bounded group and target sub-batches
sized from the single thread and memory budgets. Results are consumed in canonical
order and written incrementally, without retaining all run edges or one result object
per target in memory.
Cell-to-centroid calculations use bounded chunks with a nominal 64 MiB cap, reduced
further when current host or cgroup headroom requires it. Validation and robust cosine
normalization use those same bounds instead of expression-sized temporary arrays.
Exceptional finite values whose squared Euclidean norms would underflow or overflow
use a stable, bounded `hypot` path; ordinary Euclidean values retain the faster
vectorized path. The required group-by-group
centroid matrix is materialized because it is itself a requested output. In the two
cell-distance modes, the reusable cell-to-centroid matrix is written directly to a
temporary disk-backed memory map when it crosses the fixed size boundary or when even
a smaller heap allocation would exceed the live memory plan. That mapping is closed
before staging cleanup on success or failure. In `group-distance` mode,
cell-to-centroid distances are not computed at all: only the required
centroid-to-centroid matrix is calculated, and each per-cell `distance` recorded for
weighting is the applicable shared centroid-to-centroid distance. Automatic bandwidth
still uses the exact global positive-distance median relevant to the selected mode.

Run artifacts are first written to a private sibling staging directory. The requested
path is published only after all scientific tables and metadata are complete, so an
early validation or weighting failure does not block a retry. If individual model fits
fail, the completed diagnostic run is published with `status=failed` before SPATHI
returns an error pointing to `model_diagnostics.tsv.gz`.

The TF predictor matrix is reused across models, as are representations, centroids,
the distances needed by the selected mode, bandwidth, and one weight vector per target
group. Timing and effective resource decisions are recorded in `run_metadata.json`.

From a development checkout installed with the `dev` dependencies, run the separate
synthetic scaling benchmark with, for example:

```bash
python benchmarks/benchmark_scaling.py \
  --threads 1 2 auto \
  --targets 20 80 \
  --n-estimators 25 \
  --n-components 20 \
  --warmups 1 \
  --repeats 2 \
  --case-timeout-seconds 3600 \
  --no-checkpoint \
  --no-report
```

The explicit 25-tree/20-component settings above are a reduced development scout, not
the provisional 250-tree/50-component baseline. It executes complete CLI child processes
in balanced, reproducibly shuffled orders and
writes machine-readable CSV measurements to standard output and to a synchronously
flushed `benchmark-results.csv` inside the reported workspace. Each row includes child
wall time; sampled process-tree user/system CPU (a lower bound for descendants that
start and finish between samples); sampled peak process-tree RSS and transient run
storage; and exact final logical/allocated sizes for inputs, published output, and all
retained run artifacts. Exact input hashes, implementation and harness hashes, package
and dependency versions, requested controls, and effective controls make a row
auditable even when its successful workspace is later discarded. The benchmark also
flattens the current `run_metadata.json` contract into columns for phase timings, model
and edge counts, effective dimensions, batching, and parallelism. This keeps CLI
overhead visible while making scientific phases separately attributable.

Every SPATHI child has a positive finite timeout (one hour by default). A timeout
terminates its contained process tree, retains diagnostic artifacts, writes a
`status=timeout` row, and then continues the schedule. `Ctrl-C` and external
termination use the same cleanup path and preserve rows already fsynced. Use
`--keep-work-dir` to retain successful direct-benchmark artifacts as well. Warm-ups are
identified separately, target subsets are generated from the same synthetic dataset,
and checkpointing or report generation can be included explicitly. Benchmarks are
intentionally excluded from ordinary tests and CI.

For the validated multi-axis suites, use the orchestration entry point:

```bash
python benchmarks/run_scaling_suite.py --profile smoke
python benchmarks/run_scaling_suite.py --profile progressive
python benchmarks/run_scaling_suite.py --profile large-scale
```

`smoke` checks the measurement path, while `progressive` varies cells, genes, TFs,
targets, trees, multigroup counts, a separate one-group contract, coupled full-network
sizes, and thread budgets. `large-scale` exercises a bounded target sample at four
independently declared, rounded synthetic workloads. It contains no measurements,
identifiers, hashes, or dimensions derived from a private biological dataset. The
profile writes dense ANDREA-compatible TSV inputs and is intentionally expensive; run
it only after `progressive` succeeds.

Add `--dry-run` to validate profiles and inspect every command, outer case timeout, and
conservative disk estimate without creating output. Each executed suite first checks
that its thread budgets remain distinct on the current host and that each case fits the
available disk. It then snapshots the exact profile, benchmark harness, any optional
source document declared by a custom profile, and the importable SPATHI package used by
every child. It writes an
interruption-safe manifest, per-case logs, and one aggregate `results.csv` under the
reported output directory. Successful dense inputs and individual run directories are
discarded by default; pass `--keep-workspaces` only when those potentially large
artifacts are needed. Each complete benchmark case has an outer timeout covering data
generation plus all scheduled SPATHI children. Profiles describe their synthetic-data
limitations explicitly and do not replace measurements on prepared biological inputs.

## Scientific interpretation and limitations

A high SPATHI score means that a TF was predictively important for one target under one
group-specific weighting. It does not establish direct binding, causal regulation, or
regulatory sign. Impurity importance can favor predictors with greater variation and
can distribute importance unpredictably among correlated TFs.

Results depend on input preprocessing, cell-group assignments, representation choices,
kernel bandwidth, and—most importantly—the biological assumption that transcriptomic
proximity is informative about regulatory similarity. Compare weighting modes and
inspect weight diagnostics, external mass, and effective sample size before biological
interpretation.

## Development and distribution checks

```bash
ruff check .
ruff format --check .
mypy src/spathi
pytest -m "not distribution"
python -m build
python -m twine check dist/*
```

The excluded `distribution` test performs its own wheel build. Run plain `pytest`
when that additional local packaging assertion is desired; CI's dedicated
distribution job is the authoritative build, clean-install, and CLI smoke check.

Run the configured mypy check under Python 3.11, as CI does for the minimum supported
version. Newer interpreters are covered by the runtime test matrix.

To validate a wheel in a clean local environment:

```bash
python -m venv /tmp/spathi-wheel-check
/tmp/spathi-wheel-check/bin/python -m pip install dist/*.whl
/tmp/spathi-wheel-check/bin/spathi --help
```

Release maintainers should follow
[`docs/releasing.md`](https://github.com/AdrianSeguraOrtiz/SPATHI/blob/main/docs/releasing.md).
Building locally does not publish anything and does not require PyPI credentials.

## License

SPATHI is distributed under the
[MIT License](https://github.com/AdrianSeguraOrtiz/SPATHI/blob/main/LICENSE).
