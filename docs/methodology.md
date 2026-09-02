# SPATHI methodology

This document specifies the scientific and computational semantics of SPATHI's first
release. It is also the reference for interpreting diagnostics and testing the current
implementation.

## Software boundary

`spathi.core.infer` owns input validation, scientific computation, checkpointing,
artifact writing, and atomic publication. The public package root exposes that function
as the canonical `from spathi import infer` API together with `SpathiConfig`,
`SpathiProgressEvent`, and `SpathiRunResult`. The `spathi infer` command is a thin
adapter: it parses terminal arguments into `SpathiConfig`, renders logs and progress,
and calls the core. There is no separate command-line inference path.

## Scope and notation

Let the supplied expression matrix contain genes as rows and cells as columns. For
cell \(i\), let \(x_i\) be its expression vector and \(g_i\) its assigned group. By
default every gene is modeled as a target; an explicit `target_list` selects a nonempty
subset without changing the distance space. Candidate predictors are the genes listed
in `tf_list`, except that a target is removed from its own predictors when it is also a
TF.

For each target group \(c\), SPATHI learns a separate network using every cell. It does
not split the expression matrix into isolated group-specific training sets. A final
weight \(w_i^{(c)}\) controls cell \(i\)'s contribution to the model for group \(c\).

No ordering, trajectory, lineage, or hierarchy among groups is assumed.

## Two deliberately separate spaces

SPATHI keeps the inference space separate from the distance space:

- **Inference space:** the original preprocessed expression values supplied by the
  user. TF columns are predictors and a target gene is the response.
- **Distance space:** either the cell-by-gene expression representation or a PCA
  representation. This space is used only to compute centroids, distances, and
  weights.

`distance-standardization=standard` fits a standard scaling transform in the distance
transformation; `none` leaves values unstandardized. This does not modify values used by
the tree ensembles. SPATHI performs no implicit library-size normalization, log
transformation, scaling, or feature selection on the inference matrix.

When PCA is selected, scikit-learn PCA receives cells as observations. Its centered-data
informative-rank bound is `min(n_genes, max(0, n_cells - 1))`. The one-cell case keeps
one structural component so downstream shapes remain defined; its informative bound
and explained variance are both recorded as zero. Otherwise, the effective component
count is the smaller of the requested count and that bound. `run_metadata.json`
records the requested and effective counts, informative bound, SVD policy, and both
per-component and cumulative explained-variance ratios. When the policy is `auto`,
the concrete solver remains delegated to the recorded scikit-learn version; SPATHI
does not reproduce private solver-selection logic.

PCA distance-space runs always write `cell_embedding.tsv.gz` with the first up to three
retained PCs and `pca_explained_variance.tsv` with the complete variance summary. An
expression-distance run with `report=True` writes the same artifact names for its
clearly named `AuxiliaryPC` report projection. These auxiliary coordinates do not
participate in distance or weight calculation. They are absent for expression-distance
runs with `report=False` or `--no-report`.

### Numeric precision

Input validation and all distance, PCA, centroid, kernel, weight, response, and
diagnostic calculations use `float64`. Selected target responses are retained in a
cells-by-targets `float64` matrix so biologically valid variation smaller than one
`float32` unit is not collapsed before constant-target detection or fitting. With the
default all-gene target set this reuses the validated expression storage; a subset
materializes only its selected response columns. SPATHI extracts the reusable TF
predictor columns into a contiguous `float32` matrix, matching scikit-learn's
tree-oriented numeric path without reducing target precision. This conversion does not
normalize or otherwise transform the supplied values. Weights remain `float64` when
passed as `sample_weight`.

## Group centroids and distances

Let \(z_i\) denote cell \(i\)'s vector in the configured distance space. SPATHI uses
the arithmetic centroid:

\[
\mu_c = \frac{1}{n_c}\sum_{i:g_i=c} z_i.
\]

Centroids are computed once per run and reused by every target-group weighting pass.

For the selected Euclidean or cosine metric, SPATHI always calculates the pairwise
centroid distances \(d(\mu_h,\mu_c)\). The dense group-by-group result is materialized
once because it is both small relative to the expression matrix and a requested
output. In `cell-distance` and `cell-distance-group-anchored`, SPATHI additionally
calculates every \(d(z_i,\mu_c)\). Ordinary Euclidean calculations use
scikit-learn's vectorized path; cosine calculations use max-scaled unit-vector products
that remain finite for tiny or very large non-zero rows. Both operate in chunks with a
nominal 64 MiB cap rather than inheriting a process-wide default, and the live memory
plan reduces the budget when current host or cgroup headroom requires it. Finite-value,
zero-norm, and exceptional-magnitude checks use the same bounded blocks rather than
materializing expression-sized boolean or normalization copies. If finite coordinates
could make squared Euclidean norms underflow or overflow, SPATHI switches only that
calculation to bounded `hypot` reductions, preserving representable distances without
constructing a cell-by-group-by-dimension tensor. The resulting cell-by-group matrix is reused;
chunks are written directly into a temporary disk-backed memory map either above the
fixed size threshold or whenever the live memory plan cannot safely retain the
complete matrix on the heap. The mapping and its backing file are closed before
staging cleanup even after an exception. Arithmetic centroids similarly retain the
fast grouped mean and use a stable online mean only if finite extreme values overflow
its internal sum.

`group-distance` never needs cell-to-centroid distances and therefore does not compute
them. It uses only the centroid-to-centroid matrix. The per-cell `distance` in
`cell_weights.tsv.gz` is consequently the centroid distance shared by that cell's
source group. Automatic-bandwidth calculations retain the exact positive-distance
median relevant to the selected mode; large cell-distance calculations use bounded
scans and a temporary disk-backed selection array.

Cosine distance is undefined for a zero vector. SPATHI therefore rejects zero-norm
centroids in every mode and zero-norm cell representations in the two modes that
actually calculate cell-to-centroid distances. `group-distance` does not inspect cell
norms because those vectors never enter its distance calculation. Required zero rows
are reported by identifier rather than silently assigned distance zero or one. Non-zero
rows at exceptional magnitudes are rescaled before the cosine calculation to avoid norm
underflow or overflow. Finally, non-negative cosine values no larger than the `float64`
forward-error bound for a dot product of the configured dimension are set to exactly
zero. This prevents numerical residue from becoming an artificial automatic bandwidth
while preserving genuinely positive distances above that bound.

With one observed group, PCA and standardized expression both make the only group
centroid exactly zero. SPATHI therefore rejects either centered representation combined
with cosine distance before preprocessing; unstandardized expression or Euclidean
distance remains defined.

## Kernels and global bandwidth

Distances are converted into affinities through a kernel registry. The Gaussian kernel
is

\[
K_{\mathrm{Gaussian}}(d;h)
= \exp\left(-\frac{d^2}{2h^2}\right),
\]

and the optional exponential kernel is

\[
K_{\mathrm{exponential}}(d;h)
= \exp\left(-\frac{d}{h}\right).
\]

The bandwidth \(h\) must be positive. A numeric `bandwidth` is used directly. With
`bandwidth=auto`, one reproducible global bandwidth is selected for the entire run:

- for `cell-distance` and `cell-distance-group-anchored`, \(h\) is the median of all
  strictly positive cell-to-centroid distances;
- for `group-distance`, \(h\) is the median of all strictly positive
  centroid-to-centroid distances.

Zeros are excluded because within-group or coincident centroids may legitimately have
zero distance. If the relevant family contains no positive finite distance, SPATHI
uses a finite positive fallback, emits a warning, and records that decision. Kernel
outputs are validated to be finite and in \([0,1]\).

## Calibrated defaults

The first-release configuration defaults to `cell-distance-group-anchored` weighting,
PCA distance space, cosine distance, a Gaussian kernel with automatic bandwidth,
`cap-to-target` group-size correction, and Extra-Trees. Each ensemble uses 250 trees
and `max_features=sqrt`. The CLI reads these values from the same immutable
`SpathiConfig` used by the Python API, so the two interfaces cannot drift.

These defaults define the reproducible baseline selected by the current calibration;
they do not establish universal optimality. A new biological dataset should still be
checked for convergence and sensitivity to representation and weighting choices.

## Base weights, multiplicity correction, and final weights

SPATHI never conflates these quantities:

- `distance` is the distance selected by the weighting mode;
- `base_weight` is the direct kernel affinity, or the explicit target-group anchor;
- `group_size_factor` adjusts external population multiplicity;
- `final_weight` is passed to the estimator.

This separation matters because population-size correction changes total group
contribution without changing transcriptomic distance or base biological affinity.

### Cell distance

For `cell-distance`, group identity does not enter the base-weight definition:

\[
w_{i,\mathrm{base}}^{(c)} = K\!\left(d(z_i,\mu_c);h\right).
\]

Consequently, a cell outside \(c\) may receive a greater base weight than a peripheral
cell inside \(c\), and equally distant cells receive the same base weight regardless
of group. If size correction is disabled, the mode depends only on individual
distance.

After any size factor is applied, the complete vector is divided by its maximum so its
largest final weight is one. This rescaling is unique to `cell-distance`. A degenerate
all-zero or non-finite vector is diagnosed explicitly rather than passed silently to a
model.

### Group-anchored cell distance

For `cell-distance-group-anchored`, target-group membership fixes the base weight at
one, while external weights remain cell-specific:

\[
w_{i,\mathrm{base}}^{(c)} =
\begin{cases}
1, & g_i=c,\\
K\!\left(d(z_i,\mu_c);h\right), & g_i\ne c.
\end{cases}
\]

All target cells therefore retain final weight one. External cells from the same group
can have different weights, but none can exceed one. There is no post-correction
normalization that could reduce the target anchor.

### Group distance

For `group-distance`, each external population receives a shared base affinity:

\[
w_{i,\mathrm{base}}^{(c)} =
\begin{cases}
1, & g_i=c,\\
K\!\left(d(\mu_{g_i},\mu_c);h\right), & g_i\ne c.
\end{cases}
\]

Thus all cells from an external group have the same base weight. This mode is the most
directly interpretable at population level.

### Group-size correction

With `group-size-correction=none`, every size factor is one. With `cap-to-target`, an
external cell in group \(h\) receives

\[
f_h^{(c)} = \min\left(1,\frac{n_c}{n_h}\right),
\qquad
\widetilde{w}_i^{(c)} = w_{i,\mathrm{base}}^{(c)} f_h^{(c)}.
\]

The target group always has factor one. This caps the per-cell contribution of a larger
external group; it does not upweight smaller groups and does not modify distances. In
`cell-distance`, the final maximum-one normalization follows this multiplication. In
the two anchored modes, \(\widetilde{w}\) is already the final vector.

## Weight diagnostics

Diagnostics are computed once per target-group weight vector, not once per gene. They
include target-group size, total weight, target and per-external-group mass, target and
external mass percentages, minimum, maximum, mean, median, positive-weight count, and
effective sample size:

\[
\operatorname{ESS}(w) =
\frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
\]

The run warns about degenerate or non-finite weights, very low ESS, external mass that
exceeds target mass, and excessive concentration in one external group. These warnings
are interpretive signals. They do not stop a run unless the resulting model cannot be
trained.

`group_affinities.tsv` is deliberately a group-level centroid diagnostic. Its
`base_affinity` is the kernel of the source-to-target centroid distance, irrespective
of the selected weighting mode, and is a per-cell base model weight only in
`group-distance`. Its `group_size_factor` is the per-cell multiplicity correction for
that source/target pair. The authoritative model-level values are
`cell_weights.tsv.gz:final_weight`; their exact effective source-group mass is reported
in `weight_diagnostics.tsv`.

## Interactive report

`report=True` is the default in `SpathiConfig`. The corresponding CLI contract is
`--report`/`--no-report`, with reporting enabled unless explicitly disabled. An enabled
run writes exactly one report artifact, `report.html`; a disabled run writes none.
`SpathiRunResult.report_path` is the published `pathlib.Path` when the report exists and
`None` otherwise.

The document is self-contained and works offline. It embeds the Plotly browser runtime,
styles, interaction code, and all report data rather than loading any network resource.
Its three sections are:

- **Target explorer:** a target-group selector, weight warnings, target cell count,
  effective sample size, target and external mass, positive-weight cell count, and mean
  weight; a PC1/PC2 cell cloud; distance against final weight; exact source-group mass;
  and exact full-data weight summaries by source group.
- **Overview:** the shared cell cloud and group centroids, observed group sizes, PCA
  explained variance, the exact target-by-source weight-mass matrix, and effective
  sample size together with target-group contribution for every target group.
- **Method & provenance:** projection semantics, interpretation boundaries, the run
  summary, and requested parameters.

The HTML includes the input identifiers of sampled cells, group labels, and derived
coordinates, distances, weights, and summaries. It excludes the expression matrix and
all local input/output paths; the latter remain in `run_metadata.json`. Because
cell identifiers can still encode sample or patient information, the report must follow
the same data-governance policy as those identifiers when it is shared.

In the target cell cloud, marker fill uses the same fixed 0–1 Cividis scale for every
target group and encodes the exact `final_weight` passed to inference. Marker shape and
outline identify the observed source group; a star marks the selected target centroid.
Group envelopes are deliberately absent because a two-dimensional PCA projection does
not establish biological boundaries.

The cloud does not redefine the configured metric. Visual separation in two projected
axes equals the fitted geometry only for Euclidean distance when those axes contain the
complete fitted distance space; it is not a visual representation of cosine distance.
The adjacent distance-to-final-weight chart and the tabular outputs contain the exact
configured weighting distances.

For PCA-distance runs, PC1/PC2 are reused from the fitted distance representation. If
only one component exists, the second coordinate is fixed at zero. The cloud is only a
projection: distances and weights still use every retained component, so separation in
later components can be absent from PC1/PC2. For expression-distance runs, SPATHI fits a
deterministic auxiliary two-component PCA for the report only. It transforms the
already configured expression representation and its centroids, but never changes the
inference distances, weights, predictors, responses, or network.

Cell-level charts share one deterministic group-stratified sample across all target
groups. Allocation first reserves one cell per observed group, distributes the remaining
budget approximately in proportion to the remaining group capacities, and selects cells
by stable SHA-256 hashes of their identifiers. Reusing exactly the same cells makes
target-group views directly comparable and preserves rare-group representation. The
default budget aims to show at most 30,000 cells and about 300,000 sampled
target-by-cell values; the one-cell-per-group guarantee takes precedence when an
exceptionally large group count requires it.

The Overview is rendered lazily when its tab is first opened. On browsers without
WebGL, SVG scatter plots receive an additional deterministic, group-preserving budget
of approximately 5,000 points so opening the standalone report does not create an
unbounded DOM. Each scatter legend reports the plotted and total count for its source
group. This makes deliberate rare-group preservation visible instead of allowing point
density in the plot to be mistaken for the exact group-size distribution.

Sampling affects only cell-level chart points. Metric cards, effective sample size,
target and external mass, the target-by-source mass matrix, and per-source minimum,
quartiles, median, mean, maximum, and positive-weight count are calculated from all
cells. The report therefore does not substitute sampled estimates for its aggregate
statistics.

Report construction has explicit scalability limits and memory planning. Four sampled
target-by-cell vectors, eight exact target-by-source summary matrices, the shared
coordinates, embedded binary payloads, the Plotly runtime, and—when needed—the auxiliary
PCA workspace are included in the preflight estimate. A run fails before model fitting
if the detected memory budget cannot accommodate the bounded report. Report preparation
and rendering time is recorded separately as the `report` phase in `run_metadata.json`;
the artifact path, SHA-256 digest, byte size, total and sampled cell counts, group
count, and sampling method are recorded there as well. Use `--no-report` or
`report=False` for calibration and timing runs when this additional cost should be
excluded. That setting does not change scientific outputs.

The weight-mass charts quantify statistical contribution under the selected weighting
rule. They do not represent causal influence, lineage, or regulatory edges between
groups.

## Weighted network inference

For group \(c\) and target gene \(t\), SPATHI starts from all valid TFs, removes \(t\)
if it is among them, and excludes any remaining predictor that is constant among cells
with positive \(w_i^{(c)}\). This positive-weight restriction matches the observations
that can affect the fitted model. It then fits either scikit-learn's
`ExtraTreesRegressor` or `RandomForestRegressor` on all cells:

\[
\widehat{x_t} = F_c(X_{\mathrm{TF}}; w^{(c)}).
\]

The final vector \(w^{(c)}\) is supplied through the estimator's `sample_weight`
argument. SPATHI uses the fitted `feature_importances_`, based on weighted impurity
reduction, as scores for directed \(\mathrm{TF}\rightarrow t\) candidate edges.

Bootstrap sampling defaults are resolved per estimator: it is disabled for
Extra-Trees and enabled for Random Forest. The API value `bootstrap=None`, or omission
of both `--bootstrap` and `--no-bootstrap` in the CLI, selects that automatic policy;
an explicit boolean overrides it. `parameters.json` preserves the requested nullable
value and `run_metadata.json` records the effective boolean used by every estimator.

Every selected target is attempted for every group. Constant responses, empty or
entirely constant predictor sets, and other non-trainable models are recorded in
`skipped_targets.tsv`.
`model_diagnostics.tsv.gz` distinguishes the self-excluded predictor, constant
predictors, the complete discarded set, and the number actually used, preserving the
mapping from fitted importance columns back to TF names. No self-edges are produced,
no automatic threshold is applied, and every finite score strictly greater than zero
is retained. The feature importances are relative within each target model and are not
renormalized across targets or groups.

Predictor collections and warning collections are serialized as compact JSON arrays in
columns ending in `_json`; no delimiter is reserved inside biological identifiers.

Each output edge has `sign=?`. Tree impurity importance does not determine activation
or repression, and SPATHI does not invent a sign.

## Parallel execution and determinism

The main cost grows approximately as

\[
N_{\mathrm{groups}}\,N_{\mathrm{targets}}\,
\operatorname{cost}(\mathrm{ensemble}).
\]

`threads` is the only public CPU budget. `-1` resolves to all available logical CPUs,
`1` is sequential, and any positive value is a cap. An internal automatic plan
uses Joblib either across independent `(target group, target gene)` tasks or inside a
single ensemble, never both at full width. When outer task parallelism is active, each
ensemble receives `n_jobs=1`; threadpoolctl constrains BLAS/OpenMP libraries to prevent
nested oversubscription. The threading backend shares the read-only expression array
instead of copying it into worker processes. One persistent worker pool is reused
across all bounded target batches in an attempt; result callbacks execute only on the
orchestration thread, and completed tasks are reordered before canonical serialization.

The core also estimates a conservative peak allocation for one fitted ensemble and
detects the tightest available host or cgroup memory headroom when possible. Linux host
headroom uses `/proc/meminfo`'s `MemAvailable` (including reclaimable memory) before the
portable free-page fallback, and is still capped by every applicable cgroup limit. The
planner reserves headroom for shared arrays and caps concurrent outer models
accordingly. This heuristic cannot predict every allocator or fitted-tree shape, so it
complements rather than replaces external memory monitoring. The estimate, detected
availability, usable fraction, and selected model-concurrency cap are persisted in
`run_metadata.json`.

A task seed is derived stably from the global seed, group identifier, and target
identifier. It does not depend on scheduler order. Output is sorted by `context`,
`target`, and `source`, so changing `threads` should produce equal or numerically
equivalent results.

The core chooses bounded target-group batches large enough to expose independent
`(group, target)` tasks to Joblib while keeping output progressive. When all selected
targets fit in one batch, multiple groups are scheduled together; larger target sets
retain group-major sub-batches so output remains globally sorted without retaining one
result object for every selected target at once. PCA, centroids, distances, bandwidth
selection, and auxiliary report PCA run with one numerical-library thread. This fixes
floating-point reduction order across `threads` settings; the configured budget is
then spent on model inference, while cell-to-centroid chunks retain a separate
working-memory budget of at most 64 MiB, reduced from live memory headroom when needed.

For PCA distance runs, SPATHI creates one private C-contiguous `float64` work buffer.
Optional standardization and PCA centering reuse that buffer in place; the caller's
expression object is never mutated. This avoids retaining both an explicit transpose
copy and scikit-learn's second full centered copy. Once PCA, selected target responses,
and TF predictors are prepared, an explicit target subset no longer needs the complete
validated expression allocation; SPATHI releases it before fitting the first model.

Representations, centroids, only the distance matrices needed by the selected
weighting mode, bandwidth, TF predictors, size factors, and one weight vector per
target group are reused. In particular, `group-distance` does not allocate or scan a
cell-to-centroid matrix. Bounded model batches are consumed in canonical order and
their edges and diagnostics are written incrementally; checkpointed results are read
back in the same order without constructing a run-sized Python result collection.
Phase timings, requested/effective threads, backend, model counts, relevant dependency
versions, memory estimates, and warnings are written to metadata.

Deterministic rows are written in stable order. Compressed tables also use a fixed gzip
timestamp, so a deterministic table such as `cell_weights.tsv.gz` has identical bytes
for identical inputs and effective parameters. `model_diagnostics.tsv.gz` includes
measured fit times; its compression header is reproducible, but its time-bearing
content is not promised to be byte-identical across runs.

## Checkpoint identity and structured progress

Checkpointing is operational and does not alter the scientific configuration or final
artifacts. Each completed model is stored as a compressed, checksummed SQLite payload
under a unique `(target_group, target)` key. Transactions use WAL so an interrupted
attempt loses at most work that had not yet committed. Resume is accepted only when
the complete input digests, target/group universe, scientific parameters, dependency
versions, and installed SPATHI source fingerprint match. Recomputed `float64` final
weight vectors are also hashed per group before any committed model is skipped; this
prevents combining models fitted against numerically different weights.

The final network is reconstructed from validated checkpoint rows in lexical
`context, target, source` order inside the private staging directory. Publication is
still one atomic directory rename. Before loading scientific inputs, SPATHI exercises
both the occupied-destination and successful-rename paths in a private temporary
directory on the target filesystem. Unsupported platform, kernel, libc, or filesystem
semantics therefore fail before inference. On Linux, a missing libc `renameat2`
wrapper falls back to the architecture-specific syscall while preserving
`RENAME_NOREPLACE`. A successful run removes the checkpoint rather than including it
in the output bundle.

Structured progress reports preprocessing phases plus global completed/total model and
group counters. Events are emitted synchronously on the orchestration thread, never on
worker threads. With checkpointing enabled, a model event follows its committed
transaction; with checkpointing disabled, it follows the completed fit. This lets
wrappers map progress into their own progress contract without parsing logs, while the
checkpointed path never reports model work that cannot actually be resumed.

## Interpretation boundaries

SPATHI produces predictive regulatory hypotheses, not demonstrated causal networks.
In particular:

- impurity importance can be biased toward predictors with greater variation;
- correlated TFs can divide or exchange importance;
- transcriptomic proximity need not imply shared regulatory mechanisms;
- output quality depends on preprocessing, cell labels, TF coverage, representation,
  metric, kernel, and bandwidth;
- a centroid may summarize a heterogeneous or non-convex group poorly;
- group-size correction changes statistical contribution, not biological similarity.

Use `cell_weights.tsv.gz:final_weight` for the exact observation weights and
`weight_diagnostics.tsv` for their effective group-level contribution.
`group_affinities.tsv` should be used only as the centroid-affinity diagnostic
described above. The interactive report provides a compact view of those quantities,
but its PC1/PC2 positions are projections and do not replace the tabular values.
Independent evidence and downstream validation remain necessary.
