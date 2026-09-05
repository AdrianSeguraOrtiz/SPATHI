# SPATHI engineering benchmarks

The benchmark programs are opt-in engineering tools and are not part of the ordinary
test suite. They measure implementation behaviour; they do not measure biological
accuracy.

## Synthetic scaling

`run_scaling_suite.py` runs the validated `smoke`, `progressive`, and `large-scale`
synthetic profiles. See the project README for the complete measurement contract.

## Prepared-data reference/candidate benchmark

`benchmark_equivalence.py` compares two source snapshots on the same prepared
analysis units. Raw expression data remain in their local preparation directories.
Only a small JSON manifest containing paths, sizes, dimensions, and hashes is copied
into the result directory.

Run these commands after activating the project environment described in the root
README (`source .venv/bin/activate`). Profiles pin the complete scientific
configuration; reference and candidate differ only in their source snapshot and
operational behaviour.

First build that ignored local manifest from one or more `spathi prepare` manifests:

```bash
python benchmarks/benchmark_equivalence.py manifest \
  --prepare-manifest sample-a=/path/to/prepared/sample-a/prepare_manifest.json \
  --prepare-manifest sample-b=/path/to/prepared/sample-b/prepare_manifest.json \
  --output benchmarks/results/local-datasets.json
```

Inspect the execution matrix without hashing or executing the large inputs:

```bash
python benchmarks/benchmark_equivalence.py run \
  --profile equivalence-smoke \
  --dataset-manifest benchmarks/results/local-datasets.json \
  --reference-source /path/to/reference/SPATHI \
  --candidate-source . \
  --dataset sample-a-unit-001 \
  --dry-run
```

Remove `--dry-run` to execute. The candidate defaults to the current checkout. A source
argument may name a repository root, `src` directory, or `spathi` package directory.
Both packages are copied into the suite before any timed child starts.

The smoke profile requires exactly one `--dataset`, uses eight targets, five trees,
no warm-up, and one pair only to validate the harness contract. Comparing the same
source on both sides is deliberately allowed in smoke as an autoequivalence check;
its timing ratios are not evidence of an optimization. Run a one-group and a
multigroup check as separate suites rather than silently expanding one smoke run:

```bash
python benchmarks/benchmark_equivalence.py run \
  --profile equivalence-smoke \
  --dataset-manifest benchmarks/results/local-datasets.json \
  --reference-source . --candidate-source . \
  --dataset sample-a-single-group \
  --output-dir benchmarks/results/equivalence-smoke-single

python benchmarks/benchmark_equivalence.py run \
  --profile equivalence-smoke \
  --dataset-manifest benchmarks/results/local-datasets.json \
  --reference-source . --candidate-source . \
  --dataset sample-b-multigroup \
  --output-dir benchmarks/results/equivalence-smoke-multigroup
```

The progressive profile separately varies targets
(`20, 80, 320`), trees (`25, 250`), and threads (`1, 2, 4, 8`), and adds a combined
`320 targets x 250 trees` point. Each configuration has one warm-up pair and four
measured pairs. Progressive and full-target profiles require an explicit dataset:

```bash
python benchmarks/benchmark_equivalence.py run \
  --profile equivalence-progressive \
  --dataset-manifest benchmarks/results/local-datasets.json \
  --reference-source /path/to/reference/SPATHI \
  --candidate-source . \
  --dataset sample-a-unit-001 \
  --dataset sample-b-unit-002
```

Benchmark profiles use positive, explicit thread budgets rather than `auto`, so their
resource axis remains reproducible and is recorded unambiguously in every row.

Progressive and full-target runs reject identical package hashes by default. Use
`--allow-identical-implementations` only for a deliberate harness self-check; the
override is recorded immutably in the suite manifest.

The full-target profile is deliberately opt-in and requires exactly one dataset. It omits `--target-list`, infers every
expression gene with 250 trees, enables checkpointing, and runs ten complete processes
(one warm-up pair plus four measured pairs) for that dataset. Do not launch it
until smoke and progressive succeed. Start with the smallest one-group unit:

```bash
python benchmarks/benchmark_equivalence.py run \
  --profile equivalence-full-target \
  --dataset-manifest benchmarks/results/local-datasets.json \
  --reference-source /path/to/reference/SPATHI \
  --candidate-source . \
  --dataset sample-a-unit-001 \
  --dry-run
```

Inspect the reported process count, per-child and sequential worst-case timeout, and
conservative disk estimate; remove `--dry-run` only when that budget is acceptable.
The full-target profile allows up to 48 hours per child and its disk preflight assumes
maximally dense networks plus retained outputs for every failed comparison.

The result directory contains:

- `runs.csv`: wall time, sampled CPU, sampled process-tree RSS, transient/final disk,
  every relevant SPATHI phase, effective batching/parallelism metadata, dimensions,
  and exact input/implementation hashes;
- `comparisons.csv`: paired speed, memory, disk, and equivalence outcomes;
- `scaling.csv`: equivalent measurement pairs only, grouped by dataset and
  configuration, with paired-ratio medians, quartiles, and deterministic descriptive
  bootstrap intervals;
- `comparison-details/*.json`: per-artifact hashes, comparison mode, row counts when
  semantic parsing is needed, numeric differences, and the first mismatches;
- source, profile, and local-manifest snapshots plus complete child logs.

Each suite also snapshots the runner, resource-measurement helper, implementation
packages, and a hashed target-list manifest. Resume therefore does not regenerate or
rediscover target slices from the large expression TSV. To continue an interrupted
suite, invoke only:

```bash
python benchmarks/benchmark_equivalence.py resume \
  --output-dir benchmarks/results/the-interrupted-suite
```

The command verifies every snapshot and journal, takes an exclusive suite lock, and
continues the exact deterministic schedule. A completed suite is an idempotent no-op.
Runs recovered from a SPATHI checkpoint remain eligible for scientific equivalence,
but their partial timings are marked performance-ineligible. The same applies to a
pair whose two roles completed in different suite attempts: raw measurements remain
in `runs.csv`, while ratios are blank and the pair is excluded from `scaling.csv`
performance summaries.

Scientific tables are parsed in canonical SPATHI order. All identifiers and categorical
values must match exactly. Numeric values use the tolerances declared in the checked-in
profile. Byte-identical artifacts are schema-checked and recorded with
`comparison_mode=byte-identical`; their row counters remain zero because no row-level
comparison is necessary. Only `fit_seconds` in `model_diagnostics.tsv.gz`, output paths in
`parameters.json`, execution metadata, and the optional HTML report are excluded from
equivalence. Successful run directories are deleted after comparison unless
`--keep-outputs` is supplied; failures and mismatches are always retained.

Target subsets are deterministic and nested, and only restrict response models. They do
not alter the expression matrix or candidate TF list. Consequently they are suitable
for bounded engineering measurements, not for claiming biological target selection or
accuracy. Configuration order is deterministically shuffled and rotated within each
case/dataset; implementation order is counterbalanced across the four measured rounds.
The reported 95% bootstrap bounds describe run-to-run variability in this fixed suite;
they are not population-level confidence intervals or hypothesis tests. RSS, CPU, and
transient disk peaks are sampled lower bounds at the interval declared by the profile.
