# Generic data preparation

`spathi prepare` converts one standard 10x Genomics feature-barcode HDF5 matrix into
strict inference inputs. It is a generic package feature, not a study adapter: it has
no knowledge of study-specific annotation systems, diseases, or lineage taxonomies.

Study-specific code should resolve biological labels before this command. Its output
is the canonical annotation table described below. Keeping that boundary explicit
prevents dataset-specific assumptions from entering SPATHI's reusable core.

## Inputs

### 10x H5

`--tenx-h5` accepts the 10x feature-barcode layout rooted at `/matrix`. The reader
requires the CSC arrays `data`, `indices`, `indptr`, and `shape`, barcodes, and the
feature fields `name`, `id`, and `feature_type`. Only features whose type is exactly
`Gene Expression` participate in normalization or outputs.

Use `--gene-identifier name` (the default) to match TF symbols against
`/matrix/features/name`, or `--gene-identifier id` to use stable feature IDs. Selected
values must be non-empty; repetitions are handled according to an explicit policy. The default
`--duplicate-gene-policy sum` collapses rows with the same selected identifier by
summing their raw sparse counts before normalization. This retains library totals and
allows symbol-based TF lists to match current 10x references. Use
`--duplicate-gene-policy error` to reject repetitions instead. The manifest records
the identifiers and number of rows collapsed.

Counts must be numeric, finite, and non-negative. Each retained annotated cell must
have a positive Gene Expression library size.

### Canonical annotations

`--annotations` is a UTF-8 TSV. Its schema is intentionally small and strict:

| Column | Required | Meaning |
|---|---:|---|
| `cell` | yes | Unique barcode present in the H5 |
| `analysis_unit` | yes | Independent dataset to be passed to one inference run |
| `cluster` | yes | Context for which that run will infer a network |

Unknown columns are rejected so misspelled scientific fields cannot be ignored
accidentally. A cell cannot occur in more than one analysis unit. If a study
intentionally needs overlapping analyses, its adapter must create distinct
preparation runs and record that design explicitly.

Cells present in the H5 but absent from the table are excluded and counted. Cells in
the table but absent from the H5 are rejected. Preparation does not invent an
`unannotated` cluster.

### Optional centroid weights

`--centroid-weights` accepts a separate UTF-8 TSV with exactly this header and order:

```text
cell    centroid_weight
```

It is an explicit sensitivity input, not annotation metadata. Every annotation cell
must occur exactly once, no other cells are accepted, and every value must be positive
and finite. Row order need not match the annotations or H5 because preparation aligns
values by cell identifier before splitting them by analysis unit. SPATHI assigns no
dataset-specific meaning to the scalar.

### TF list

`--tf-list` contains one unique identifier per line, without blank lines or surrounding
whitespace. Identifiers absent from the selected H5 gene field are recorded in the
manifest. For each analysis unit, the remaining list is intersected again with genes
that pass that unit's detection filter. Its original order is retained.

## Transformation

The first implementation exposes one normalization with one unambiguous definition:

```text
--normalization library-size-log1p
```

For raw count `x[g,c]`, Gene Expression library size `L[c]`, and `--target-sum T`:

```text
y[g,c] = log(1 + T * x[g,c] / L[c])
```

The default is `T = 10000`. `L[c]` is calculated from every Gene Expression feature
before splitting cells or filtering genes. Consequently, two preparation runs select
the same normalized value for a cell and gene even if their analysis-unit membership
differs.

Within each analysis unit, a gene is retained when its raw sparse count is non-zero in
at least `--min-gene-cells` cells (default: 1). An analysis unit is emitted only when:

- it contains at least `--min-cells` annotated cells (default: 300);
- at least one gene passes the detection filter; and
- at least one supplied TF remains after intersection.

Ineligible units are listed with a machine-readable reason in the manifest. If no unit
is eligible, preparation fails and publishes no output directory.

## Outputs and provenance

Every eligible unit contains:

- `expression.tsv`: normalized genes by cells, accepted by `spathi infer` and ANDREA;
- `groups.tsv`: exact `cell` and `cluster` columns with one row per expression cell;
- `tf_list.txt`: unit-specific TF intersection; and
- `centroid_weights.tsv`, only if `--centroid-weights` was supplied.

`prepare_manifest.json` records:

- SHA-256, byte size, and resolved path for every input;
- the installed SPATHI version and every transformation parameter;
- original matrix, Gene Expression, annotated, and excluded-cell counts;
- pre-normalization library-size summaries;
- present and absent TF identifiers;
- every analysis unit, group size, eligibility decision, output dimensions, paths, and
  SHA-256 fingerprints.

The result directory follows the same never-overwrite policy as inference and is
published only after all files and the manifest have been completed. The H5 data stay
sparse; dense storage is bounded to a single output row while serializing the TSV.

## CLI reference

```bash
spathi prepare \
  --tenx-h5 filtered_feature_bc_matrix.h5 \
  --annotations annotations.tsv \
  --centroid-weights centroid_weights.tsv \
  --tf-list tf_list.txt \
  --output-dir prepared/sample \
  --min-cells 300 \
  --min-gene-cells 1 \
  --normalization library-size-log1p \
  --target-sum 10000 \
  --gene-identifier name \
  --duplicate-gene-policy sum
```

The generated `expression.tsv`, `groups.tsv`, and `tf_list.txt` are then ordinary
`spathi infer` inputs. When emitted, per-unit `centroid_weights.tsv` can be passed
explicitly as `--centroid-weights`; omitting both preparation and inference options
retains the primary uniform-centroid analysis.
Direct H5 input is intentionally unavailable on `infer`.
