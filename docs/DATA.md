# Training data contract

`orbit.common.data` builds training Parquet from **user-supplied** JSON arrays or JSONL records. It does not recover, fabricate, download, or publish experimental data. `synthetic_records("train")` and `synthetic_records("val")` return one deliberately trivial record each, with distinct IDs, for plumbing tests only. No binary dataset is committed.

## Required fields

| Field | Contract and consumer |
| --- | --- |
| `prompt` (or explicit `prompt_key`) | Nonempty chat message list; vendored `RLHFDataset` reads `data.prompt_key`, default `prompt`, and applies the tokenizer chat template. |
| `data_source` | Nonempty user-supplied string; reward manager dispatches through `reward_fn_key`. |
| `reward_model.ground_truth` | Explicit string reference (may be empty if user deliberately supplies it); reward managers read this field. |
| `reward_model.style` | Nonempty string describing the supplied reference/reward convention; retained as dataset metadata, not inferred. |
| `extra_info.prompt_history` | Nonempty chat messages consumed by rubric scoring. |
| `extra_info.rubrics` | Nonempty list of objects with nonempty string `criterion` and finite numeric `points`; positive, zero, and negative weights are preserved. |
| `extra_info.query_id` | Explicit nonempty unique string, stable across repeated sampling of a query; curriculum state is keyed by this ID. |
| `extra_info.sorted_rubric_indices` | Optional for ordinary scoring; required with `require_curriculum=True`. A permutation of zero-based rubric list positions. The injector indexes `all_rubrics[orig_idx]`; values are positions, not arbitrary rubric IDs. |

Messages use `system`, `user`, or `assistant` with string `content`. This text-only builder rejects multimodal/tool message structures; those need an explicit compatible extension. Additional user fields are retained, not silently stripped. User-provided `extra_info.index` and rubric original-index metadata are preserved. No guessed IDs, reordered rubrics, reference answers, or curriculum order are generated. Prompt and prompt history are independently required because their experiment-specific relation must be supplied by the dataset owner.

The strict production validator rejects missing/duplicate IDs, empty datasets, malformed messages, invalid/missing references, nonfinite/nonnumeric/boolean weights, and invalid curriculum indices before writing. This is intentionally stricter than the runtime scorer's permissive handling of malformed rubrics. Numeric strings must be converted explicitly by the data owner; silent coercion could conceal source errors.

## Python API

```python
from pathlib import Path
from orbit.common.data import read_records, read_parquet, write_parquet, synthetic_records

records = read_records(Path("user_train.jsonl"), require_curriculum=True)
write_parquet(Path("train.parquet"), records, require_curriculum=True)
assert read_parquet(Path("train.parquet"), require_curriculum=True) == records

# Explicit synthetic fixtures, never real training or evaluation evidence:
write_parquet(Path("synthetic_train.parquet"), synthetic_records("train"))
write_parquet(Path("synthetic_val.parquet"), synthetic_records("val"))
```

Install the project's optional data dependency in an isolated environment. `pyarrow` imports lazily: JSON validation and package imports do not require it. Parquet output uses a temporary file in the destination directory and atomic no-clobber publication; existing files raise `FileExistsError` unless `overwrite=True` is explicit. Parent directories must already exist. Arrow preserves nested structures; heterogeneous additional fields must be Arrow-compatible and conversion errors abort without publishing an output file. Read-back revalidates the contract.

## Limits

Dataset loading with actual transformers tokenizers, GPU training, real corpus quality, split leakage and curriculum effectiveness require target-environment verification. The builder does not auto-split real data: supply disjoint train/validation IDs and check semantic duplicates and publication rights separately.

## CLI workflow with synthetic inputs

```sh
orbit data-build --input examples/prepared-train.json --output synthetic-train.parquet --curriculum
orbit data-build --input examples/prepared-val.json --output synthetic-val.parquet --curriculum
orbit data-validate --input synthetic-val.parquet --curriculum
orbit evaluate --config configs/offline.json --input synthetic-val.parquet --responses examples/prepared-responses.json --recorded examples/recorded-judge.json
```

Fixed responses join by `query_id`, with exact identity-set equality and no duplicate IDs; neither response order nor model completion order is used. This evaluates supplied responses; it does not run policy inference.

For generation-to-data, include a stable `query_id` in each generation context. `generate` retains it in the output. Then `data-build --input prepared-contexts.json --rubrics generated.json --output train.parquet` attaches `evaluation_criteria` to `extra_info.rubrics` by exact IDs before schema validation. Prepared contexts must still supply prompt, source, reference and history. If using curriculum, supply the verified rubric ordering and `--curriculum`; the builder does not guess difficulty/order from scores. Generation records without IDs are deliberately rejected instead of aligned by position. Existing rubrics are replaced only when this explicit `--rubrics` option is used.

The training execution preflight reads approved train/validation Parquet using this validator, rejects duplicate IDs across files and train/validation overlap, and requires course ordering for all three curriculum/stochastic variants. This checks identity leakage, not semantic duplicate leakage. All historical dataset selection and filtering thresholds remain owner-confirmed inputs.

## Explicit legacy identity migration

Some historical prepared files have missing or split-local row IDs. Use `orbit data-build --input legacy.parquet --output prepared.parquet --identity-namespace source_label` to assign a stable source-label plus SHA-256 message identity. Original IDs are retained as `extra_info.source_query_id`. Migration is opt-in, preserves the original records, and does not deduplicate, select splits or infer curriculum difficulty. Duplicate histories receive identical IDs and fail normal unique-ID validation; resolve them through an approved selection procedure. When joining generated rubrics, generate using the migrated IDs. Required prompt/reference/rubric metadata and curriculum ordering must still be supplied explicitly.

RAG generation retains a supplied `query_id`; `data-build --rubrics` accepts a single RAG result or an array of generated records. Training preflight checks both cross-split IDs and exact message identity, so different row IDs cannot hide identical dialogue histories.
