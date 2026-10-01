# Configuration

A config is a JSON object or a YAML mapping whose sections correspond to workflow responsibilities. YAML requires `pip install -e '.[config]'`; JSON needs no extra dependency. Omitted values use the Python configuration classes' defaults. Secrets are environment variables, not configuration values. YAML uses a safe loader and rejects duplicate keys, executable tags, nonfinite numbers and non-JSON value types.

| Section | Common options |
| --- | --- |
| `judge` | `base_url`, `model`, `api_key_env`, `temperature`, `max_tokens`, `timeout`, `max_concurrency` |
| `scoring` | `mode`, `chunk_size`, `max_workers`, `request_attempts`, `failure_policy`, `normalization_policy`, `criteria_policy`, `prompt_template` |
| `generation` | `max_criteria`, `minimum_points`, `maximum_points` |
| `retrieval` | `profile`, `k_cases`, `k_rubrics`, `k_rerank`, `k_examples`, `case_char_limit` |
| `backend` | `mode`, `embedding`, optional `reranker` |
| `training` | Direct Hydra key/value mapping |

`variant` selects the experiment's reward strategy. Its mapping to trainer manager names is defined once in `rubrics_rl/variants.py`. `status` is optional provenance metadata; checked-in settings remain unconfirmed engineering recipes.

## Shared settings

`extends` references one base JSON or YAML file relative to the config file, not the current working directory. Sections merge by key, and child values win. Lists and scalar values replace inherited values. A base cannot itself extend another file; recursive inheritance is intentionally absent.

Training strategy files share `configs/training.json`. Retrieval profiles share `configs/generation.json`. Edit the shared recipe to change common service/path/trainer settings, or place the differing fields in an experiment file. No environment-variable aliases or implicit model locations are needed.

For a config placed in the repository root:

```json
{
  "extends": "configs/training.json",
  "variant": "curriculum",
  "training": {
    "trainer.experiment_name": "my-run",
    "reward_model.launch_reward_fn_async": false
  }
}
```

This demonstrates the merge; use the curriculum example's scoring policies for that strategy. An explicit `reward_model.reward_manager` is still accepted if it matches `variant`; otherwise the launcher supplies it.

## Judge and scoring

Transport retries default to zero; item scoring defaults to two request attempts. Chunk scoring can retry a JSON-object request and individually score missing/duplicate criterion IDs. `max_workers` limits scoring work, while `judge.max_concurrency` limits service calls through a shared transport. `failure_policy` chooses zero rewards or safe exceptions.

The default normalized reward divides awarded points by total positive points. `signed_fallback` uses absolute negative points when no positive denominator exists. Negative-weight criteria describe mistakes: meeting them subtracts from the reward. `criteria_policy` chooses permissive recovered truth parsing (`legacy`) or strict boolean decisions (`boolean_only`). `ordered_batch` returns raw points and uses its own ordered-array prompt.

Use `scoring.prompt_template` to supply an item prompt. It must preserve `<<conversation>>` and `<<rubric_item>>` placeholders for substitution. The default item and chunk templates live in `rubrics_rl/prompts.py`.

## Retrieval profiles

| Profile | Actual distinction |
| --- | --- |
| `standard` | Case examples and a reranked rubric candidate pool |
| `hard` | Different generation system prompt; supplied hard-case corpus |
| `no_hard` | Standard prompt/retrieval; externally filtered corpus |
| `consensus` | Standard prompt with five displayed examples in its recipe; supplied consensus corpus |
| `no_rag` | No retrieved examples or candidate pool |
| `supp_no_rag` | Supplementary backfill prompt without retrieval |

These profiles do not calculate difficulty thresholds or select a clinical corpus automatically. Pre-ranked retrieval consumes the corpus's supplied order/scores. Semantic retrieval requires an explicitly supplied embedding model and optionally a reranker:

```json
{
  "backend": {
    "mode": "semantic",
    "embedding": {"model_path": "${ORBIT_EMBEDDING_PATH}"},
    "reranker": {"model_path": "${ORBIT_RERANKER_PATH}"}
  }
}
```

Model settings include device, dtype, batch size, token limit, pooling, local-file loading and an overridable retrieval `instruction`. Their defaults preserve the recovered neural prompt/pooling behavior. Optional libraries and weights load only when the backend runs. Vector caches require a caller-supplied `encoder_id` reflecting the actual model/settings; they do not silently identify or download models.

## Existing config migration

- Move the contents of `training.overrides` directly into `training`; the old wrapper remains accepted for standalone configs.
- Omit `schema_version`; the loader accepts and discards the former supported value at the input boundary.
- Named scoring presets had identical text. Remove `scoring.prompt`, or use `scoring.prompt_template` for an actual custom template. Former names remain accepted at the read boundary.
- The admission-gated curriculum is named `curriculum_admission`; use its config and manager registration rather than a release-suffixed name.
- The former InfoBench recipe used the baseline manager and identical prompt. Use the baseline recipe for that behavior.

`check-config --offline` validates settings without resolving external resources. Recorded-response workflows also leave service paths unresolved. Training execution uses resolved absolute input/model/output paths, real Parquet and a separately installed trainer. Output replacement, secret handling and checkpoint ownership remain explicit.

## One configured invocation

`orbit run --config task.yaml` invokes the existing command named by `workflow.command`:

```yaml
variant: generation
workflow:
  command: generate
  input: ./contexts.json
  recorded: ./recorded-generation.json
  output: ./generated.json
```

Workflow file paths and training data/model/output/resume paths resolve relative to the invoked config's directory, including inherited values. Explicit `./`, `../` and `~` retrieval model paths resolve there; a Hugging Face model ID retains its meaning. `${UPPERCASE_NAME}` supplies resources from the environment. Credentials remain in the named API-key environment variable.

Hydra training settings use dotted keys. Lists and option mappings are serialized using Hydra syntax. Prefix an added key with `+`, or use `++` to add/override an optional field. For example, `++reward_model.reward_kwargs: {active_window_size: 7}` configures a curriculum manager; only supply parameters accepted by the selected manager. Use that strategy's scoring policies and set `reward_model.launch_reward_fn_async: false` for stateful curricula.

Supported commands are `generate`, `rag`, `supplementary`, `score`, `evaluate`, `data-build`, `data-validate` and `train`. Options reuse CLI names with underscores, such as `compare_recorded`, `prompt_key`, `identity_namespace`, `generate`, `curriculum` and `overwrite`. Only options applying to the chosen command are accepted. A workflow runs one command; sequencing remains explicit. Training previews by default and requires `orbit run --config train.yaml --execute` to start. Check `configs/offline-generation.yaml`, `configs/generate.yaml` and `configs/train.yaml` for concise examples.
