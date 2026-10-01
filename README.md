# ORBIT

**Open-ended rubric-based incremental training** for medical dialogue.

[Paper](https://arxiv.org/abs/2510.15859) · [Quick start](#installation-and-first-run)

ORBIT constructs case-specific criteria from expert-written rubric seeds, judges policy responses against those criteria, and uses the resulting rewards for reinforcement learning. Positive criteria identify desired actions; negative criteria identify mistakes to penalize. This makes the feedback explicit at the level of clinical decisions rather than relying solely on a holistic response score.

This repository focuses on the executable core: rubric generation and retrieval, response scoring, curriculum strategies, prepared-data interfaces, fixed-response evaluation, and integration with verl. Selected figures from the public paper and project page illustrate the method and reported results below. Research plotting code, paper-specific statistical analyses, raw annotation data and result files are outside the code release. Historical dialogue simulation and final dataset selection are not reconstructed here, and GPU experiment reproduction has not been established.

![ORBIT framework: dialogue construction, rubric-guided reinforcement learning, and retrieval-augmented rubric generation](https://raw.githubusercontent.com/pidneuralode/ORBIT/main/docs/static/assets/orbit-pipeline-1.png)

*Paper overview: construct dialogue queries, generate case-specific rubrics, and use criterion-level feedback to guide reinforcement learning. The data-construction stages shown in the figure describe the research pipeline; their historical implementation is not reconstructed in this code release.*

## Core workflow

1. **Generate rubrics.** Retrieve related cases and rubric candidates, rerank them, and render evidence for a generator. Alternatively, supply evidence directly. The generated criteria carry integer importance weights.
2. **Prepare data.** Attach generated criteria to supplied dialogue records by query identity. Provide references, prompt history, disjoint train/validation identities and any curriculum ordering explicitly.
3. **Score responses and train.** Judge each criterion against the dialogue and response. Aggregate signed rewards, optionally select criteria through a curriculum, and pass rewards to GRPO through the retained verl runtime.
4. **Evaluate fixed responses.** Join supplied responses to prepared data, score them with the same engine, and optionally compare judge decisions. This interface does not perform policy inference.

## Paper results

![HealthBench-Hard comparison of ORBIT and baseline models across clinical themes and evaluation axes](https://raw.githubusercontent.com/pidneuralode/ORBIT/main/docs/static/assets/performance-compare-1.png)

*Results reported in the [paper](https://arxiv.org/abs/2510.15859), grouped by clinical theme and evaluation axis. See the paper for the model, dataset and judge settings; these figures are not new reproduction results from this code release.*

## Repository layout

```text
src/orbit/
  common/              # Judge transport, rubric/data contracts, JSON I/O
  rubrics_generator/   # Generation, evidence retrieval, neural adapters, backfill
  rubrics_rl/          # Rewards, curriculum state, process execution, trainer launch
    managers/          # Optional verl tensor managers for each reward strategy
  evaluation/          # Fixed responses and aligned criterion reports
  config.py            # Parse task settings and resolve required resources
  runtime.py           # Construct the configured scoring engine
  cli.py               # Command orchestration
vendor/verl/           # Retained GPU runtime; ORBIT registration bridges
configs/
  training.json        # Shared recovered training settings
  generation.json      # Shared generator service settings
  experiments/         # Small files containing strategy/profile differences
examples/              # Synthetic input and recorded judge responses
```

Each algorithm has one implementation. Previous flat Python import paths are thin aliases to the same module objects; they contain no alternate implementation. Internal code and the documentation use the grouped modules. The vendored runtime remains separate to preserve its dependency and licensing boundary. See [architecture and call flow](docs/ARCHITECTURE.md).

## Installation and first run

Use Python 3.10 or later in an isolated environment:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

orbit score --config configs/offline.json \
  --input examples/synthetic_case.json \
  --recorded examples/recorded-judge.json
```

The example uses synthetic inputs and recorded judge responses. It makes no service calls. The output contains criterion judgments, awarded points and the normalized score. The core package has no third-party runtime dependencies.

| Extra | Needed for |
| --- | --- |
| `.[api]` | An OpenAI-compatible judge or generator |
| `.[config]` | YAML configuration and one configured invocation |
| `.[data]` | Reading and writing training Parquet |
| `.[models]` | Local Transformers embedding and reranking models |
| `.[dev]` | Contributor tooling |

The optional GPU trainer has its own dependencies; install it in a compatible training environment as described in [training setup](docs/TRAINING.md).

## Run a configured workflow

```sh
python -m pip install -e '.[config]'
orbit run --config configs/offline-generation.yaml
```

This example uses synthetic inputs and recorded responses. For a service, install `.[api,config]`, edit `configs/generate.yaml` or set its generator service/input/output environment variables, then run `orbit run --config configs/generate.yaml`. Credentials stay in `ORBIT_GENERATOR_API_KEY`. File paths are relative to the YAML directory. [Configuration](docs/CONFIGURATION.md) lists supported workflow options.

Configure `configs/train.yaml` and run `orbit run --config configs/train.yaml` to preview training. Add `--execute` in a provisioned GPU environment to launch. Relative data/model/output/resume paths resolve against the config directory, so shared recipes need no personal absolute paths. For another strategy, use its scoring policies, explicit rubric ordering and worker mode described in the training guide.

## Generate rubrics and inspect evidence

Direct generation accepts `query`, `top_cases_text` and `candidate_rubrics_text`. A supplied `query_id` is retained in its output:

```sh
orbit generate --config configs/offline.json \
  --input examples/generation-context.json \
  --recorded examples/recorded-generation.json --output generated.json
```

For retrieval, inspect the evidence first:

```sh
orbit rag --config configs/offline.json \
  --input examples/rag-query.json --corpus examples/rag-corpus.json
```

Add `--generate` with recorded responses or service settings to generate from that evidence. The default backend uses the ranking supplied by the corpus. Semantic search is opt-in through explicitly configured embedding and optional reranker models. `supplementary` previews missing cases; `--generate` produces backfill patches and a merged result.

The Python entry points live in `orbit.rubrics_generator.generation`, `rag`, `retrieval`, `model_backends` and `supplementary`. [Configuration](docs/CONFIGURATION.md) describes retrieval profiles and model options.

## Prepare data and evaluate responses

A training record contains `prompt`, `data_source`, `reward_model` reference metadata, and `extra_info` with `prompt_history`, weighted `rubrics` and a stable `query_id`. Curriculum strategies require an explicit permutation of rubric positions in `sorted_rubric_indices`.

```sh
python -m pip install -e '.[data]'
orbit data-build --input examples/prepared-train.json \
  --output synthetic-train.parquet --curriculum
orbit data-build --input examples/prepared-val.json \
  --output synthetic-val.parquet --curriculum

orbit evaluate --config configs/offline.json \
  --input synthetic-val.parquet --responses examples/prepared-responses.json \
  --recorded examples/recorded-judge.json
```

Use `orbit data-build --input examples/prepared-train.json --rubrics generated.json --output generated-train.parquet` to attach the generation result by identity. When replacing rubrics, also supply curriculum ordering that matches the resulting list. The builder does not infer difficulty or select dataset splits. [The data contract](docs/DATA.md) covers JSONL, custom prompt keys and required metadata.

`score` accepts one `{solution_str, extra_info}` object. `evaluate` also accepts arrays of `{sample_id, solution_str, extra_info}` without Parquet. `--compare-recorded` compares a second judge's criterion decisions, reporting agreement and Cohen's kappa. These are core evaluation outputs; paper analyses and figure production are not included. Use `--output` to save a result and `--overwrite` to replace an existing file explicitly.

Research benchmark comparisons reference OpenAI **simple-evals**. A [pinned acquisition script and evaluation guide](docs/EVALUATION.md) keep its source, licenses and metric boundary explicit. `orbit evaluate` is the configured core scoring interface and must not be labeled the full official HealthBench evaluation. Real case–rubrics pairs and scale subsets remain outside this candidate pending the [data release decisions](docs/DATA_RELEASE.md); only synthetic fixtures are bundled.

## Configure a service

Install `.[api]` and set the endpoint, model and credential in your local environment. A service config needs only:

```json
{
  "judge": {
    "base_url": "${ORBIT_JUDGE_BASE_URL}",
    "model": "${ORBIT_JUDGE_MODEL}"
  }
}
```

The credential defaults to the environment variable `ORBIT_JUDGE_API_KEY`. Set `judge.api_key_env` to use another name. The same transport interface serves scoring and generation; generator recipes explicitly choose their own service settings.

For example, after setting the judge variables:

```sh
orbit score --config configs/experiments/baseline.json --input your-case.json
```

Training paths in that config are resolved only by the training command. Optional model imports and client creation occur when their workflow actually needs them.

## Training strategies and configuration

Common experiment settings live once in `configs/training.json`. A strategy file contains only its differences, with one relative base reference:

```json
{
  "extends": "../training.json",
  "variant": "curriculum_admission",
  "scoring": {
    "normalization_policy": "signed_fallback",
    "criteria_policy": "boolean_only"
  },
  "training": {
    "trainer.experiment_name": "curriculum_admission",
    "reward_model.launch_reward_fn_async": false
  }
}
```

The loader merges each section once; it does not recursively inherit configurations. `variant` selects the reward manager from one strategy table, so the manager name need not be repeated in every recipe. Explicit trainer overrides remain available. The shared settings are recovered engineering choices, not verified final paper parameters.

| Strategy | Selection and execution |
| --- | --- |
| `baseline` | Per-response rubric scoring with a bounded process request window |
| `adaptive_all` | Whole-batch all-rubric scoring through a shared judge/scorer |
| `curriculum` | Sequential admission to an active criterion window |
| `curriculum_admission` | Admission gated by active-window mastery, with an admission limit |
| `stochastic_review` | Review of mastered criteria and EMA-based soft weighting |

`ordered_batch` is a separate raw-score evaluation path, with no training recipe supplied. Scoring prompt text is configurable through `scoring.prompt_template`; the default item prompt has one definition. The former InfoBench preset contained the same prompt and does not require a separate implementation.

In a compatible GPU environment, set approved absolute paths and judge variables, then preview:

```sh
export ORBIT_TRAIN_FILE=/absolute/path/train.parquet
export ORBIT_VAL_FILE=/absolute/path/val.parquet
export ORBIT_MODEL_PATH=/absolute/path/model
export ORBIT_OUTPUT_DIR=/absolute/path/run
# Set ORBIT_JUDGE_BASE_URL, ORBIT_JUDGE_MODEL and ORBIT_JUDGE_API_KEY locally.
orbit train --config configs/experiments/baseline.json
```

Add `--execute` to launch training. The launcher does not download weights or start service servers. [Training setup](docs/TRAINING.md) explains checkout discovery, Ray environments and curriculum checkpoints. [Configuration](docs/CONFIGURATION.md) covers scoring, retrieval, generator limits and migration of existing settings.

## Scope and release status

Real datasets, model weights and service access are not bundled. Service-backed scoring, neural retrieval, distributed training and published-result reproduction require verification in the target environment. Package metadata, dependency compatibility bounds and persistent-data format identifiers remain where the runtime needs them; implementation and experiment names do not carry release suffixes.

Author-owned ORBIT code is released under the [MIT license](LICENSE). Third-party code retains its own licensing: the [verl Apache 2.0 license](vendor/verl/LICENSE) and [OpenAI prompt material MIT license](docs/licenses/simple-evals-MIT.txt) remain intact. [NOTICE](NOTICE) describes these boundaries. Research data and weights require separate source/data rights decisions; the code license does not relicense them. See [remaining limits](docs/GAPS.md) and [source export](docs/RELEASE.md).
