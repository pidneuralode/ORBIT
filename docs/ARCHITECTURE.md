# Architecture

The package separates business workflows from resource access and GPU integration.

| Module group | Owns | Depends on |
| --- | --- | --- |
| `common` | Judge protocol/client, response parsing, rubric validation, JSON and prepared data | Standard library; optional API/Arrow imports at use time |
| `rubrics_generator` | Generation prompts/limits, retrieval, model adapters and backfill | `common`; optional neural libraries at model load time |
| `rubrics_rl` | Scoring, curriculum state/persistence and process execution | `common` |
| `rubrics_rl.managers` | Tensor rewards and strategy-specific curriculum selection | Core rewards/state, plus the optional verl/GPU runtime |
| `evaluation` | Fixed-response identity joins, score reports and judge agreement | `common`, reward result types |
| `config`, `runtime`, `cli` | Configuration resolution, object construction and command dispatch | The selected workflow modules |
| `vendor/verl` | Distributed training infrastructure and ORBIT registration bridges | Its separately provisioned training dependencies |

The package initializers do not import GPU managers. Importing ORBIT creates no API clients, model instances, executors or jobs. The previous flat imports redirect to the canonical module object so callers and dependency injection still refer to one implementation.

## Call flow

- Generation: CLI input → `Generator.generate` → prompt rendering → `Judge.judge` → generated-rubric validation.
- Retrieval generation: corpus/backend → `RagPipeline.retrieve` → reranker → evidence context → profile prompt → judge → generated-rubric validation.
- Data preparation: supplied records + optional generated criteria → exact identity join → contract validation → Parquet writer.
- Scoring: `make_scorer` → rubric validation → dialogue transcript → ordered item/chunk requests → criterion judgments → signed reward aggregation.
- Evaluation: fixed responses + prepared metadata → `EvaluationCase` → scoring → report; comparison requires aligned sample and rubric definitions.
- Training preview: config/base merge → resource expansion → strategy manager selection → Hydra arguments → `TrainingPlan`.
- Training execution: plan/input/runtime checks → verl driver → registration bridge → ORBIT tensor manager → shared reward executor/scorer → reward tensor → policy update.
- Resume: trainer checkpoint hook → train/validation manager state hooks → curriculum state and RNG restoration.

Configuration is read once per CLI command. Neural paths can be resolved later from the parsed backend object when semantic retrieval is selected. Transport credentials remain in environment variables and are accessed by the transport when it is constructed.

## Strategy boundaries

Sequential admission, gated admission and stochastic review are separate algorithms, not duplicated releases. They share query-state initialization, EMA helpers, persistence and worker execution. Their selection loops remain distinct so admission gates and stochastic weighting are visible to a reviewer.

The baseline uses a bounded in-flight request window. The shared process executor retains input ordering, timeout results and worker ownership. Other managers retain whole-batch scheduling when no limit is supplied. Ordered-batch scoring remains separate because it returns raw weighted points rather than the normalized item reward.

GPU managers accept the constructor surface required by verl. ORBIT supplies its own rubric scorer on that path; an arbitrary external custom reward callback is not a documented ORBIT manager workflow. The upstream trainer's general reward APIs remain in the vendored runtime.

## Source boundary

Paper plots, statistical analysis scripts, raw clinical records, physician annotations, output datasets, checkpoints and report images are not dependencies of the package and are not included. Runtime reward metrics and criterion agreement are retained because training and evaluation consume them.

A runtime data-format identifier is not a second implementation: checkpoints and embedding caches use it to reject incompatible serialized state. Dependency and build metadata are also retained. Source export copies the approved Git tree without history or local research evidence; it does not publish the result.

Configured invocations read a JSON/YAML document once, resolve `workflow` to an existing CLI command, then construct only its required service/model/training resources. Training path resolution precedes Hydra serialization and Parquet split preflight. The driver forwards only required reward environment variables to Ray; credentials remain out of configuration, launch arguments and printed runtime settings.

`common.data` offers explicit legacy identity migration and exact conversation identity checks. These protect plumbing consistency and do not implement historical clinical selection, semantic contamination analysis or privacy approval. External official benchmark sources and reporting boundaries are documented in [EVALUATION.md](EVALUATION.md).
