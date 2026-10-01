# Evaluation sources and metric boundaries

ORBIT's research manuscript references [OpenAI simple-evals](https://github.com/openai/simple-evals)
for HealthBench-Hard evaluation. A recovered research copy contains additional
local-model samplers and batch/inference variants, and has no Git metadata. Its
historical upstream revision and the exact final experiment entry point have not
been established. That copy is not bundled in this release candidate.

For a fixed official reference, obtain the upstream code separately:

```sh
python scripts/fetch_simple_evals.py
```

The script checks out commit `652c89d0ca9df547706735883097e9537d40dc47`
from the official repository into ignored `.external/simple-evals`. It retains
the upstream MIT license, refuses an existing destination, and does not install
dependencies, download evaluation data or call models. This pin is the source
reviewed during release preparation; it is not a claim about the historical
experiment revision. Follow that checkout's README for optional sampler/eval
dependencies and service setup. Upstream ceased updating new model results in
July 2025 but retains the reference implementations.

`orbit evaluate` scores explicitly supplied policy responses with the configured
ORBIT reward engine. It does not perform policy inference or implement the full
official HealthBench reporting protocol. Recovered HealthBench scoring divides
signed achieved points by positive possible points, returns no score when that
denominator is zero, and clips the aggregate mean to [0, 1]. ORBIT's
configured reward policies and report aggregation differ, so an ORBIT report
must not be labeled an official HealthBench-Hard score. Use the pinned reference
for benchmark comparisons and record policy sampler, grader model, prompt,
sampling settings, subset, source revision and response identity alignment.

# Data rights and benchmark integrity

The [simple-evals code license](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/LICENSE)
does not establish one uniform license for every dataset used by its evaluators.
Check each source release and keep its attribution and applicable conditions.
The official README lists HealthBench, SimpleQA and BrowseComp as MIT, while
MGSM is CC BY 4.0. Its summary links are not sufficient evidence for all other
dataset redistribution rights.

HealthBench rubrics are physician-authored upstream criteria. ORBIT-generated
rubrics for other cases are a separate artifact. OpenAI
[requests that HealthBench examples not be exposed as text or images online](https://openai.com/index/healthbench/)
to reduce benchmark contamination. This is a benchmark-integrity request,
distinct from the license. Use the official local acquisition path and source
identities instead of committing a crawlable copy of the benchmark.

Keep evaluation cases out of training and RAG seed selection. A validation set
used to tune checkpoints/hyperparameters is not an untouched final evaluation
set. Record exact and semantic overlap checks independently; disjoint row IDs
alone do not establish disjoint conversations.
