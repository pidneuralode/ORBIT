# Running ORBIT

The [README](../README.md) provides installation, offline examples, data preparation, scoring and training commands. Use the [configuration guide](CONFIGURATION.md) for judge transport, retrieval profiles, model options and shared experiment settings.

`score` accepts `{solution_str, extra_info}`. `generate` accepts a context or context array. `rag` accepts `{query}` and an explicit corpus for retrieval profiles. `supplementary` accepts case/result arrays and previews missing cases before explicit generation. `evaluate` accepts fixed-response case arrays or prepared Parquet with `--responses`.

Prepared records and generated rubrics join by query identity. See [data contracts](DATA.md). For the separately provisioned GPU environment, worker paths and resume, see [training setup](TRAINING.md).

The CLI resolves only resources used by a command. Recorded judges make no service calls; pre-ranked evidence preview does not initialize a judge. Semantic retrieval loads its configured models when run. Training previews a launch plan; `--execute` starts it explicitly. Existing output files require `--overwrite`.

`orbit run --config task.yaml` reads `workflow.command` and its file/options mapping, then uses the same command implementations above. File paths resolve against the configuration directory. Install `.[config]` for YAML. Start with `configs/offline-generation.yaml`; configure `configs/generate.yaml` for an explicit service or `configs/train.yaml` for a training preview. External models, data, service credentials and compatible GPU dependencies must still be supplied.

Use [EVALUATION.md](EVALUATION.md) for official benchmark source/metric boundaries and [DATA_RELEASE.md](DATA_RELEASE.md) before preparing any real dataset release.
