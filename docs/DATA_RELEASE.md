# Dataset release decisions

The public candidate includes synthetic fixtures only. Real case–rubrics pairs,
scale subsets, physician annotations, service logs and benchmark examples are
not included. Generating a rubric does not remove the source case's conditions
or establish that its contents are suitable for publication.

| Source | Evidence | Remaining decision |
| --- | --- | --- |
| HealthBench | [Official evaluation repository](https://github.com/openai/simple-evals), MIT-listed; physician-written rubrics | Respect the [no-online-example request](https://openai.com/index/healthbench/); obtain locally and exclude Hard cases from training/RAG seeds |
| DoctorAgent-RL / MTMedDialog | [Dataset description](https://github.com/JarvisUSTC/DoctorAgent-RL/blob/main/DATASET.md); repository declares [Apache 2.0](https://github.com/JarvisUSTC/DoctorAgent-RL/blob/main/LICENSE) | Verify the exact acquired revision, underlying IMCS21/CHIP-MDCFNPC/MedDG conditions, transformation lineage and privacy review |
| ReMeDi | [Source README](https://github.com/yanguojun123/Medical-Dialogue#7-license) states that all resources use MIT | Verify the acquired release, subset/translation lineage, retained attribution and privacy review for the proposed processed files |
| ORBIT-generated rubrics / mixed-scale pairs | Derived research artifacts | Confirm source case rights, generator/service output conditions, final selection and privacy; specify attribution and a dataset license explicitly |

For each proposed release, complete a separate data card with:

- Dataset name, source URLs and fixed revisions/checksums; rights evidence per component.
- Intended task, language/domain, record count, schema and train/validation/evaluation roles.
- Exact transformations, translation/generator settings and selection procedure; mark unrecovered stages.
- Stable source and query identities, duplicate policy and exact/semantic cross-split overlap checks.
- Separate privacy review of dialogue, references, annotations, rubric text and metadata; document approved removals without publishing removed content.
- Owner approval of exact file manifest and dataset license, required notices, restrictions and release channel.

The builder in [DATA.md](DATA.md) validates the core schema and aligns generated
rubrics by identity. It does not recreate historical dialogue simulation,
clinical selection or an unrecovered train split. Prepare locally acquired,
approved cases with explicit prompts/references and IDs, generate criteria,
then use `data-build` to produce Parquet. For curriculum training, provide
reviewed ordering corresponding to the generated criterion list. Never invent
difficulty ordering to make a configuration pass.
