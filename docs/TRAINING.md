# Training checkout and runtime

The wheel contains the ORBIT reusable package, not the vendored GPU trainer.
Use the checked-out candidate's `vendor/verl` for the project-specific reward
manager integration. Installing a different upstream verl does not establish
compatibility with these variants.

In a separate training environment, install the ORBIT package with its optional
API and data dependencies and the vendored trainer in editable mode:

```bash
python -m pip install -e '/path/to/ORBIT-candidate[api,data,config]'
python -m pip install -e /path/to/ORBIT-candidate/vendor/verl
export ORBIT_REPO_ROOT=/path/to/ORBIT-candidate
```

Choose CUDA/PyTorch/vLLM/Ray versions for the actual GPU runtime following the
vendored dependency declarations. No GPU runtime combination has been certified
by the offline checks, and this document does not invent an experimental version
lock. Configure data/model/output paths and judge environment
variables described in the main README; `orbit train --config ...` previews the
command and `--execute` explicitly starts it.

An installed wheel can locate a checkout with `ORBIT_REPO_ROOT` (candidate root)
or `ORBIT_VERL_ROOT` (directory containing `verl/trainer/main_dapo.py`). The Python
API also accepts `build_training_plan(..., repo_root=Path(...))`. Without an
available trainer checkout, preview fails explicitly. Execution checks the
configuration file, input/model paths and an explicitly supplied resume checkpoint
before starting a subprocess. The launcher adds the actual installed ORBIT package
location and selected trainer checkout to `PYTHONPATH`, and sets `ORBIT_CONFIG`.

## Stage, restart and Ray workers

Stage schedules remain explicit in the ORBIT configuration. Experiment profiles
are unconfirmed engineering variants; none is identified as the final paper run.
To restart, set the existing Hydra `trainer.resume_mode` and, when needed,
`trainer.resume_from_path` to an existing checkpoint. Relative paths resolve against the config file. Preview exposes
these overrides. The launcher does not select a checkpoint or derive a stage
schedule from the paper.

Ray reward workers must be able to import the same ORBIT package and read the
same configuration path, model/data paths, and credential environment variable.
The driver subprocess inherits the caller's environment and explicitly forwards the required reward service variables to its Ray task, including attachment to an existing cluster. Runtime environment mappings are not printed. Install the same ORBIT package on every node and provide shared file paths. Credentials are supplied through environment variables; never put them into Hydra overrides.
The offline tests validate
launch planning and environment assembly, not remote Ray propagation or GPU runs.

## Data preflight

Execution reads both configured training and validation Parquet files through
`orbit.data.read_parquet` before starting the trainer. The required prompt,
reward metadata, prompt history, rubric and explicit unique query IDs must be
present. IDs must be unique across files within each split and disjoint between
training and validation. The preflight also rejects identical dialogue histories across splits even when IDs differ. This exact equality check does not establish absence of semantic contamination. Curriculum variants additionally require the rubric
position permutation `sorted_rubric_indices`; no ordering is invented. JSON or
JSONL preparation inputs must first be validated and exported using the data
preparation command. This check reads the datasets and therefore requires the
optional data dependency and enough host memory for the configured records.
Synthetic Parquet fixtures verify this preflight without research data or GPU.

## Curriculum checkpoint boundary

All three stateful managers implement validated `state_dict`/`load_state_dict`. The trainer stores `curriculum.json` atomically in the same `global_step_*` directory as actor/dataloader checkpoints, before updating the latest-checkpoint marker. Data-format identity, manager class, selection parameters, per-query EMA/status and stochastic RNG are checked before loading; train and validation manager states are separate.

These managers explicitly reject `reward_model.launch_reward_fn_async=true`: Ray remote reward copies would otherwise lose updates from the driver-owned state. The three engineering templates set it to false. This is an explicit supported-mode restriction, not proof of prior experiment equivalence. Missing state in an older checkpoint emits a warning and keeps fresh curriculum state; such a resume is not a faithful continuation. Stochastic review retains the shared global RNG by default; an independently seeded RNG is supported at manager construction. Full distributed/GPU save/resume remains untested.


The admission-gated strategy uses the canonical manager identity
`DAPORewardManagerCurriculumAdmission`. A checkpoint from a differently named
historical manager requires an explicit migration before resume; the loader does
not silently rewrite its identity or selection parameters.

Manager-specific options use `++reward_model.reward_kwargs` in the training mapping. The reward loader applies them to driver and asynchronous instances; configured maximum response length and overlong penalties are preserved in both paths. CPU tests exercise this loader with mocked managers, not tensor/GPU execution.
