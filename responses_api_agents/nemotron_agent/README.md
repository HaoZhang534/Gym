# Nemotron OSWorld Agent

This response-api agent runs real OSWorld desktop tasks with the Nemotron-VL
agent vendored under this directory. It also vendors the OSWorld desktop
environment package and task metadata needed by the default datasets, so it does
not require a sibling `osworld` checkout.

Example:

```bash
cd nemo_chunking_support/Nemo_GYM
UV_CACHE_DIR=/tmp/$USER/nemo_gym_uv uv run ng_run \
  "+config_paths=[responses_api_agents/nemotron_agent/configs/nemotron_agent.yaml]"

UV_CACHE_DIR=/tmp/$USER/nemo_gym_uv uv run ng_collect_rollouts \
  +agent_name=nemotron_agent \
  +input_jsonl_fpath=responses_api_agents/nemotron_agent/data/osworld_test_test.jsonl \
  +output_jsonl_fpath=results/osworld_nemotron_rollouts.jsonl \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

The input JSONL contains real OSWorld task IDs and local task config paths. Each
rollout writes `traj.jsonl`, screenshots, `recording.mp4` when available, and
`trajectory.gif` under `result_dir`.

## One-click local vLLM collection

From the project root, this starts local vLLM, starts the Nemo_GYM server, then
collects rollouts:

```bash
bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_rollouts.sh
```

By default the wrapper submits the existing Slurm job when `sbatch` is
available outside an allocation, and runs directly when already inside a Slurm
job. Useful overrides:

```bash
LIMIT=1 RUN_MODE=local \
  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_rollouts.sh

RUN_MODE=slurm WAIT_FOR_SLURM=1 NUM_SAMPLES_IN_PARALLEL=12 \
  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_rollouts.sh
```

Artifacts are written under `RUN_ROOT`, defaulting to
`results/nemotron_agent_rollouts_<timestamp>`.

For RFC1 subtask 2 Qwen/Qwen3-VL-8B-Instruct full-set evaluation, use the
model-specific wrapper. It reuses the same Nemo-GYM OSWorld rollout path and
overrides the served model name at collection time:

```bash
RUN_MODE=slurm WAIT_FOR_SLURM=1 \
  MODEL_SOURCE=/path/to/Qwen3-VL-8B-Instruct \
  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_qwen3_vl_8b_rollouts.sh
```

If `MODEL_SOURCE` is left unset, vLLM receives the Hugging Face model id
`Qwen/Qwen3-VL-8B-Instruct` directly and `HF_HUB_OFFLINE=0` by default. Use a
local model path for reproducible cluster runs.

## Interactive debugging

Reserve one debug node and container:

```bash
bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/reserve_debug_node.sh
```

After it prints `attach: bash <RUN_ROOT>/attach.sh`, attach three times for the
fast debug workflow. In the first shell, start vLLM and leave it running:

```bash
bash <RUN_ROOT>/attach.sh
source <RUN_ROOT>/debug_env.sh
bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh vllm
```

In the second shell, run the Gym/agent server in the foreground:

```bash
bash <RUN_ROOT>/attach.sh
source <RUN_ROOT>/debug_env.sh
bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh gym
```

After the Gym server prints `All 1 / 1 servers ready`, use a third attached
shell to trigger one rollout:

```bash
bash <RUN_ROOT>/attach.sh
source <RUN_ROOT>/debug_env.sh
bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh collect
```

Breakpoints in `nemotron_agent` code are interactive in the Gym server shell.
After editing agent code, restart only the `gym` command; keep `vllm` running.
The debug scripts reuse existing Gym virtualenvs by default; set
`NEMOTRON_REUSE_CLI_VENV=false` or
`NEMO_GYM_SKIP_SERVER_VENV_IF_PRESENT=false` if dependencies changed.
