#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run Qwen/Qwen3-VL-8B-Instruct on the OSWorld full set through the existing
Nemo-GYM nemotron_agent rollout pipeline.

This wrapper only sets model/run defaults, then delegates to run_rollouts.sh.
It does not change the OSWorld task set or agent server implementation.

Common overrides:
  RUN_MODE=auto|local|slurm
  WAIT_FOR_SLURM=1
  MODEL_SOURCE=/path/to/Qwen3-VL-8B-Instruct
  ALLOW_REMOTE_MODEL_SOURCE=true
  HF_HOME=/path/to/hf/cache
  NUM_SAMPLES_IN_PARALLEL=12
  AGENT_CONCURRENCY=12

Examples:
  RUN_MODE=slurm WAIT_FOR_SLURM=1 \
    MODEL_SOURCE=/lustre/path/Qwen3-VL-8B-Instruct \
    bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_qwen3_vl_8b_rollouts.sh

  RUN_MODE=local LIMIT=1 MODEL_SOURCE=/lustre/path/Qwen3-VL-8B-Instruct \
    bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_qwen3_vl_8b_rollouts.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen3_vl_8b_instruct_responses_agent_local}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_SOURCE="${MODEL_SOURCE:-${QWEN3_VL_8B_MODEL_SOURCE:-Qwen/Qwen3-VL-8B-Instruct}}"
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/tmp/qwen3_vl_8b_instruct}"
export ALLOW_REMOTE_MODEL_SOURCE="${ALLOW_REMOTE_MODEL_SOURCE:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export MAX_IMAGES="${MAX_IMAGES:-3}"
export DATASET="${DATASET:-responses_api_agents/nemotron_agent/data/osworld_test_all_qwen3vl_235b_a22b_instruct.jsonl}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.9}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export EXPECTED_ACCURACY="${EXPECTED_ACCURACY:-0.339}"
export EXTRA_VLLM_PIP_PACKAGES="${EXTRA_VLLM_PIP_PACKAGES:-qwen-vl-utils}"

exec bash "${SCRIPT_DIR}/run_rollouts.sh"
