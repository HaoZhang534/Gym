#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run nemotron_agent OSWorld rollouts end to end.

This wrapper starts the local vLLM server, starts the Nemo_GYM server, and
collects rollouts by delegating to slurm/collect_local_nano_omni_inner.sh.

Usage:
  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_rollouts.sh

Common overrides:
  RUN_MODE=auto|local|slurm    auto chooses slurm outside an allocation when sbatch exists
  WAIT_FOR_SLURM=1             block until the submitted Slurm job finishes
  RUN_ROOT=/path/to/run        output root; defaults under Nemo_GYM/results
  DATASET=path.jsonl           input JSONL relative to Nemo_GYM root or absolute
  LIMIT=1                      optional number of tasks to collect
  NUM_SAMPLES_IN_PARALLEL=12   rollout collection concurrency
  AGENT_CONCURRENCY=12         nemotron_agent server concurrency
  MODEL_SOURCE=/path/model     source model directory copied to LOCAL_MODEL_PATH
  LOCAL_MODEL_PATH=/tmp/model  node-local model directory
  VLLM_PORT=8000               local vLLM port
  TP=8                         vLLM tensor parallel size
  MAX_STEPS=100                OSWorld max steps per task

Examples:
  LIMIT=1 RUN_MODE=local bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_rollouts.sh
  RUN_MODE=slurm WAIT_FOR_SLURM=1 bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/run_rollouts.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${NEMO_GYM_ROOT}/../.." && pwd)}"

INNER_SCRIPT="${INNER_SCRIPT:-${SCRIPT_DIR}/slurm/collect_local_nano_omni_inner.sh}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/slurm/collect_local_nano_omni_sbatch.sh}"

RUN_MODE="${RUN_MODE:-auto}"
WAIT_FOR_SLURM="${WAIT_FOR_SLURM:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-nemotron_agent_rollouts}"
RUN_ROOT="${RUN_ROOT:-${NEMO_GYM_ROOT}/results/${RUN_NAME}_${RUN_STAMP}}"

DATASET="${DATASET:-responses_api_agents/nemotron_agent/data/osworld_test_all_nvidia_nemotron_nano_12b_v2_vl.jsonl}"
LIMIT="${LIMIT:-}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-12}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-${NUM_SAMPLES_IN_PARALLEL}}"

export REPO_ROOT
export NEMO_GYM_ROOT
export RUN_ROOT
export RUN_NAME
export RUN_STAMP
export DATASET
export LIMIT
export NUM_SAMPLES_IN_PARALLEL
export AGENT_CONCURRENCY

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${description} not found: ${path}" >&2
    exit 2
  fi
}

dataset_path() {
  if [[ "${DATASET}" = /* ]]; then
    printf '%s\n' "${DATASET}"
  else
    printf '%s/%s\n' "${NEMO_GYM_ROOT}" "${DATASET}"
  fi
}

resolve_run_mode() {
  case "${RUN_MODE}" in
    auto)
      if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        printf 'local\n'
      elif command -v sbatch >/dev/null 2>&1; then
        printf 'slurm\n'
      else
        printf 'local\n'
      fi
      ;;
    local|slurm)
      printf '%s\n' "${RUN_MODE}"
      ;;
    *)
      echo "Invalid RUN_MODE=${RUN_MODE}; expected auto, local, or slurm" >&2
      exit 2
      ;;
  esac
}

print_config() {
  cat <<EOF
run_mode=${MODE}
repo_root=${REPO_ROOT}
nemo_gym_root=${NEMO_GYM_ROOT}
run_root=${RUN_ROOT}
run_name=${RUN_NAME}
dataset=${DATASET}
limit=${LIMIT:-<none>}
num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL}
agent_concurrency=${AGENT_CONCURRENCY}
inner_script=${INNER_SCRIPT}
sbatch_script=${SBATCH_SCRIPT}
EOF
}

require_file "${INNER_SCRIPT}" "inner rollout script"
require_file "$(dataset_path)" "dataset"

MODE="$(resolve_run_mode)"
mkdir -p "${RUN_ROOT}"
print_config

case "${MODE}" in
  local)
    echo "Starting local end-to-end rollout collection..."
    exec bash "${INNER_SCRIPT}"
    ;;
  slurm)
    require_file "${SBATCH_SCRIPT}" "sbatch rollout script"
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "RUN_MODE=slurm requested, but sbatch is not available" >&2
      exit 2
    fi

    mkdir -p "${NEMO_GYM_ROOT}/results/slurm_logs"
    submit_args=(
      --parsable
      --export=ALL
      --output="${NEMO_GYM_ROOT}/results/slurm_logs/nemo-osw-local-%j.out"
      --error="${NEMO_GYM_ROOT}/results/slurm_logs/nemo-osw-local-%j.err"
    )
    if [[ "${WAIT_FOR_SLURM}" == "1" || "${WAIT_FOR_SLURM}" == "true" ]]; then
      submit_args+=(--wait)
    fi

    echo "Submitting Slurm rollout job..."
    submit_output="$(sbatch "${submit_args[@]}" "${SBATCH_SCRIPT}")"
    job_id="${submit_output%%$'\n'*}"
    job_id="${job_id%%;*}"
    echo "job_id=${job_id}"
    echo "slurm_stdout=${NEMO_GYM_ROOT}/results/slurm_logs/nemo-osw-local-${job_id}.out"
    echo "slurm_stderr=${NEMO_GYM_ROOT}/results/slurm_logs/nemo-osw-local-${job_id}.err"
    echo "rollouts=${RUN_ROOT}/rollouts/rollouts.jsonl"
    echo "summary=${RUN_ROOT}/summary.json"
    ;;
esac
