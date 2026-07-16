#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL}"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-${REPO_ROOT}/nemo_chunking_support/Nemo_GYM}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_ROOT="${RUN_ROOT:-${NEMO_GYM_ROOT}/results/nemotron_nvcf_singularity_slurm_${JOB_TAG}}"
LOG_DIR="${RUN_ROOT}/logs"
ROLLOUT_DIR="${RUN_ROOT}/rollouts"

mkdir -p "${LOG_DIR}" "${ROLLOUT_DIR}" "${RUN_ROOT}/osworld_results"
umask 0002

export HOME="${HOME:-/tmp/nemotron_home_${JOB_TAG}}"
mkdir -p "${HOME}"

unset VIRTUAL_ENV
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/nemo_gym_uv_${JOB_TAG}}"
export UV_PROJECT_ENVIRONMENT="/tmp/nemo_gym_project_venv_${JOB_TAG}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/nemo_gym_xdg_cache_${JOB_TAG}}"
mkdir -p "${UV_CACHE_DIR}" "${UV_PROJECT_ENVIRONMENT}" "${XDG_CACHE_HOME}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp}"
export NEMO_GYM_RAY_TMPDIR="${NEMO_GYM_RAY_TMPDIR:-/tmp}"
export NEMO_GYM_RAY_NUM_CPUS="${NEMO_GYM_RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-8}}"
export NEMO_GYM_RAY_INCLUDE_DASHBOARD="${NEMO_GYM_RAY_INCLUDE_DASHBOARD:-0}"

export NVCF_SINGULARITY_SIF_PATH="${NVCF_SINGULARITY_SIF_PATH:-/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/nvcf-osworld-eval/osworld-linux.sif}"
export OSWORLD_PROVIDER="${OSWORLD_PROVIDER:-nvcf_singularity}"
export OSWORLD_NEMOTRON_RESULT_DIR="${OSWORLD_NEMOTRON_RESULT_DIR:-${RUN_ROOT}/osworld_results}"
export OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-/tmp/osworld_nemotron_cache_${JOB_TAG}}"
export OSWORLD_TEST_CONFIG_BASE_DIR="${OSWORLD_TEST_CONFIG_BASE_DIR:-${NEMO_GYM_ROOT}/responses_api_agents/nemotron_agent/data/evaluation_examples}"
export NEMOTRON_MODEL="${NEMOTRON_MODEL:-nvidia/nemotron-nano-12b-v2-vl}"
if [[ -z "${VLLM_API_ENDPOINT:-}" && -n "${OPENAI_API_BASE:-}" ]]; then
  export VLLM_API_ENDPOINT="${OPENAI_API_BASE}"
fi
export VLLM_API_ENDPOINT="${VLLM_API_ENDPOINT:-https://integrate.api.nvidia.com/v1/chat/completions}"

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  if [[ -n "${OPENAI_API_BASE:-}" && "${VLLM_API_ENDPOINT}" == "${OPENAI_API_BASE}" && -n "${OPENAI_API_KEY:-}" ]]; then
    export VLLM_API_KEY="${OPENAI_API_KEY}"
  elif [[ -n "${NGC_API_KEY:-}" ]]; then
    export VLLM_API_KEY="${NGC_API_KEY}"
  elif [[ -n "${MY_NGC_API_KEY:-}" ]]; then
    export VLLM_API_KEY="${MY_NGC_API_KEY}"
  elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    export VLLM_API_KEY="${OPENAI_API_KEY}"
  fi
fi
if [[ -z "${NGC_API_KEY:-}" && -n "${VLLM_API_KEY:-}" ]]; then
  export NGC_API_KEY="${VLLM_API_KEY}"
fi
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "VLLM_API_KEY is required. Export VLLM_API_KEY, NGC_API_KEY, or MY_NGC_API_KEY before sbatch." >&2
  exit 2
fi

DATASET="${DATASET:-responses_api_agents/nemotron_agent/data/osworld_test_test_multi.jsonl}"
LIMIT="${LIMIT:-3}"
MAX_STEPS="${MAX_STEPS:-3}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM="${USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM:-true}"

echo "node=$(hostname)"
echo "run_root=${RUN_ROOT}"
echo "dataset=${DATASET}"
echo "limit=${LIMIT}"
echo "model=${NEMOTRON_MODEL}"
echo "provider=${OSWORLD_PROVIDER}"
echo "use_builtin_task_bootstrap_without_llm=${USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM}"
echo "sif=${NVCF_SINGULARITY_SIF_PATH}"
echo "uv_project_environment=${UV_PROJECT_ENVIRONMENT}"
echo "uv_cache_dir=${UV_CACHE_DIR}"
python --version
uv --version
singularity --version
ls -lh "${NVCF_SINGULARITY_SIF_PATH}"

cd "${NEMO_GYM_ROOT}"

COMMON_OVERRIDES=(
  "+config_paths=[responses_api_agents/nemotron_agent/configs/nemotron_agent.yaml]"
  "+uv_venv_dir=/tmp/nemo_gym_server_venvs_${JOB_TAG}"
  "+uv_cache_dir=${UV_CACHE_DIR}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.provider_name=${OSWORLD_PROVIDER}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.model=${NEMOTRON_MODEL}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.max_steps=${MAX_STEPS}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.thinking=false"
  "+nemotron_agent.responses_api_agents.nemotron_agent.temperature=${TEMPERATURE}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.top_p=${TOP_P}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.max_tokens=${MAX_TOKENS}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.startup_sleep=10.0"
  "+nemotron_agent.responses_api_agents.nemotron_agent.settle_sleep=5.0"
  "+nemotron_agent.responses_api_agents.nemotron_agent.test_config_base_dir=${OSWORLD_TEST_CONFIG_BASE_DIR}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.result_dir=${OSWORLD_NEMOTRON_RESULT_DIR}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.use_builtin_task_bootstrap_without_llm=${USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.concurrency=1"
)

uv run ng_dump_config "${COMMON_OVERRIDES[@]}" >"${LOG_DIR}/dump_config.log" 2>&1

RUN_LOG="${LOG_DIR}/ng_run.log"
uv run ng_run "${COMMON_OVERRIDES[@]}" >"${RUN_LOG}" 2>&1 &
NG_RUN_PID=$!

cleanup() {
  if [[ -n "${NG_RUN_PID:-}" ]] && kill -0 "${NG_RUN_PID}" 2>/dev/null; then
    kill "${NG_RUN_PID}" 2>/dev/null || true
    sleep 5
    kill -9 "${NG_RUN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "ng_run pid=${NG_RUN_PID}"
for i in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:11000/docs" >/dev/null 2>&1; then
    echo "Nemo_GYM head server is ready"
    break
  fi
  if ! kill -0 "${NG_RUN_PID}" 2>/dev/null; then
    echo "ng_run exited before head server became ready" >&2
    tail -200 "${RUN_LOG}" >&2 || true
    exit 3
  fi
  if (( i % 12 == 0 )); then
    echo "waiting_for_ng_run_${i}"
    tail -60 "${RUN_LOG}" || true
  fi
  sleep 5
done

if ! curl -fsS --max-time 2 "http://127.0.0.1:11000/docs" >/dev/null 2>&1; then
  echo "Timed out waiting for Nemo_GYM head server" >&2
  tail -200 "${RUN_LOG}" >&2 || true
  exit 4
fi

COLLECT_LOG="${LOG_DIR}/ng_collect_rollouts.log"
uv run ng_collect_rollouts \
  +agent_name=nemotron_agent \
  +input_jsonl_fpath="${DATASET}" \
  +output_jsonl_fpath="${ROLLOUT_DIR}/rollouts.jsonl" \
  +limit="${LIMIT}" \
  +num_repeats=1 \
  +num_samples_in_parallel=1 \
  >"${COLLECT_LOG}" 2>&1

echo "rollouts=${ROLLOUT_DIR}/rollouts.jsonl"
if [[ -f "${ROLLOUT_DIR}/rollouts_aggregate_metrics.json" ]]; then
  cat "${ROLLOUT_DIR}/rollouts_aggregate_metrics.json"
fi

find "${OSWORLD_NEMOTRON_RESULT_DIR}" -path "*/trajectory.gif" -printf "%T@ %p\n" 2>/dev/null | sort -nr >"${RUN_ROOT}/trajectory_gifs.txt" || true
cat "${RUN_ROOT}/trajectory_gifs.txt" || true
