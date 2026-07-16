#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run/debug nemotron_agent from inside an attached debug container.

Recommended pdb workflow:
  1. In attach shell #1:
       bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh vllm
  2. In attach shell #2:
       bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh gym
  3. Wait until shell #2 prints "All 1 / 1 servers ready".
  4. In attach shell #3:
       bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh collect

Subcommands:
  vllm     Start local vLLM and leave it running.
  gym      Start only the Nemo_GYM server in the foreground.
           Python breakpoints in nemotron_agent code are interactive here.
  collect  Send one rollout request to the foreground server.
  server   Compatibility mode: start local vLLM, then run Nemo_GYM in foreground.
  all      Non-interactive end-to-end run, equivalent to the normal rollout script.

Common overrides:
  RUN_ROOT=/path/to/run
  DATASET=responses_api_agents/nemotron_agent/data/osworld_test_test.jsonl
  LIMIT=1
  MAX_STEPS=3
  NUM_SAMPLES_IN_PARALLEL=1
  AGENT_CONCURRENCY=1
  DEBUG_BREAKPOINTS=all
  NEMOTRON_USE_TTY_BREAKPOINT=true
  NEMOTRON_PYTHONBREAKPOINT=pdb.set_trace
  NEMOTRON_REUSE_CLI_VENV=true
  NEMO_GYM_SKIP_SERVER_VENV_IF_PRESENT=true
USAGE
}

MODE="${1:-server}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${NEMO_GYM_ROOT}/../.." && pwd)}"
INNER_SCRIPT="${INNER_SCRIPT:-${SCRIPT_DIR}/slurm/collect_local_nano_omni_inner.sh}"

JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${NEMO_GYM_ROOT}/results/nemotron_agent_debug_${JOB_TAG}_${RUN_STAMP}}"
LOG_DIR="${RUN_ROOT}/logs"
ROLLOUT_DIR="${RUN_ROOT}/rollouts"

export REPO_ROOT
export NEMO_GYM_ROOT
export RUN_STAMP
export RUN_ROOT
export JOB_TAG

apply_debug_defaults() {
  export DATASET="${DATASET:-${NEMO_GYM_ROOT}/responses_api_agents/nemotron_agent/data/osworld_test_test.jsonl}"
  export LIMIT="${LIMIT:-1}"
  export NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-1}"
  export AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-1}"
  export MAX_STEPS="${MAX_STEPS:-3}"
  export TEMPERATURE="${TEMPERATURE:-0.1}"
  export TOP_P="${TOP_P:-0.9}"
  export MAX_TOKENS="${MAX_TOKENS:-2048}"
  export STARTUP_SLEEP="${STARTUP_SLEEP:-5.0}"
  export SETTLE_SLEEP="${SETTLE_SLEEP:-2.0}"
  export CREATE_GIF="${CREATE_GIF:-false}"
  export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-false}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export PYTHONBREAKPOINT="${NEMOTRON_PYTHONBREAKPOINT:-pdb.set_trace}"
  export NEMOTRON_USE_TTY_BREAKPOINT="${NEMOTRON_USE_TTY_BREAKPOINT:-true}"
  export NEMOTRON_DEBUG_SYNC_RUN="${NEMOTRON_DEBUG_SYNC_RUN:-true}"
  export NEMOTRON_FORCE_PDB_ON_RUN="${NEMOTRON_FORCE_PDB_ON_RUN:-true}"

  if [[ -n "${DEBUG_BREAKPOINTS:-}" ]]; then
    export NEMO_ROLLOUT_DEBUG_BREAKPOINTS="${DEBUG_BREAKPOINTS}"
  fi
  if [[ -n "${DEBUG_DUMP_ONLY:-}" ]]; then
    export NEMO_ROLLOUT_DEBUG_DUMP_ONLY="${DEBUG_DUMP_ONLY}"
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local tries="${3:-180}"
  for i in $(seq 1 "${tries}"); do
    if curl -sS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if (( i % 12 == 0 )); then
      echo "waiting_for_${label}_${i}: ${url}"
    fi
    sleep 5
  done
  echo "Timed out waiting for ${label}: ${url}" >&2
  return 1
}

agent_url_from_head() {
  local head_url="$1"
  python - "${head_url}" <<'PY'
import json
import sys
import urllib.request
from urllib.parse import urlparse, urlunparse

head_url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{head_url}/server_instances", timeout=5) as response:
    instances = json.load(response)

for instance in instances:
    if instance.get("name") == "nemotron_agent" or instance.get("config_path") == "nemotron_agent":
        url = instance.get("url")
        if not url and instance.get("host") and instance.get("port"):
            url = f"http://{instance['host']}:{instance['port']}"
        if url:
            parsed = urlparse(url)
            if parsed.hostname in {"0.0.0.0", "::"}:
                netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
                url = urlunparse(parsed._replace(netloc=netloc))
            print(url)
            raise SystemExit(0)

raise SystemExit("nemotron_agent not found in /server_instances")
PY
}

run_collect() {
  apply_debug_defaults
  mkdir -p "${LOG_DIR}" "${ROLLOUT_DIR}"

  local nemo_gym_python="${NEMO_GYM_CLI_VENV:-/tmp/nemo_gym_cli_venv_${JOB_TAG}}/bin/python"
  if [[ ! -x "${nemo_gym_python}" ]]; then
    echo "Nemo_GYM CLI venv not found: ${nemo_gym_python}" >&2
    echo "Start the Gym server first in another attached shell:" >&2
    echo "  bash ${SCRIPT_DIR}/debug_inside_container.sh gym" >&2
    exit 2
  fi

  local head_url="${NEMO_GYM_HEAD_URL:-http://127.0.0.1:11000}"
  wait_for_url "${head_url}/docs" "nemo_gym_head"

  local agent_url
  agent_url="$(agent_url_from_head "${head_url}")"
  wait_for_url "${agent_url}" "nemotron_agent"

  local collect_stamp
  collect_stamp="$(date +%Y%m%d_%H%M%S)"
  local output_jsonl="${OUTPUT_JSONL_FPATH:-${ROLLOUT_DIR}/debug_rollouts_${collect_stamp}.jsonl}"
  local collect_log="${LOG_DIR}/debug_collect_${collect_stamp}.log"

  local served_model_name="${SERVED_MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
  local collect_cmd=(
    "${nemo_gym_python}" -c 'from nemo_gym.rollout_collection import collect_rollouts; collect_rollouts()'
    "+agent_name=nemotron_agent"
    "+input_jsonl_fpath=${DATASET}"
    "+output_jsonl_fpath=${output_jsonl}"
    "+num_repeats=1"
    "+num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL}"
    "+resume_from_cache=${RESUME_FROM_CACHE}"
    "+responses_create_params.model=${served_model_name}"
  )
  if [[ -n "${LIMIT}" ]]; then
    collect_cmd+=("+limit=${LIMIT}")
  fi

  echo "head_url=${head_url}"
  echo "agent_url=${agent_url}"
  echo "dataset=${DATASET}"
  echo "output_jsonl=${output_jsonl}"
  echo "collect_log=${collect_log}"
  set +e
  "${collect_cmd[@]}" 2>&1 | tee "${collect_log}"
  local status=${PIPESTATUS[0]}
  set -e

  echo "rollouts=${output_jsonl}"
  if [[ -f "${output_jsonl%.*}_aggregate_metrics.json" ]]; then
    echo "aggregate_metrics=${output_jsonl%.*}_aggregate_metrics.json"
    cat "${output_jsonl%.*}_aggregate_metrics.json"
  fi
  return "${status}"
}

case "${MODE}" in
  vllm)
    apply_debug_defaults
    export NEMOTRON_INNER_MODE=vllm
    exec bash "${INNER_SCRIPT}"
    ;;
  gym)
    apply_debug_defaults
    export NEMOTRON_INNER_MODE=gym
    exec bash "${INNER_SCRIPT}"
    ;;
  server)
    apply_debug_defaults
    export NEMOTRON_INNER_MODE=server
    exec bash "${INNER_SCRIPT}"
    ;;
  collect)
    run_collect
    ;;
  all)
    apply_debug_defaults
    export NEMOTRON_INNER_MODE=all
    exec bash "${INNER_SCRIPT}"
    ;;
  *)
    echo "Invalid subcommand: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
