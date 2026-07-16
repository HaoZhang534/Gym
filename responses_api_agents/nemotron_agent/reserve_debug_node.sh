#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Reserve one Ray/Pyxis debug node for nemotron_agent debugging.

Usage from a login node:
  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/reserve_debug_node.sh

Useful overrides:
  SBATCH_PARTITION=interactive
  SBATCH_TIME=04:00:00
  GPUS_PER_NODE=8
  RUN_ROOT=/path/to/debug_run
  CONTAINER_IMAGE=/path/to/nemo-rl-apptainer_20260419.sqsh
  WAIT_FOR_ATTACH_SCRIPT=1

After it prints an attach command:
  bash <RUN_ROOT>/attach.sh

Inside the attached shell:
  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh vllm
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${NEMO_GYM_ROOT}/../.." && pwd)}"
NEMORL="${NEMORL:-${REPO_ROOT}/nemo-rl-nano-v3-omni}"
RAY_SUB="${RAY_SUB:-${NEMORL}/ray.sub}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nvr_lpr_agentic}"
SBATCH_PARTITION="${SBATCH_PARTITION:-interactive}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${NEMO_GYM_ROOT}/results/nemotron_agent_debug_${RUN_STAMP}}"
JOB_NAME="${JOB_NAME:-nemotron-debug}"
WAIT_FOR_ATTACH_SCRIPT="${WAIT_FOR_ATTACH_SCRIPT:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"

export CONTAINER="${CONTAINER:-${CONTAINER_IMAGE:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/data/images/nemo-rl-apptainer_20260419.sqsh}}"
export MOUNTS="${MOUNTS:-/lustre:/lustre,/tmp:/tmp}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_ROOT}/ray_logs}"
export GPUS_PER_NODE
export REPO_ROOT
export NEMO_GYM_ROOT
export RUN_ROOT
export RUN_STAMP

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this script from a login node. Inside an allocation, use debug_inside_container.sh directly." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to reserve a debug node" >&2
  exit 2
fi
for path in "${RAY_SUB}" "${CONTAINER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/slurm" "${BASE_LOG_DIR}"

echo "Submitting nemotron debug Ray allocation..."
echo "run_root=${RUN_ROOT}"
echo "container=${CONTAINER}"
echo "ray_sub=${RAY_SUB}"

cd "${REPO_ROOT}"
submit_output="$(
  sbatch \
    --parsable \
    --nodes="${NUM_NODES}" \
    --account="${SBATCH_ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${SBATCH_PARTITION}" \
    --time="${SBATCH_TIME}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --output="${RUN_ROOT}/slurm/%x-%j.out" \
    --error="${RUN_ROOT}/slurm/%x-%j.err" \
    "${RAY_SUB}"
)"
job_id="${submit_output%%$'\n'*}"
job_id="${job_id%%;*}"

generated_attach="${REPO_ROOT}/${job_id}-attach.sh"
stable_attach="${RUN_ROOT}/attach.sh"

echo "job_id=${job_id}"
echo "generated_attach=${generated_attach}"
echo "stable_attach=${stable_attach}"
echo "slurm_stdout=${RUN_ROOT}/slurm/${JOB_NAME}-${job_id}.out"
echo "slurm_stderr=${RUN_ROOT}/slurm/${JOB_NAME}-${job_id}.err"

if [[ "${WAIT_FOR_ATTACH_SCRIPT}" != "1" && "${WAIT_FOR_ATTACH_SCRIPT}" != "true" ]]; then
  echo "Submitted. Attach after ${generated_attach} appears."
  exit 0
fi

deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  if [[ -f "${generated_attach}" ]]; then
    cp "${generated_attach}" "${stable_attach}"
    chmod +x "${stable_attach}"
    cat >"${RUN_ROOT}/debug_env.sh" <<EOF
export REPO_ROOT="${REPO_ROOT}"
export NEMO_GYM_ROOT="${NEMO_GYM_ROOT}"
export RUN_ROOT="${RUN_ROOT}"
export RUN_STAMP="${RUN_STAMP}"
EOF
    echo "ready=true"
    echo "attach: bash ${stable_attach}"
    echo ""
    echo "Then inside attach shell:"
    echo "  source ${RUN_ROOT}/debug_env.sh"
    echo "  bash nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/debug_inside_container.sh vllm"
    exit 0
  fi

  state="$(squeue -h -j "${job_id}" -o '%T %R' 2>/dev/null || true)"
  if [[ -z "${state}" ]]; then
    echo "Job ${job_id} is no longer in squeue before attach script appeared." >&2
    sacct -j "${job_id}" --format=JobID,State,Elapsed,ExitCode -n -X || true
    echo "stdout=${RUN_ROOT}/slurm/${JOB_NAME}-${job_id}.out" >&2
    echo "stderr=${RUN_ROOT}/slurm/${JOB_NAME}-${job_id}.err" >&2
    exit 1
  fi

  if (( SECONDS % 30 < 2 )); then
    echo "waiting_for_attach_script job_id=${job_id} state=${state}"
  fi
  sleep 2
done

echo "Timed out waiting for attach script after ${WAIT_SECONDS}s" >&2
echo "job_id=${job_id}" >&2
echo "generated_attach=${generated_attach}" >&2
exit 1
