#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL}"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-${REPO_ROOT}/nemo_chunking_support/Nemo_GYM}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${NEMO_GYM_ROOT}/responses_api_agents/nemotron_agent/slurm/collect_local_nano_omni_sbatch.sh}"
RUN_ROOT="${RUN_ROOT:-$(readlink -f "${NEMO_GYM_ROOT}/latest_nemotron3_nano_omni_responses_agent_local_osworld_run")}"
CURRENT_JOB_ID="${CURRENT_JOB_ID:-4476119}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MAX_RESUBMITS="${MAX_RESUBMITS:-3}"
TOTAL="${TOTAL:-369}"

resubmits=0

rollouts_path() {
  printf '%s/rollouts/rollouts.jsonl' "${RUN_ROOT}"
}

print_stats() {
  python - "${RUN_ROOT}" "${TOTAL}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_root = Path(sys.argv[1])
total = int(sys.argv[2])
p = run_root / "rollouts" / "rollouts.jsonl"
rows = []
if p.exists():
    with p.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
success_sum = sum(float(row.get("reward") or 0.0) for row in rows)
errors = sum(1 for row in rows if row.get("error"))
domains = Counter(row.get("domain") or "<unknown>" for row in rows)
completed = len(rows)
acc_completed = success_sum / completed if completed else 0.0
acc_total = success_sum / total if total else 0.0
domain_summary = ", ".join(f"{k}:{v}" for k, v in sorted(domains.items()))
print(
    f"completed={completed}/{total} success_sum={success_sum:.3f} "
    f"acc_completed={acc_completed:.4f} acc_total_so_far={acc_total:.4f} "
    f"errors={errors}"
)
print(f"domains={domain_summary}")
PY
}

completed_count() {
  local path
  path="$(rollouts_path)"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}"
  else
    echo 0
  fi
}

job_active() {
  [[ -n "${CURRENT_JOB_ID}" ]] && [[ -n "$(squeue -h -j "${CURRENT_JOB_ID}" 2>/dev/null)" ]]
}

while true; do
  echo "---- $(date '+%Y-%m-%d %H:%M:%S %Z') ----"
  echo "run_root=${RUN_ROOT}"
  echo "current_job_id=${CURRENT_JOB_ID:-<none>}"
  if job_active; then
    squeue -j "${CURRENT_JOB_ID}" -o '%.18i %.9P %.32j %.8T %.10M %.9l %.6D %R'
  else
    echo "job_active=false"
    if [[ -n "${CURRENT_JOB_ID}" ]]; then
      sacct -j "${CURRENT_JOB_ID}" --format=JobID,State,Elapsed,ExitCode -n -X | head -n 5 || true
    fi
  fi
  print_stats

  completed="$(completed_count)"
  if (( completed >= TOTAL )); then
    echo "complete=true"
    exit 0
  fi

  if ! job_active; then
    if (( resubmits >= MAX_RESUBMITS )); then
      echo "max_resubmits_reached=${MAX_RESUBMITS}; not submitting again"
      exit 2
    fi
    echo "resubmitting_with_same_run_root=true"
    submit_output="$(sbatch --parsable --export=ALL,RUN_ROOT="${RUN_ROOT}" "${SBATCH_SCRIPT}")"
    CURRENT_JOB_ID="${submit_output%%;*}"
    resubmits=$((resubmits + 1))
    echo "new_job_id=${CURRENT_JOB_ID}"
  fi

  sleep "${CHECK_INTERVAL}"
done
