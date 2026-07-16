#!/usr/bin/env bash
#SBATCH --job-name=nemo-osw-local
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL/nemo_chunking_support/Nemo_GYM/results/slurm_logs/nemo-osw-local-%j.out
#SBATCH --error=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL/nemo_chunking_support/Nemo_GYM/results/slurm_logs/nemo-osw-local-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/data/images/nemo-rl-apptainer_20260419.sqsh}"
INNER_SCRIPT="${REPO_ROOT}/nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/slurm/collect_local_nano_omni_inner.sh"

mkdir -p "${REPO_ROOT}/nemo_chunking_support/Nemo_GYM/results/slurm_logs"

echo "host_node=$(hostname)"
echo "container_image=${CONTAINER_IMAGE}"
echo "inner_script=${INNER_SCRIPT}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "job_id=${SLURM_JOB_ID:-manual}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/lustre:/lustre,/tmp:/tmp \
  --container-workdir="${REPO_ROOT}/nemo_chunking_support/Nemo_GYM" \
  --container-writable \
  bash "${INNER_SCRIPT}"
