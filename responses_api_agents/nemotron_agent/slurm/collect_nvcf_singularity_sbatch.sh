#!/usr/bin/env bash
#SBATCH --job-name=nemotron-osw-sing
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL/nemo_chunking_support/Nemo_GYM/results/slurm_logs/nemotron-osw-sing-%j.out
#SBATCH --error=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL/nemo_chunking_support/Nemo_GYM/results/slurm_logs/nemotron-osw-sing-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/data/images/nemo-rl-apptainer_20260419.sqsh}"
INNER_SCRIPT="${REPO_ROOT}/nemo_chunking_support/Nemo_GYM/responses_api_agents/nemotron_agent/slurm/collect_nvcf_singularity_inner.sh"

mkdir -p "${REPO_ROOT}/nemo_chunking_support/Nemo_GYM/results/slurm_logs"

echo "host_node=$(hostname)"
echo "container_image=${CONTAINER_IMAGE}"
echo "inner_script=${INNER_SCRIPT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/lustre:/lustre,/tmp:/tmp \
  --container-workdir="${REPO_ROOT}/nemo_chunking_support/Nemo_GYM" \
  --container-writable \
  bash
