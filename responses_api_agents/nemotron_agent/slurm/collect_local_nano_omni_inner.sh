#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/CUA_RL}"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-${REPO_ROOT}/nemo_chunking_support/Nemo_GYM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-nemotron3_nano_omni_responses_agent_local}"
RUN_ROOT="${RUN_ROOT:-${NEMO_GYM_ROOT}/results/${RUN_NAME}_${JOB_TAG}_${RUN_STAMP}}"
LOG_DIR="${RUN_ROOT}/logs"
ROLLOUT_DIR="${RUN_ROOT}/rollouts"
OSWORLD_RESULTS_DIR="${RUN_ROOT}/osworld_results"
LATEST_LINK="${LATEST_LINK:-${NEMO_GYM_ROOT}/latest_${RUN_NAME}_osworld_run}"

mkdir -p "${LOG_DIR}" "${ROLLOUT_DIR}" "${OSWORLD_RESULTS_DIR}"
ln -sfn "${RUN_ROOT}" "${LATEST_LINK}"
umask 0002

if [[ "${DUMP_VLLM_REQUESTS:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  export NEMOTRON_AGENT_VLLM_DUMP_DIR="${NEMOTRON_AGENT_VLLM_DUMP_DIR:-${RUN_ROOT}/request_dumps/agent}"
  export NEMO_GYM_VLLM_DUMP_FULL_REQUEST="${NEMO_GYM_VLLM_DUMP_FULL_REQUEST:-1}"
  export NEMO_GYM_VLLM_DUMP_TOKENIZE="${NEMO_GYM_VLLM_DUMP_TOKENIZE:-1}"
  mkdir -p "${NEMOTRON_AGENT_VLLM_DUMP_DIR}"
fi

export HOME="${NEMO_GYM_HOME:-/tmp/nemo_gym_home_${JOB_TAG}}"
mkdir -p "${HOME}"

unset VIRTUAL_ENV
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/nemo_gym_uv_${JOB_TAG}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/nemo_gym_project_venv_${JOB_TAG}}"
export NEMO_GYM_CLI_VENV="${NEMO_GYM_CLI_VENV:-/tmp/nemo_gym_cli_venv_${JOB_TAG}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/nemo_gym_xdg_cache_${JOB_TAG}}"
export EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-/tmp/nemo_gym_extra_site_${JOB_TAG}}"
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${EXTRA_SITE_PACKAGES}"
if [[ -d "${UV_PROJECT_ENVIRONMENT}" && ! -f "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg" ]]; then
  rm -rf "${UV_PROJECT_ENVIRONMENT}"
fi
ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp}"
export NEMO_GYM_RAY_TMPDIR="${NEMO_GYM_RAY_TMPDIR:-/tmp}"
export NEMO_GYM_RAY_NUM_CPUS="${NEMO_GYM_RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
export NEMO_GYM_RAY_INCLUDE_DASHBOARD="${NEMO_GYM_RAY_INCLUDE_DASHBOARD:-0}"

MODEL_SOURCE="${MODEL_SOURCE:-${REPO_ROOT}/nemo-rl-nano-v3-omni/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/tmp/nemotron3_nano_omni_bf16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
VLLM_ROOT="${VLLM_ROOT:-${REPO_ROOT}/nemo-rl-nano-v3-omni/3rdparty/vllm}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_CLIENT_HOST="${VLLM_CLIENT_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_IMAGES="${MAX_IMAGES:-3}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
ALLOW_REMOTE_MODEL_SOURCE="${ALLOW_REMOTE_MODEL_SOURCE:-false}"
EXTRA_VLLM_PIP_PACKAGES="${EXTRA_VLLM_PIP_PACKAGES:-}"

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_API_ENDPOINT="http://${VLLM_CLIENT_HOST}:${VLLM_PORT}/v1/chat/completions"
export VLLM_SERVER_DEV_MODE=1
export USE_TORCH=1
export USE_TF=0
export USE_FLAX=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torch_inductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${XDG_CACHE_HOME}/flashinfer}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

export NVCF_SINGULARITY_SIF_PATH="${NVCF_SINGULARITY_SIF_PATH:-/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/nvcf-osworld-eval/osworld-linux.sif}"
export OSWORLD_PROVIDER="${OSWORLD_PROVIDER:-nvcf_singularity}"
export OSWORLD_NEMOTRON_RESULT_DIR="${OSWORLD_NEMOTRON_RESULT_DIR:-${OSWORLD_RESULTS_DIR}}"
export OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-/tmp/osworld_nemotron_cache_${JOB_TAG}}"
export OSWORLD_TEST_CONFIG_BASE_DIR="${OSWORLD_TEST_CONFIG_BASE_DIR:-${NEMO_GYM_ROOT}/responses_api_agents/nemotron_agent/data/evaluation_examples}"
export OSWORLD_PASSWORD="${OSWORLD_PASSWORD:-password}"
export NEMOTRON_MODEL="${NEMOTRON_MODEL:-${SERVED_MODEL_NAME}}"
export NEMOTRON_USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM="${NEMOTRON_USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM:-false}"

DATASET="${DATASET:-responses_api_agents/nemotron_agent/data/osworld_test_all_nvidia_nemotron_nano_12b_v2_vl.jsonl}"
LIMIT="${LIMIT:-}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-12}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-${NUM_SAMPLES_IN_PARALLEL}}"
MAX_STEPS="${MAX_STEPS:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
STARTUP_SLEEP="${STARTUP_SLEEP:-10.0}"
SETTLE_SLEEP="${SETTLE_SLEEP:-10.0}"
CREATE_GIF="${CREATE_GIF:-false}"
RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-true}"
NEMOTRON_REUSE_CLI_VENV="${NEMOTRON_REUSE_CLI_VENV:-true}"
NEMO_GYM_SKIP_SERVER_VENV_IF_PRESENT="${NEMO_GYM_SKIP_SERVER_VENV_IF_PRESENT:-true}"
export NEMO_GYM_SKIP_SERVER_VENVS="${NEMO_GYM_SKIP_SERVER_VENVS:-1}"
NEMOTRON_INSTALL_AGENT_REQS_IN_CLI_VENV="${NEMOTRON_INSTALL_AGENT_REQS_IN_CLI_VENV:-true}"
NEMOTRON_INNER_MODE="${NEMOTRON_INNER_MODE:-all}"
case "${NEMOTRON_INNER_MODE}" in
  all|server|vllm|gym|wrapper_smoke) ;;
  *)
    echo "Invalid NEMOTRON_INNER_MODE=${NEMOTRON_INNER_MODE}; expected all, server, vllm, gym, or wrapper_smoke" >&2
    exit 2
    ;;
esac

echo "node=$(hostname)"
echo "run_root=${RUN_ROOT}"
echo "latest_link=${LATEST_LINK}"
echo "dataset=${DATASET}"
echo "limit=${LIMIT:-<none>}"
echo "model_source=${MODEL_SOURCE}"
echo "local_model_path=${LOCAL_MODEL_PATH}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "vllm_root=${VLLM_ROOT}"
echo "vllm_endpoint=${VLLM_API_ENDPOINT}"
echo "vllm_request_dump_agent=${NEMOTRON_AGENT_VLLM_DUMP_DIR:-<disabled>}"
echo "provider=${OSWORLD_PROVIDER}"
echo "sif=${NVCF_SINGULARITY_SIF_PATH}"
echo "num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL}"
echo "agent_concurrency=${AGENT_CONCURRENCY}"
echo "max_steps=${MAX_STEPS}"
echo "max_tokens=${MAX_TOKENS}"
echo "temperature=${TEMPERATURE}"
echo "top_p=${TOP_P}"
echo "create_gif=${CREATE_GIF}"
echo "vllm_dtype=${VLLM_DTYPE}"
echo "allow_remote_model_source=${ALLOW_REMOTE_MODEL_SOURCE}"
echo "extra_vllm_pip_packages=${EXTRA_VLLM_PIP_PACKAGES:-<none>}"
echo "inner_mode=${NEMOTRON_INNER_MODE}"
echo "reuse_cli_venv=${NEMOTRON_REUSE_CLI_VENV}"
echo "skip_server_venv_if_present=${NEMO_GYM_SKIP_SERVER_VENV_IF_PRESENT}"
echo "skip_server_venvs=${NEMO_GYM_SKIP_SERVER_VENVS}"
echo "install_agent_reqs_in_cli_venv=${NEMOTRON_INSTALL_AGENT_REQS_IN_CLI_VENV}"
python --version
uv --version
singularity --version
nvidia-smi || true
ls -lh "${NVCF_SINGULARITY_SIF_PATH}"

if [[ "${NEMOTRON_INNER_MODE}" != "gym" ]]; then
if [[ ! -d "${VLLM_ROOT}/vllm" ]]; then
  echo "vLLM checkout not found: ${VLLM_ROOT}" >&2
  exit 2
fi
MODEL_PATH_FOR_VLLM="${LOCAL_MODEL_PATH}"
if [[ ! -d "${MODEL_SOURCE}" && "${ALLOW_REMOTE_MODEL_SOURCE}" != "true" && "${ALLOW_REMOTE_MODEL_SOURCE}" != "1" ]]; then
  echo "Model source is not a local directory: ${MODEL_SOURCE}" >&2
  echo "Set MODEL_SOURCE to a local model path, or set ALLOW_REMOTE_MODEL_SOURCE=true to let vLLM resolve a Hugging Face model id." >&2
  exit 2
elif [[ ! -d "${MODEL_SOURCE}" ]]; then
  MODEL_PATH_FOR_VLLM="${MODEL_SOURCE}"
  echo "Model source is not local; vLLM will resolve it directly: ${MODEL_PATH_FOR_VLLM}"
fi

python -m pip install --disable-pip-version-check --no-cache-dir \
  --target "${EXTRA_SITE_PACKAGES}" --upgrade \
  "typing-extensions>=4.14" \
  "dill>=0.3.8" \
  "numpy==1.26.4" \
  "packaging>=24.0" \
  "fastapi>=0.116.1" \
  "starlette>=0.49.1,<1.0.0" \
  "uvicorn>=0.35.0" \
  "huggingface-hub>=0.34,<1.0" \
  "safetensors>=0.5.0" \
  "simplejson>=3.19.0" \
  "transformers==4.57.1" \
  "jinja2>=3.1.6" \
  "markupsafe>=2.1.5" \
  "Pillow>=10.0.0" \
  "psutil>=5.9.0" \
  "cffi>=1.16.0" \
  "soundfile>=0.12.1" \
  "scipy>=1.12.0" \
  "scikit-learn>=1.4.0" \
  "pyarrow>=16.0.0" \
  "aiohttp>=3.9.0" \
  "cachetools" \
  "sentencepiece" \
  "blake3" \
  "py-cpuinfo" \
  "openai>=1.99.1" \
  "prometheus_client>=0.18.0" \
  "prometheus-fastapi-instrumentator>=7.0.0" \
  "tiktoken>=0.6.0" \
  "lm-format-enforcer==0.11.3" \
  "outlines_core==0.2.11" \
  "diskcache==5.6.3" \
  "lark==1.2.2" \
  "partial-json-parser" \
  "pyzmq>=25.0.0" \
  "msgspec" \
  "gguf>=0.17.0" \
  "opencv-python-headless>=4.11.0" \
  "setuptools>=77.0.3,<81.0.0" \
  "einops" \
  "depyf==0.20.0" \
  "cloudpickle" \
  "watchfiles" \
  "python-json-logger" \
  "ninja" \
  "pybase64" \
  "cbor2" \
  "ijson" \
  "setproctitle" \
  "openai-harmony>=0.0.3" \
  "anthropic>=0.71.0" \
  "model-hosting-container-standards>=0.1.13,<1.0.0" \
  "mcp" \
  "grpcio>=1.76.0" \
  "grpcio-reflection>=1.76.0" \
  "protobuf>=6.30.0" \
  "six>=1.16.0" \
  "networkx>=3.0" \
  "msgpack>=1.0.0" \
  "nvidia-ml-py>=12.0.0" \
  "websockets>=10.4,<14.0"
python -m pip install --disable-pip-version-check --no-cache-dir \
  --target "${EXTRA_SITE_PACKAGES}" --upgrade --no-deps \
  "accelerate>=0.34.0"
python -m pip install --disable-pip-version-check --no-cache-dir \
  --target "${EXTRA_SITE_PACKAGES}" --upgrade --no-deps \
  "torchvision==0.24.1"
if [[ -n "${EXTRA_VLLM_PIP_PACKAGES}" ]]; then
  read -r -a extra_vllm_pip_packages <<< "${EXTRA_VLLM_PIP_PACKAGES}"
  python -m pip install --disable-pip-version-check --no-cache-dir \
    --target "${EXTRA_SITE_PACKAGES}" --upgrade \
    "${extra_vllm_pip_packages[@]}"
fi

python - "${EXTRA_SITE_PACKAGES}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
generation_init = root / "transformers" / "generation" / "__init__.py"
instrumentator_routing = root / "prometheus_fastapi_instrumentator" / "routing.py"
sitecustomize = root / "sitecustomize.py"
structure_marker = "\n\nif TYPE_CHECKING:"
structure_patch = """

# Compatibility for container torch installs that can import torch but are not
# detected by Transformers metadata checks.
_import_structure.setdefault("utils", [])
if "GenerationMixin" not in _import_structure["utils"]:
    _import_structure["utils"].append("GenerationMixin")
"""
snippet = """

# Compatibility shim for vLLM checkouts that import GenerationMixin from transformers.generation.
try:
    from .utils import GenerationMixin as _GenerationMixin
    import sys as _sys
    GenerationMixin = _GenerationMixin
    setattr(_sys.modules[__name__], "GenerationMixin", _GenerationMixin)
except Exception:
    pass
"""
text = generation_init.read_text(encoding="utf-8")
if "container torch installs that can import torch" not in text:
    text = text.replace(structure_marker, structure_patch + structure_marker, 1)
if 'setattr(_sys.modules[__name__], "GenerationMixin"' not in text:
    text = text.rstrip() + snippet + "\n"
generation_init.write_text(text, encoding="utf-8")
if instrumentator_routing.exists():
    routing_text = instrumentator_routing.read_text(encoding="utf-8")
    if "Nemo-GYM compatibility: skip pathless routes" not in routing_text:
        routing_text = routing_text.replace(
            """    for route in routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_name = route.path
""",
            """    for route in routes:
        # Nemo-GYM compatibility: skip pathless routes added by newer Starlette.
        route_path = getattr(route, "path", None)
        if route_path is None:
            continue
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_name = route_path
""",
            1,
        )
        routing_text = routing_text.replace(
            """        elif match == Match.PARTIAL and route_name is None:
            route_name = route.path
""",
            """        elif match == Match.PARTIAL and route_name is None:
            route_name = route_path
""",
            1,
        )
        instrumentator_routing.write_text(routing_text, encoding="utf-8")
sitecustomize.write_text(
    """
try:
    import transformers.generation as _generation
    from transformers.generation.utils import GenerationMixin as _GenerationMixin
    setattr(_generation, "GenerationMixin", _GenerationMixin)
except Exception:
    pass
""".lstrip(),
    encoding="utf-8",
)
PY

cd "${NEMO_GYM_ROOT}"
export PYTHONPATH="${EXTRA_SITE_PACKAGES}:${VLLM_ROOT}:${ORIGINAL_PYTHONPATH}"
python - <<'PY'
import sys
import typing_extensions
import dill
import numpy
import torch
from packaging import version as packaging_version
from fastapi import FastAPI
from huggingface_hub import get_safetensors_metadata
from transformers import GenerationConfig, PretrainedConfig
from transformers.generation.utils import GenerationMixin
from torchvision.transforms import InterpolationMode
import vllm
import vllm.entrypoints.openai.api_server as api_server

print(f"python_executable={sys.executable}")
print(f"typing_extensions={typing_extensions.__file__}")
print(f"dill={dill.__file__}")
print(f"numpy={numpy.__file__} {numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"packaging_version={packaging_version.Version('24.0')}")
print(f"fastapi={FastAPI}")
print(f"get_safetensors_metadata={get_safetensors_metadata}")
print(f"transformers_generation_config={GenerationConfig}")
print(f"transformers_generation_mixin={GenerationMixin}")
print(f"torchvision_interpolation={InterpolationMode.BILINEAR}")
print(f"vllm={vllm.__file__}")
print(f"vllm_api_server={api_server.__file__}")
PY

if [[ -d "${MODEL_SOURCE}" && ! -f "${LOCAL_MODEL_PATH}/config.json" ]]; then
  echo "Copying model to node-local path: ${LOCAL_MODEL_PATH}"
  rm -rf "${LOCAL_MODEL_PATH}"
  mkdir -p "$(dirname "${LOCAL_MODEL_PATH}")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${MODEL_SOURCE}/" "${LOCAL_MODEL_PATH}/"
  else
    cp -a "${MODEL_SOURCE}" "${LOCAL_MODEL_PATH}"
  fi
elif [[ ! -d "${MODEL_SOURCE}" ]]; then
  echo "Skipping node-local model copy for remote/model-id source: ${MODEL_SOURCE}"
fi

VLLM_LOG="${LOG_DIR}/vllm.log"
echo "Starting local vLLM server; log=${VLLM_LOG}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH_FOR_VLLM}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --dtype "${VLLM_DTYPE}" \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

NG_RUN_PID=""
cleanup() {
  if [[ -n "${NG_RUN_PID}" ]] && kill -0 "${NG_RUN_PID}" 2>/dev/null; then
    kill "${NG_RUN_PID}" 2>/dev/null || true
    sleep 5
    kill -9 "${NG_RUN_PID}" 2>/dev/null || true
  fi
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 5
    kill -9 "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Waiting for local vLLM server..."
VLLM_READY=0
for i in $(seq 1 120); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready" >&2
    tail -160 "${VLLM_LOG}" >&2 || true
    exit 3
  fi
  if python - "${VLLM_CLIENT_HOST}" "${VLLM_PORT}" "${VLLM_API_KEY}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

host, port, api_key = sys.argv[1:4]
headers = {}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
req = urllib.request.Request(f"http://{host}:{port}/v1/models", headers=headers)
with urllib.request.urlopen(req, timeout=5) as r:
    json.load(r)
PY
  then
    VLLM_READY=1
    echo "vLLM server is ready"
    break
  fi
  if (( i % 12 == 0 )); then
    echo "waiting_for_vllm_${i}"
    tail -80 "${VLLM_LOG}" || true
  fi
  sleep 10
done
if [[ "${VLLM_READY}" != "1" ]]; then
  echo "Timed out waiting for vLLM" >&2
  tail -160 "${VLLM_LOG}" >&2 || true
  exit 4
fi

if [[ "${NEMOTRON_INNER_MODE}" == "vllm" ]]; then
  echo "vLLM server is running in foreground mode. Leave this shell open."
  echo "endpoint=${VLLM_API_ENDPOINT}"
  wait "${VLLM_PID}"
  exit "$?"
fi
else
echo "Skipping local vLLM startup; expecting existing vLLM at ${VLLM_API_ENDPOINT}"
VLLM_READY=0
for i in $(seq 1 120); do
  if python - "${VLLM_CLIENT_HOST}" "${VLLM_PORT}" "${VLLM_API_KEY}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

host, port, api_key = sys.argv[1:4]
headers = {}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
req = urllib.request.Request(f"http://{host}:{port}/v1/models", headers=headers)
with urllib.request.urlopen(req, timeout=5) as r:
    json.load(r)
PY
  then
    VLLM_READY=1
    echo "Existing vLLM server is ready"
    break
  fi
  if (( i % 12 == 0 )); then
    echo "waiting_for_existing_vllm_${i}"
  fi
  sleep 5
done
if [[ "${VLLM_READY}" != "1" ]]; then
  echo "Timed out waiting for existing vLLM at ${VLLM_API_ENDPOINT}" >&2
  exit 4
fi
fi

if [[ "${NEMOTRON_INNER_MODE}" == "wrapper_smoke" ]]; then
  unset PYTHONPATH
  export VLLM_API_ENDPOINT="http://${VLLM_CLIENT_HOST}:${VLLM_PORT}/v1/chat/completions"
  export VLLM_API_KEY="${VLLM_API_KEY}"
  export NEMOTRON_MODEL="${SERVED_MODEL_NAME}"
  WRAPPER_SMOKE_RESULT_DIR="${WRAPPER_SMOKE_RESULT_DIR:-${RUN_ROOT}/wrapper_smoke}"
  WRAPPER_SMOKE_DATASET="${WRAPPER_SMOKE_DATASET:-responses_api_agents/nemotron_agent/data/osworld_test_test.jsonl}"
  WRAPPER_SMOKE_LINE="${WRAPPER_SMOKE_LINE:-1}"
  WRAPPER_SMOKE_LIMIT="${WRAPPER_SMOKE_LIMIT:-1}"
  WRAPPER_SMOKE_LOG="${LOG_DIR}/wrapper_smoke.log"
  echo "Starting nemo_wrapper rollout-smoke; log=${WRAPPER_SMOKE_LOG}"
  cd "${REPO_ROOT}"
  set +e
  uv run --no-project \
    --with httpx --with backoff --with loguru \
    --with gymnasium --with pillow --with requests --with requests-toolbelt \
    --with numpy --with pyyaml --with python-dotenv --with playwright \
    --with pydrive --with boto3 --with aiohttp --with filelock \
    --with docker --with prettytable --with formulas --with lxml \
    --with cssselect --with beautifulsoup4 --with pandas --with rapidfuzz \
    --with xmltodict --with openpyxl --with tldextract --with psutil --with ngcsdk \
    python "${REPO_ROOT}/nemo_wrapper/examples/nemotron_osworld/train.py" rollout-smoke \
      --data-file "${NEMO_GYM_ROOT}/${WRAPPER_SMOKE_DATASET}" \
      --line "${WRAPPER_SMOKE_LINE}" \
      --limit "${WRAPPER_SMOKE_LIMIT}" \
      --provider-name "${OSWORLD_PROVIDER}" \
      --max-steps "${MAX_STEPS}" \
      --startup-sleep "${STARTUP_SLEEP}" \
      --settle-sleep "${SETTLE_SLEEP}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --max-tokens "${MAX_TOKENS}" \
      --policy-base-url "${VLLM_API_ENDPOINT}" \
      --policy-api-key "${VLLM_API_KEY}" \
      --policy-model-name "${SERVED_MODEL_NAME}" \
      --cache-dir "${OSWORLD_CACHE_DIR}" \
      --result-dir "${WRAPPER_SMOKE_RESULT_DIR}" \
      --no-gif \
      >"${WRAPPER_SMOKE_LOG}" 2>&1
  status=$?
  set -e
  tail -240 "${WRAPPER_SMOKE_LOG}" || true
  if [[ "${status}" != "0" ]]; then
    echo "wrapper_smoke failed with status=${status}" >&2
    exit "${status}"
  fi
  echo "wrapper_smoke_result_dir=${WRAPPER_SMOKE_RESULT_DIR}"
  echo "wrapper_smoke_summary=${WRAPPER_SMOKE_RESULT_DIR}/rollout_smoke_summary.json"
  find "${WRAPPER_SMOKE_RESULT_DIR}" -name 'traj.jsonl' -print || true
  exit 0
fi

unset PYTHONPATH
NEMO_GYM_PYTHON="${NEMO_GYM_CLI_VENV}/bin/python"

if [[ "${NEMOTRON_REUSE_CLI_VENV}" == "true" && -x "${NEMO_GYM_PYTHON}" ]]; then
  echo "Reusing isolated Nemo_GYM CLI venv: ${NEMO_GYM_CLI_VENV}"
else
  echo "Creating isolated Nemo_GYM CLI venv: ${NEMO_GYM_CLI_VENV}"
  uv venv --clear --seed --system-site-packages --python "$(command -v python)" "${NEMO_GYM_CLI_VENV}" >"${LOG_DIR}/cli_venv_create.log" 2>&1
  uv pip install --python "${NEMO_GYM_PYTHON}" \
    -e "${NEMO_GYM_ROOT}" \
    "ray[default]==2.52.1" \
    >"${LOG_DIR}/cli_venv_install.log" 2>&1
fi

CLI_AGENT_REQS_MARKER="${NEMO_GYM_CLI_VENV}/.nemotron_agent_requirements.ok"
if [[ "${NEMOTRON_INSTALL_AGENT_REQS_IN_CLI_VENV}" == "true" || "${NEMOTRON_INSTALL_AGENT_REQS_IN_CLI_VENV}" == "1" ]]; then
  if [[ ! -f "${CLI_AGENT_REQS_MARKER}" ]]; then
    echo "Installing nemotron_agent requirements into CLI venv"
    CLI_AGENT_REQS_FILTERED="${LOG_DIR}/nemotron_agent_requirements_cli.txt"
    (
      cd "${NEMO_GYM_ROOT}/responses_api_agents/nemotron_agent"
      grep -vE '^[[:space:]]*-e[[:space:]]+nemo-gym(\[dev\])?[[:space:]]*@' requirements.txt > "${CLI_AGENT_REQS_FILTERED}"
      uv pip install --python "${NEMO_GYM_PYTHON}" \
        -r "${CLI_AGENT_REQS_FILTERED}"
    ) >"${LOG_DIR}/cli_venv_agent_requirements_install.log" 2>&1
    touch "${CLI_AGENT_REQS_MARKER}"
  else
    echo "CLI venv already has nemotron_agent requirements marker: ${CLI_AGENT_REQS_MARKER}"
  fi
fi

mkdir -p "/tmp/nemo_gym_server_venvs_${JOB_TAG}"

COMMON_OVERRIDES=(
  "+config_paths=[responses_api_agents/nemotron_agent/configs/nemotron_agent.yaml]"
  "+uv_venv_dir=/tmp/nemo_gym_server_venvs_${JOB_TAG}"
  "+uv_cache_dir=${UV_CACHE_DIR}"
  "+uv_pip_set_python=true"
  "+skip_venv_if_present=${NEMO_GYM_SKIP_SERVER_VENV_IF_PRESENT}"
  "+python_version=${NEMO_GYM_PYTHON}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.provider_name=${OSWORLD_PROVIDER}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.model=${SERVED_MODEL_NAME}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.max_steps=${MAX_STEPS}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.thinking=true"
  "+nemotron_agent.responses_api_agents.nemotron_agent.temperature=${TEMPERATURE}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.top_p=${TOP_P}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.max_tokens=${MAX_TOKENS}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.startup_sleep=${STARTUP_SLEEP}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.settle_sleep=${SETTLE_SLEEP}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.test_config_base_dir=${OSWORLD_TEST_CONFIG_BASE_DIR}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.result_dir=${OSWORLD_NEMOTRON_RESULT_DIR}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.cache_dir=${OSWORLD_CACHE_DIR}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.vllm_api_endpoint=${VLLM_API_ENDPOINT}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.vllm_api_key=${VLLM_API_KEY}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.allow_openai_env_fallback=false"
  "+nemotron_agent.responses_api_agents.nemotron_agent.allow_my_ngc_api_key_fallback=false"
  "+nemotron_agent.responses_api_agents.nemotron_agent.use_builtin_task_bootstrap_without_llm=${NEMOTRON_USE_BUILTIN_TASK_BOOTSTRAP_WITHOUT_LLM}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.create_gif=${CREATE_GIF}"
  "+nemotron_agent.responses_api_agents.nemotron_agent.concurrency=${AGENT_CONCURRENCY}"
)

"${NEMO_GYM_PYTHON}" - <<'PY' >"${LOG_DIR}/uv_project_env.log" 2>&1
import annotated_types
import hydra
import omegaconf
import pydantic
import ray
import sys

print(f"python_executable={sys.executable}")
print(f"pydantic={pydantic.__file__}")
print(f"annotated_types={annotated_types.__file__}")
print(f"omegaconf={omegaconf.__file__}")
print(f"hydra={hydra.__file__}")
print(f"ray={ray.__version__}")
PY

"${NEMO_GYM_PYTHON}" -c 'from nemo_gym.cli import dump_config; dump_config()' "${COMMON_OVERRIDES[@]}" >"${LOG_DIR}/dump_config.log" 2>&1

if [[ "${NEMOTRON_INNER_MODE}" == "server" || "${NEMOTRON_INNER_MODE}" == "gym" ]]; then
  RUN_LOG="${LOG_DIR}/ng_run_foreground.log"
  echo "Starting Nemo_GYM server in foreground for debugging"
  echo "Run rollout collection from another attached shell after this prints: All 1 / 1 servers ready"
  if [[ "${NEMOTRON_FOREGROUND_TEE_LOG:-0}" == "1" ]]; then
    echo "Mirroring foreground server output to ${RUN_LOG}"
    set +e
    "${NEMO_GYM_PYTHON}" -c 'from nemo_gym.cli import run; run()' "${COMMON_OVERRIDES[@]}" 2>&1 | tee "${RUN_LOG}"
    status=${PIPESTATUS[0]}
    set -e
    exit "${status}"
  fi
  "${NEMO_GYM_PYTHON}" -c 'from nemo_gym.cli import run; run()' "${COMMON_OVERRIDES[@]}"
  exit "$?"
fi

RUN_LOG="${LOG_DIR}/ng_run.log"
echo "Starting Nemo_GYM server; log=${RUN_LOG}"
"${NEMO_GYM_PYTHON}" -c 'from nemo_gym.cli import run; run()' "${COMMON_OVERRIDES[@]}" >"${RUN_LOG}" 2>&1 &
NG_RUN_PID=$!

echo "ng_run pid=${NG_RUN_PID}"
for i in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:11000/docs" >/dev/null 2>&1; then
    echo "Nemo_GYM head server is ready"
    break
  fi
  if ! kill -0 "${NG_RUN_PID}" 2>/dev/null; then
    echo "ng_run exited before head server became ready" >&2
    tail -240 "${RUN_LOG}" >&2 || true
    exit 5
  fi
  if (( i % 12 == 0 )); then
    echo "waiting_for_ng_run_${i}"
    tail -80 "${RUN_LOG}" || true
  fi
  sleep 5
done

if ! curl -fsS --max-time 2 "http://127.0.0.1:11000/docs" >/dev/null 2>&1; then
  echo "Timed out waiting for Nemo_GYM head server" >&2
  tail -240 "${RUN_LOG}" >&2 || true
  exit 6
fi

echo "Waiting for Nemo_GYM agent servers..."
for i in $(seq 1 180); do
  if grep -q "All 1 / 1 servers ready" "${RUN_LOG}"; then
    echo "Nemo_GYM agent servers are ready"
    break
  fi
  if ! kill -0 "${NG_RUN_PID}" 2>/dev/null; then
    echo "ng_run exited before agent servers became ready" >&2
    tail -240 "${RUN_LOG}" >&2 || true
    exit 7
  fi
  if (( i % 12 == 0 )); then
    echo "waiting_for_ng_agent_servers_${i}"
    tail -80 "${RUN_LOG}" || true
  fi
  sleep 5
done

if ! grep -q "All 1 / 1 servers ready" "${RUN_LOG}"; then
  echo "Timed out waiting for Nemo_GYM agent servers" >&2
  tail -240 "${RUN_LOG}" >&2 || true
  exit 8
fi

COLLECT_LOG="${LOG_DIR}/ng_collect_rollouts.log"
COLLECT_CMD=(
  "${NEMO_GYM_PYTHON}" -c 'from nemo_gym.rollout_collection import collect_rollouts; collect_rollouts()'
  "+agent_name=nemotron_agent"
  "+input_jsonl_fpath=${DATASET}"
  "+output_jsonl_fpath=${ROLLOUT_DIR}/rollouts.jsonl"
  "+num_repeats=1"
  "+num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL}"
  "+resume_from_cache=${RESUME_FROM_CACHE}"
  "+responses_create_params.model=${SERVED_MODEL_NAME}"
)
if [[ -n "${LIMIT}" ]]; then
  COLLECT_CMD+=("+limit=${LIMIT}")
fi

echo "Starting rollout collection; log=${COLLECT_LOG}"
"${COLLECT_CMD[@]}" >"${COLLECT_LOG}" 2>&1

SUMMARY_JSON="${RUN_ROOT}/summary.json"
SUMMARY_CMD=(
  python "${SCRIPT_DIR}/../summarize_osworld_rollouts.py"
  "${ROLLOUT_DIR}/rollouts.jsonl"
  --summary-json "${SUMMARY_JSON}"
  --report-md "${RUN_ROOT}/REPORT.md"
  --label "${RUN_NAME}"
)
if [[ -n "${EXPECTED_ACCURACY:-}" ]]; then
  SUMMARY_CMD+=(--expected-accuracy "${EXPECTED_ACCURACY}")
fi
"${SUMMARY_CMD[@]}"

if [[ -f "${ROLLOUT_DIR}/rollouts_aggregate_metrics.json" ]]; then
  echo "aggregate_metrics=${ROLLOUT_DIR}/rollouts_aggregate_metrics.json"
  cat "${ROLLOUT_DIR}/rollouts_aggregate_metrics.json"
fi

echo "rollouts=${ROLLOUT_DIR}/rollouts.jsonl"
echo "summary=${SUMMARY_JSON}"
echo "run_root=${RUN_ROOT}"
