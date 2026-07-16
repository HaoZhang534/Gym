# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym wrapper for the OSWorld Nemotron-VL agent.

This agent intentionally implements `/run` directly. An OSWorld rollout is a
stateful desktop episode, so the response-api agent owns environment reset,
model calls, actions, evaluation, trajectory files, and GIF rendering.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import pdb
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import Body
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


REPO_ROOT = Path(__file__).resolve().parents[4]
APP_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = APP_DIR.parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


logger = logging.getLogger("nemo_gym.nemotron_agent")

_DEBUG_TTY_OBJECTS: list[Any] = []


def _preload_thread_sensitive_modules() -> None:
    try:
        import rapidfuzz.fuzz as rapidfuzz_fuzz

        _ = rapidfuzz_fuzz.WRatio
    except Exception:
        logger.warning("Failed to preload rapidfuzz.fuzz", exc_info=True)


_preload_thread_sensitive_modules()


def _debug_env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _tty_pdb_set_trace(frame: Any) -> None:
    tty_in = open("/dev/tty", "r")
    tty_out = open("/dev/tty", "w")
    debugger = pdb.Pdb(stdin=tty_in, stdout=tty_out)
    _DEBUG_TTY_OBJECTS.extend([tty_in, tty_out, debugger])
    debugger.set_trace(frame)


def _tty_breakpointhook(*_args: Any, **_kwargs: Any) -> None:
    _tty_pdb_set_trace(sys._getframe().f_back)


if _debug_env_flag("NEMOTRON_USE_TTY_BREAKPOINT"):
    sys.breakpointhook = _tty_breakpointhook


def _force_pdb_if_requested(name: str) -> None:
    if not _debug_env_flag("NEMOTRON_FORCE_PDB_ON_RUN"):
        return

    print(
        "[NEMOTRON_DEBUG] "
        f"force_pdb={name} pid={os.getpid()} file={__file__} "
        f"PYTHONBREAKPOINT={os.environ.get('PYTHONBREAKPOINT')} "
        f"NEMOTRON_DEBUG_SYNC_RUN={os.environ.get('NEMOTRON_DEBUG_SYNC_RUN')}",
        flush=True,
    )
    try:
        _tty_pdb_set_trace(sys._getframe().f_back)
    except OSError:
        pdb.Pdb().set_trace(sys._getframe().f_back)


def _debug_value_summary(value: Any, depth: int = 0) -> str:
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, dict):
        fields = []
        for key, item in list(value.items())[:10]:
            if key in {"prompt_token_ids", "generation_token_ids", "generation_log_probs"}:
                try:
                    fields.append(f"{key}=len({len(item)})")
                except TypeError:
                    fields.append(f"{key}={type(item).__name__}")
            elif key in {"chunks", "trajectory", "input", "output"} and isinstance(item, list):
                fields.append(f"{key}=list(len={len(item)})")
            elif key == "content" and isinstance(item, str):
                fields.append(f"{key}={item[:160]!r}")
            else:
                fields.append(f"{key}={_debug_value_summary(item, depth + 1)}")
        suffix = ", ..." if len(value) > 10 else ""
        return "{" + ", ".join(fields) + suffix + "}"
    if isinstance(value, list):
        if not value:
            return "list(len=0)"
        return f"list(len={len(value)}, first={_debug_value_summary(value[0], depth + 1)})"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"{type(value).__name__}(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, str):
        return repr(value[:200])
    return repr(value)


def _maybe_debug_breakpoint(name: str, **context: Any) -> None:
    requested = {
        item.strip()
        for item in os.environ.get("NEMO_ROLLOUT_DEBUG_BREAKPOINTS", "").split(",")
        if item.strip()
    }
    if "all" not in requested and name not in requested:
        return

    print(f"[NEMO_ROLLOUT_DEBUG] breakpoint={name}", flush=True)
    for key, value in context.items():
        print(f"[NEMO_ROLLOUT_DEBUG]   {key}: {_debug_value_summary(value)}", flush=True)

    dump_only = os.environ.get("NEMO_ROLLOUT_DEBUG_DUMP_ONLY", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not dump_only:
        breakpoint()


class _RolloutTimeoutError(TimeoutError):
    pass


def _rollout_deadline(timeout_sec: Optional[float]) -> Optional[float]:
    if timeout_sec is None or timeout_sec <= 0:
        return None
    return time.monotonic() + float(timeout_sec)


def _check_rollout_deadline(deadline: Optional[float], where: str) -> None:
    if deadline is None:
        return
    if time.monotonic() >= deadline:
        raise _RolloutTimeoutError(f"Rollout timed out while {where}")


def _remaining_rollout_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _sleep_with_rollout_deadline(seconds: float, deadline: Optional[float], where: str) -> None:
    if seconds <= 0:
        return
    remaining = _remaining_rollout_seconds(deadline)
    if remaining is None:
        time.sleep(seconds)
    else:
        if remaining <= 0:
            _check_rollout_deadline(deadline, where)
        time.sleep(min(seconds, remaining))
    _check_rollout_deadline(deadline, where)


class NemotronAgentConfig(BaseResponsesAPIAgentConfig):
    model: str = "nvidia/nemotron-vl"
    provider_name: Literal[
        "aws",
        "virtualbox",
        "vmware",
        "docker",
        "azure",
        "singularity",
        "nvcf",
        "nvcf_dummy",
        "nvcf_enroot",
        "nvcf_singularity",
    ] = "nvcf"
    path_to_vm: Optional[str] = None
    headless: bool = True
    action_space: Literal["pyautogui"] = "pyautogui"
    observation_type: Literal["screenshot"] = "screenshot"
    sleep_after_execution: float = 5.0
    max_steps: int = 100
    test_config_base_dir: str = str(APP_DIR / "data" / "evaluation_examples")
    result_dir: str = "results/osworld_nemotron_agent"
    region: str = "us-east-1"
    client_password: str = ""
    screen_width: int = 1920
    screen_height: int = 1080
    password: str = "password"
    coordinate_type: Literal["relative", "qwen25", "absolute"] = "relative"
    max_image_history_length: int = 3
    thinking: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 16384
    cache_dir: str = "/tmp/osworld_nemotron_cache"
    ui_only: bool = False
    startup_sleep: float = 10.0
    settle_sleep: float = 10.0
    create_gif: bool = True
    gif_frame_duration_ms: int = 900
    concurrency: int = 1
    rollout_timeout_sec: Optional[float] = None
    close_env_after_run: bool = True
    vllm_model_server_name: Optional[str] = None
    vllm_api_endpoint: Optional[str] = None
    vllm_api_key: Optional[str] = None
    allow_openai_env_fallback: bool = True
    allow_my_ngc_api_key_fallback: bool = True
    use_builtin_task_bootstrap_without_llm: bool = False
    return_step_chunks_for_training: bool = False
    training_chunk_reward_source: Literal["final", "step"] = "final"


class NemotronAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NemotronAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class NemotronAgent(SimpleResponsesAPIAgent):
    config: NemotronAgentConfig
    sem: asyncio.Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        del body
        return NeMoGymResponse.model_validate(
            _response_object(
                model=self.config.model,
                output_text="Nemotron OSWorld agent exposes full desktop rollouts through /run.",
            )
        )

    async def run(self, body: NemotronAgentRunRequest = Body()) -> NemotronAgentVerifyResponse:
        async with self.sem:
            debug_sync_run = _debug_env_flag("NEMOTRON_DEBUG_SYNC_RUN")
            if debug_sync_run:
                result = self._run_blocking(body)
            else:
                result = await asyncio.to_thread(self._run_blocking, body)
            return NemotronAgentVerifyResponse.model_validate(result)

    def _run_blocking(self, body: NemotronAgentRunRequest) -> dict[str, Any]:
        print(
            "[NEMOTRON_DEBUG] "
            f"entered _run_blocking pid={os.getpid()} file={__file__} "
            f"PYTHONBREAKPOINT={os.environ.get('PYTHONBREAKPOINT')} "
            f"NEMOTRON_DEBUG_SYNC_RUN={os.environ.get('NEMOTRON_DEBUG_SYNC_RUN')}",
            flush=True,
        )
        _force_pdb_if_requested("_run_blocking")
        _prepare_environment(self.config)
        row = body.model_dump(exclude_unset=False)
        domain, example_id, example = _load_osworld_example(row, self.config)
        instruction = _instruction_from_body(body) or example["instruction"]

        task_index = row.get("task_index")
        rollout_index = row.get("rollout_index")
        run_id = _run_id(domain, example_id, task_index, rollout_index)
        example_result_dir = _result_dir(self.config, domain, example_id, run_id)
        example_result_dir.mkdir(parents=True, exist_ok=True)

        runtime_logger = _setup_task_logger(example, example_result_dir)
        env = None
        reward = 0.0
        trajectory: list[dict[str, Any]] = []
        final_error = None
        gif_path = None
        recording_path = example_result_dir / "recording.mp4"
        deadline = _rollout_deadline(self.config.rollout_timeout_sec)

        try:
            _check_rollout_deadline(deadline, "creating OSWorld environment")
            env = _make_env(self.config)
            _check_rollout_deadline(deadline, "creating Nemotron agent")
            agent = _make_agent(self.config)
            agent.reset(runtime_logger)

            _check_rollout_deadline(deadline, "resetting OSWorld environment")
            env.reset(task_config=example)
            _sleep_with_rollout_deadline(self.config.startup_sleep, deadline, "startup sleep")
            _check_rollout_deadline(deadline, "getting initial observation")
            obs = env._get_obs()
            _append_trajectory(
                example_result_dir=example_result_dir,
                trajectory=trajectory,
                step_num=0,
                action=None,
                natural_language_action="Initial state",
                response=None,
                reward=0.0,
                done=False,
                info={},
                screenshot=obs["screenshot"],
            )

            _start_recording(env)
            done = False
            step_idx = 0
            while not done and step_idx < self.config.max_steps:
                _check_rollout_deadline(deadline, f"predicting action for step {step_idx + 1}")
                response, actions, info_dict = agent.predict(instruction, obs, step_idx=step_idx)
                _check_rollout_deadline(deadline, f"processing model response for step {step_idx + 1}")
                runtime_logger.info("Step %s actions: %s", step_idx + 1, actions)

                first_action = str(actions[0]).lower() if actions else ""
                if not actions or actions[0] == "" or first_action.startswith(("error", "fail")):
                    final_error = f"No executable action returned at step {step_idx + 1}: {actions}"
                    runtime_logger.error(final_error)
                    break

                for action in actions:
                    _check_rollout_deadline(deadline, f"executing action for step {step_idx + 1}")
                    sleep_after_execution = self.config.sleep_after_execution
                    remaining = _remaining_rollout_seconds(deadline)
                    if remaining is not None:
                        sleep_after_execution = min(sleep_after_execution, remaining)
                    obs, step_reward, done, info = env.step(action, sleep_after_execution)
                    _check_rollout_deadline(deadline, f"recording result for step {step_idx + 1}")
                    _append_trajectory(
                        example_result_dir=example_result_dir,
                        trajectory=trajectory,
                        step_num=step_idx + 1,
                        action=action,
                        natural_language_action=(info_dict or {}).get("action"),
                        response=response,
                        reward=step_reward,
                        done=done,
                        info=info,
                        screenshot=obs["screenshot"],
                    )
                    if done:
                        break

                step_idx += 1

            _sleep_with_rollout_deadline(self.config.settle_sleep, deadline, "settle sleep")
            _check_rollout_deadline(deadline, "evaluating rollout")
            reward = float(env.evaluate())
            (example_result_dir / "result.txt").write_text(f"{reward}\n", encoding="utf-8")
        except Exception as exc:
            reward = 0.0
            final_error = f"{type(exc).__name__}: {exc}"
            runtime_logger.error("%s\n%s", final_error, traceback.format_exc())
            (example_result_dir / "result.txt").write_text("0.0\n", encoding="utf-8")
            _write_jsonl(
                example_result_dir / "traj.jsonl",
                {
                    "Error": final_error,
                    "traceback": traceback.format_exc(),
                    "domain": domain,
                    "example_id": example_id,
                },
            )
        finally:
            if env is not None:
                _end_recording(env, recording_path)
                if self.config.close_env_after_run:
                    try:
                        env.close()
                    except Exception as exc:
                        runtime_logger.warning("Failed to close OSWorld env: %s", exc)

        if self.config.create_gif:
            try:
                gif_path = _make_trajectory_gif(example_result_dir, trajectory, self.config.gif_frame_duration_ms)
            except Exception as exc:
                runtime_logger.warning("Failed to create trajectory GIF: %s", exc)

        summary = {
            "domain": domain,
            "example_id": example_id,
            "instruction": instruction,
            "reward": reward,
            "steps": max(0, len(trajectory) - 1),
            "result_dir": str(example_result_dir),
            "trajectory_jsonl": str(example_result_dir / "traj.jsonl"),
            "trajectory_gif": str(gif_path) if gif_path else None,
            "recording": str(recording_path) if recording_path.exists() else None,
            "error": final_error,
        }
        output_text = json.dumps(summary, ensure_ascii=False, indent=2)

        result = {
            "responses_create_params": body.responses_create_params.model_dump(mode="json"),
            "reward": reward,
            "response": _response_object(model=self.config.model, output_text=output_text),
            **summary,
            "trajectory": trajectory,
        }
        _write_json(example_result_dir / "rollout_result.json", result)
        if self.config.return_step_chunks_for_training:
            chunks = _training_chunks_from_trajectory(
                config=self.config,
                body=body,
                trajectory=trajectory,
                final_reward=reward,
                instruction=instruction,
                result_dir=example_result_dir,
                summary=summary,
            )
            if not chunks:
                terminal_action = next(
                    (
                        entry.get("action")
                        for entry in reversed(trajectory)
                        if entry.get("action") in {"DONE", "FAIL"}
                    ),
                    None,
                )
                last_response = next(
                    (
                        entry.get("response")
                        for entry in reversed(trajectory)
                        if entry.get("response") is not None
                    ),
                    None,
                )
                chunk_skip_reason = {
                    "final_error": final_error,
                    "terminal_action": terminal_action,
                    "last_response_excerpt": str(last_response)[:1000]
                    if last_response is not None
                    else None,
                }
                runtime_logger.warning(
                    "Rollout for %s/%s produced no trainable chunks; returning an "
                    "empty chunk list so the RL batch can skip this sample. "
                    "reason=%s",
                    domain,
                    example_id,
                    chunk_skip_reason,
                )
                result["chunks"] = []
                result["num_training_chunks"] = 0
                result["chunk_skip_reason"] = chunk_skip_reason
                _write_json(example_result_dir / "rollout_result.json", result)
                return result
            result["chunks"] = chunks
            result["num_training_chunks"] = len(chunks)
            _write_json(example_result_dir / "rollout_result.json", result)
        return result


def _prepare_environment(config: NemotronAgentConfig) -> None:
    if config.vllm_model_server_name:
        from nemo_gym.server_utils import get_server_url

        os.environ["VLLM_API_ENDPOINT"] = f"{get_server_url(config.vllm_model_server_name)}/v1/chat/completions"
    elif config.vllm_api_endpoint:
        os.environ["VLLM_API_ENDPOINT"] = config.vllm_api_endpoint
    if config.vllm_api_key is not None:
        os.environ["VLLM_API_KEY"] = config.vllm_api_key

    if config.allow_my_ngc_api_key_fallback:
        ngc_key = os.environ.get("NGC_API_KEY") or os.environ.get("MY_NGC_API_KEY")
        if ngc_key:
            os.environ.setdefault("NGC_API_KEY", ngc_key)
            os.environ.setdefault("VLLM_API_KEY", ngc_key)
            os.environ.setdefault("VLLM_API_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")

    if config.allow_openai_env_fallback:
        if "VLLM_API_ENDPOINT" not in os.environ and os.environ.get("OPENAI_API_BASE"):
            os.environ["VLLM_API_ENDPOINT"] = os.environ["OPENAI_API_BASE"]
        if "VLLM_API_KEY" not in os.environ and os.environ.get("OPENAI_API_KEY"):
            os.environ["VLLM_API_KEY"] = os.environ["OPENAI_API_KEY"]

    missing = [name for name in ("VLLM_API_ENDPOINT", "VLLM_API_KEY") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required Nemotron vLLM environment variables: {missing}")


def _load_osworld_example(row: dict[str, Any], config: NemotronAgentConfig) -> tuple[str, str, dict[str, Any]]:
    metadata = row.get("verifier_metadata") or {}
    domain = row.get("domain") or row.get("osworld_domain") or metadata.get("domain") or metadata.get("osworld_domain")
    example_id = (
        row.get("example_id")
        or row.get("osworld_example_id")
        or row.get("id")
        or metadata.get("example_id")
        or metadata.get("osworld_example_id")
    )

    if isinstance(example_id, str) and "/" in example_id and not domain:
        domain, example_id = example_id.split("/", 1)

    config_path = row.get("task_config_path") or metadata.get("task_config_path")
    config_file: Optional[Path] = None
    if config_path:
        candidate = _resolve_path(config_path)
        if candidate.exists():
            config_file = candidate

    if config_file is None:
        if not domain or not example_id:
            raise ValueError("OSWorld rows must include domain/example_id or task_config_path.")
        config_file = _resolve_path(config.test_config_base_dir) / "examples" / str(domain) / f"{example_id}.json"

    with config_file.open("r", encoding="utf-8") as f:
        example = json.load(f)

    return str(domain or config_file.parent.name), str(example["id"]), example


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path

    for base in (APP_DIR, Path.cwd(), NEMO_GYM_ROOT, REPO_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate

    return (APP_DIR / path).resolve()


def _instruction_from_body(body: NemotronAgentRunRequest) -> Optional[str]:
    input_items = body.responses_create_params.input
    if isinstance(input_items, str):
        return input_items

    for item in reversed(input_items or []):
        role = getattr(item, "role", None)
        if role != "user":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            return content
    return None


def _make_env(config: NemotronAgentConfig):
    from desktop_env.desktop_env import DesktopEnv

    screen_size = (config.screen_width, config.screen_height)
    common_kwargs = dict(
        path_to_vm=config.path_to_vm,
        action_space=config.action_space,
        provider_name=config.provider_name,
        screen_size=screen_size,
        headless=config.headless,
        os_type="Ubuntu",
        require_a11y_tree=False,
        client_password=config.client_password,
    )

    if config.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        ami_id = IMAGE_ID_MAP[config.region].get(screen_size, IMAGE_ID_MAP[config.region][(1920, 1080)])
        return DesktopEnv(
            **common_kwargs,
            region=config.region,
            snapshot_name=ami_id,
            enable_proxy=True,
        )

    return DesktopEnv(**common_kwargs, cache_dir=config.cache_dir)


def _make_agent(config: NemotronAgentConfig):
    from mm_agents.nvidia import NemotronAgent as OSWorldNemotronAgent

    return OSWorldNemotronAgent(
        model=config.model,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        temperature=config.temperature,
        action_space=config.action_space,
        observation_type=config.observation_type,
        screen_size=(config.screen_width, config.screen_height),
        coordinate_type=config.coordinate_type,
        max_image_history_length=config.max_image_history_length,
        max_steps=config.max_steps,
        thinking=config.thinking,
        password=config.password,
        ui_only=config.ui_only,
        use_builtin_task_bootstrap_without_llm=config.use_builtin_task_bootstrap_without_llm,
    )


def _run_id(domain: str, example_id: str, task_index: Any, rollout_index: Any) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d@%H%M%S")
    if task_index is not None and rollout_index is not None:
        return f"task{task_index}_rollout{rollout_index}_{timestamp}"
    return f"{domain}_{example_id}_{timestamp}_{uuid4().hex[:8]}"


def _result_dir(config: NemotronAgentConfig, domain: str, example_id: str, run_id: str) -> Path:
    model_name = config.model.replace("/", "__")
    return (
        Path(config.result_dir)
        / config.action_space
        / config.observation_type
        / model_name
        / domain
        / example_id
        / run_id
    )


def _setup_task_logger(example: dict[str, Any], result_dir: Path) -> logging.Logger:
    runtime_logger = logging.getLogger(f"nemo_gym.nemotron_agent.{example['id']}.{uuid4().hex[:8]}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.propagate = True
    handler = logging.FileHandler(result_dir / "runtime.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s %(module)s/%(lineno)d] %(message)s"))
    runtime_logger.addHandler(handler)
    return runtime_logger


def _append_trajectory(
    *,
    example_result_dir: Path,
    trajectory: list[dict[str, Any]],
    step_num: int,
    action: Any,
    natural_language_action: Optional[str],
    response: Any,
    reward: float,
    done: bool,
    info: dict[str, Any],
    screenshot: bytes,
) -> None:
    timestamp = dt.datetime.now().strftime("%Y%m%d@%H%M%S%f")
    screenshot_file = f"step_{step_num}_{timestamp}.png"
    (example_result_dir / screenshot_file).write_bytes(screenshot)
    entry = {
        "step_num": step_num,
        "action": action,
        "natural_language_action": natural_language_action,
        "action_timestamp": timestamp,
        "response": response,
        "reward": reward,
        "done": done,
        "info": info,
        "screenshot_file": screenshot_file,
        "screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
    }
    trajectory.append(entry)
    _write_jsonl(example_result_dir / "traj.jsonl", entry)


def _write_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str))
        f.write("\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2), encoding="utf-8")


def _training_chunks_from_trajectory(
    *,
    config: NemotronAgentConfig,
    body: NemotronAgentRunRequest,
    trajectory: list[dict[str, Any]],
    final_reward: float,
    instruction: str,
    result_dir: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert each generated OSWorld action into an independent NeMo-RL sample.

    The OSWorld agent talks to the policy through Chat Completions, so the
    original /run result is not token-contiguous across steps. Returning chunks
    lets Nemo_RL train each model call with its own prompt token ids/logprobs.
    """
    base_params = body.responses_create_params.model_dump(mode="json")
    screenshot_paths_by_sha = _trajectory_screenshot_paths_by_sha(
        trajectory, result_dir
    )
    chunks: list[dict[str, Any]] = []
    _maybe_debug_breakpoint(
        "agent_before_chunks",
        trajectory_len=len(trajectory),
        final_reward=final_reward,
        instruction=instruction,
        result_dir=str(result_dir),
        summary=summary,
    )

    for entry_idx, entry in enumerate(trajectory):
        response = entry.get("response")
        if not _has_training_token_info(response):
            continue

        prompt_has_image = bool(response.get("_nemo_gym_prompt_has_image", True))
        input_screenshot_entry = _input_screenshot_entry_for_entry(trajectory, entry_idx)
        input_screenshot = (
            _screenshot_path_from_entry(input_screenshot_entry, result_dir)
            if prompt_has_image
            else None
        )
        chunk_params = dict(base_params)
        chunk_params["model"] = config.model
        request_messages = _training_input_from_request_messages(
            response=response,
            screenshot_paths_by_sha=screenshot_paths_by_sha,
        )
        if request_messages is not None:
            chunk_params["input"] = request_messages
            training_input_source = "actual_request_messages"
        else:
            chunk_params["input"] = _training_chunk_input(
                instruction=instruction,
                step_num=int(entry.get("step_num") or len(chunks) + 1),
                screenshot_path=input_screenshot,
            )
            training_input_source = "synthetic_training_chunk"

        chunk_reward = (
            float(entry.get("reward") or 0.0)
            if config.training_chunk_reward_source == "step"
            else float(final_reward)
        )
        metadata = {
            "domain": summary["domain"],
            "example_id": summary["example_id"],
            "result_dir": summary["result_dir"],
            "trajectory_jsonl": summary["trajectory_jsonl"],
            "trajectory_gif": summary["trajectory_gif"],
            "step_num": entry.get("step_num"),
            "action": entry.get("action"),
            "natural_language_action": entry.get("natural_language_action"),
            "screenshot_file": entry.get("screenshot_file"),
            "input_screenshot": str(input_screenshot) if input_screenshot else None,
            "input_screenshot_sha256": (
                input_screenshot_entry.get("screenshot_sha256")
                if input_screenshot_entry
                else None
            ),
            "prompt_has_image": prompt_has_image,
            "prompt_image_count": response.get("_nemo_gym_prompt_image_count"),
            "prompt_image_sha256": response.get("_nemo_gym_prompt_image_sha256"),
            "training_input_source": training_input_source,
        }
        chunk = {
            "responses_create_params": chunk_params,
            "response": _training_response_object(config.model, response),
            "reward": chunk_reward,
            "chunk_index": len(chunks),
            "metadata": metadata,
        }
        _maybe_debug_breakpoint(
            "agent_chunk",
            entry_idx=entry_idx,
            chunk_index=chunk["chunk_index"],
            entry=entry,
            chunk=chunk,
        )
        chunks.append(chunk)

    _maybe_debug_breakpoint("agent_after_chunks", num_chunks=len(chunks), chunks=chunks)
    return chunks


def _trajectory_screenshot_paths_by_sha(
    trajectory: list[dict[str, Any]],
    result_dir: Path,
) -> dict[str, str]:
    paths_by_sha: dict[str, str] = {}
    for entry in trajectory:
        sha = entry.get("screenshot_sha256")
        path = _screenshot_path_from_entry(entry, result_dir)
        if isinstance(sha, str) and path is not None:
            paths_by_sha[sha] = str(path)
    return paths_by_sha


def _training_input_from_request_messages(
    *,
    response: dict[str, Any],
    screenshot_paths_by_sha: dict[str, str],
) -> Optional[list[dict[str, Any]]]:
    messages = response.get("_nemo_gym_request_messages")
    if not isinstance(messages, list):
        return None

    resolved_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str):
            return None
        content = message.get("content")
        if isinstance(content, list):
            resolved_content = []
            for part in content:
                resolved_part = _resolve_training_message_part(
                    part, screenshot_paths_by_sha
                )
                if resolved_part is None:
                    return None
                resolved_content.append(resolved_part)
            content = resolved_content
        elif not isinstance(content, str):
            return None

        resolved_messages.append(
            {
                "role": role,
                "content": content,
                "type": message.get("type", "message"),
            }
        )

    return resolved_messages


def _resolve_training_message_part(
    part: Any,
    screenshot_paths_by_sha: dict[str, str],
) -> Optional[dict[str, Any]]:
    if not isinstance(part, dict):
        return None

    part_type = part.get("type")
    if part_type in {"input_image", "image_url", "image"}:
        image_url = part.get("image_url") or part.get("image")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        image_sha = part.get("image_sha256")
        if isinstance(image_sha, str):
            image_url = screenshot_paths_by_sha.get(image_sha)
        if not image_url:
            return None
        return {"type": "input_image", "image_url": str(image_url)}

    if part_type in {"input_text", "text"}:
        return {"type": "input_text", "text": str(part.get("text", ""))}

    return json.loads(json.dumps(part, ensure_ascii=False, default=str))


def _has_training_token_info(response: Any) -> bool:
    return isinstance(response, dict) and all(
        key in response
        for key in ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
    )


def _input_screenshot_entry_for_entry(
    trajectory: list[dict[str, Any]],
    entry_idx: int,
) -> Optional[dict[str, Any]]:
    if entry_idx <= 0:
        return None

    return trajectory[entry_idx - 1]


def _screenshot_path_from_entry(
    entry: Optional[dict[str, Any]],
    result_dir: Path,
) -> Optional[Path]:
    if entry is None:
        return None

    screenshot_file = entry.get("screenshot_file")
    if not screenshot_file:
        return None

    screenshot_path = result_dir / str(screenshot_file)
    if screenshot_path.exists():
        return screenshot_path
    return None


def _training_chunk_input(
    *,
    instruction: str,
    step_num: int,
    screenshot_path: Optional[Path],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"{instruction}\n\nTraining chunk for OSWorld step {step_num}.",
        }
    ]
    if screenshot_path is not None:
        content.insert(
            0,
            {
                "type": "input_image",
                "image_url": str(screenshot_path),
            },
        )
    return [
        {
            "role": "user",
            "content": content,
            "type": "message",
        }
    ]


def _training_response_object(model: str, message: dict[str, Any]) -> dict[str, Any]:
    output_text = message.get("content") or ""
    response_obj = _response_object(model=model, output_text=output_text)
    output_item = response_obj["output"][-1]
    output_item["prompt_token_ids"] = _coerce_int_list(message["prompt_token_ids"])
    output_item["generation_token_ids"] = _coerce_int_list(message["generation_token_ids"])
    output_item["generation_log_probs"] = [float(v) for v in message["generation_log_probs"]]
    return response_obj


def _coerce_int_list(values: Any) -> list[int]:
    return [int(v) for v in values]


def _start_recording(env: Any) -> None:
    try:
        env.controller.start_recording()
    except Exception as exc:
        logger.warning("Failed to start OSWorld recording: %s", exc)


def _end_recording(env: Any, recording_path: Path) -> None:
    try:
        env.controller.end_recording(str(recording_path))
    except Exception as exc:
        logger.warning("Failed to end OSWorld recording: %s", exc)


def _make_trajectory_gif(result_dir: Path, trajectory: list[dict[str, Any]], frame_duration_ms: int) -> Optional[Path]:
    if not trajectory:
        return None

    from PIL import Image, ImageDraw

    frames = []
    for step in trajectory:
        screenshot_file = step.get("screenshot_file")
        if not screenshot_file:
            continue
        image_path = result_dir / screenshot_file
        if not image_path.exists():
            continue
        img = Image.open(image_path).convert("RGB")
        overlay_height = max(44, img.height // 18)
        canvas = Image.new("RGB", (img.width, img.height + overlay_height), (18, 18, 18))
        canvas.paste(img, (0, overlay_height))
        draw = ImageDraw.Draw(canvas)
        label = _gif_label(step)
        draw.text((12, 12), label[:240], fill=(255, 255, 255))
        frames.append(canvas)

    if not frames:
        return None

    gif_path = result_dir / "trajectory.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return gif_path


def _gif_label(step: dict[str, Any]) -> str:
    action = step.get("natural_language_action") or step.get("action") or "Initial state"
    action = " ".join(str(action).split())
    return f"Step {step.get('step_num')} | reward={step.get('reward')} | done={step.get('done')} | {action}"


def _response_object(model: str, output_text: str) -> dict[str, Any]:
    response_id = f"resp_{uuid4().hex}"
    return {
        "id": response_id,
        "created_at": time.time(),
        "model": model,
        "object": "response",
        "output": [
            {
                "id": f"msg_{uuid4().hex}",
                "content": [
                    {
                        "annotations": [],
                        "text": output_text,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


if __name__ == "__main__":
    NemotronAgent.run_webserver()
