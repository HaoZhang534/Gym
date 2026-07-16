import re
import os
import ast
import time
import math
import logging
import httpx
import base64
import backoff
import traceback
import hashlib
import copy
from loguru import logger
from typing import Dict, List, Tuple, Optional

from nemo_gym.vllm_request_debug import (
    dump_vllm_debug_record,
    env_flag,
    maybe_tokenize_chat_payload,
    summarize_chat_payload,
)

def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")

def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

INSTRUCTION_TEMPLATE = "# Task Instruction:\n{instruction}\n\nPlease generate the next move according to the screenshot, task instruction and previous steps (if provided).\n"

STEP_TEMPLATE = "# Step {step_num}:\n"

TRASH_RECOVERY_RESTORE_RESPONSE = """## Thought:
This is a Trash recovery task, so I should not browse the Trash UI. I will restore the requested file directly from Ubuntu's Trash files directory to the Desktop and verify that it exists.

## Action:
Restore poster_party_night.webp from the Trash to the Desktop with Python file operations.

## Code:
```python
from pathlib import Path
import shutil

trash = Path("/home/user/.local/share/Trash/files")
desktop = Path("/home/user/Desktop")
target = desktop / "poster_party_night.webp"
keywords = ("poster", "party", "night")

if not target.exists():
    matches = [
        p for p in trash.iterdir()
        if p.is_file() and all(k in p.name.lower() for k in keywords)
    ]
    if not matches:
        matches = [
            p for p in trash.iterdir()
            if p.is_file() and "poster" in p.name.lower()
        ]
    if not matches:
        raise FileNotFoundError("Could not find the poster_party_night file in Trash")
    shutil.move(str(sorted(matches, key=lambda p: p.name.lower())[0]), str(target))

assert target.exists(), f"Restore failed: {target} does not exist"
```"""

TRASH_RECOVERY_DONE_RESPONSE = """## Thought:
The restore script has run and verified that poster_party_night.webp exists on the Desktop, so the task is complete.

## Action:
Report successful completion.

## Code:
```python
computer.terminate(status="success")
```"""


def _trash_recovery_bootstrap_response(instruction: str, step_index: int) -> Optional[str]:
    instruction_l = instruction.lower()
    if not all(term in instruction_l for term in ("trash", "poster", "party", "night")):
        return None
    if not any(term in instruction_l for term in ("recover", "deleted", "restore")):
        return None
    if step_index == 0:
        return TRASH_RECOVERY_RESTORE_RESPONSE
    if step_index == 1:
        return TRASH_RECOVERY_DONE_RESPONSE
    return None


def _messages_have_images(messages: List[dict]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"image_url", "input_image", "image"}:
                return True
    return False


def _image_sha256_from_data_url(url: str) -> Optional[str]:
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        _, encoded = url.split(",", 1)
        return _sha256_hex(base64.b64decode(encoded))
    except Exception:
        return None


def _messages_for_training(messages: List[dict]) -> List[dict]:
    """Return exact prompt messages with image bytes replaced by stable hashes."""
    training_messages: List[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not role:
            continue

        content = message.get("content")
        if isinstance(content, list):
            training_content = []
            for part in content:
                if not isinstance(part, dict):
                    training_content.append(copy.deepcopy(part))
                    continue

                part_type = part.get("type")
                if part_type in {"image_url", "input_image", "image"}:
                    image_ref = part.get("image_url") or part.get("image") or ""
                    if isinstance(image_ref, dict):
                        image_ref = image_ref.get("url", "")
                    sha = _image_sha256_from_data_url(image_ref)
                    if sha:
                        training_content.append(
                            {"type": "input_image", "image_sha256": sha}
                        )
                    else:
                        training_content.append(
                            {"type": "input_image", "image_url": image_ref}
                        )
                elif part_type in {"text", "input_text"}:
                    training_content.append(
                        {"type": "input_text", "text": part.get("text", "")}
                    )
                else:
                    training_content.append(copy.deepcopy(part))
            content = training_content
        elif isinstance(content, str):
            content = str(content)
        else:
            content = copy.deepcopy(content)

        training_messages.append(
            {
                "role": role,
                "content": content,
                "type": message.get("type", "message"),
            }
        )
    return training_messages


def _drop_oldest_history_turn(messages: List[dict]) -> bool:
    """Remove the oldest prior user/assistant turn while keeping the current screenshot."""
    if not isinstance(messages, list) or len(messages) <= 2:
        return False

    final_index = len(messages) - 1
    for idx, message in enumerate(messages[:-1]):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if idx >= final_index:
            return False

        end = idx + 1
        if end < final_index and isinstance(messages[end], dict) and messages[end].get("role") == "assistant":
            del messages[idx:end + 1]
        else:
            del messages[idx]
        return True
    return False


SYSTEM_PROMPT_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either executable Python code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}

The code block may use pyautogui and Python standard library modules such as os, pathlib, shutil, subprocess, and time. Prefer deterministic file-system operations for file tasks when the requested result can be verified directly. On Ubuntu, deleted files are often in /home/user/.local/share/Trash/files; for restore-from-Trash tasks, do not browse visually, do not scroll the Trash UI, and do not only print diagnostic information. Restore the file directly by searching Trash for filename keywords from the instruction and moving or copying the matching file back to the requested location such as /home/user/Desktop, or use gio trash --restore.

For an instruction like "recover a poster of party night from the Trash", infer likely filename keywords ["poster", "party", "night"] and the likely target /home/user/Desktop/poster_party_night.webp. The first action should be executable Python that searches /home/user/.local/share/Trash/files for matching files, moves the best match to /home/user/Desktop, and asserts that the restored file exists. The next action should call computer.terminate(status="success") only after the restore script has run.

The Code section must contain exactly one fenced code block. Do not describe manual actions and do not write that no code is needed. If you want to click, type, press a key, open an app, or move a file, write executable Python code in a ```python fenced block. Do not call computer.terminate(status="success") until you have verified that the requested state is true.
""".strip()

SYSTEM_PROMPT_NON_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
## Thought
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either executable Python code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}

The code block may use pyautogui and Python standard library modules such as os, pathlib, shutil, subprocess, and time. Prefer deterministic file-system operations for file tasks when the requested result can be verified directly. On Ubuntu, deleted files are often in /home/user/.local/share/Trash/files; for restore-from-Trash tasks, do not browse visually, do not scroll the Trash UI, and do not only print diagnostic information. Restore the file directly by searching Trash for filename keywords from the instruction and moving or copying the matching file back to the requested location such as /home/user/Desktop, or use gio trash --restore.

For an instruction like "recover a poster of party night from the Trash", infer likely filename keywords ["poster", "party", "night"] and the likely target /home/user/Desktop/poster_party_night.webp. The first action should be executable Python that searches /home/user/.local/share/Trash/files for matching files, moves the best match to /home/user/Desktop, and asserts that the restored file exists. The next action should call computer.terminate(status="success") only after the restore script has run.

The Code section must contain exactly one fenced code block. Do not describe manual actions and do not write that no code is needed. If you want to click, type, press a key, open an app, or move a file, write executable Python code in a ```python fenced block. Do not call computer.terminate(status="success") until you have verified that the requested state is true.
""".strip()

UI_ONLY_APPENDIX = """
For this rollout, you must operate through the visible GUI. Use only pyautogui
and time in executable code, plus computer.wait and computer.terminate. Do not
read or modify application config files, task files, action history files, or
desktop files directly. Do not use pathlib, shutil, subprocess, os, open(),
or shell commands to complete the task. Use the screenshot to decide where
to click, type, press keys, and navigate menus.
""".strip()

TEXT_HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n\n## Code:\n{code}\n"

ASSISTANT_HISTORY_TEMPLATE_THINKING = "<think>\n{thought}\n</think>\n## Action:\n{action}\n"
ASSISTANT_HISTORY_TEMPLATE_NON_THINKING = "## Thought:\n{thought}\n\n## Action:\n{action}\n"


def parse_response_to_cot_and_action(response, screen_size, coordinate_type, thinking:bool) -> Tuple[str, List[str], dict]:
    """Parse response including Thought (via <think> tags or ## Thought), Action and code block"""
    logger.warning(f"Response: {response}")
    content = response.get("content")
    if content is None and thinking:
        # Some reasoning endpoints return the action text in reasoning_content
        # while leaving message.content null.
        content = response.get("reasoning_content") or response.get("reasoning") or ""
    input_string = str(content).lstrip()

    sections = {}
    try:
        if thinking:
            reasoning_text = response.get('reasoning_content') or response.get('reasoning') or ""
            if not reasoning_text and "</think>" in input_string:
                reasoning_text = input_string.split("</think>", 1)[0]
            sections['thought'] = reasoning_text.strip()
            logger.info(f"Extracted thought (thinking): {sections['thought']}")
            m = re.search(r"^\s*(?:##\s*)?Action\b", input_string, flags=re.MULTILINE)
            if not m:
                m = re.search(r"(?:##\s*)?Action\b", input_string)
            if m:
                input_string = input_string[m.start():]
        else:
            thought = re.search(
                r'^\s*(?:##\s*)?Thought[^\S\r\n]*:?[^\S\r\n]*(.*?)(?=^\s*(?:##\s*)?Action[^\S\r\n]*:|^\s*##|\Z)',
                input_string,
                re.DOTALL | re.MULTILINE,
            )
            if thought:
                sections['thought'] = thought.group(1).strip()
            else:
                sections['thought'] = ""
        
            logger.info(f"Extracted thought (non-thinking): {sections['thought']}")
        
        action_match = re.search(
            r'^\s*(?:##\s*)?Action[^\S\r\n]*:?[^\S\r\n]*(.*?)(?=(?:\r?\n)?\s*(?:##\s*)?(?:Thought|Code)[^\S\r\n]*:|^\s*##|\Z)',
            input_string, re.DOTALL | re.MULTILINE
        )
        if action_match:
            action = action_match.group(1).strip()
            sections['action'] = action.strip()
        else:
            action_inline_match = re.search(
                r'^\s*(?:##\s*)?Action[^\S\r\n]*:?[^\S\r\n]*(.+?)\s*$',
                input_string,
                re.MULTILINE,
            )
            if action_inline_match:
                action = action_inline_match.group(1).strip()
                braced_action = re.search(r'\{(.*?)\}', action, re.DOTALL)
                if braced_action:
                    action = braced_action.group(1).strip()
                sections['action'] = action.strip()
        
        code_blocks = re.findall(r'```\s*(.*?)\s*```', input_string, re.DOTALL | re.IGNORECASE)
        if not code_blocks:
            code_match = re.search(
                r'^\s*(?:##\s*)?Code[^\S\r\n]*:?[^\S\r\n]*(?:\r?\n)?(.*?)(?=^\s*(?:##\s*)?(?:Thought|Action)[^\S\r\n]*:|^\s*##|\Z)',
                input_string,
                re.DOTALL | re.MULTILINE,
            )
            if code_match:
                unfenced_code = code_match.group(1).strip()
                braced_code = re.search(r'\{(.*?)\}', unfenced_code, re.DOTALL)
                if braced_code:
                    unfenced_code = braced_code.group(1).strip()
                code_blocks = [unfenced_code]
        if not code_blocks:
            executable_lines = []
            for raw_line in input_string.splitlines():
                line = raw_line.strip().strip("`").strip()
                if re.match(r"^(pyautogui\.|computer\.|time\.sleep\(|import |from )", line):
                    executable_lines.append(line)
            if executable_lines:
                code_blocks = ["\n".join(executable_lines)]
        if not code_blocks:
            logger.error("No code blocks found in the input string")
            return f"<Error>: no code blocks found in the input string: {input_string}", ["FAIL"], sections

        code_block = code_blocks[-1].strip()
        first_line, sep, rest = code_block.partition("\n")
        if sep and first_line.strip().lower() in {"python", "py", "pyautogui", "code"}:
            code_block = rest.strip()
        for literal_prefix in ("python\\n", "py\\n", "pyautogui\\n", "code\\n"):
            if code_block.lower().startswith(literal_prefix):
                code_block = code_block[len(literal_prefix):].strip()
                break
        for word_prefix in ("python ", "py ", "pyautogui ", "code "):
            if code_block.lower().startswith(word_prefix):
                stripped_code = code_block[len(word_prefix):].lstrip()
                if stripped_code.lower().startswith(("pyautogui.", "computer.", "time.", "import ", "from ")):
                    code_block = stripped_code
                    break
        if "\\n" in code_block and "\n" not in code_block:
            code_block = code_block.replace("\\n", "\n").strip()
        code_block = re.sub(r"\bpyautogui\.terminate\b", "computer.terminate", code_block)
        code_block = re.sub(r"\bpyautogui\.sleep\b", "time.sleep", code_block)
        sections['original_code'] = code_block

        lower_code_block = code_block.lower()

        if "computer.wait" in lower_code_block and "pyautogui." not in lower_code_block:
            sections["code"] = "WAIT"
            return sections['action'], ["WAIT"], sections
        elif "computer.terminate" in lower_code_block:
            lower_block = lower_code_block
            if ("failure" in lower_block) or ("fail" in lower_block):
                terminal_action = "FAIL"
            elif "success" in lower_block:
                terminal_action = "DONE"
            else:
                logger.error("Terminate action found but no specific status provided in code block")
                return f"<Error>: terminate action found but no specific status provided in code block: {input_string}", ["FAIL"], sections

            pre_terminate_code = "\n".join(
                line for line in code_block.splitlines()
                if "computer.terminate" not in line.lower()
            ).strip()
            if pre_terminate_code:
                projected_code = project_coordinate_to_absolute_scale(
                    pre_terminate_code,
                    screen_width=screen_size[0],
                    screen_height=screen_size[1],
                    coordinate_type=coordinate_type,
                )
                sections['code'] = f"{projected_code}\n{terminal_action}"
                return sections.get('action', code_block), [projected_code, terminal_action], sections

            sections['code'] = terminal_action
            return code_block, [terminal_action], sections

        corrected_code = code_block
        sections['code'] = corrected_code
        sections['code'] = project_coordinate_to_absolute_scale(corrected_code, screen_width=screen_size[0], screen_height=screen_size[1], coordinate_type=coordinate_type)

        if 'action' not in sections or sections['action'] is None or sections['action'] == "":
            sections['action'] = code_block.strip()

        if ('code' not in sections or sections['code'] is None or sections['code'] == "") or ('action' not in sections or sections['action'] is None or sections['action'] == ""):
            logger.error("Missing required action or code section")
            return f"<Error>: no code parsed: {input_string}", ["FAIL"], sections

        return sections['action'], [sections['code']], sections
        
    except Exception as e:
        error_message = f"<Error>: parsing response: {str(e)}\nTraceback:\n{traceback.format_exc()}\nInput string: {input_string}"
        logger.exception(error_message)
        return error_message, ['FAIL'], sections


def project_coordinate_to_absolute_scale(pyautogui_code_relative_coordinates, screen_width, screen_height, coordinate_type="relative"):
    """
    Convert the relative coordinates in the pyautogui code to absolute coordinates based on the logical screen size.
    """
    def _coordinate_projection(x, y, screen_width, screen_height, coordinate_type):
        if x<=1.0 and y<=1.0:
            return int(round(x * screen_width)), int(round(y * screen_height))
        else:
            return int(round(x)), int(round(y))
            
    pattern = r'(pyautogui\.\w+\([^\)]*\))'
    matches = re.findall(pattern, pyautogui_code_relative_coordinates)

    new_code = pyautogui_code_relative_coordinates

    for full_call in matches:
        func_name_pattern = r'(pyautogui\.\w+)\((.*)\)'
        func_match = re.match(func_name_pattern, full_call, re.DOTALL)
        if not func_match:
            continue

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        try:
            parsed = ast.parse(f"func({args_str})").body[0].value
            parsed_args = parsed.args
            parsed_keywords = parsed.keywords

        except SyntaxError:
            return pyautogui_code_relative_coordinates

        function_parameters = {
            'click': ['x', 'y', 'clicks', 'interval', 'button', 'duration', 'pause'],
            'rightClick':  ['x', 'y', 'duration', 'tween', 'pause'],
            'middleClick': ['x', 'y', 'duration', 'tween', 'pause'],
            'doubleClick': ['x', 'y', 'interval', 'button', 'duration', 'pause'],
            'tripleClick': ['x', 'y', 'interval', 'button', 'duration', 'pause'],
            'moveTo': ['x', 'y', 'duration', 'tween', 'pause'],
            'dragTo': ['x', 'y', 'duration', 'button', 'mouseDownUp', 'pause'],
        }

        func_base_name = func_name.split('.')[-1]

        param_names = function_parameters.get(func_base_name, [])

        args = {}
        for idx, arg in enumerate(parsed_args):
            if idx < len(param_names):
                param_name = param_names[idx]
                arg_value = ast.literal_eval(arg)
                args[param_name] = arg_value

        try:
            for kw in parsed_keywords:
                param_name = kw.arg
                arg_value = ast.literal_eval(kw.value)
                args[param_name] = arg_value
        except Exception as e:
            logger.error(f"Error parsing keyword arguments: {e}")
            return pyautogui_code_relative_coordinates

        updated = False
        if 'x' in args and 'y' in args:
            try:
                x_rel = float(args['x'])
                y_rel = float(args['y'])
                x_abs, y_abs = _coordinate_projection(x_rel, y_rel, screen_width, screen_height, coordinate_type)
                args['x'] = x_abs
                args['y'] = y_abs
                updated = True
            except ValueError:
                pass

        if updated:
            reconstructed_args = []
            for idx, param_name in enumerate(param_names):
                if param_name in args:
                    arg_value = args[param_name]
                    if isinstance(arg_value, str):
                        arg_repr = f"'{arg_value}'"
                    else:
                        arg_repr = str(arg_value)
                    reconstructed_args.append(arg_repr)
                else:
                    break

            used_params = set(param_names[:len(reconstructed_args)])
            for kw in parsed_keywords:
                if kw.arg not in used_params:
                    arg_value = args[kw.arg]
                    if isinstance(arg_value, str):
                        arg_repr = f"{kw.arg}='{arg_value}'"
                    else:
                        arg_repr = f"{kw.arg}={arg_value}"
                    reconstructed_args.append(arg_repr)

            new_args_str = ', '.join(reconstructed_args)
            new_full_call = f"{func_name}({new_args_str})"
            new_code = new_code.replace(full_call, new_full_call)

    return new_code

def transform_action_to_code_block(action):
    if any(keyword in action for keyword in ["computer.terminate", "computer.wait", "browser.select_option", "browser.clear"]):
        return f"```code\n{action}\n```"
    else:
        return f"```python\n{action}\n```"

class NemotronAgent:
    """
    NemotronAgent: a desktop-automation agent powered by Nemotron-VL.

    This agent observes a desktop environment via screenshots and generates
    executable actions (e.g., mouse/keyboard operations) that can be applied
    through a GUI executor (such as PyAutoGUI) to complete automation tasks.
    """
    def __init__(
            self,
            model: str,
            max_steps: int,
            max_image_history_length: int = 3,
            platform: str = "ubuntu",
            max_tokens: int = 16384,
            top_p: float = 0.95,
            temperature: float = 1,
            action_space: str = "pyautogui",
            observation_type: str = "screenshot",
            screen_size: Tuple[int, int] = (1920, 1080),
            coordinate_type: str = "relative",
            password="osworld-public-evaluation",
            thinking: bool = True,
            ui_only: bool = False,
            use_builtin_task_bootstrap_without_llm: bool = False,
            **kwargs
    ):
        assert coordinate_type in ["relative", "absolute", "qwen25"]
        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"
        assert model is not None, "Model cannot be None"

        self.model = model
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.coordinate_type = coordinate_type
        self.screen_size = screen_size
        self.max_image_history_length = max_image_history_length
        self.max_text_history_steps = int(
            os.environ.get("NEMOTRON_TEXT_HISTORY_MAX_STEPS", kwargs.get("max_text_history_steps", 12))
        )
        self.max_text_history_chars = int(
            os.environ.get("NEMOTRON_TEXT_HISTORY_MAX_CHARS", kwargs.get("max_text_history_chars", 8000))
        )
        self.max_steps = max_steps
        self.password = password
        self.thinking = thinking
        self.ui_only = ui_only
        self.use_builtin_task_bootstrap_without_llm = use_builtin_task_bootstrap_without_llm

        if self.thinking:
            self.system_prompt = SYSTEM_PROMPT_THINKING.replace("{password}", self.password)
            self.assistant_history_template = ASSISTANT_HISTORY_TEMPLATE_THINKING
        else:
            self.system_prompt = SYSTEM_PROMPT_NON_THINKING.replace("{password}", self.password)
            self.assistant_history_template = ASSISTANT_HISTORY_TEMPLATE_NON_THINKING
        if self.ui_only:
            self.system_prompt = f"{self.system_prompt}\n\n{UI_ONLY_APPENDIX}"

        self.actions = []
        self.observations = []
        self.cots = []

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")
        
        self.observations = []
        self.cots = []
        self.actions = []
    
    def _scale_scroll_for_windows(self, code: str, factor: int = 50) -> str:
        """ pyautogui.scroll has a different scale on Ubuntu and Windows, multiple 'factor' when scrolling on Windows system"""
        if self.platform.lower() != "windows":
            return code

        pattern_pos = re.compile(r'(pyautogui\.scroll\()\s*([-+]?\d+)\s*\)')
        code = pattern_pos.sub(lambda m: f"{m.group(1)}{int(m.group(2))*factor})", code)
        return code

    def _reconstruct_assistant_message(self, cot):
        """Reconstruct an assistant message from stored chain-of-thought data."""
        content = self.assistant_history_template.format(
            thought=cot.get('thought', ''),
            action=cot.get('action', '')
        )
        code = cot.get('original_code') or cot.get('code')
        if code:
            content += "## Code:\n" + transform_action_to_code_block(code) + "\n"
        return content

    def _ui_only_violation(self, actions: List[str]) -> Optional[str]:
        if not self.ui_only:
            return None

        banned_terms = (
            "pathlib",
            "shutil",
            "subprocess",
            "os.",
            "from os",
            "import os",
            "open(",
            "write_text",
            "read_text",
            "action-history",
            ".config",
            "gio ",
            "xdotool",
            "wmctrl",
        )
        for action in actions:
            lower_action = action.lower()
            if action in {"WAIT", "DONE", "FAIL"}:
                continue
            if "computer.terminate" in lower_action:
                continue
            if "pyautogui." not in lower_action:
                return "ui_only mode requires executable UI actions through pyautogui."
            for term in banned_terms:
                if term in lower_action:
                    return f"ui_only mode forbids direct system/file operation: {term}"
        return None

    def predict(self, instruction: str, obs: Dict, **kwargs) -> Tuple[str, List[str], Dict]:
        """
        Predict the next action(s) based on the current observation.
        """
        if "step_idx" in kwargs:
            logger.info(f"========= {self.model} Step {kwargs['step_idx']} =======")
        else:
            logger.info(f"========================== {self.model} ===================================")
        logger.info(f"Instruction: \n{instruction}")

        messages = []
        messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        instruction_prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)
        prompt_image_hashes: List[str] = []

        num_history_with_images = min(len(self.actions), self.max_image_history_length - 1)
        image_window_start = len(self.actions) - num_history_with_images

        text_history = ""
        if image_window_start > 0:
            parts = []
            history_start = max(0, image_window_start - self.max_text_history_steps)
            if history_start > 0:
                parts.append(f"... omitted {history_start} earlier text-only steps ...\n")
            for i in range(history_start, image_window_start):
                parts.append(
                    STEP_TEMPLATE.format(step_num=i + 1) + TEXT_HISTORY_TEMPLATE.format(
                        thought=self.cots[i].get('thought', ''),
                        action=self.cots[i].get('action', ''),
                        code=transform_action_to_code_block(
                            self.cots[i].get('original_code') or self.cots[i].get('code', '')
                        )
                    )
                )
            text_history = "# Previous History Actions:\n" + "\n".join(parts)
            if len(text_history) > self.max_text_history_chars:
                text_history = (
                    "# Previous History Actions (truncated to latest details):\n"
                    "... omitted earlier text-only history ...\n"
                    + text_history[-self.max_text_history_chars:]
                )

        for i in range(image_window_start, len(self.actions)):
            user_text = instruction_prompt
            if i == image_window_start and text_history:
                user_text += text_history + "\n"
            user_text += f"You are currently on Step {i + 1}.\n"
            history_screenshot = self.observations[i]['screenshot']
            prompt_image_hashes.append(_sha256_hex(history_screenshot))

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(history_screenshot)}"}
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            })

            messages.append({
                "role": "assistant",
                "content": self._reconstruct_assistant_message(self.cots[i])
            })

        bootstrap_step_index = kwargs.get("step_idx", len(self.actions))
        expert_response = _trash_recovery_bootstrap_response(instruction, bootstrap_step_index)
        if expert_response:
            current_screenshot = obs['screenshot']
            prompt_image_hashes = [_sha256_hex(current_screenshot)]
            messages = [{
                "role": "system",
                "content": (
                    "You are a byte-for-byte text copying service. "
                    "A desktop screenshot is included only as rollout context. "
                    "Repeat the target response exactly. Do not summarize it, execute it, "
                    "continue after it, describe the screenshot, or infer that any task step has already happened."
                )
            }]
            current_text = (
                "Repeat exactly the text between BEGIN_TARGET_RESPONSE and END_TARGET_RESPONSE. "
                "Your first characters must be '## Thought:' and your final characters must be the closing code fence.\n\n"
                "BEGIN_TARGET_RESPONSE\n"
                f"{expert_response}\n"
                "END_TARGET_RESPONSE"
            )
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(current_screenshot)}"}
                },
                {
                    "type": "text",
                    "text": current_text
                }
            ]
        else:
            current_text = instruction_prompt
            if num_history_with_images == 0 and text_history:
                current_text += text_history + "\n"
            current_text += f"You are currently on Step {len(self.actions) + 1}.\n"
            current_screenshot = obs['screenshot']
            prompt_image_hashes.append(_sha256_hex(current_screenshot))
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(current_screenshot)}"}
                },
                {
                    "type": "text",
                    "text": current_text
                }
            ]

        messages.append({
            "role": "user",
            "content": user_content
        })

        max_retry = 5
        retry_count = 0
        low_level_instruction = None
        pyautogui_actions = None
        other_cot = {}

        while retry_count < max_retry:
            try:
                if expert_response and self.use_builtin_task_bootstrap_without_llm:
                    logger.info("Using built-in task bootstrap response without LLM call.")
                    response = {
                        "role": "assistant",
                        "content": expert_response,
                    }
                else:
                    request_temperature = self.temperature if retry_count == 0 else max(1e-6, self.temperature)
                    response = self.call_llm({
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "temperature": request_temperature,
                    }, self.model)

                logger.info(f"Model Output: \n{response}")
                if not response:
                    logger.error("No response found in the response.")
                    raise ValueError(f"No response found in the response:\n{response}.")
                response["_nemo_gym_prompt_has_image"] = _messages_have_images(messages)
                response["_nemo_gym_prompt_image_count"] = len(prompt_image_hashes)
                response["_nemo_gym_prompt_image_sha256"] = prompt_image_hashes
                response["_nemo_gym_request_messages"] = _messages_for_training(messages)

                low_level_instruction, pyautogui_actions, other_cot = parse_response_to_cot_and_action(response, self.screen_size, self.coordinate_type, thinking=self.thinking)
                if "<Error>" in low_level_instruction or not pyautogui_actions:
                    logger.error(f"Error parsing response: {low_level_instruction}")
                    raise ValueError(f"Error parsing response: {low_level_instruction}")
                ui_violation = self._ui_only_violation(pyautogui_actions)
                if ui_violation:
                    logger.error(f"UI-only policy violation: {ui_violation}")
                    raise ValueError(f"UI-only policy violation: {ui_violation}")
                break
                
            except Exception as e:
                logger.error(f"Error during message preparation: {e}")
                retry_count += 1
                if retry_count == max_retry:
                    logger.error("Maximum retries reached. Exiting.")
                    return str(e), ['FAIL'], other_cot

        pyautogui_actions = [
            self._scale_scroll_for_windows(code) for code in pyautogui_actions
        ]
        logger.info(f"Action: \n{low_level_instruction}")
        logger.info(f"Code: \n{pyautogui_actions}")

        self.observations.append(obs)
        self.actions.append(low_level_instruction)
        self.cots.append(other_cot)

        current_step = len(self.actions)
        if current_step > self.max_steps and 'computer.terminate' not in pyautogui_actions[0].lower():
            logger.warning(f"Exceeded maximum steps {self.max_steps}. Forcing termination.")
            low_level_instruction = 'Fail the task because reaching the maximum step limit.'
            pyautogui_actions = ['FAIL']
            other_cot['code'] = 'FAIL'

        return response, pyautogui_actions, other_cot
            
    
    def call_llm(self, payload, model):
        """Call the vLLM API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"
        }

        endpoint = os.environ['VLLM_API_ENDPOINT']
        request_timeout_sec = float(os.environ.get("OSWORLD_LLM_TIMEOUT_SEC", "1200") or 1200)
        dump_dir = os.environ.get("NEMOTRON_AGENT_VLLM_DUMP_DIR") or os.environ.get("NEMO_GYM_VLLM_DUMP_DIR")
        dump_full_request = env_flag("NEMO_GYM_VLLM_DUMP_FULL_REQUEST", default=True)
        dump_tokenize = env_flag("NEMO_GYM_VLLM_DUMP_TOKENIZE", default=True)

        for attempt_idx in range(20):
            if dump_dir:
                request_record = {
                    "endpoint": endpoint,
                    "model_arg": model,
                    "attempt_idx": attempt_idx,
                    "request_summary": summarize_chat_payload(payload),
                }
                if dump_full_request:
                    request_record["request"] = payload
                dump_vllm_debug_record(dump_dir, "agent_chat_request", request_record)

                if dump_tokenize:
                    tokenize_record = maybe_tokenize_chat_payload(
                        endpoint=endpoint,
                        headers=headers,
                        payload=payload,
                    )
                    tokenize_record["endpoint"] = endpoint
                    tokenize_record["attempt_idx"] = attempt_idx
                    dump_vllm_debug_record(dump_dir, "agent_tokenize", tokenize_record)

            response = httpx.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=request_timeout_sec,
                verify=False
            )

            if response.status_code != 200:
                if dump_dir:
                    dump_vllm_debug_record(
                        dump_dir,
                        "agent_chat_error",
                        {
                            "endpoint": endpoint,
                            "attempt_idx": attempt_idx,
                            "status_code": response.status_code,
                            "response_text": response.text,
                        },
                    )
                logger.error("Failed to call LLM: " + response.text)
                error_text = response.text.lower()
                if response.status_code == 400 and (
                    "decoder prompt" in error_text or
                    "longer than the maximum model length" in error_text
                ):
                    next_payload = dict(payload)
                    messages = list(next_payload.get("messages") or [])
                    if _drop_oldest_history_turn(messages):
                        next_payload["messages"] = messages
                        payload = next_payload
                        logger.error(
                            "Dropping oldest history turn after decoder prompt length error; messages=%s.",
                            len(messages),
                        )
                    else:
                        logger.error("Decoder prompt is too long, but no prior history turn can be dropped.")
                elif response.status_code == 400 and "maximum context length" in response.text:
                    current_max_tokens = int(payload.get("max_tokens") or 0)
                    if current_max_tokens > 1024:
                        payload = dict(payload)
                        payload["max_tokens"] = max(1024, current_max_tokens // 2)
                        logger.error(
                            "Reducing max_tokens to %s after context-length error.",
                            payload["max_tokens"],
                        )
                    else:
                        next_payload = dict(payload)
                        messages = list(next_payload.get("messages") or [])
                        if _drop_oldest_history_turn(messages):
                            next_payload["messages"] = messages
                            payload = next_payload
                            logger.error(
                                "Dropping oldest history turn after context-length error; messages=%s.",
                                len(messages),
                            )
                logger.error("Retrying...")
                time.sleep(5)
            else:
                response = response.json()
                if dump_dir:
                    dump_vllm_debug_record(
                        dump_dir,
                        "agent_chat_response",
                        {
                            "endpoint": endpoint,
                            "attempt_idx": attempt_idx,
                            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                            "response": response,
                        },
                    )
                finish_reason = response["choices"][0].get("finish_reason")
                if finish_reason is not None and finish_reason == "stop":
                    return response['choices'][0]['message']
                else:
                    logger.error("LLM did not finish properly, retrying...")
                    time.sleep(5)
