# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remote desktop environment tool (session-based HTTP API).

Communicates with an externally-managed desktop environment service via the
following endpoints::

    POST /session/create                           body: {"task_id": ...}
    POST /session/{session_id}/step                body: {"action": "pyautogui...", "pause": 2.0}
    POST /session/{session_id}/evaluate
    POST /session/{session_id}/close

Lifecycle: ``create → (step)* → evaluate → close``.

The tool exposes the Qwen-VL ``computer_use`` OpenAI function schema to the
LLM (structured ``{action, coordinate, text, ...}`` args). It translates
those structured actions into ``pyautogui`` code snippets that the remote
service executes in its sandboxed desktop. This keeps prompt/action formats
stable even if the backend protocol evolves.

The service is expected to return a ``screenshot`` field (base64
encoded PNG) from ``/session/create`` and ``/session/{id}/step`` so that the
agent can feed the observation back to the LLM. Missing screenshots are retried
once with a short WAIT observation step, then treated as environment failures if
the retry also fails.
"""

import asyncio
import base64
import io
import importlib.util
import json
import logging
import math
import os
import random
import re
import socket
import sys
import time
from typing import Any, Optional
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

try:
    from recipe.fully_async_gui_agent.computer_use_schema import (
        _COMPUTER_USE_TOOL,
        build_computer_use_system_prompt,
        build_computer_use_tool_dict,
    )
except ModuleNotFoundError:
    _schema_path = os.path.join(os.path.dirname(__file__), "computer_use_schema.py")
    _schema_spec = importlib.util.spec_from_file_location("_gui_agent_computer_use_schema", _schema_path)
    if _schema_spec is None or _schema_spec.loader is None:
        raise RuntimeError(f"Unable to load computer_use_schema.py from {_schema_path}")
    _schema_module = importlib.util.module_from_spec(_schema_spec)
    sys.modules[_schema_spec.name] = _schema_module
    _schema_spec.loader.exec_module(_schema_module)
    _COMPUTER_USE_TOOL = _schema_module._COMPUTER_USE_TOOL
    build_computer_use_system_prompt = _schema_module.build_computer_use_system_prompt
    build_computer_use_tool_dict = _schema_module.build_computer_use_tool_dict

# We bypass the ``logging`` framework for INFO/DEBUG lines because verl's
# global ``basicConfig(WARNING)`` plus Ray's early-attached handlers silently
# drop them regardless of per-logger levels. ``print`` goes straight to stdout
# (picked up by Ray's log forwarder) and is unaffected by any of that.
# ``logger`` is still kept for ERROR / exc_info warnings where the stack trace
# is valuable.
logger = logging.getLogger(__name__)


class DesktopEnvStepError(RuntimeError):
    """Raised when a backend /step failure should discard the trajectory."""


_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_LOG_LEVEL = os.getenv("GUI_AGENT_LOGGING_LEVEL", "ERROR").upper()
_LOG_THRESHOLD = _LOG_LEVELS.get(_LOG_LEVEL, _LOG_LEVELS["ERROR"])


def _ts() -> str:
    """Short timestamp prefix for error logs, e.g. ``2026-04-24 14:47:03.123``."""
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', lt)}.{ms:03d}"


def _log(msg: str, *, level: str = "DEBUG", debug: bool | None = None) -> None:
    """Print-based logger controlled by ``GUI_AGENT_LOGGING_LEVEL``.

    Normal DesktopEnv traces use DEBUG level; failures use ERROR level.
    ``flush=True`` + ``stderr`` ensures lines survive worker crashes.
    """
    if debug:
        level = "DEBUG"
    level = level.upper()
    if _LOG_LEVELS.get(level, _LOG_LEVELS["INFO"]) < _LOG_THRESHOLD:
        return
    print(f"[{_ts()}] {msg}", file=sys.stderr, flush=True)


# Truncate very long payload/response strings in logs to keep output readable.
_MAX_LOG_BODY = int(os.getenv("DESKTOP_ENV_LOG_MAX_BODY", "4096"))


def _short_repr(obj: Any, limit: int = _MAX_LOG_BODY) -> str:
    """Return a truncated ``repr`` suitable for logs.

    Screenshot base64 blobs can be huge; we elide them so logs stay useful.
    """
    try:
        if isinstance(obj, dict):
            redacted = {}
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > 200 and k in ("screenshot", "image", "a11y_tree"):
                    redacted[k] = f"<{k}: {len(v)} chars elided>"
                elif isinstance(v, dict):
                    redacted[k] = {
                        sk: (
                            f"<{sk}: {len(sv)} chars elided>"
                            if isinstance(sv, str) and len(sv) > 200 and sk in ("screenshot", "image", "a11y_tree")
                            else sv
                        )
                        for sk, sv in v.items()
                    }
                else:
                    redacted[k] = v
            s = repr(redacted)
        else:
            s = repr(obj)
    except Exception:
        s = f"<unreprable {type(obj).__name__}>"
    if len(s) > limit:
        return s[:limit] + f"... <{len(s) - limit} more chars>"
    return s


def _request_log_context(request_body: dict) -> str:
    request_id = request_body.get("request_id", "<none>")
    session_id = request_body.get("session_id")
    if session_id is None:
        return f"request_id={request_id}"
    return f"request_id={request_id} session_id={session_id}"


def _format_http_error_body(body: str) -> str:
    if not body:
        return "<empty>"
    try:
        payload = json.loads(body)
    except Exception:
        return _short_repr(body)
    if isinstance(payload, dict) and "detail" in payload:
        return f"proxy_detail={_short_repr(payload['detail'])}"
    return _short_repr(payload)


def _build_tool_schema(screen_width: int, screen_height: int) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        build_computer_use_tool_dict(screen_width, screen_height)
    )


# ---------------------------------------------------------------------------
# Action → pyautogui code translation
# ---------------------------------------------------------------------------


def _denorm_coord(
    coord,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Scale model coordinates from a 0..(source-1) grid to real desktop pixels."""
    x, y = coord
    source_x_max = max(1, source_width - 1)
    source_y_max = max(1, source_height - 1)
    abs_x = int(float(x) / float(source_x_max) * target_width)
    abs_y = int(float(y) / float(source_y_max) * target_height)
    abs_x = max(0, min(target_width - 1, abs_x))
    abs_y = max(0, min(target_height - 1, abs_y))
    return abs_x, abs_y


def _clean_keys(keys: Any) -> list[str]:
    """Normalize model-emitted key arrays into pyautogui key names."""
    if not keys:
        return []
    cleaned_keys = []
    for key in keys:
        if isinstance(key, str):
            if key.startswith("keys=["):
                key = key[6:]
            if key.endswith("]"):
                key = key[:-1]
            if key.startswith("['") or key.startswith('["'):
                key = key[2:] if len(key) > 2 else key
            if key.endswith("']") or key.endswith('"]'):
                key = key[:-2] if len(key) > 2 else key
            key = key.strip()
        cleaned_keys.append(str(key))
    return cleaned_keys


def _with_held_keys(code: str, keys: list[str]) -> str:
    """Wrap pyautogui code with keyDown/keyUp for modifier-style actions."""
    if not keys:
        return code
    lines = [f"pyautogui.keyDown({key!r})" for key in keys]
    lines.extend(line for line in code.split("\n") if line)
    lines.extend(f"pyautogui.keyUp({key!r})" for key in reversed(keys))
    return "\n".join(lines)


_ACTION_ALIASES: dict[str, str] = {
    "click": "left_click",
    "drag": "left_click_drag",
}


def _normalize_action_alias(action: Any) -> Any:
    """Rewrite known OWL-style aliases; leave all other actions unchanged."""
    if not isinstance(action, str):
        return action
    return _ACTION_ALIASES.get(action, action)


def _translate_action_to_pyautogui(
    parameters: dict[str, Any],
    source_screen_width: int,
    source_screen_height: int,
    real_screen_width: int,
    real_screen_height: int,
) -> Optional[str]:
    """Translate a Qwen-VL computer_use structured action into pyautogui code.

    Coordinates output by Qwen-VL are interpreted in the prompt screen space
    and scaled to the real desktop pixels before being embedded in the
    generated pyautogui snippet. The action semantics intentionally mirror
    OSWorld's ``Qwen3VLAgent.parse_response``.

    Returns ``None`` for virtual actions that are handled by the agent loop
    directly (``terminate``, ``answer``) and do not map to a backend step.
    """
    action = _normalize_action_alias(parameters.get("action", ""))
    coord = parameters.get("coordinate")
    keys = _clean_keys(parameters.get("keys", []))

    if action == "terminate" or action == "answer":
        return None

    def adjusted_coordinate(default: tuple[int, int] | None = None) -> tuple[int, int] | None:
        if coord is None:
            return default
        return _denorm_coord(coord, source_screen_width, source_screen_height, real_screen_width, real_screen_height)

    if action == "mouse_move":
        x, y = adjusted_coordinate(default=(0, 0))
        return f"pyautogui.moveTo({x}, {y})"

    if action == "left_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return _with_held_keys("pyautogui.click()", keys)
        x, y = adjusted
        return _with_held_keys(f"pyautogui.click({x}, {y})", keys)

    if action == "right_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return _with_held_keys("pyautogui.rightClick()", keys)
        x, y = adjusted
        return _with_held_keys(f"pyautogui.rightClick({x}, {y})", keys)

    if action == "middle_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return _with_held_keys("pyautogui.middleClick()", keys)
        x, y = adjusted
        return _with_held_keys(f"pyautogui.middleClick({x}, {y})", keys)

    if action == "double_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return _with_held_keys("pyautogui.doubleClick()", keys)
        x, y = adjusted
        return _with_held_keys(f"pyautogui.doubleClick({x}, {y})", keys)

    if action == "triple_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return _with_held_keys("pyautogui.tripleClick()", keys)
        x, y = adjusted
        return _with_held_keys(f"pyautogui.tripleClick({x}, {y})", keys)

    if action == "left_click_drag":
        adjusted = adjusted_coordinate(default=(0, 0))
        x, y = adjusted
        duration = parameters.get("duration", 0.5)
        return f"pyautogui.dragTo({x}, {y}, duration={duration})"

    if action == "type":
        text = parameters.get("text", "")
        code_lines = []
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if line:
                code_lines.append(f"pyautogui.typewrite({line!r}, interval=0.03)")
            if idx < len(lines) - 1:
                code_lines.append("pyautogui.press('enter')")
        return "\n".join(code_lines)

    if action == "key":
        cleaned_keys = keys
        keys_str = ", ".join(repr(str(key)) for key in cleaned_keys)
        if len(cleaned_keys) > 1:
            return f"pyautogui.hotkey({keys_str})"
        return f"pyautogui.press({keys_str})"

    if action == "scroll":
        pixels = int(parameters.get("pixels", 0) or 0)
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return _with_held_keys(f"pyautogui.scroll({pixels})", keys)
        x, y = adjusted
        scroll_code = _with_held_keys(f"pyautogui.scroll({pixels})", keys)
        return f"pyautogui.moveTo({x}, {y})\n{scroll_code}"

    if action == "hscroll":
        pixels = int(parameters.get("pixels", 0) or 0)
        return _with_held_keys(f"pyautogui.hscroll({pixels})", keys)

    if action == "wait":
        return "WAIT"

    raise ValueError(f"Unknown computer_use action: {action!r}")


# Whitelist of action names understood by ``_translate_action_to_pyautogui``
# plus the virtual actions handled in ``DesktopEnvTool.execute``. Used to
# give structured feedback on unknown actions without aborting the rollout.
_VALID_COMPUTER_USE_ACTIONS: tuple[str, ...] = (
    # physical actions
    "mouse_move",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
    "type",
    "key",
    "scroll",
    "hscroll",
    "wait",
    # virtual actions (no backend step)
    "terminate",
    "answer",
)
_COORD_ACTIONS = {
    "mouse_move",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
}
_KEYS_ACTIONS = {
    "key",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "hscroll",
}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_coordinate(coord: Any) -> str | None:
    if not isinstance(coord, list | tuple) or len(coord) != 2:
        return "coordinate must be an array [x, y] with two numbers"
    if not all(_is_finite_number(value) for value in coord):
        return "coordinate must be an array [x, y] with two finite numbers"
    return None


def _validate_action_parameters(parameters: dict[str, Any]) -> str | None:
    action = _normalize_action_alias(parameters.get("action"))
    if not isinstance(action, str) or not action:
        return "action must be a non-empty string"

    if action not in _VALID_COMPUTER_USE_ACTIONS:
        valid_list = ", ".join(_VALID_COMPUTER_USE_ACTIONS)
        return f"unknown action {action!r}. Valid computer_use actions are: {valid_list}"

    keys = parameters.get("keys")
    if keys is not None and action in _KEYS_ACTIONS and not isinstance(keys, list):
        return f"action {action!r} requires keys as an array when provided"

    if action in _COORD_ACTIONS:
        coord = parameters.get("coordinate")
        if coord is not None:
            if error := _validate_coordinate(coord):
                return f"action {action!r} has invalid {error}"
    if action == "type":
        if "text" in parameters and not isinstance(parameters.get("text"), str):
            return "action 'type' requires text as a string"
    elif action in {"scroll", "hscroll"}:
        coord = parameters.get("coordinate")
        if coord is not None:
            if error := _validate_coordinate(coord):
                return f"action {action!r} has invalid {error}"
        if "pixels" in parameters and not _is_finite_number(parameters.get("pixels")):
            return f"action {action!r} requires pixels as a finite number"
    elif action == "wait":
        wait_time = parameters.get("time")
        if wait_time is not None and (not _is_finite_number(wait_time) or float(wait_time) < 0):
            return "action 'wait' time must be a non-negative finite number when provided"
    elif action == "answer":
        has_text = "text" in parameters
        has_status = "status" in parameters
        if has_text and not isinstance(parameters.get("text"), str):
            return "action 'answer' requires text as a string when provided"
        if has_status and parameters.get("status") not in {"success", "failure"}:
            return "action 'answer' requires status to be either 'success' or 'failure' when provided"
    # NOTE: `terminate` intentionally performs no status validation. Matching the
    # qwen3vl agent, any status other than "failure" (including an invalid or
    # missing value) is treated as a successful completion ("DONE") downstream.

    return None


def _decode_screenshot(b64_png: Optional[str]) -> Optional[Image.Image]:
    """Decode a base64-encoded PNG screenshot field if present."""
    if not b64_png:
        return None
    try:
        raw = base64.b64decode(b64_png)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        logger.warning("Failed to decode screenshot field from response", exc_info=True)
        return None


def _require_screenshot(observation: dict[str, Any] | None, *, context: str) -> Image.Image:
    """Return the decoded screenshot or fail the env step/create."""
    screenshot = _decode_screenshot((observation or {}).get("screenshot"))
    if screenshot is None:
        raise DesktopEnvStepError(f"{context} returned no valid screenshot")
    return screenshot


# ---------------------------------------------------------------------------
# DesktopEnvTool
# ---------------------------------------------------------------------------


class DesktopEnvTool(BaseTool):
    """Session-based HTTP desktop environment tool.

    Config keys:
        api_base_url (str): Base URL of the desktop env service (e.g.
            ``http://localhost:2354``).
        screen_width (int): Width shown to the model in the prompt (Qwen-VL
            normalized space, default 1000).
        screen_height (int): Height shown to the model in the prompt (Qwen-VL
            normalized space, default 1000).
        real_screen_width (int): Actual desktop resolution width used to
            denormalize model coordinates (default: same as screen_width).
        real_screen_height (int): Actual desktop resolution height used to
            denormalize model coordinates (default: same as screen_height).
        close_timeout (int): HTTP request timeout in seconds for ``/close``
            calls (default 60).
        connect_timeout (float): Connection timeout in seconds. This is
            intentionally short so connection failures retry quickly (default
            10.0).
        step_timeout (int): Client-side HTTP request timeout in seconds for
            ``/step`` calls (default 400).
        step_server_timeout (int): Timeout seconds forwarded to the desktop
            service for ``/step`` (default is slightly below step_timeout).
        evaluate_timeout (int): HTTP request timeout in seconds for
            ``/evaluate`` (default 605).
        evaluate_server_timeout (int): Timeout seconds forwarded to the
            desktop service for ``/evaluate`` (default 600).
        pause (float): ``pause`` value forwarded to ``/step`` after each
            action (default 2.0).
        evaluate_settle_seconds (int): local sleep before ``/evaluate`` when
            computing the terminal reward (default 3).
        step_reward (float): Per-step reward returned by ``execute()``
            (default 0.0).
        slow_step_log_threshold (float): Log successful ``/step`` calls whose
            wall time is at least this many seconds. ``0`` disables the log
            (default 30, or ``$DESKTOP_STEP_SLOW_LOG_THRESHOLD``).
        http_reuse_session (bool): If true, keep one aiohttp ClientSession
            alive for the tool lifetime. Defaults to false, so every request
            gets a fresh ClientSession/TCP connection.
        http_keepalive_timeout (float): Seconds to keep idle TCP connections
            in the aiohttp connector pool when http_reuse_session is true
            (default 30.0).
        http_session_max_age (float): Maximum age of the aiohttp ClientSession
            in seconds when http_reuse_session is true. ``0`` disables
            age-based reset (default 0.0).
        http_force_close (bool): If true, close TCP connections after every
            request (default True).
        connect_max_retries (int): Retry budget for connection-stage failures
            (default follows ``max_retries``).
        connect_retry_jitter_min (float): Minimum sleep before retrying
            connection-stage failures (default 3.0).
        connect_retry_jitter_max (float): Maximum sleep before retrying
            connection-stage failures (default 8.0). Other retryable failures
            use ``retry_interval``.
        create_jitter_seconds (float): Maximum random sleep before
            ``/session/create``. ``0`` disables create jitter (default 0.0).
        create_status_max_retries (int): Retry budget for retryable HTTP
            status errors from ``/session/create`` (default 0).
        create_retry_statuses (list[int]): HTTP statuses treated as retryable
            for ``/session/create`` (default [429, 500, 502, 503, 504]).
        auth_token (str): Optional Bearer token for the desktop service.
            Defaults to ``$DESKTOP_API_AUTH_TOKEN`` or ``$RL_PROXY_AUTH_TOKEN``.
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        screen_width = config.get("screen_width", 1000)
        screen_height = config.get("screen_height", 1000)

        if tool_schema is None:
            tool_schema = _build_tool_schema(screen_width, screen_height)

        super().__init__(config=config, tool_schema=tool_schema)

        self.api_base_url = config["api_base_url"].rstrip("/")
        self.auth_token = str(
            config.get("auth_token")
            or os.getenv("DESKTOP_API_AUTH_TOKEN")
            or os.getenv("RL_PROXY_AUTH_TOKEN")
            or ""
        ).strip() or None
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Real desktop resolution for coordinate denormalization.
        # Falls back to screen_width/screen_height if not configured.
        self.real_screen_width = int(config.get("real_screen_width", screen_width))
        self.real_screen_height = int(config.get("real_screen_height", screen_height))
        self.close_timeout_seconds = float(config.get("close_timeout", 60))
        self.pause = float(config.get("pause", 2.0))
        self.missing_screenshot_recovery_wait_seconds = float(config.get("missing_screenshot_recovery_wait", 2.0))
        if self.missing_screenshot_recovery_wait_seconds < 0:
            raise ValueError("missing_screenshot_recovery_wait must be non-negative")
        self.evaluate_settle_seconds = int(config.get("evaluate_settle_seconds", 3))
        self.step_reward = float(config.get("step_reward", 0.0))
        self.slow_step_log_threshold_seconds = float(
            config.get("slow_step_log_threshold", os.getenv("DESKTOP_STEP_SLOW_LOG_THRESHOLD", 30.0))
        )
        if self.slow_step_log_threshold_seconds < 0:
            raise ValueError("slow_step_log_threshold must be non-negative")

        # HTTP retry config. _post itself does not catch/retry; callers decide
        # whether an endpoint is safe to retry.
        self.max_retries = int(config.get("max_retries", 1))
        self.connect_max_retries = int(config.get("connect_max_retries", self.max_retries))
        self.connect_retry_jitter_min = float(config.get("connect_retry_jitter_min", 3.0))
        self.connect_retry_jitter_max = float(config.get("connect_retry_jitter_max", 8.0))
        if self.connect_retry_jitter_max < self.connect_retry_jitter_min:
            raise ValueError("connect_retry_jitter_max must be >= connect_retry_jitter_min")
        self.retry_interval = float(config.get("retry_interval", 30.0))
        self.connect_timeout_seconds = float(config.get("connect_timeout", 10.0))
        self.create_jitter_seconds = float(config.get("create_jitter_seconds", 0.0))
        if self.create_jitter_seconds < 0:
            raise ValueError("create_jitter_seconds must be non-negative")
        self.create_status_max_retries = int(config.get("create_status_max_retries", 0))
        if self.create_status_max_retries < 0:
            raise ValueError("create_status_max_retries must be non-negative")
        self.create_retry_statuses = tuple(int(status) for status in config.get("create_retry_statuses", [429, 500, 502, 503, 504]))
        self.create_status_retry_interval = float(config.get("create_status_retry_interval", 30.0))

        # HTTP connection config. By default, each request gets a fresh
        # ClientSession/TCP connection. Session reuse is opt-in.
        # B2 routes the hot-path step/evaluate directly to each session's own
        # worker process/port, so connections are spread across workers (each
        # serving ~1 at a time) instead of funneled through one central event
        # loop. Per-request short connections are therefore not a bottleneck and
        # connection reuse is unnecessary; keeping it off also avoids long-lived
        # keep-alive connections going half-open through NAT. SO_KEEPALIVE
        # (socket_factory) still guards the single long /evaluate or /step
        # request held open while the worker runs.
        self.http_reuse_session = bool(config.get("http_reuse_session", False))
        self.http_keepalive_timeout = float(config.get("http_keepalive_timeout", 30.0))
        self.http_session_max_age = float(config.get("http_session_max_age", 0.0))
        self.http_force_close = bool(config.get("http_force_close", True))

        # instance_id → {"session_id": str, "task_id": str}
        self._instances: dict[str, dict[str, Any]] = {}

        # Optional persistent aiohttp session per DesktopEnvTool/rollout event
        # loop. Only used when http_reuse_session=true.
        self._http_session: aiohttp.ClientSession | None = None
        self._http_session_loop: asyncio.AbstractEventLoop | None = None
        self._http_session_created_at = 0.0
        self._http_inflight = 0
        self._http_reset_pending = False

        self.close_timeout = self._make_client_timeout(self.close_timeout_seconds)
        self.create_timeout = self._make_client_timeout(float(config.get("create_timeout", 600)))
        step_timeout_seconds = float(config.get("step_timeout", 400))
        self.step_timeout = self._make_client_timeout(step_timeout_seconds)
        self.step_server_timeout_seconds = float(
            config.get("step_server_timeout", self._default_server_timeout(step_timeout_seconds))
        )
        if self.step_server_timeout_seconds <= 0:
            raise ValueError("step_server_timeout must be positive")
        if self.step_server_timeout_seconds >= step_timeout_seconds:
            raise ValueError("step_server_timeout must be smaller than step_timeout")
        self.evaluate_timeout = self._make_client_timeout(float(config.get("evaluate_timeout", 605)))
        self.evaluate_server_timeout_seconds = float(config.get("evaluate_server_timeout", 600))

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_server_timeout(client_timeout_seconds: float) -> float:
        """Keep server-side work timeout below the client total timeout."""
        client_timeout_seconds = float(client_timeout_seconds)
        if client_timeout_seconds > 60:
            return max(1.0, client_timeout_seconds - 40.0)
        return max(1.0, client_timeout_seconds * 0.9)

    def _make_client_timeout(self, read_timeout_seconds: float) -> aiohttp.ClientTimeout:
        """Build aiohttp timeout with short connect timeout and endpoint-specific total timeout."""
        read_timeout_seconds = float(read_timeout_seconds)
        return aiohttp.ClientTimeout(
            total=read_timeout_seconds,
            connect=self.connect_timeout_seconds,
        )

    async def _get_http_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        session_expired = (
            self._http_session is not None
            and self.http_session_max_age > 0
            and now - self._http_session_created_at > self.http_session_max_age
            and self._http_inflight == 0
        )
        if (self._http_reset_pending or session_expired) and self._http_inflight == 0:
            await self._close_http_session()
        if self._http_session is None or self._http_session.closed or self._http_session_loop is not loop:
            await self._close_http_session()
            self._http_session = aiohttp.ClientSession(
                connector=self._make_http_connector(),
            )
            self._http_session_loop = loop
            self._http_session_created_at = now
            _log(
                f"[DesktopEnvTool] HTTP session created keepalive_timeout={self.http_keepalive_timeout} "
                f"max_age={self.http_session_max_age} force_close={self.http_force_close} "
                f"reuse={self.http_reuse_session}",
                debug=True,
            )
        return self._http_session

    @staticmethod
    def _make_keepalive_socket(addr_info):
        """Socket factory enabling OS-level TCP keepalive.

        Keeps the NAT/firewall mapping alive during long quiet requests
        (e.g. /evaluate, slow /step) where aiohttp's pool keepalive_timeout
        does not apply because the connection is checked out, not idle.
        Must only create the socket and set options (aiohappyeyeballs handles
        setblocking/connect).
        """
        family, type_, proto = addr_info[0], addr_info[1], addr_info[2]
        sock = socket.socket(family=family, type=type_, proto=proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, "TCP_KEEPINTVL"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
            if hasattr(socket, "TCP_KEEPCNT"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4)
        except OSError:
            pass
        return sock

    def _make_http_connector(self) -> aiohttp.TCPConnector:
        connector_kwargs: dict[str, Any] = {
            "limit": 128,
            "limit_per_host": 128,
            "enable_cleanup_closed": True,
            "ttl_dns_cache": 300,
            "socket_factory": self._make_keepalive_socket,
        }
        if self.http_force_close:
            connector_kwargs["force_close"] = True
        else:
            connector_kwargs["keepalive_timeout"] = self.http_keepalive_timeout
        return aiohttp.TCPConnector(**connector_kwargs)

    def _log_slow_step(
        self,
        *,
        elapsed_s: float,
        request_id: str,
        session_id: str,
        action: str,
        actual_action: str | None,
        timeout_s: float,
        server_timeout_s: float,
        pause_s: float,
        meta: dict[str, Any],
        worker_base: str | None = None,
    ) -> None:
        threshold = self.slow_step_log_threshold_seconds
        if threshold <= 0 or elapsed_s < threshold:
            return
        _log(
            f"[DesktopEnvTool][SLOW_STEP] elapsed={elapsed_s:.3f}s "
            f"threshold={threshold:.3f}s request_id={request_id} session_id={session_id} "
            f"action={action!r} actual_action={actual_action!r} "
            f"timeout={timeout_s}s server_timeout={server_timeout_s}s pause={pause_s} "
            f"done={meta.get('done')} step_count={meta.get('step_count')} "
            f"worker_base={worker_base or '<central>'}",
            level="ERROR",
        )

    async def _close_http_session(self) -> None:
        self._http_reset_pending = False
        session = self._http_session
        self._http_session = None
        self._http_session_loop = None
        self._http_session_created_at = 0.0
        if session is not None and not session.closed:
            await session.close()

    async def _reset_http_session(self) -> None:
        if self._http_inflight == 0:
            await self._close_http_session()
        else:
            self._http_reset_pending = True

    @staticmethod
    def _is_retryable_transport_error(exc: BaseException) -> bool:
        return DesktopEnvTool._http_failure_kind(exc) in {"connect", "timeout", "transport"}

    @staticmethod
    def _is_connect_error(exc: BaseException) -> bool:
        return isinstance(exc, (aiohttp.ConnectionTimeoutError, aiohttp.ClientConnectorError))

    @staticmethod
    def _http_failure_kind(exc: BaseException) -> str:
        if isinstance(exc, aiohttp.ClientResponseError):
            return "http_status"
        if DesktopEnvTool._is_connect_error(exc):
            return "connect"
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(
            exc,
            (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                OSError,
            ),
        ):
            return "transport"
        if isinstance(exc, RuntimeError) and "is used by transport" in str(exc):
            return "transport"
        return "protocol"

    def _retry_plan_for_http_failure(
        self,
        exc: BaseException,
        *,
        retry_transport: bool,
        retry_statuses: set[int],
        transport_max_retries: int,
        connect_max_retries: int,
        status_max_retries: int,
        status_retry_interval: float,
    ) -> tuple[str, bool, int, float]:
        kind = self._http_failure_kind(exc)
        if kind == "http_status":
            retryable = (
                isinstance(exc, aiohttp.ClientResponseError)
                and exc.status in retry_statuses
                and status_max_retries > 0
            )
            retry_interval = status_retry_interval * random.uniform(0.75, 1.25)
            error_kind = f"http {exc.status}" if isinstance(exc, aiohttp.ClientResponseError) else kind
            return error_kind, retryable, status_max_retries + 1, retry_interval
        if kind == "connect":
            retry_interval = random.uniform(self.connect_retry_jitter_min, self.connect_retry_jitter_max)
            return self._transport_error_kind(exc), retry_transport, connect_max_retries + 1, retry_interval
        if kind in {"timeout", "transport"}:
            return self._transport_error_kind(exc), retry_transport, transport_max_retries + 1, self.retry_interval
        return kind, False, 1, 0.0

    @staticmethod
    def _transport_error_kind(exc: BaseException) -> str:
        if isinstance(exc, aiohttp.ClientResponseError):
            return f"http {exc.status}"
        if isinstance(exc, aiohttp.ConnectionTimeoutError):
            return "connect timeout"
        if isinstance(exc, aiohttp.ClientConnectorError):
            return "connect error"
        if isinstance(exc, TimeoutError):
            return "timeout"
        return "transport error"

    @staticmethod
    def _error_detail(exc: BaseException) -> str:
        if isinstance(exc, aiohttp.ClientResponseError):
            url = exc.request_info.url if exc.request_info else "<unknown>"
            return (
                f"status={exc.status} url={url} "
                f"body={_format_http_error_body(exc.message or '')}"
            )
        return str(exc) or repr(exc)

    async def _post(
        self,
        path: str,
        payload: dict | None = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        base_url: str | None = None,
    ) -> dict:
        """POST JSON once and return the JSON response.

        This method does not catch/retry errors. Callers choose endpoint-specific
        retry policy based on whether the operation is idempotent. ``base_url``
        overrides the central service URL — used to route hot-path step/evaluate
        directly to the per-session worker (B2).
        """
        url = f"{base_url or self.api_base_url}{path}"
        request_body = payload or {}
        request_context = _request_log_context(request_body)
        if timeout is None:
            raise ValueError(f"POST {path} requires an explicit timeout")
        effective_timeout = timeout

        _log(f"[DesktopEnvTool] -> POST path={path} {request_context} payload={_short_repr(request_body)}")

        if self.http_reuse_session:
            session = await self._get_http_session()
            self._http_inflight += 1
            close_session = False
        else:
            session = aiohttp.ClientSession(
                connector=self._make_http_connector(),
            )
            close_session = True

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else None
            async with session.post(url, json=request_body, timeout=effective_timeout, headers=headers) as resp:
                status = resp.status
                content_type = resp.content_type
                text_body = await resp.text()
                if status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=status,
                        message=text_body or resp.reason or "",
                        headers=resp.headers,
                    )
                if content_type == "application/json" and text_body:
                    import json as _json

                    try:
                        data = _json.loads(text_body)
                    except _json.JSONDecodeError as je:
                        _log(
                            f"[DesktopEnvTool] <- POST {url} status={status} "
                            f"non-JSON body (content_type={content_type}): {_short_repr(text_body)}",
                            level="ERROR",
                        )
                        raise RuntimeError(f"POST {path} returned non-JSON body: {text_body!r}") from je
                    _log(
                        f"[DesktopEnvTool] <- POST path={path} {request_context} "
                        f"status={status} response={_short_repr(data)}"
                    )
                    return data
                _log(
                    f"[DesktopEnvTool] <- POST path={path} {request_context} "
                    f"status={status} empty body (content_type={content_type})"
                )
                return {}
        finally:
            if close_session:
                await session.close()
            else:
                self._http_inflight = max(0, self._http_inflight - 1)

    async def _post_with_retries(
        self,
        path: str,
        payload: dict | None = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        *,
        base_url: str | None = None,
        max_retries: int | None = None,
        retry_transport: bool = True,
        retry_statuses: tuple[int, ...] | None = None,
        status_max_retries: int = 0,
        status_retry_interval: float | None = None,
        error_context: str | None = None,
    ) -> dict:
        transport_max_retries = self.max_retries if max_retries is None else max_retries
        connect_max_retries = self.connect_max_retries if max_retries is None else max_retries
        request_body = payload or {}
        request_context = _request_log_context(request_body)
        attempt = 0
        failures_by_kind: dict[str, int] = {}
        retry_status_set = set(retry_statuses or ())
        status_retry_interval = self.retry_interval if status_retry_interval is None else status_retry_interval
        while True:
            attempt += 1
            try:
                return await self._post(path, request_body, timeout=timeout, base_url=base_url)
            except Exception as exc:
                error_kind, retryable, max_failures, retry_interval = self._retry_plan_for_http_failure(
                    exc,
                    retry_transport=retry_transport,
                    retry_statuses=retry_status_set,
                    transport_max_retries=transport_max_retries,
                    connect_max_retries=connect_max_retries,
                    status_max_retries=status_max_retries,
                    status_retry_interval=status_retry_interval,
                )
                failures_by_kind[error_kind] = failures_by_kind.get(error_kind, 0) + 1
                current_failures = failures_by_kind[error_kind]
                # Do NOT tear down the whole keep-alive session on a transient
                # failure: aiohttp already evicts the single failed connection
                # and the retry transparently acquires a healthy one. Resetting
                # the session would drop every warm connection in the pool.
                if (not retryable) or current_failures >= max_failures:
                    context = f" | {error_context}" if error_context else ""
                    _log(
                        f"[DesktopEnvTool] POST {path} {request_context} "
                        f"failed after {attempt} attempt(s) ({error_kind}, "
                        f"{error_kind} failures={current_failures}/{max_failures}): {self._error_detail(exc)} | "
                        f"payload={_short_repr(request_body)}{context}",
                        level="ERROR",
                    )
                    logger.exception(
                        "[DesktopEnvTool] POST %s failed after %d attempt(s)%s",
                        path,
                        attempt,
                        context,
                    )
                    raise
                _log(
                    f"[DesktopEnvTool] POST {path} {request_context} "
                    f"{error_kind} (attempt {attempt}, {error_kind} failures "
                    f"{current_failures}/{max_failures}): {self._error_detail(exc)}. "
                    f"Retrying in {retry_interval:.1f}s"
                    f"{' | ' + error_context if error_context else ''}",
                    level="ERROR",
                )
                await asyncio.sleep(retry_interval)

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _build_tool_schema(self.screen_width, self.screen_height)

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a new desktop session and return the initial screenshot.

        Args:
            instance_id: Optional pre-assigned ID. A UUID is generated if None.
            create_kwargs: Must contain ``task_id``.

        Returns:
            ``(instance_id, ToolResponse with initial screenshot)``.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = create_kwargs or {}
        task_id = create_kwargs.get("task_id")
        if not task_id:
            raise ValueError("create_kwargs must contain 'task_id'")

        _log(f"[DesktopEnvTool] create session task_id={task_id} instance_id={instance_id}")
        if self.create_jitter_seconds > 0:
            jitter_seconds = random.uniform(0.0, self.create_jitter_seconds)
            _log(
                f"[DesktopEnvTool] create jitter sleep={jitter_seconds:.3f}s "
                f"task_id={task_id} instance_id={instance_id}",
                debug=True,
            )
            await asyncio.sleep(jitter_seconds)

        try:
            payload = {
                "session_id": instance_id,
                "task_id": task_id,
                "require_a11y_tree": False,
            }
            resp = await self._post_with_retries(
                "/session/create",
                payload,
                timeout=self.create_timeout,
                retry_statuses=self.create_retry_statuses,
                status_max_retries=self.create_status_max_retries,
                status_retry_interval=self.create_status_retry_interval,
            )
            session_id = resp.get("session_id")
            if not session_id:
                raise RuntimeError(f"/session/create did not return session_id; response={resp!r}")
            if session_id != instance_id:
                raise RuntimeError(
                    f"/session/create returned unexpected session_id={session_id!r}; "
                    f"expected instance_id={instance_id!r}"
                )

            # Once server returned a session_id, the slot is IN_USE on the
            # server side. Anything failing between here and the dict
            # assignment (incl. CancelledError) must close the server-side
            # session, else we leak. Catch BaseException to cover
            # CancelledError too.
            instance_info = {
                "session_id": session_id,
                "task_id": task_id,
            }
            # B2: if the server returned a per-session worker HTTP port, route the
            # hot-path step/evaluate calls directly to that worker (same host as
            # the central service, different port). Falls back to the central
            # service when absent.
            worker_port = resp.get("worker_port")
            if worker_port:
                split = urlsplit(self.api_base_url)
                instance_info["worker_base"] = f"{split.scheme}://{split.hostname}:{int(worker_port)}"
                _log(
                    f"[DesktopEnvTool] session_id={session_id} routing step/evaluate "
                    f"to worker {instance_info['worker_base']}"
                )
                self._instances[instance_id] = instance_info
            elif resp.get("worker_http_enabled", True):
                # Server runs per-session worker HTTP (the expected, guaranteed-on
                # case) but this session got no port — worker bind failure or port
                # exhaustion. Abort the create so the outer rerun re-creates the
                # session; the ``except BaseException`` below closes this orphan
                # server session. Default True so a missing flag also aborts rather
                # than silently degrading to the slow central path.
                raise RuntimeError(
                    f"create returned no worker_port for session_id={session_id} "
                    f"(worker_http_error={resp.get('worker_http_error')!r}); aborting for env rerun"
                )
            else:
                # Worker HTTP explicitly disabled server-side (rollback switch):
                # the central /step + /evaluate path is intended here.
                _log(
                    f"[DesktopEnvTool] session_id={session_id} worker HTTP disabled "
                    f"server-side; using central step/evaluate at {self.api_base_url}",
                    debug=True,
                )
                self._instances[instance_id] = instance_info
        except BaseException:
            # Best-effort cleanup. If we already have a server session_id,
            # try to close it on the server.
            self._instances.pop(instance_id, None)
            server_session_id = locals().get("session_id", instance_id)
            if server_session_id:
                try:
                    await self._post_with_retries(
                        f"/session/{server_session_id}/close",
                        timeout=self.close_timeout,
                    )
                except Exception:
                    _log(
                        f"[DesktopEnvTool] create cleanup: failed to close orphan "
                        f"server session_id={server_session_id}",
                        level="ERROR",
                    )
            raise

        _log(f"[DesktopEnvTool] create session OK task_id={task_id} instance_id={instance_id} session_id={session_id}")

        observation = resp.get("observation") or {}
        try:
            screenshot = await self._decode_or_recover_screenshot(
                observation,
                session_id=session_id,
                worker_base=instance_info.get("worker_base"),
                context=f"/session/create session_id={session_id}",
            )
        except BaseException:
            self._instances.pop(instance_id, None)
            try:
                await self._post_with_retries(
                    f"/session/{session_id}/close",
                    timeout=self.close_timeout,
                )
            except Exception:
                _log(
                    f"[DesktopEnvTool] create cleanup: failed to close no-screenshot "
                    f"server session_id={session_id}",
                    level="ERROR",
                )
            raise
        images = [screenshot]
        _log(f"[DesktopEnvTool] created session_id={session_id} has_screenshot=True")
        return instance_id, ToolResponse(image=images)

    async def _recover_missing_screenshot_with_wait(
        self,
        *,
        session_id: str,
        worker_base: str | None,
        context: str,
    ) -> Image.Image:
        wait_seconds = self.missing_screenshot_recovery_wait_seconds
        request_id = str(uuid4())
        _log(
            f"[DesktopEnvTool] missing screenshot recovery request_id={request_id} "
            f"session_id={session_id} context={context} action=WAIT pause={wait_seconds} "
            f"timeout={self.step_timeout.total} server_timeout={self.step_server_timeout_seconds}",
            level="ERROR",
        )
        try:
            resp = await self._post_with_retries(
                f"/session/{session_id}/step",
                {
                    "request_id": request_id,
                    "action": "WAIT",
                    "pause": wait_seconds,
                    "timeout_seconds": self.step_server_timeout_seconds,
                },
                timeout=self.step_timeout,
                error_context=f"missing_screenshot_recovery context={context}",
                base_url=worker_base,
            )
        except Exception as exc:
            raise DesktopEnvStepError(
                f"{context} returned no valid screenshot and recovery WAIT failed; "
                f"cause={self._error_detail(exc)}"
            ) from exc
        observation = resp.get("observation") or {}
        screenshot = _require_screenshot(observation, context=f"{context} recovery WAIT")
        _log(
            f"[DesktopEnvTool] missing screenshot recovery OK request_id={request_id} "
            f"session_id={session_id} context={context}"
        )
        return screenshot

    async def _decode_or_recover_screenshot(
        self,
        observation: dict[str, Any] | None,
        *,
        session_id: str,
        worker_base: str | None,
        context: str,
    ) -> Image.Image:
        screenshot = _decode_screenshot((observation or {}).get("screenshot"))
        if screenshot is not None:
            return screenshot
        return await self._recover_missing_screenshot_with_wait(
            session_id=session_id,
            worker_base=worker_base,
            context=context,
        )

    async def screenshot(self, instance_id: str) -> list:
        """Take a screenshot of the current desktop without executing any action.

        Sends a no-op ``time.sleep(0)`` step to the backend and returns the
        observation screenshot. This is used to attach a fresh screenshot to
        error responses so the agent always receives visual context.

        Returns:
            list of PIL.Image (length 0 or 1).
        """
        info = self._instances.get(instance_id)
        if info is None:
            return []
        session_id = info["session_id"]
        try:
            request_id = str(uuid4())
            _log(
                f"[DesktopEnvTool] screenshot step request_id={request_id} "
                f"session_id={session_id} timeout={self.step_timeout.total} "
                f"server_timeout={self.step_server_timeout_seconds}"
            )
            resp = await self._post_with_retries(
                f"/session/{session_id}/step",
                {
                    "request_id": request_id,
                    "action": "import time; time.sleep(0)",
                    "pause": 0,
                    "timeout_seconds": self.step_server_timeout_seconds,
                },
                timeout=self.step_timeout,
                base_url=info.get("worker_base"),
            )
            observation = resp.get("observation") or {}
            screenshot = _decode_screenshot(observation.get("screenshot"))
            return [screenshot] if screenshot is not None else []
        except Exception:
            _log(f"[DesktopEnvTool] screenshot failed for session_id={session_id}", level="ERROR")
            return []

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute an action on the desktop environment.

        Structured ``computer_use`` parameters are translated into
        pyautogui code and sent to ``/step``. For virtual actions
        (``terminate``, ``answer``) no backend call is made.
        """
        info = self._instances.get(instance_id)
        if info is None:
            raise ValueError(f"Unknown instance_id: {instance_id}")
        session_id = info["session_id"]

        raw_action = parameters.get("action", "")
        action = _normalize_action_alias(raw_action)
        if action != raw_action:
            parameters = {**parameters, "action": action}
            _log(
                f"[DesktopEnvTool] normalized action alias raw_action={raw_action!r} "
                f"action={action!r} session_id={session_id}"
            )

        # Reject schema/format-invalid actions with a structured tool response
        # instead of raising. No backend call is needed because the screen state
        # has not changed; the next turn will reuse the previous screenshot.
        if validation_error := _validate_action_parameters(parameters):
            _log(
                f"[DesktopEnvTool] invalid action parameters action={action!r} "
                f"session_id={session_id}: {validation_error}",
                level="ERROR",
            )
            return (
                ToolResponse(
                    text=(
                        f"Error: invalid computer_use action arguments: {validation_error}. "
                        f"No screen state change. Please output a valid computer_use action."
                    ),
                ),
                0.0,
                {"action": action, "invalid_action": True, "invalid_action_error": validation_error},
            )

        # Virtual actions that the agent loop handles directly.
        if action == "terminate":
            status = parameters.get("status")
            code = "FAIL" if status == "failure" else "DONE"
            request_id = str(uuid4())
            terminate_timeout = self.step_timeout
            error_context = (
                f"step_action={action} actual_action={code!r} "
                f"timeout={terminate_timeout.total}s "
                f"server_timeout={self.step_server_timeout_seconds}s pause={self.pause}"
            )
            _log(
                f"[DesktopEnvTool] terminate step request_id={request_id} "
                f"session_id={session_id} status={status} code={code} "
                f"timeout={terminate_timeout.total} server_timeout={self.step_server_timeout_seconds}"
            )
            try:
                step_started_at = time.monotonic()
                resp = await self._post_with_retries(
                    f"/session/{session_id}/step",
                    {
                        "request_id": request_id,
                        "action": code,
                        "pause": self.pause,
                        "timeout_seconds": self.step_server_timeout_seconds,
                    },
                    timeout=terminate_timeout,
                    error_context=error_context,
                    base_url=info.get("worker_base"),
                )
                step_elapsed_s = time.monotonic() - step_started_at
            except Exception as exc:
                raise DesktopEnvStepError(
                    f"/step failed for action={action!r}, actual_action={code!r}, "
                    f"timeout={terminate_timeout.total}s; cause={self._error_detail(exc)}"
                ) from exc
            observation = resp.get("observation") or {}
            screenshot = _decode_screenshot(observation.get("screenshot"))
            images = [screenshot] if screenshot is not None else []
            meta = {k: v for k, v in resp.items() if k != "observation"}
            meta["code"] = code
            self._log_slow_step(
                elapsed_s=step_elapsed_s,
                request_id=request_id,
                session_id=session_id,
                action=action,
                actual_action=code,
                timeout_s=terminate_timeout.total,
                server_timeout_s=self.step_server_timeout_seconds,
                pause_s=self.pause,
                meta=meta,
                worker_base=info.get("worker_base"),
            )
            _log(
                f"[DesktopEnvTool] terminate step done request_id={request_id} "
                f"session_id={session_id} status={status} done={meta.get('done')} "
                f"step_count={meta.get('step_count')} has_screenshot={bool(images)}"
            )
            return (
                ToolResponse(image=images, text=f"Task terminated with status: {status}; code: {code}"),
                self.step_reward,
                {"action": action, **meta},
            )
        if action == "answer":
            answer_text = parameters.get("text", "")
            status = parameters.get("status", "success")
            code = "FAIL" if status == "failure" else "DONE"
            _log(f"[DesktopEnvTool] virtual action=answer session_id={session_id}")
            response_text = f"Answer: {answer_text}" if answer_text else f"Answer status: {status}; code: {code}"
            return (
                ToolResponse(text=response_text),
                0.0,
                {
                    "action": action,
                    "answer": answer_text,
                    "code": code,
                    "done": True,
                    "status": status,
                },
            )

        code = _translate_action_to_pyautogui(
            parameters,
            self.screen_width,
            self.screen_height,
            self.real_screen_width,
            self.real_screen_height,
        )
        actual_coordinate: tuple[int, int] | None = None
        raw_coordinate = parameters.get("coordinate")
        if raw_coordinate is not None:
            actual_coordinate = _denorm_coord(
                raw_coordinate,
                self.screen_width,
                self.screen_height,
                self.real_screen_width,
                self.real_screen_height,
            )
        pause = self.pause
        step_timeout = self.step_timeout
        request_id = str(uuid4())
        _log(
            f"[DesktopEnvTool] step request_id={request_id} "
            f"session_id={session_id} action={action} raw_coordinate={raw_coordinate} "
            f"actual_coordinate={actual_coordinate} code={code} pause={pause} "
            f"timeout={step_timeout.total} server_timeout={self.step_server_timeout_seconds}"
        )
        error_context = (
            f"step_action={action} actual_action={code!r} "
            f"timeout={step_timeout.total}s server_timeout={self.step_server_timeout_seconds}s pause={pause}"
        )
        try:
            step_started_at = time.monotonic()
            resp = await self._post_with_retries(
                f"/session/{session_id}/step",
                {
                    "request_id": request_id,
                    "action": code,
                    "pause": pause,
                    "timeout_seconds": self.step_server_timeout_seconds,
                },
                timeout=step_timeout,
                error_context=error_context,
                base_url=info.get("worker_base"),
            )
            step_elapsed_s = time.monotonic() - step_started_at
        except Exception as exc:
            raise DesktopEnvStepError(
                f"/step failed for action={action!r}, actual_action={code!r}, "
                f"timeout={step_timeout.total}s; cause={self._error_detail(exc)}"
            ) from exc

        observation = resp.get("observation") or {}
        screenshot = await self._decode_or_recover_screenshot(
            observation,
            session_id=session_id,
            worker_base=info.get("worker_base"),
            context=f"/step action={action!r} actual_action={code!r} session_id={session_id}",
        )
        images = [screenshot]

        action_summary = f"Executed action: {action}"
        if "coordinate" in parameters:
            action_summary += f" at {parameters['coordinate']}"
            if actual_coordinate is not None:
                action_summary += f" -> actual {list(actual_coordinate)}"
        action_summary += f"; code: {code}"

        # Forward non-observation metadata (reward / done / info / step_count).
        meta = {k: v for k, v in resp.items() if k != "observation"}
        meta["code"] = code
        if raw_coordinate is not None:
            meta["raw_coordinate"] = raw_coordinate
        if actual_coordinate is not None:
            meta["actual_coordinate"] = list(actual_coordinate)
            meta["source_screen_size"] = [self.screen_width, self.screen_height]
            meta["real_screen_size"] = [self.real_screen_width, self.real_screen_height]
        self._log_slow_step(
            elapsed_s=step_elapsed_s,
            request_id=request_id,
            session_id=session_id,
            action=action,
            actual_action=code,
            timeout_s=step_timeout.total,
            server_timeout_s=self.step_server_timeout_seconds,
            pause_s=pause,
            meta=meta,
            worker_base=info.get("worker_base"),
        )
        _log(
            f"[DesktopEnvTool] step done request_id={request_id} "
            f"session_id={session_id} action={action} "
            f"has_screenshot=True done={meta.get('done')} "
            f"step_count={meta.get('step_count')}"
        )
        return (
            ToolResponse(image=images, text=action_summary),
            self.step_reward,
            {"action": action, **meta},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Compute the terminal reward via ``/evaluate``."""
        info = self._instances.get(instance_id)
        if info is None:
            _log(f"[DesktopEnvTool] calc_reward: unknown instance_id={instance_id} (already released?)", level="ERROR")
            return 0.0
        session_id = info["session_id"]

        evaluate_server_timeout = self.evaluate_server_timeout_seconds
        _log(
            f"[DesktopEnvTool] evaluate session_id={session_id} "
            f"local_settle={self.evaluate_settle_seconds}s "
            f"timeout={self.evaluate_timeout.total} server_timeout={evaluate_server_timeout}"
        )
        if self.evaluate_settle_seconds > 0:
            await asyncio.sleep(self.evaluate_settle_seconds)
        try:
            resp = await self._post_with_retries(
                f"/session/{session_id}/evaluate",
                {"timeout_seconds": evaluate_server_timeout},
                timeout=self.evaluate_timeout,
                error_context=f"timeout={self.evaluate_timeout.total}s server_timeout={evaluate_server_timeout}s",
                base_url=info.get("worker_base"),
            )
        except Exception:
            _log(
                f"[DesktopEnvTool] Evaluation failed for session {session_id} after retries; treating reward as 0",
                level="ERROR",
            )
            logger.warning("Evaluation failed for session %s after retries; reward=0", session_id, exc_info=True)
            return 0.0

        try:
            reward = float(resp.get("reward", 0.0))
        except (TypeError, ValueError):
            _log(
                f"[DesktopEnvTool] evaluate session_id={session_id} "
                f"returned non-numeric reward: {resp.get('reward')!r}",
                level="ERROR",
            )
            return 0.0
        _log(f"[DesktopEnvTool] evaluate session_id={session_id} reward={reward:.4f}")
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """Close the session. MUST be called in a ``finally`` block."""
        info = self._instances.get(instance_id)
        if info is None:
            _log(
                f"[DesktopEnvTool] release: no-op for unknown "
                f"instance_id={instance_id} (already released or never registered)",
                level="ERROR",
            )
            return
        session_id = info["session_id"]
        _log(
            f"[DesktopEnvTool] release session_id={session_id} task_id={info.get('task_id')} instance_id={instance_id}"
        )
        try:
            await self._post_with_retries(
                f"/session/{session_id}/close",
                timeout=self.close_timeout,
            )
            _log(f"[DesktopEnvTool] release OK session_id={session_id} instance_id={instance_id}")
        except Exception:
            _log(f"[DesktopEnvTool] Failed to close session {session_id}", level="ERROR")
            logger.warning("Failed to close session %s", session_id, exc_info=True)
        finally:
            self._instances.pop(instance_id, None)
            if not self._instances:
                await self._close_http_session()
