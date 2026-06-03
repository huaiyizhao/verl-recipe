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

The service is expected to return an optional ``screenshot`` field (base64
encoded PNG) from ``/session/create`` and ``/session/{id}/step`` so that the
agent can feed the observation back to the LLM. The tool tolerates missing
screenshots gracefully.
"""

import asyncio
import base64
import copy
import io
import logging
import math
import os
import re
import sys
import time
from typing import Any, Optional
from uuid import uuid4

import aiohttp
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

# We bypass the ``logging`` framework for INFO/DEBUG lines because verl's
# global ``basicConfig(WARNING)`` plus Ray's early-attached handlers silently
# drop them regardless of per-logger levels. ``print`` goes straight to stdout
# (picked up by Ray's log forwarder) and is unaffected by any of that.
# ``logger`` is still kept for ERROR / exc_info warnings where the stack trace
# is valuable.
logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# computer_use tool schema (Qwen-VL compatible)
# Ref: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/utils/agent_function_call.py
# ---------------------------------------------------------------------------
_COMPUTER_USE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or "
            "applications menu. You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait "
            "and take successive screenshots to see the results of your actions.\n"
            "* The screen's resolution is {screen_width}x{screen_height}.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should "
            "consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of "
            "the element. Don't click boxes on their edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. Available actions:\n"
                        "* `key`: Key down presses on the arguments passed in order, then key releases in reverse.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate.\n"
                        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate.\n"
                        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `scroll`: Scroll the mouse wheel.\n"
                        "* `hscroll`: Horizontal scroll.\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status.\n"
                        "* `answer`: Answer a question."
                    ),
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "triple_click",
                        "scroll",
                        "hscroll",
                        "wait",
                        "terminate",
                        "answer",
                    ],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"},
                "coordinate": {
                    "description": "(x, y): pixel coordinates.",
                    "type": "array",
                },
                "pixels": {
                    "description": "Scrolling amount; positive scrolls up, negative scrolls down.",
                    "type": "number",
                },
                "time": {"description": "Seconds to wait. Required only by `action=wait`.", "type": "number"},
                "status": {
                    "description": "Status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
        },
    },
}


def _build_tool_schema(screen_width: int, screen_height: int) -> OpenAIFunctionToolSchema:
    tool = copy.deepcopy(_COMPUTER_USE_TOOL)
    tool["function"]["description"] = tool["function"]["description"].format(
        screen_width=screen_width,
        screen_height=screen_height,
    )
    return OpenAIFunctionToolSchema.model_validate(tool)


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
    action = parameters.get("action", "")
    coord = parameters.get("coordinate")

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
            return "pyautogui.click()"
        x, y = adjusted
        return f"pyautogui.click({x}, {y})"

    if action == "right_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return "pyautogui.rightClick()"
        x, y = adjusted
        return f"pyautogui.rightClick({x}, {y})"

    if action == "middle_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return "pyautogui.middleClick()"
        x, y = adjusted
        return f"pyautogui.middleClick({x}, {y})"

    if action == "double_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return "pyautogui.doubleClick()"
        x, y = adjusted
        return f"pyautogui.doubleClick({x}, {y})"

    if action == "triple_click":
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return "pyautogui.tripleClick()"
        x, y = adjusted
        return f"pyautogui.tripleClick({x}, {y})"

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
        keys = parameters.get("keys", []) or []
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
            cleaned_keys.append(key)
        keys_str = ", ".join(repr(str(key)) for key in cleaned_keys)
        if len(cleaned_keys) > 1:
            return f"pyautogui.hotkey({keys_str})"
        return f"pyautogui.press({keys_str})"

    if action == "scroll":
        pixels = int(parameters.get("pixels", 0) or 0)
        adjusted = adjusted_coordinate()
        if adjusted is None:
            return f"pyautogui.scroll({pixels})"
        x, y = adjusted
        return f"pyautogui.moveTo({x}, {y})\npyautogui.scroll({pixels})"

    if action == "hscroll":
        pixels = int(parameters.get("pixels", 0) or 0)
        return f"pyautogui.scroll({pixels})"

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


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_coordinate(coord: Any) -> str | None:
    if not isinstance(coord, list | tuple) or len(coord) != 2:
        return "coordinate must be an array [x, y] with two numbers"
    if not all(_is_finite_number(value) for value in coord):
        return "coordinate must be an array [x, y] with two finite numbers"
    return None


def _validate_action_parameters(parameters: dict[str, Any]) -> str | None:
    action = parameters.get("action")
    if not isinstance(action, str) or not action:
        return "action must be a non-empty string"

    if action not in _VALID_COMPUTER_USE_ACTIONS:
        valid_list = ", ".join(_VALID_COMPUTER_USE_ACTIONS)
        return f"unknown action {action!r}. Valid computer_use actions are: {valid_list}"

    if action in _COORD_ACTIONS:
        coord = parameters.get("coordinate")
        if coord is not None:
            if error := _validate_coordinate(coord):
                return f"action {action!r} has invalid {error}"
    if action == "type":
        if "text" in parameters and not isinstance(parameters.get("text"), str):
            return "action 'type' requires text as a string"
    elif action == "key":
        keys = parameters.get("keys")
        if keys is not None and not isinstance(keys, list):
            return "action 'key' requires keys as an array when provided"
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
        if "text" not in parameters or not isinstance(parameters.get("text"), str):
            return "action 'answer' requires text as a string"
    elif action == "terminate":
        if parameters.get("status") is not None and parameters.get("status") not in {"success", "failure"}:
            return "action 'terminate' requires status to be either 'success' or 'failure'"

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
        timeout (int): HTTP request timeout in seconds (default 30).
        pause (float): ``pause`` value forwarded to ``/step`` after each
            action (default 2.0).
        evaluate_settle_seconds (int): local sleep before ``/evaluate`` when
            computing the terminal reward (default 3).
        step_reward (float): Per-step reward returned by ``execute()``
            (default 0.0).
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
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        screen_width = config.get("screen_width", 1000)
        screen_height = config.get("screen_height", 1000)

        if tool_schema is None:
            tool_schema = _build_tool_schema(screen_width, screen_height)

        super().__init__(config=config, tool_schema=tool_schema)

        self.api_base_url = config["api_base_url"].rstrip("/")
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Real desktop resolution for coordinate denormalization.
        # Falls back to screen_width/screen_height if not configured.
        self.real_screen_width = int(config.get("real_screen_width", screen_width))
        self.real_screen_height = int(config.get("real_screen_height", screen_height))
        self.timeout = aiohttp.ClientTimeout(total=config.get("timeout", 30))
        self.create_timeout = aiohttp.ClientTimeout(total=config.get("create_timeout", 30))
        self.pause = float(config.get("pause", 2.0))
        self.evaluate_settle_seconds = int(config.get("evaluate_settle_seconds", 3))
        self.step_reward = float(config.get("step_reward", 0.0))

        # HTTP retry config. _post itself does not catch/retry; callers decide
        # whether an endpoint is safe to retry.
        self.max_retries = int(config.get("max_retries", 2))
        self.retry_interval = float(config.get("retry_interval", 30.0))

        # HTTP connection config. By default, each request gets a fresh
        # ClientSession/TCP connection. Session reuse is opt-in.
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

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

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
            connector_kwargs: dict[str, Any] = {
                "limit": 128,
                "limit_per_host": 128,
                "enable_cleanup_closed": True,
                "ttl_dns_cache": 300,
            }
            if self.http_force_close:
                connector_kwargs["force_close"] = True
            else:
                connector_kwargs["keepalive_timeout"] = self.http_keepalive_timeout
            connector = aiohttp.TCPConnector(**connector_kwargs)
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
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

    def _make_http_connector(self) -> aiohttp.TCPConnector:
        connector_kwargs: dict[str, Any] = {
            "limit": 128,
            "limit_per_host": 128,
            "enable_cleanup_closed": True,
            "ttl_dns_cache": 300,
        }
        if self.http_force_close:
            connector_kwargs["force_close"] = True
        else:
            connector_kwargs["keepalive_timeout"] = self.http_keepalive_timeout
        return aiohttp.TCPConnector(**connector_kwargs)

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
        if isinstance(exc, TimeoutError):
            return True
        if isinstance(
            exc,
            (
                aiohttp.ClientConnectorError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                OSError,
            ),
        ):
            return True
        return isinstance(exc, RuntimeError) and "is used by transport" in str(exc)

    @staticmethod
    def _error_detail(exc: BaseException) -> str:
        if isinstance(exc, aiohttp.ClientResponseError):
            return (
                f"status={exc.status} message={exc.message!r} "
                f"url={exc.request_info.url if exc.request_info else '<unknown>'}"
            )
        return str(exc) or repr(exc)

    async def _post(
        self,
        path: str,
        payload: dict | None = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> dict:
        """POST JSON once and return the JSON response.

        This method does not catch/retry errors. Callers choose endpoint-specific
        retry policy based on whether the operation is idempotent.
        """
        url = f"{self.api_base_url}{path}"
        request_body = payload or {}
        request_id = request_body.get("request_id", "<none>")
        effective_timeout = timeout if timeout is not None else self.timeout

        _log(f"[DesktopEnvTool] -> POST path={path} request_id={request_id} payload={_short_repr(request_body)}")

        if self.http_reuse_session:
            session = await self._get_http_session()
            self._http_inflight += 1
            close_session = False
        else:
            session = aiohttp.ClientSession(
                connector=self._make_http_connector(),
                timeout=self.timeout,
            )
            close_session = True

        try:
            async with session.post(url, json=request_body, timeout=effective_timeout) as resp:
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
                        f"[DesktopEnvTool] <- POST path={path} request_id={request_id} "
                        f"status={status} response={_short_repr(data)}"
                    )
                    return data
                _log(
                    f"[DesktopEnvTool] <- POST path={path} request_id={request_id} "
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
        max_retries: int | None = None,
        retry_transport: bool = True,
    ) -> dict:
        attempts = (self.max_retries if max_retries is None else max_retries) + 1
        request_body = payload or {}
        request_id = request_body.get("request_id", "<none>")
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._post(path, request_body, timeout=timeout)
            except Exception as exc:
                last_exc = exc
                retryable = retry_transport and self._is_retryable_transport_error(exc)
                if retryable:
                    await self._reset_http_session()
                if (not retryable) or attempt >= attempts:
                    _log(
                        f"[DesktopEnvTool] POST {path} request_id={request_id} "
                        f"failed after {attempt} attempt(s): {self._error_detail(exc)} | "
                        f"payload={_short_repr(request_body)}",
                        level="ERROR",
                    )
                    logger.exception(
                        "[DesktopEnvTool] POST %s failed after %d attempt(s)",
                        path,
                        attempt,
                    )
                    raise
                _log(
                    f"[DesktopEnvTool] POST {path} request_id={request_id} "
                    f"transport error (attempt {attempt}/{attempts}): {self._error_detail(exc)}. "
                    f"Retrying in {self.retry_interval:.1f}s",
                    level="ERROR",
                )
                await asyncio.sleep(self.retry_interval)
        assert last_exc is not None
        raise last_exc

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
            self._instances[instance_id] = {
                "session_id": session_id,
                "task_id": task_id,
            }
        except BaseException:
            # Best-effort cleanup. If we already have a server session_id,
            # try to close it on the server.
            server_session_id = locals().get("session_id", instance_id)
            if server_session_id:
                try:
                    await self._post_with_retries(
                        f"/session/{server_session_id}/close",
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
        screenshot = _decode_screenshot(observation.get("screenshot"))
        images = [screenshot] if screenshot is not None else []
        _log(f"[DesktopEnvTool] created session_id={session_id} has_screenshot={bool(screenshot)}")
        return instance_id, ToolResponse(image=images)

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
            _log(f"[DesktopEnvTool] screenshot step request_id={request_id} session_id={session_id}")
            resp = await self._post_with_retries(
                f"/session/{session_id}/step",
                {
                    "request_id": request_id,
                    "action": "import time; time.sleep(0)",
                    "pause": 0,
                },
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

        action = parameters.get("action", "")

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
            _log(
                f"[DesktopEnvTool] terminate step request_id={request_id} "
                f"session_id={session_id} status={status} code={code}"
            )
            resp = await self._post_with_retries(
                f"/session/{session_id}/step",
                {"request_id": request_id, "action": code, "pause": self.pause},
            )
            observation = resp.get("observation") or {}
            screenshot = _decode_screenshot(observation.get("screenshot"))
            images = [screenshot] if screenshot is not None else []
            meta = {k: v for k, v in resp.items() if k != "observation"}
            meta["code"] = code
            _log(
                f"[DesktopEnvTool] terminate step done request_id={request_id} "
                f"session_id={session_id} status={status} done={meta.get('done')} "
                f"step_count={meta.get('step_count')}"
            )
            return (
                ToolResponse(image=images, text=f"Task terminated with status: {status}; code: {code}"),
                self.step_reward,
                {"action": action, **meta},
            )
        if action == "answer":
            _log(f"[DesktopEnvTool] virtual action=answer session_id={session_id}")
            return (
                ToolResponse(text=f"Answer: {parameters.get('text', '')}"),
                0.0,
                {"action": action},
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
        pause = float(parameters.get("time", self.pause)) if action == "wait" else self.pause
        request_id = str(uuid4())
        _log(
            f"[DesktopEnvTool] step request_id={request_id} "
            f"session_id={session_id} action={action} raw_coordinate={raw_coordinate} "
            f"actual_coordinate={actual_coordinate} code={code} pause={pause}"
        )
        resp = await self._post_with_retries(
            f"/session/{session_id}/step",
            {"request_id": request_id, "action": code, "pause": pause},
        )

        observation = resp.get("observation") or {}
        screenshot = _decode_screenshot(observation.get("screenshot"))
        images = [screenshot] if screenshot is not None else []

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
        _log(
            f"[DesktopEnvTool] step done request_id={request_id} "
            f"session_id={session_id} action={action} "
            f"has_screenshot={bool(screenshot)} done={meta.get('done')} "
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

        _log(f"[DesktopEnvTool] evaluate session_id={session_id} local_settle={self.evaluate_settle_seconds}s")
        if self.evaluate_settle_seconds > 0:
            await asyncio.sleep(self.evaluate_settle_seconds)
        try:
            resp = await self._post_with_retries(
                f"/session/{session_id}/evaluate",
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
            )
            _log(f"[DesktopEnvTool] release OK session_id={session_id} instance_id={instance_id}")
        except Exception:
            _log(f"[DesktopEnvTool] Failed to close session {session_id}", level="ERROR")
            logger.warning("Failed to close session %s", session_id, exc_info=True)
        finally:
            self._instances.pop(instance_id, None)
            if not self._instances:
                await self._close_http_session()
