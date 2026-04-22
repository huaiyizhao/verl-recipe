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
    POST /session/{session_id}/evaluate            body: {"settle_seconds": 20}
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
import os
import sys
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

_DEBUG_ENABLED = os.getenv("VERL_LOGGING_LEVEL", "INFO").upper() == "DEBUG"


def _log(msg: str, *, debug: bool = False) -> None:
    """Print-based logger that bypasses the ``logging`` framework entirely.

    ``flush=True`` + ``stderr`` ensures lines survive worker crashes.
    Honors ``VERL_LOGGING_LEVEL=DEBUG`` for debug-gated messages.
    """
    if debug and not _DEBUG_ENABLED:
        return
    print(msg, file=sys.stderr, flush=True)


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
                        sk: (f"<{sk}: {len(sv)} chars elided>" if isinstance(sv, str) and len(sv) > 200 and sk in ("screenshot", "image", "a11y_tree") else sv)
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
                        "key", "type", "mouse_move", "left_click", "left_click_drag",
                        "right_click", "middle_click", "double_click", "triple_click",
                        "scroll", "hscroll", "wait", "terminate", "answer",
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


def _translate_action_to_pyautogui(parameters: dict[str, Any]) -> Optional[str]:
    """Translate a Qwen-VL computer_use structured action into a pyautogui code line.

    Returns ``None`` for virtual actions that are handled by the agent loop
    directly (``terminate``, ``answer``) and do not map to a backend step.
    """
    action = parameters.get("action", "")
    coord = parameters.get("coordinate")

    if action == "terminate" or action == "answer":
        return None

    if action == "mouse_move":
        x, y = coord
        return f"pyautogui.moveTo({int(x)}, {int(y)})"

    if action == "left_click":
        x, y = coord
        return f"pyautogui.click(x={int(x)}, y={int(y)}, button='left')"

    if action == "right_click":
        x, y = coord
        return f"pyautogui.click(x={int(x)}, y={int(y)}, button='right')"

    if action == "middle_click":
        x, y = coord
        return f"pyautogui.click(x={int(x)}, y={int(y)}, button='middle')"

    if action == "double_click":
        x, y = coord
        return f"pyautogui.doubleClick(x={int(x)}, y={int(y)})"

    if action == "triple_click":
        x, y = coord
        return f"pyautogui.tripleClick(x={int(x)}, y={int(y)})"

    if action == "left_click_drag":
        x, y = coord
        # Drag *to* the target from the current cursor position, mouse button held left.
        return f"pyautogui.dragTo({int(x)}, {int(y)}, button='left')"

    if action == "type":
        text = parameters.get("text", "")
        # Use write() for plain ASCII typing; repr to safely embed the string.
        return f"pyautogui.write({text!r}, interval=0.02)"

    if action == "key":
        keys = parameters.get("keys", []) or []
        # hotkey() presses keys together (e.g. ctrl+c); this matches the
        # "press-then-release-in-reverse" semantics described by the schema.
        key_repr = ", ".join(repr(str(k)) for k in keys)
        return f"pyautogui.hotkey({key_repr})"

    if action == "scroll":
        pixels = int(parameters.get("pixels", 0) or 0)
        return f"pyautogui.scroll({pixels})"

    if action == "hscroll":
        pixels = int(parameters.get("pixels", 0) or 0)
        return f"pyautogui.hscroll({pixels})"

    if action == "wait":
        seconds = float(parameters.get("time", 0) or 0)
        return f"import time; time.sleep({seconds})"

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
        screen_width (int): Screen width in pixels (default 1000).
        screen_height (int): Screen height in pixels (default 1000).
        timeout (int): HTTP request timeout in seconds (default 30).
        pause (float): ``pause`` value forwarded to ``/step`` after each
            action (default 2.0).
        evaluate_settle_seconds (int): ``settle_seconds`` value forwarded to
            ``/evaluate`` when computing the terminal reward (default 20).
        step_reward (float): Per-step reward returned by ``execute()``
            (default 0.0).
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
        self.timeout = aiohttp.ClientTimeout(total=config.get("timeout", 30))
        self.pause = float(config.get("pause", 2.0))
        self.evaluate_settle_seconds = int(config.get("evaluate_settle_seconds", 20))
        self.step_reward = float(config.get("step_reward", 0.0))

        # HTTP retry config: retry ``max_retries`` times with a fixed
        # ``retry_interval`` seconds between attempts. After that, _post
        # raises and the caller (agent loop) will abort the rollout.
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_interval = float(config.get("retry_interval", 30.0))

        # instance_id → {"session_id": str, "task_id": str}
        self._instances: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post(
        self,
        path: str,
        payload: dict | None = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> dict:
        """POST JSON to the desktop env service and return the JSON response.

        Logs the request payload and response body at INFO level. On any
        failure, retry up to ``self.max_retries`` times, sleeping
        ``self.retry_interval`` seconds between attempts. If all retries
        fail, raise so the caller (agent loop) can abort the rollout.
        """
        url = f"{self.api_base_url}{path}"
        attempts = self.max_retries + 1  # initial attempt + retries
        last_exc: Optional[BaseException] = None
        effective_timeout = timeout if timeout is not None else self.timeout
        request_body = payload or {}

        _log(f"[DesktopEnvTool] -> POST {url} payload={_short_repr(request_body)}")

        for attempt in range(1, attempts + 1):
            try:
                async with aiohttp.ClientSession(timeout=effective_timeout) as session:
                    async with session.post(url, json=request_body) as resp:
                        status = resp.status
                        content_type = resp.content_type
                        # Always read the full body so we can log it (and show
                        # it in error messages).
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
                                    f"non-JSON body (content_type={content_type}): {_short_repr(text_body)}"
                                )
                                raise RuntimeError(
                                    f"POST {path} returned non-JSON body: {text_body!r}"
                                ) from je
                            _log(
                                f"[DesktopEnvTool] <- POST {url} status={status} "
                                f"response={_short_repr(data)}"
                            )
                            return data
                        # Empty body is allowed for endpoints like /close.
                        _log(
                            f"[DesktopEnvTool] <- POST {url} status={status} "
                            f"empty body (content_type={content_type})"
                        )
                        return {}
            except Exception as exc:
                last_exc = exc
                # Extract as much detail as possible from the exception.
                detail = str(exc) or repr(exc)
                if isinstance(exc, aiohttp.ClientResponseError):
                    detail = (
                        f"status={exc.status} message={exc.message!r} "
                        f"url={exc.request_info.url if exc.request_info else url}"
                    )
                if attempt >= attempts:
                    _log(
                        f"[DesktopEnvTool] POST {path} failed after {attempts} attempts: "
                        f"{detail} | payload={_short_repr(request_body)}"
                    )
                    # Preserve stack trace via the logging framework (ERROR
                    # is not filtered out by the default WARNING config).
                    logger.error(
                        "[DesktopEnvTool] POST %s failed after %d attempts",
                        path, attempts,
                        exc_info=True,
                    )
                    break
                _log(
                    f"[DesktopEnvTool] POST {path} failed (attempt {attempt}/{attempts}): "
                    f"{detail}. Retrying in {self.retry_interval:.1f}s"
                )
                await asyncio.sleep(self.retry_interval)

        raise RuntimeError(
            f"POST {path} failed after {attempts} attempts: {last_exc}"
        ) from last_exc

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

        _log(
            f"[DesktopEnvTool] create session task_id={task_id} instance_id={instance_id}"
        )
        resp = await self._post(
            "/session/create",
            {"task_id": task_id, "require_a11y_tree": False},
            timeout=aiohttp.ClientTimeout(total=120),
        )
        session_id = resp.get("session_id")
        if not session_id:
            raise RuntimeError(
                f"/session/create did not return session_id; response={resp!r}"
            )

        self._instances[instance_id] = {"session_id": session_id, "task_id": task_id}

        observation = resp.get("observation") or {}
        screenshot = _decode_screenshot(observation.get("screenshot"))
        images = [screenshot] if screenshot is not None else []
        _log(
            f"[DesktopEnvTool] created session_id={session_id} "
            f"has_screenshot={bool(screenshot)}"
        )
        return instance_id, ToolResponse(image=images)

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
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

        # Reject unknown actions with a structured tool response instead of
        # raising. Aborting the rollout here was the root cause of many
        # cold-start training failures where the untrained model kept
        # emitting schema-invalid action names (``click``, ``press``, ...),
        # causing the entire rollout to be discarded and the batch to be
        # empty. Returning a descriptive error lets the agent learn the
        # valid action vocabulary through tool feedback.
        if action not in _VALID_COMPUTER_USE_ACTIONS:
            valid_list = ", ".join(_VALID_COMPUTER_USE_ACTIONS)
            _log(
                f"[DesktopEnvTool] invalid action={action!r} "
                f"session_id={session_id} (returning soft error)"
            )
            return (
                ToolResponse(
                    text=(
                        f"Error: unknown action {action!r}. "
                        f"No screen state change. "
                        f"Valid computer_use actions are: {valid_list}."
                    )
                ),
                0.0,
                {"action": action, "invalid_action": True},
            )

        # Virtual actions that the agent loop handles directly.
        if action == "terminate":
            _log(
                f"[DesktopEnvTool] virtual action=terminate session_id={session_id} "
                f"status={parameters.get('status', 'unknown')}"
            )
            return (
                ToolResponse(
                    text=f"Task terminated with status: {parameters.get('status', 'unknown')}"
                ),
                0.0,
                {"action": action},
            )
        if action == "answer":
            _log(f"[DesktopEnvTool] virtual action=answer session_id={session_id}")
            return (
                ToolResponse(text=f"Answer: {parameters.get('text', '')}"),
                0.0,
                {"action": action},
            )

        code = _translate_action_to_pyautogui(parameters)
        _log(
            f"[DesktopEnvTool] step session_id={session_id} action={action} code={code}",
            debug=True,
        )
        resp = await self._post(
            f"/session/{session_id}/step",
            {"action": code, "pause": self.pause},
        )

        observation = resp.get("observation") or {}
        screenshot = _decode_screenshot(observation.get("screenshot"))
        images = [screenshot] if screenshot is not None else []

        action_summary = f"Executed action: {action}"
        if "coordinate" in parameters:
            action_summary += f" at {parameters['coordinate']}"

        # Forward non-observation metadata (reward / done / info / step_count).
        meta = {k: v for k, v in resp.items() if k != "observation"}
        _log(
            f"[DesktopEnvTool] step done session_id={session_id} action={action} "
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
            _log(
                f"[DesktopEnvTool] calc_reward: unknown instance_id={instance_id} "
                f"(already released?)"
            )
            return 0.0
        session_id = info["session_id"]

        _log(
            f"[DesktopEnvTool] evaluate session_id={session_id} "
            f"settle={self.evaluate_settle_seconds}s"
        )
        try:
            resp = await self._post(
                f"/session/{session_id}/evaluate",
                {"settle_seconds": self.evaluate_settle_seconds},
            )
        except Exception:
            _log(f"[DesktopEnvTool] Failed to evaluate session {session_id}")
            logger.warning(
                "Failed to evaluate session %s", session_id, exc_info=True
            )
            return 0.0

        try:
            reward = float(resp.get("reward", 0.0))
        except (TypeError, ValueError):
            _log(
                f"[DesktopEnvTool] evaluate session_id={session_id} "
                f"returned non-numeric reward: {resp.get('reward')!r}"
            )
            return 0.0
        _log(f"[DesktopEnvTool] evaluate session_id={session_id} reward={reward:.4f}")
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """Close the session. MUST be called in a ``finally`` block."""
        info = self._instances.pop(instance_id, None)
        if info is None:
            _log(
                f"[DesktopEnvTool] release: no-op for unknown instance_id={instance_id}",
                debug=True,
            )
            return
        session_id = info["session_id"]
        _log(
            f"[DesktopEnvTool] release session_id={session_id} "
            f"task_id={info.get('task_id')}"
        )
        try:
            await self._post(f"/session/{session_id}/close")
        except Exception:
            _log(f"[DesktopEnvTool] Failed to close session {session_id}")
            logger.warning(
                "Failed to close session %s", session_id, exc_info=True
            )
