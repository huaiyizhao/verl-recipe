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

import base64
import copy
import io
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import aiohttp
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


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

        # instance_id → {"session_id": str, "task_id": str}
        self._instances: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict | None = None) -> dict:
        """POST JSON to the desktop env service and return the JSON response."""
        url = f"{self.api_base_url}{path}"
        logger.debug("[DesktopEnvTool] POST %s payload_keys=%s", path, list((payload or {}).keys()))
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload or {}) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error(
                        "[DesktopEnvTool] POST %s -> HTTP %d body=%s",
                        path,
                        resp.status,
                        body[:500],
                    )
                resp.raise_for_status()
                if resp.content_type == "application/json":
                    data = await resp.json()
                    logger.debug(
                        "[DesktopEnvTool] POST %s -> keys=%s", path, list(data.keys())
                    )
                    return data
                # Empty body is allowed for endpoints like /close.
                logger.debug("[DesktopEnvTool] POST %s -> empty body", path)
                return {}

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

        logger.info(
            "[DesktopEnvTool] create session task_id=%s instance_id=%s",
            task_id,
            instance_id,
        )
        resp = await self._post("/session/create", {"task_id": task_id})
        session_id = resp.get("session_id")
        if not session_id:
            raise RuntimeError(
                f"/session/create did not return session_id; response={resp!r}"
            )

        self._instances[instance_id] = {"session_id": session_id, "task_id": task_id}

        observation = resp.get("observation") or {}
        screenshot = _decode_screenshot(observation.get("screenshot"))
        images = [screenshot] if screenshot is not None else []
        logger.info(
            "[DesktopEnvTool] created session_id=%s has_screenshot=%s",
            session_id,
            bool(screenshot),
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

        # Virtual actions that the agent loop handles directly.
        if action == "terminate":
            logger.info(
                "[DesktopEnvTool] virtual action=terminate session_id=%s status=%s",
                session_id,
                parameters.get("status", "unknown"),
            )
            return (
                ToolResponse(
                    text=f"Task terminated with status: {parameters.get('status', 'unknown')}"
                ),
                0.0,
                {"action": action},
            )
        if action == "answer":
            logger.info(
                "[DesktopEnvTool] virtual action=answer session_id=%s", session_id
            )
            return (
                ToolResponse(text=f"Answer: {parameters.get('text', '')}"),
                0.0,
                {"action": action},
            )

        code = _translate_action_to_pyautogui(parameters)
        logger.debug(
            "[DesktopEnvTool] step session_id=%s action=%s code=%s",
            session_id,
            action,
            code,
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
        logger.info(
            "[DesktopEnvTool] step done session_id=%s action=%s has_screenshot=%s done=%s step_count=%s",
            session_id,
            action,
            bool(screenshot),
            meta.get("done"),
            meta.get("step_count"),
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
            logger.warning(
                "[DesktopEnvTool] calc_reward: unknown instance_id=%s (already released?)",
                instance_id,
            )
            return 0.0
        session_id = info["session_id"]

        logger.info(
            "[DesktopEnvTool] evaluate session_id=%s settle=%ds",
            session_id,
            self.evaluate_settle_seconds,
        )
        try:
            resp = await self._post(
                f"/session/{session_id}/evaluate",
                {"settle_seconds": self.evaluate_settle_seconds},
            )
        except Exception:
            logger.warning(
                "Failed to evaluate session %s", session_id, exc_info=True
            )
            return 0.0

        try:
            reward = float(resp.get("reward", 0.0))
        except (TypeError, ValueError):
            logger.warning(
                "[DesktopEnvTool] evaluate session_id=%s returned non-numeric reward: %r",
                session_id,
                resp.get("reward"),
            )
            return 0.0
        logger.info(
            "[DesktopEnvTool] evaluate session_id=%s reward=%.4f", session_id, reward
        )
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """Close the session. MUST be called in a ``finally`` block."""
        info = self._instances.pop(instance_id, None)
        if info is None:
            logger.debug(
                "[DesktopEnvTool] release: no-op for unknown instance_id=%s", instance_id
            )
            return
        session_id = info["session_id"]
        logger.info(
            "[DesktopEnvTool] release session_id=%s task_id=%s",
            session_id,
            info.get("task_id"),
        )
        try:
            await self._post(f"/session/{session_id}/close")
        except Exception:
            logger.warning(
                "Failed to close session %s", session_id, exc_info=True
            )
