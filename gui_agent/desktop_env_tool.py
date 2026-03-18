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
"""Remote desktop environment tool for GUI agent training.

This tool communicates with an externally-managed desktop environment pool
via HTTP API.  The lifecycle is:

    acquire → reset → (action + screenshot)* → task_status → release

The pool service is responsible for managing the actual desktop VMs; this
tool is a thin client.
"""

import copy
import io
import json
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
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Computer-use tool schema (Qwen-VL compatible)
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
            "and take successive screenshots to see the results of your actions. E.g. if you click on "
            "Firefox and a window doesn't open, try wait and taking another screenshot.\n"
            "* The screen's resolution is {screen_width}x{screen_height}.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should "
            "consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after waiting, "
            "try adjusting your cursor position so that the tip of the cursor visually falls on the "
            "element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of "
            "the element. Don't click boxes on their edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. The available actions are:\n"
                        "* `key`: Performs key down presses on the arguments passed in order, "
                        "then performs key releases in reverse order.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate.\n"
                        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate.\n"
                        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                        "* `hscroll`: Performs a horizontal scroll.\n"
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
                "keys": {
                    "description": "Required only by `action=key`.",
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type` and `action=answer`.",
                    "type": "string",
                },
                "coordinate": {
                    "description": (
                        "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                        "coordinates to move the mouse to."
                    ),
                    "type": "array",
                },
                "pixels": {
                    "description": (
                        "The amount of scrolling to perform. Positive values scroll up, "
                        "negative values scroll down. Required only by `action=scroll` and `action=hscroll`."
                    ),
                    "type": "number",
                },
                "time": {
                    "description": "The seconds to wait. Required only by `action=wait`.",
                    "type": "number",
                },
                "status": {
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
        },
    },
}


def _build_tool_schema(screen_width: int, screen_height: int) -> OpenAIFunctionToolSchema:
    """Build the OpenAI-format tool schema with concrete resolution."""
    tool = copy.deepcopy(_COMPUTER_USE_TOOL)
    tool["function"]["description"] = tool["function"]["description"].format(
        screen_width=screen_width,
        screen_height=screen_height,
    )
    return OpenAIFunctionToolSchema.model_validate(tool)


class DesktopEnvTool(BaseTool):
    """Remote desktop environment tool for GUI agent training.

    Communicates with an external desktop environment pool via HTTP.

    Config keys:
        api_base_url (str): Base URL of the environment pool API.
        screen_width (int): Screen width in pixels (default 1000).
        screen_height (int): Screen height in pixels (default 1000).
        timeout (int): HTTP request timeout in seconds (default 30).
        step_reward (float): Reward returned on each successful ``execute()``
            call (default 0.0).  Set to a small positive value (e.g. 0.1) to
            provide per-step reward shaping.  Subclasses can override
            :meth:`calc_reward` for custom task-level reward logic.
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
        self.step_reward = float(config.get("step_reward", 0.0))

        # instance_id → {"env_id": str, "task_id": str}
        self._instances: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict | None = None) -> dict:
        """POST JSON to the env pool API and return the JSON response."""
        url = f"{self.api_base_url}{path}"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload or {}) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _get(self, path: str) -> aiohttp.ClientResponse:
        """GET from the env pool API (for binary content like screenshots)."""
        url = f"{self.api_base_url}{path}"
        session = aiohttp.ClientSession(timeout=self.timeout)
        try:
            resp = await session.get(url)
            resp.raise_for_status()
            return resp, session
        except Exception:
            await session.close()
            raise

    async def _get_screenshot(self, env_id: str) -> Image.Image:
        """Download a screenshot from the environment and return a PIL Image."""
        resp, session = await self._get(f"/envs/{env_id}/screenshot")
        try:
            data = await resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        finally:
            resp.close()
            await session.close()

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _build_tool_schema(self.screen_width, self.screen_height)

    async def create(self, instance_id: Optional[str] = None, create_kwargs: dict | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Acquire an env from the pool, reset it, and take an initial screenshot.

        Args:
            instance_id: Optional pre-assigned ID. A UUID is generated if None.
            create_kwargs: Must contain ``task_id``.

        Returns:
            (instance_id, ToolResponse with initial screenshot)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = create_kwargs or {}
        task_id = create_kwargs.get("task_id")
        if not task_id:
            raise ValueError("create_kwargs must contain 'task_id'")

        # Step 1: Acquire an env from the external pool
        acquire_resp = await self._post("/envs/acquire")
        env_id = acquire_resp["env_id"]

        try:
            # Step 2: Reset the env for this task
            await self._post(f"/envs/{env_id}/reset", {"task_id": task_id})

            # Step 3: Take initial screenshot
            screenshot = await self._get_screenshot(env_id)
        except Exception:
            # If reset or screenshot fails, release the env
            try:
                await self._post(f"/envs/{env_id}/release")
            except Exception:
                logger.warning(f"Failed to release env {env_id} after create error")
            raise

        self._instances[instance_id] = {"env_id": env_id, "task_id": task_id}
        return instance_id, ToolResponse(image=[screenshot])

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute an action on the desktop environment.

        Args:
            instance_id: The instance ID from ``create``.
            parameters: Action parameters (``action``, ``coordinate``, etc.).

        Returns:
            (ToolResponse with screenshot, step reward 0.0, metrics dict)
        """
        info = self._instances.get(instance_id)
        if info is None:
            raise ValueError(f"Unknown instance_id: {instance_id}")
        env_id = info["env_id"]

        action = parameters.get("action", "")

        # For "terminate" action we don't send to env, just return empty response
        if action == "terminate":
            return ToolResponse(text=f"Task terminated with status: {parameters.get('status', 'unknown')}"), 0.0, {}

        # Send action to the env
        await self._post(f"/envs/{env_id}/action", parameters)

        # Get new screenshot
        screenshot = await self._get_screenshot(env_id)

        action_summary = f"Executed action: {action}"
        if "coordinate" in parameters:
            action_summary += f" at {parameters['coordinate']}"

        return ToolResponse(image=[screenshot], text=action_summary), self.step_reward, {"action": action}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Check task completion via the environment API.

        Returns:
            1.0 if the task is completed, 0.0 otherwise.
        """
        info = self._instances.get(instance_id)
        if info is None:
            return 0.0
        env_id = info["env_id"]

        try:
            resp = await self._post(f"/envs/{env_id}/task_status")
            return 1.0 if resp.get("completed", False) else 0.0
        except Exception:
            logger.warning(f"Failed to check task status for env {env_id}", exc_info=True)
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the environment back to the pool.

        This MUST be called in a finally block to avoid leaking envs.
        """
        info = self._instances.pop(instance_id, None)
        if info is None:
            return
        env_id = info["env_id"]

        try:
            await self._post(f"/envs/{env_id}/release")
        except Exception:
            logger.warning(f"Failed to release env {env_id}", exc_info=True)
