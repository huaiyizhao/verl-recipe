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
"""MCP-based desktop environment tool for GUI agent training.

This tool is a drop-in replacement for :class:`DesktopEnvTool` that uses the
`Model Context Protocol (MCP) <https://modelcontextprotocol.io>`_ instead of
a custom REST API.  It reuses Galileo's MCP utilities:

* :class:`SimpleMCPClient` — wraps ``fastmcp.Client`` for tool execution and
  screenshot capture.
* :class:`ComputerUseMCPAddressClient` — centralized HTTP service for MCP
  address allocation (acquire / release).
* :class:`MCPEnvironmentManager` — environment reset / reboot via MCP server
  HTTP endpoints.

Configuration
-------------

.. code-block:: yaml

   tools:
     - class_name: recipe.gui_agent.mcp_desktop_env_tool.MCPDesktopEnvTool
       config:
         type: native
         allocator_base_url: "http://allocator:8080"
         allocator_env: "a4861800"
         allocator_namespace: "Development"
         expire_min: 60
         auth_token: null          # optional MCP auth token
         timeout: 30
         reboot_max_retries: 12
         reboot_retry_interval: 2.0
         screen_width: 1000
         screen_height: 1000
"""

import base64
import io
import logging
import os
from typing import Any, ClassVar, Optional
from uuid import uuid4

from PIL import Image

from recipe.gui_agent.desktop_env_tool import DesktopEnvTool, _build_tool_schema
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _base64_to_pil(data_uri: str) -> Image.Image:
    """Convert a base64 data URI to a PIL Image.

    Handles both raw base64 strings and ``data:image/...;base64,...`` URIs
    as returned by the MCP screenshot endpoint.

    Args:
        data_uri: Base64-encoded image data, optionally prefixed with
            ``data:<mime>;base64,``.

    Returns:
        PIL Image in RGB mode.
    """
    if data_uri.startswith("data:"):
        # Strip the data URI header: data:image/png;base64,<data>
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    raw_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


class MCPDesktopEnvTool(DesktopEnvTool):
    """MCP-based desktop environment tool.

    Uses MCP protocol for environment communication instead of the REST API
    used by the parent :class:`DesktopEnvTool`.

    The class-level ``_address_client`` is shared across all instances (one
    per process) following the same pattern as Galileo's implementation.

    Config keys (in addition to :class:`DesktopEnvTool` keys):
        allocator_base_url (str): MCP address allocator service URL.
        allocator_env (str): Environment identifier (default ``"a4861800"``).
        allocator_namespace (str): Namespace identifier (default ``"Development"``).
        expire_min (int): Address lease timeout in minutes (default 60).
        auth_token (str | None): Optional MCP authentication token.
        timeout (int): MCP/HTTP request timeout in seconds (default 30).
        reboot_max_retries (int): Max retries when waiting for reboot (default 12).
        reboot_retry_interval (float): Seconds between reboot poll attempts (default 2.0).
    """

    # Shared across all instances within a process (lazy-initialised)
    _address_client: ClassVar[Any] = None

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        # We override the parent init to avoid requiring api_base_url
        screen_width = config.get("screen_width", 1000)
        screen_height = config.get("screen_height", 1000)

        if tool_schema is None:
            tool_schema = _build_tool_schema(screen_width, screen_height)

        # Call BaseTool.__init__ directly (skip DesktopEnvTool.__init__ which
        # requires api_base_url)
        from verl.tools.base_tool import BaseTool

        BaseTool.__init__(self, config=config, tool_schema=tool_schema)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.timeout = config.get("timeout", 30)
        self.step_reward = config.get("step_reward", 0.0)

        # MCP-specific config
        self.allocator_base_url = config["allocator_base_url"]
        self.allocator_env = config.get("allocator_env", "a4861800")
        self.allocator_namespace = config.get("allocator_namespace", "Development")
        self.expire_min = config.get("expire_min", 60)
        self.auth_token = config.get("auth_token")
        self.reboot_max_retries = config.get("reboot_max_retries", 12)
        self.reboot_retry_interval = config.get("reboot_retry_interval", 2.0)

        # instance_id → {"address": str, "mcp_client": SimpleMCPClient, "ground_truth": dict}
        self._instances: dict[str, dict[str, Any]] = {}

    @classmethod
    def _get_address_client(cls, config: dict) -> Any:
        """Lazy-initialise the class-level address client."""
        if cls._address_client is None:
            from cua.galileo.tools.utils import ComputerUseMCPAddressClient

            cls._address_client = ComputerUseMCPAddressClient(
                base_url=config["allocator_base_url"],
                env=config.get("allocator_env", "a4861800"),
                namespace=config.get("allocator_namespace", "Development"),
                expire_min=config.get("expire_min", 60),
            )
        return cls._address_client

    async def create(self, instance_id: Optional[str] = None, create_kwargs: dict | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Acquire an MCP address, reboot the environment, and take an initial screenshot.

        Args:
            instance_id: Optional pre-assigned ID.  A UUID is generated if None.
            create_kwargs: Must contain ``task_id``.  May contain ``ground_truth``
                for reward calculation.

        Returns:
            (instance_id, ToolResponse with initial screenshot)

        Raises:
            ValueError: If ``task_id`` is missing from create_kwargs.
            TimeoutError: If the environment fails to come back after reboot.
        """
        from cua.galileo.tools.utils import MCPEnvironmentManager, SimpleMCPClient, take_screenshot

        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = create_kwargs or {}
        task_id = create_kwargs.get("task_id")
        if not task_id:
            raise ValueError("create_kwargs must contain 'task_id'")

        ground_truth = create_kwargs.get("ground_truth", {})

        # 1. Get/create shared address client
        address_client = self._get_address_client(self.config)

        # 2. Allocate an MCP address
        address = await address_client.allocate_address(instance_id)
        mcp_url = address_client.get_mcp_url(address)

        try:
            # 3. Reboot the environment to a clean state
            await MCPEnvironmentManager.reboot_environment(
                address,
                max_retries=self.reboot_max_retries,
                retry_interval=self.reboot_retry_interval,
                timeout=self.timeout,
            )

            # 4. Create MCP client
            mcp_client = SimpleMCPClient(mcp_url, auth_token=self.auth_token)

            # 5. Take initial screenshot
            screenshot_b64 = await take_screenshot(mcp_client)
            if screenshot_b64 is None:
                raise RuntimeError(f"Failed to take initial screenshot from {address}")
            screenshot = _base64_to_pil(screenshot_b64)

        except Exception:
            # Release the address on failure
            try:
                await address_client.release_address(instance_id)
            except Exception:
                logger.warning(f"Failed to release address for {instance_id} after create error")
            raise

        self._instances[instance_id] = {
            "address": address,
            "mcp_url": mcp_url,
            "mcp_client": mcp_client,
            "task_id": task_id,
            "ground_truth": ground_truth,
        }
        return instance_id, ToolResponse(image=[screenshot])

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute an action via MCP tool call.

        Args:
            instance_id: The instance ID from ``create``.
            parameters: Action parameters (``action``, ``coordinate``, etc.).

        Returns:
            (ToolResponse with screenshot, step reward, metrics dict)

        Raises:
            ValueError: If instance_id is unknown.
            TimeoutError: If the MCP call times out (fatal — env is broken).
        """
        from cua.galileo.tools.utils import take_screenshot

        info = self._instances.get(instance_id)
        if info is None:
            raise ValueError(f"Unknown instance_id: {instance_id}")

        mcp_client = info["mcp_client"]
        action = parameters.get("action", "")

        # For "terminate" action we don't send to env
        if action == "terminate":
            return (
                ToolResponse(text=f"Task terminated with status: {parameters.get('status', 'unknown')}"),
                0.0,
                {},
            )

        # Call the MCP tool — TimeoutError propagates as fatal
        await mcp_client.call_tool("computer_use", parameters)

        # Take screenshot after action
        screenshot_b64 = await take_screenshot(mcp_client)
        if screenshot_b64 is None:
            # Non-fatal: screenshot failed but env may still be alive
            logger.warning(f"Screenshot failed after action '{action}' for {instance_id}")
            return ToolResponse(text=f"Executed action: {action} (screenshot unavailable)"), self.step_reward, {"action": action}

        screenshot = _base64_to_pil(screenshot_b64)

        action_summary = f"Executed action: {action}"
        if "coordinate" in parameters:
            action_summary += f" at {parameters['coordinate']}"

        return ToolResponse(image=[screenshot], text=action_summary), self.step_reward, {"action": action}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate task completion reward.

        If ground_truth was provided in create_kwargs it can be used for
        automated evaluation.  Currently returns 0.0 as a default — override
        in a subclass for custom reward logic.

        Returns:
            Reward score (0.0 by default).
        """
        info = self._instances.get(instance_id)
        if info is None:
            return 0.0
        # TODO: integrate ground-truth based reward evaluation
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Close the MCP client and release the address back to the pool.

        This MUST be called in a finally block to avoid leaking addresses.
        """
        info = self._instances.pop(instance_id, None)
        if info is None:
            return

        mcp_client = info.get("mcp_client")
        if mcp_client is not None:
            try:
                await mcp_client.close()
            except Exception:
                logger.warning(f"Failed to close MCP client for {instance_id}", exc_info=True)

        address_client = self._address_client
        if address_client is not None:
            try:
                await address_client.release_address(instance_id)
            except Exception:
                logger.warning(f"Failed to release address for {instance_id}", exc_info=True)
