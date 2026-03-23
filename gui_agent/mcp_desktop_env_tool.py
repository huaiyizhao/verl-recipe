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

Drop-in replacement for :class:`DesktopEnvTool` using the
`Model Context Protocol <https://modelcontextprotocol.io>`_.

Self-contained — all MCP helpers (address allocation, environment
reboot, screenshot capture, task validation) are inlined here.

External dependencies: ``aiohttp``, ``requests``, ``fastmcp`` (lazy),
``PIL``.
"""

import asyncio
import base64
import io
import logging
import os
from typing import Any, ClassVar, Optional
from uuid import uuid4

import aiohttp
import requests
from PIL import Image

from recipe.gui_agent.desktop_env_tool import _build_tool_schema
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _to_base_url(address: str) -> str:
    """``"host:port"`` or ``"http://host:port"`` → ``"http://host:port"``."""
    if address.startswith(("http://", "https://")):
        return address.rstrip("/")
    return f"http://{address}"


def _build_initial_url(ground_truth: dict, url_rewrite: dict | None) -> str | None:
    """Build initial navigation URL from url_rewrite config, or None."""
    if not url_rewrite:
        return None
    target = url_rewrite.get("target", "").rstrip("/")
    if not target:
        return None
    params = []
    for arg in url_rewrite.get("args", []):
        key = arg.get("key", "")
        param = arg.get("param", key)
        value = ground_truth.get(key, "")
        if value:
            params.append(f"{param}={value}")
    return f"{target}/?{'&'.join(params)}" if params else f"{target}/"


def _base64_to_pil(data_uri: str) -> Image.Image:
    """Decode a base64 (or ``data:image/…;base64,…``) string to a PIL Image."""
    encoded = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


# ============================================================================
# _AddressClient — address allocation + environment reboot
# ============================================================================


class _AddressClient:
    """Manages MCP address lifecycle: allocate, reboot, release."""

    def __init__(
        self,
        base_url: str,
        env: str = None,
        namespace: str = None,
        expire_min: int = 60,
        retry_interval: float = 1.0,
        max_retries: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.env = env
        self.namespace = namespace
        self.expire_min = expire_min
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self._mapping: dict[str, str] = {}  # instance_id → address

    async def allocate(self, instance_id: str) -> str:
        """Request an address via ``POST /mcp-assign`` with automatic retry."""
        if instance_id in self._mapping:
            return self._mapping[instance_id]

        payload = {"env": self.env, "namespace": self.namespace, "expire_min": self.expire_min}
        logger.info("Allocating address for %s: POST %s/mcp-assign payload=%s", instance_id, self.base_url, payload)
        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.base_url}/mcp-assign", json=payload, timeout=_HTTP_TIMEOUT) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            address = data.get("address") if isinstance(data, dict) else None
                            if address:
                                self._mapping[instance_id] = address
                                logger.info("Allocated %s -> %s (attempt %s)", address, instance_id, attempt)
                                return address
                            logger.warning(
                                "Allocate: no address in response (attempt %s/%s): %s",
                                attempt, self.max_retries, data,
                            )
                        else:
                            body = await resp.text()
                            logger.warning(
                                "Allocate: HTTP %s (attempt %s/%s) url=%s body=%s",
                                resp.status, attempt, self.max_retries, self.base_url, body[:200],
                            )
            except Exception as exc:
                logger.warning("Allocate error (attempt %s/%s) url=%s: %s", attempt, self.max_retries, self.base_url, exc)
            await asyncio.sleep(self.retry_interval)

        raise RuntimeError(f"Failed to allocate address for {instance_id} after {self.max_retries} attempts")

    async def reboot(self, address: str, max_retries: int = 30, retry_interval: float = 2.0, timeout: int = 10) -> None:
        """POST ``/instruction/reboot`` then poll ``/instruction/status`` until 200."""
        base = _to_base_url(address)
        to = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base}/instruction/reboot", json={}, timeout=to) as resp:
                logger.info("Reboot %s -> HTTP %s", address, resp.status)

        await asyncio.sleep(5)  # wait for reboot to begin

        for attempt in range(1, max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base}/instruction/status", timeout=to) as resp:
                        if resp.status == 200:
                            logger.info("Node %s back online", address)
                            await asyncio.sleep(2)
                            return
            except Exception as exc:
                logger.debug("Waiting for %s (%s/%s): %s", address, attempt, max_retries, exc)
            await asyncio.sleep(retry_interval)

        raise TimeoutError(f"Node {address} did not come back after {max_retries} attempts")

    async def release(self, instance_id: str) -> None:
        """Release an address via ``POST /mcp-lock``.  No-op if nothing allocated."""
        address = self._mapping.get(instance_id)
        if address is None:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/mcp-lock", json={"address": address, "unlock": True}, timeout=_HTTP_TIMEOUT
                ) as resp:
                    if resp.status == 200 and (await resp.json()).get("code") == 0:
                        del self._mapping[instance_id]
                        logger.info("Released %s from %s", address, instance_id)
                    else:
                        logger.error("Failed to release %s: HTTP %s", address, resp.status)
        except Exception as exc:
            logger.error("release error for %s: %s", address, exc)

    @staticmethod
    def to_mcp_url(address: str) -> str:
        """``"host:port"`` → ``"http://host:port/mcp"``."""
        base = _to_base_url(address.strip())
        return base if base.endswith("/mcp") else f"{base}/mcp"


# ============================================================================
# _MCPClient — MCP tool calls + screenshot
# ============================================================================


class _MCPClient:
    """Thin wrapper around :class:`fastmcp.Client` (imported lazily)."""

    def __init__(self, mcp_url: str, auth_token: Optional[str] = None):
        from fastmcp import Client
        from fastmcp.client.transports import SSETransport

        self.mcp_url = mcp_url
        if auth_token:
            self._client = Client(SSETransport(url=mcp_url, headers={"Authorization": f"Bearer {auth_token}"}))
        else:
            self._client = Client({"mcpServers": {"default": {"url": mcp_url}}})

    async def call_tool(self, tool_name: str, parameters: dict[str, Any]) -> Any:
        """Call an MCP tool and return the raw result."""
        try:
            async with self._client:
                return await self._client.call_tool_mcp(tool_name, parameters)
        except Exception:
            logger.error("MCP call %s failed (url=%s)", tool_name, self.mcp_url)
            raise

    async def take_screenshot(self) -> Optional[str]:
        """Call ``browser_take_screenshot`` and return a ``data:image/…;base64,…`` URI, or *None*."""
        try:
            result = await self.call_tool("browser_take_screenshot", {})
        except Exception as exc:
            logger.warning("Screenshot failed: %s", exc)
            return None

        for item in getattr(result, "content", None) or []:
            if hasattr(item, "type") and item.type == "image" and isinstance(getattr(item, "data", None), str):
                raw = item.data
                if raw.startswith("data:"):
                    return raw
                mime = getattr(item, "mimeType", None) or getattr(item, "mime_type", None) or "image/png"
                return f"data:{mime};base64,{raw}"
        return None

    async def close(self) -> None:
        """Best-effort cleanup of underlying client resources."""
        try:
            await asyncio.sleep(0.1)
            if hasattr(self._client, "close"):
                await self._client.close()
            for task in getattr(self._client, "_tasks", []):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        except Exception as exc:
            logger.warning("Client cleanup error: %s", exc)


# ============================================================================
# Task validation (reward) — sync HTTP, not tied to a class
# ============================================================================


def _check_task(base_url: str, seed: str, qseed: str, session: str, mock_date: str = "", timeout: int = 15) -> dict[str, Any]:
    """``GET <base_url>/api/__task__?…&check=true`` → ``{"score": int, "reward": float}``."""
    url = f"{base_url}/api/__task__?seed={seed}&qseed={qseed}&_session_={session}&_mockdate_={mock_date}&check=true"
    try:
        resp = requests.get(url, timeout=timeout, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("Validation API code=%s message=%s", data.get("code"), data.get("message"))
            return {"score": 0, "reward": 0.0}
        result = data.get("result", {})
        score = result.get("score", 0)
        reward = max(0.0, min(score / 100.0, 1.0))
        logger.info("Validation: score=%s reward=%s", score, reward)
        return {"score": score, "reward": reward}
    except Exception as exc:
        logger.error("check_task failed: %s", exc, exc_info=True)
        return {"score": 0, "reward": 0.0}


def _reset_task(base_url: str, seed: str, qseed: str, session: str, mock_date: str = "", timeout: int = 10) -> None:
    """Best-effort reset after reward evaluation."""
    try:
        url = f"{base_url}/__reset__?seed={seed}&qseed={qseed}&_session_={session}&_mockdate_={mock_date}"
        requests.get(url, timeout=timeout, verify=False)
    except Exception as exc:
        logger.warning("reset_task failed (non-critical): %s", exc)


# ============================================================================
# MCPDesktopEnvTool
# ============================================================================


class MCPDesktopEnvTool(BaseTool):
    """MCP-based desktop environment tool.

    Config keys:
        allocator_base_url, allocator_env, allocator_namespace, expire_min,
        auth_token, timeout, reboot_max_retries, reboot_retry_interval,
        screen_width, screen_height, step_reward.
    """

    _address_client: ClassVar[Optional[_AddressClient]] = None

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        screen_width = config.get("screen_width", 1000)
        screen_height = config.get("screen_height", 1000)
        if tool_schema is None:
            tool_schema = _build_tool_schema(screen_width, screen_height)

        super().__init__(config=config, tool_schema=tool_schema)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.timeout = config.get("timeout", 30)
        self.step_reward = config.get("step_reward", 0.0)
        self.auth_token = config.get("auth_token")
        self.reboot_max_retries = config.get("reboot_max_retries", 12)
        self.reboot_retry_interval = config.get("reboot_retry_interval", 2.0)
        self._instances: dict[str, dict[str, Any]] = {}

        if MCPDesktopEnvTool._address_client is None:
            MCPDesktopEnvTool._address_client = _AddressClient(
                base_url=config["allocator_base_url"],
                env=config.get("allocator_env", "a4861800"),
                namespace=config.get("allocator_namespace", "Development"),
                expire_min=config.get("expire_min", 60),
            )

    async def create(self, instance_id: Optional[str] = None, create_kwargs: dict | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Acquire an MCP address, reboot, and return an initial screenshot."""
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = create_kwargs or {}
        task_id = create_kwargs.get("task_id")
        if not task_id:
            raise ValueError("create_kwargs must contain 'task_id'")

        # Generate session and inject into ground_truth
        ground_truth = dict(create_kwargs.get("ground_truth", {}))
        ground_truth["session"] = str(uuid4())

        addr_client = self._address_client
        address = await addr_client.allocate(instance_id)

        try:
            await addr_client.reboot(address, self.reboot_max_retries, self.reboot_retry_interval, self.timeout)
            mcp = _MCPClient(_AddressClient.to_mcp_url(address), auth_token=self.auth_token)

            # Navigate browser to initial URL if url_rewrite is configured
            initial_url = _build_initial_url(ground_truth, create_kwargs.get("url_rewrite"))
            if initial_url:
                await mcp.call_tool("browser_navigate", {"url": initial_url})

            screenshot_b64 = await mcp.take_screenshot()
            if screenshot_b64 is None:
                raise RuntimeError(f"Failed to take initial screenshot from {address}")
        except Exception:
            try:
                await addr_client.release(instance_id)
            except Exception:
                logger.warning("Failed to release address for %s after create error", instance_id)
            raise

        self._instances[instance_id] = {
            "address": address,
            "mcp_client": mcp,
            "task_id": task_id,
            "ground_truth": ground_truth,
        }
        return instance_id, ToolResponse(image=[_base64_to_pil(screenshot_b64)])

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute an action via MCP and return (response, step_reward, metrics)."""
        info = self._instances.get(instance_id)
        if info is None:
            raise ValueError(f"Unknown instance_id: {instance_id}")

        action = parameters.get("action", "")
        if action == "terminate":
            return ToolResponse(text=f"Task terminated with status: {parameters.get('status', 'unknown')}"), 0.0, {}

        mcp: _MCPClient = info["mcp_client"]
        await mcp.call_tool("computer_use", parameters)

        screenshot_b64 = await mcp.take_screenshot()
        if screenshot_b64 is None:
            logger.warning("Screenshot failed after '%s' for %s", action, instance_id)
            return ToolResponse(text=f"Executed action: {action} (screenshot unavailable)"), self.step_reward, {"action": action}

        summary = f"Executed action: {action}"
        if "coordinate" in parameters:
            summary += f" at {parameters['coordinate']}"
        return ToolResponse(image=[_base64_to_pil(screenshot_b64)], text=summary), self.step_reward, {"action": action}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate task reward via the ground-truth validation API.

        Requires ``ground_truth`` to contain ``base_url``, ``seed``,
        ``qseed``, ``session``, and optionally ``mock_date``.
        Returns 0.0 if any required key is missing.
        """
        info = self._instances.get(instance_id)
        if info is None:
            return 0.0

        gt = info.get("ground_truth", {})
        base_url, seed, qseed, session = gt.get("base_url", ""), gt.get("seed"), gt.get("qseed"), gt.get("session")
        if not (seed and qseed and session):
            return 0.0

        mock_date = gt.get("mock_date", "")
        result = _check_task(base_url, seed, qseed, session, mock_date)
        _reset_task(base_url, seed, qseed, session, mock_date)
        logger.info("calc_reward for %s: score=%s reward=%s", instance_id, result.get("score"), result["reward"])
        return result["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Close the MCP client and release the address back to the pool."""
        info = self._instances.pop(instance_id, None)
        if info is None:
            return
        try:
            mcp = info.get("mcp_client")
            if mcp is not None:
                await mcp.close()
        except Exception:
            logger.warning("Failed to close MCP client for %s", instance_id, exc_info=True)
        finally:
            if self._address_client is not None:
                try:
                    await self._address_client.release(instance_id)
                except Exception:
                    logger.warning("Failed to release address for %s", instance_id, exc_info=True)
