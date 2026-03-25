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
"""CPU-only unit tests for GUI Agent Loop components.

Tests:
- keep_last_k_images: image pruning logic
- DesktopEnvTool: mocked HTTP interactions
- Multi-trajectory output structure
- Context strategies: KeepLastKImagesStrategy, SlidingWindowStrategy
- Per-step reward accumulation
- Fatal error handling (TimeoutError propagation)
- MCPDesktopEnvTool with mocked MCP utilities
"""

import asyncio
import io
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from recipe.gui_agent.context_manager import keep_last_k_images


# ---------------------------------------------------------------------------
# Helper: create a small test image
# ---------------------------------------------------------------------------


def _make_image(color: str = "red", size: tuple = (10, 10)) -> Image.Image:
    return Image.new("RGB", size, color)


# ===========================================================================
# Tests for keep_last_k_images
# ===========================================================================


class TestKeepLastKImages:
    """Tests for the keep_last_k_images helper function."""

    def test_no_pruning_when_under_k(self):
        """When total images <= k, nothing changes."""
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}, {"type": "text", "text": "hi"}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("blue")]

        result_msgs = keep_last_k_images(messages, k=3)

        # Nothing should change
        # Both image blocks should be preserved
        assert result_msgs[0]["content"][0]["type"] == "image_url"
        assert result_msgs[1]["content"][0]["type"] == "image_url"

    def test_prune_oldest_images(self):
        """When total images > k, oldest are replaced with placeholders."""
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}, {"type": "text", "text": "turn 1"}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("green"), _make_image("blue"), _make_image("orange")]

        result_msgs = keep_last_k_images(messages, k=2)

        # Only 2 images should remain

        # First two messages' images should be replaced
        assert result_msgs[0]["content"][0] == {"type": "text", "text": "[screenshot omitted]"}
        assert result_msgs[1]["content"][0] == {"type": "text", "text": "[screenshot omitted]"}

        # Last two should be preserved
        assert result_msgs[2]["content"][0]["type"] == "image_url"
        assert result_msgs[3]["content"][0]["type"] == "image_url"

    def test_prune_with_k_equals_1(self):
        """Only keep the very last image."""
        messages = [
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("green"), _make_image("blue")]

        result_msgs = keep_last_k_images(messages, k=1)

        # Only last message should have an image
        assert result_msgs[0]["content"][0]["type"] == "text"
        assert result_msgs[1]["content"][0]["type"] == "text"
        assert result_msgs[2]["content"][0]["type"] == "image_url"

    def test_mixed_content_preserved(self):
        """Non-image content blocks in the same message are untouched."""
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}},
                    {"type": "text", "text": "action result"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}},
                    {"type": "text", "text": "latest result"},
                ],
            },
        ]
        images = [_make_image("red"), _make_image("blue")]

        result_msgs = keep_last_k_images(messages, k=1)

        # First message: image replaced, text preserved
        assert result_msgs[0]["content"][0] == {"type": "text", "text": "[screenshot omitted]"}
        assert result_msgs[0]["content"][1] == {"type": "text", "text": "action result"}

        # Second message: image preserved, text preserved
        assert result_msgs[1]["content"][0]["type"] == "image_url"
        assert result_msgs[1]["content"][1]["type"] == "text"

    def test_invalid_k_raises(self):
        """k <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            keep_last_k_images([], k=0)

    def test_no_images(self):
        """Messages without images are left untouched."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        images: list = []

        result_msgs = keep_last_k_images(messages, k=3)
        assert result_msgs == messages

    def test_string_content_ignored(self):
        """Messages with string content (not list) are skipped."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("blue")]

        result_msgs = keep_last_k_images(messages, k=1)

        # String content message untouched
        assert result_msgs[0]["content"] == "hello"
        # First image replaced
        assert result_msgs[1]["content"][0]["type"] == "text"
        # Last image kept
        assert result_msgs[2]["content"][0]["type"] == "image_url"


# ===========================================================================
# Tests for DesktopEnvTool (mocked HTTP)
# ===========================================================================


class TestDesktopEnvTool:
    """Tests for DesktopEnvTool with mocked HTTP calls."""

    @pytest.fixture
    def tool(self):
        """Create a DesktopEnvTool with mocked internals."""
        from recipe.gui_agent.desktop_env_tool import DesktopEnvTool

        config = {
            "api_base_url": "http://mock-desktop:8000",
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 5,
        }
        return DesktopEnvTool(config=config)

    @pytest.mark.asyncio
    async def test_create_acquires_and_resets(self, tool):
        """create() should acquire an env, reset it, and return initial screenshot."""
        test_image = _make_image("blue", (100, 100))

        tool._post = AsyncMock(side_effect=[
            {"env_id": "env-123"},       # acquire response
            {"status": "ok"},            # reset response
        ])
        tool._get_screenshot = AsyncMock(return_value=test_image)

        instance_id, response = await tool.create(create_kwargs={"task_id": "task-001"})

        assert instance_id is not None
        assert response.image is not None
        assert len(response.image) == 1

        # Verify API calls
        tool._post.assert_any_call("/envs/acquire")
        tool._post.assert_any_call("/envs/env-123/reset", {"task_id": "task-001"})
        tool._get_screenshot.assert_called_once_with("env-123")

        # Verify internal state
        assert instance_id in tool._instances
        assert tool._instances[instance_id]["env_id"] == "env-123"

    @pytest.mark.asyncio
    async def test_create_without_task_id_raises(self, tool):
        """create() without task_id should raise ValueError."""
        with pytest.raises(ValueError, match="task_id"):
            await tool.create(create_kwargs={})

    @pytest.mark.asyncio
    async def test_execute_sends_action(self, tool):
        """execute() should send action and return screenshot."""
        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}

        test_image = _make_image("green", (100, 100))
        tool._post = AsyncMock(return_value={"status": "ok"})
        tool._get_screenshot = AsyncMock(return_value=test_image)

        response, reward, metrics = await tool.execute(
            "inst-1",
            {"action": "left_click", "coordinate": [500, 300]},
        )

        assert response.image is not None
        assert len(response.image) == 1
        assert reward == 0.0
        assert "action" in metrics

        tool._post.assert_called_once_with(
            "/envs/env-123/action",
            {"action": "left_click", "coordinate": [500, 300]},
        )

    @pytest.mark.asyncio
    async def test_execute_terminate_action(self, tool):
        """execute() with terminate action should not call env API."""
        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool._post = AsyncMock()

        response, reward, metrics = await tool.execute(
            "inst-1",
            {"action": "terminate", "status": "success"},
        )

        assert "terminated" in response.text.lower()
        tool._post.assert_not_called()

    @pytest.mark.asyncio
    async def test_calc_reward_completed(self, tool):
        """calc_reward() should return 1.0 when task is completed."""
        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool._post = AsyncMock(return_value={"completed": True})

        reward = await tool.calc_reward("inst-1")
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_calc_reward_not_completed(self, tool):
        """calc_reward() should return 0.0 when task is not completed."""
        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool._post = AsyncMock(return_value={"completed": False})

        reward = await tool.calc_reward("inst-1")
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_calc_reward_unknown_instance(self, tool):
        """calc_reward() for unknown instance returns 0.0."""
        reward = await tool.calc_reward("nonexistent")
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_release(self, tool):
        """release() should call API and remove internal state."""
        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool._post = AsyncMock(return_value={})

        await tool.release("inst-1")

        tool._post.assert_called_once_with("/envs/env-123/release")
        assert "inst-1" not in tool._instances

    @pytest.mark.asyncio
    async def test_release_unknown_instance(self, tool):
        """release() for unknown instance is a no-op."""
        tool._post = AsyncMock()
        await tool.release("nonexistent")
        tool._post.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_releases_on_reset_failure(self, tool):
        """If reset fails, the env should be released."""
        tool._post = AsyncMock(side_effect=[
            {"env_id": "env-123"},                     # acquire OK
            Exception("reset failed"),                  # reset fails
        ])

        # Create a new mock for the release call after error
        release_mock = AsyncMock(return_value={})
        original_post = tool._post

        async def side_effect_with_release(path, payload=None):
            if path == "/envs/env-123/release":
                return await release_mock(path, payload)
            return await original_post(path, payload)

        tool._post = AsyncMock(side_effect=[
            {"env_id": "env-123"},           # acquire
            Exception("reset failed"),        # reset fails
            {},                              # release
        ])

        with pytest.raises(Exception, match="reset failed"):
            await tool.create(create_kwargs={"task_id": "task-001"})

    def test_tool_schema(self, tool):
        """Tool schema should be properly constructed."""
        schema = tool.get_openai_tool_schema()
        assert schema.function.name == "computer_use"
        assert "1000x1000" in schema.function.description


# ===========================================================================
# Tests for AgentLoopGroupOutput structure
# ===========================================================================


class TestMultiTrajectoryOutput:
    """Verify the structure of multi-trajectory outputs."""

    def test_group_output_structure(self):
        """AgentLoopGroupOutput should hold multiple trajectories with shared reward."""
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopGroupOutput,
            AgentLoopMetrics,
            AgentLoopOutput,
        )

        metrics = AgentLoopMetrics()

        trajs = []
        for i in range(5):
            traj = AgentLoopOutput(
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                response_mask=[1, 1, 1],
                num_turns=(i + 1) * 2,
                metrics=metrics,
                extra_fields={
                    "trajectory_role": "intermediate" if i < 4 else "final",
                    "turn_number": i + 1,
                },
            )
            trajs.append(traj)

        group = AgentLoopGroupOutput(
            trajectories=trajs,
            shared_reward=1.0,
        )

        assert len(group.trajectories) == 5
        assert group.shared_reward == 1.0
        assert group.trajectories[-1].extra_fields["trajectory_role"] == "final"
        assert group.trajectories[0].extra_fields["trajectory_role"] == "intermediate"

    def test_group_output_none_reward(self):
        """shared_reward=None should be allowed (framework computes from final traj)."""
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopGroupOutput,
            AgentLoopMetrics,
            AgentLoopOutput,
        )

        metrics = AgentLoopMetrics()
        traj = AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            num_turns=2,
            metrics=metrics,
            reward_score=0.5,
            extra_fields={"trajectory_role": "final"},
        )

        group = AgentLoopGroupOutput(trajectories=[traj], shared_reward=None)
        assert group.shared_reward is None
        assert group.trajectories[0].reward_score == 0.5


# ===========================================================================
# Tests for context strategies
# ===========================================================================


class TestKeepLastKImagesStrategy:
    """Tests for KeepLastKImagesStrategy (pluggable wrapper)."""

    def test_basic_pruning(self):
        """Strategy should prune oldest images, keeping last k."""
        from recipe.gui_agent.context_manager import KeepLastKImagesStrategy

        strategy = KeepLastKImagesStrategy(k=2)
        messages = [
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("green"), _make_image("blue")]

        result_msgs = strategy.prepare_context(messages)

        assert result_msgs[0]["content"][0] == {"type": "text", "text": "[screenshot omitted]"}
        assert result_msgs[1]["content"][0]["type"] == "image_url"
        assert result_msgs[2]["content"][0]["type"] == "image_url"

    def test_no_pruning_under_k(self):
        """Strategy should not prune when under k images."""
        from recipe.gui_agent.context_manager import KeepLastKImagesStrategy

        strategy = KeepLastKImagesStrategy(k=5)
        messages = [
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("green")]

        result_msgs = strategy.prepare_context(messages)

    def test_invalid_k(self):
        """k=0 should raise ValueError."""
        from recipe.gui_agent.context_manager import KeepLastKImagesStrategy

        with pytest.raises(ValueError):
            KeepLastKImagesStrategy(k=0)


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy."""

    def test_basic_sliding_window(self):
        """Strategy should keep only the last N rounds."""
        from recipe.gui_agent.context_manager import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(max_conversation_rounds=2, max_image_rounds=2)

        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Do the task."},
            # Round 1
            {"role": "assistant", "content": "Clicking..."},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}, {"type": "text", "text": "clicked"}]},
            # Round 2
            {"role": "assistant", "content": "Scrolling..."},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}, {"type": "text", "text": "scrolled"}]},
            # Round 3
            {"role": "assistant", "content": "Typing..."},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}, {"type": "text", "text": "typed"}]},
        ]
        images = [_make_image("red"), _make_image("green"), _make_image("blue")]

        result_msgs = strategy.prepare_context(messages)

        # Should keep system + user prefix, plus last 2 rounds
        # Prefix: system + user = 2 msgs
        # Rounds kept: round 2 + round 3 = 4 msgs
        assert len(result_msgs) == 6

        # System and user should be preserved
        assert result_msgs[0]["role"] == "system"
        assert result_msgs[1]["role"] == "user"

    def test_image_pruning_within_window(self):
        """Images in rounds beyond max_image_rounds should be replaced."""
        from recipe.gui_agent.context_manager import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(max_conversation_rounds=3, max_image_rounds=1)

        messages = [
            {"role": "user", "content": "Do the task."},
            # Round 1
            {"role": "assistant", "content": "Action 1"},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            # Round 2
            {"role": "assistant", "content": "Action 2"},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
            # Round 3
            {"role": "assistant", "content": "Action 3"},
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}]},
        ]
        images = [_make_image("red"), _make_image("green"), _make_image("blue")]

        result_msgs = strategy.prepare_context(messages)

        # All 3 rounds kept, but only last 1 round's images preserved
        # Rounds 1 & 2 images should be replaced with placeholders
        # Round 1 tool message
        assert result_msgs[2]["content"][0] == {"type": "text", "text": "[screenshot omitted]"}
        # Round 2 tool message
        assert result_msgs[4]["content"][0] == {"type": "text", "text": "[screenshot omitted]"}
        # Round 3 tool message: image preserved
        assert result_msgs[6]["content"][0]["type"] == "image_url"

    def test_no_messages(self):
        """Empty messages should work without errors."""
        from recipe.gui_agent.context_manager import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(max_conversation_rounds=5, max_image_rounds=3)
        result_msgs = strategy.prepare_context([])
        assert result_msgs == []

    def test_invalid_params(self):
        """Invalid parameters should raise ValueError."""
        from recipe.gui_agent.context_manager import SlidingWindowStrategy

        with pytest.raises(ValueError):
            SlidingWindowStrategy(max_conversation_rounds=0)
        with pytest.raises(ValueError):
            SlidingWindowStrategy(max_image_rounds=0)


# ===========================================================================
# Tests for per-step reward in DesktopEnvTool
# ===========================================================================


class TestPerStepReward:
    """Tests for step_reward configuration in DesktopEnvTool."""

    @pytest.fixture
    def tool_with_reward(self):
        """Create a DesktopEnvTool with step_reward > 0."""
        from recipe.gui_agent.desktop_env_tool import DesktopEnvTool

        config = {
            "api_base_url": "http://mock-desktop:8000",
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 5,
            "step_reward": 0.1,
        }
        return DesktopEnvTool(config=config)

    @pytest.fixture
    def tool_default_reward(self):
        """Create a DesktopEnvTool with default step_reward (0.0)."""
        from recipe.gui_agent.desktop_env_tool import DesktopEnvTool

        config = {
            "api_base_url": "http://mock-desktop:8000",
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 5,
        }
        return DesktopEnvTool(config=config)

    @pytest.mark.asyncio
    async def test_step_reward_returned(self, tool_with_reward):
        """execute() should return the configured step_reward."""
        tool_with_reward._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool_with_reward._post = AsyncMock(return_value={"status": "ok"})
        tool_with_reward._get_screenshot = AsyncMock(return_value=_make_image("green"))

        _, reward, _ = await tool_with_reward.execute(
            "inst-1", {"action": "left_click", "coordinate": [100, 200]}
        )
        assert reward == 0.1

    @pytest.mark.asyncio
    async def test_default_step_reward(self, tool_default_reward):
        """Default step_reward should be 0.0."""
        tool_default_reward._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool_default_reward._post = AsyncMock(return_value={"status": "ok"})
        tool_default_reward._get_screenshot = AsyncMock(return_value=_make_image("green"))

        _, reward, _ = await tool_default_reward.execute(
            "inst-1", {"action": "left_click", "coordinate": [100, 200]}
        )
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_terminate_always_zero_reward(self, tool_with_reward):
        """Terminate action should always return 0.0 reward regardless of config."""
        tool_with_reward._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool_with_reward._post = AsyncMock()

        _, reward, _ = await tool_with_reward.execute(
            "inst-1", {"action": "terminate", "status": "success"}
        )
        assert reward == 0.0


# ===========================================================================
# Tests for fatal error handling
# ===========================================================================


class TestFatalErrorHandling:
    """Tests for error handling patterns in DesktopEnvTool."""

    @pytest.fixture
    def tool(self):
        from recipe.gui_agent.desktop_env_tool import DesktopEnvTool

        config = {
            "api_base_url": "http://mock-desktop:8000",
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 5,
        }
        return DesktopEnvTool(config=config)

    @pytest.mark.asyncio
    async def test_timeout_propagates(self, tool):
        """TimeoutError from _post should propagate (fatal error)."""
        import asyncio

        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool._post = AsyncMock(side_effect=asyncio.TimeoutError("MCP timeout"))

        with pytest.raises(asyncio.TimeoutError):
            await tool.execute("inst-1", {"action": "left_click", "coordinate": [100, 200]})

    @pytest.mark.asyncio
    async def test_screenshot_failure_propagates(self, tool):
        """Screenshot failure should propagate."""
        tool._instances["inst-1"] = {"env_id": "env-123", "task_id": "task-001"}
        tool._post = AsyncMock(return_value={"status": "ok"})
        tool._get_screenshot = AsyncMock(side_effect=ConnectionError("connection lost"))

        with pytest.raises(ConnectionError):
            await tool.execute("inst-1", {"action": "left_click", "coordinate": [100, 200]})


# ===========================================================================
# Tests for MCPDesktopEnvTool (mocked MCP utilities)
# ===========================================================================


class TestMCPDesktopEnvTool:
    """Tests for MCPDesktopEnvTool with fully mocked MCP dependencies."""

    def _make_mcp_tool(self):
        """Create an MCPDesktopEnvTool instance."""
        from recipe.gui_agent.mcp_desktop_env_tool import MCPDesktopEnvTool

        config = {
            "allocator_base_url": "http://mock-allocator:8080",
            "allocator_env": "test_env",
            "allocator_namespace": "Test",
            "expire_min": 30,
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 10,
            "step_reward": 0.1,
        }
        MCPDesktopEnvTool._address_client = None
        return MCPDesktopEnvTool(config=config)

    def test_tool_schema(self):
        """MCPDesktopEnvTool should have computer_use schema."""
        tool = self._make_mcp_tool()
        schema = tool.get_openai_tool_schema()
        assert schema.function.name == "computer_use"
        assert "1000x1000" in schema.function.description

    def test_step_reward_config(self):
        """MCPDesktopEnvTool should respect step_reward config."""
        tool = self._make_mcp_tool()
        assert tool.step_reward == 0.1

    @pytest.mark.asyncio
    async def test_calc_reward_unknown_instance(self):
        """calc_reward for unknown instance returns 0.0."""
        tool = self._make_mcp_tool()
        reward = await tool.calc_reward("nonexistent")
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_release_unknown_instance(self):
        """release for unknown instance is a no-op."""
        tool = self._make_mcp_tool()
        await tool.release("nonexistent")

    @pytest.mark.asyncio
    async def test_create_missing_task_id_raises(self):
        """create() without task_id should raise ValueError."""
        tool = self._make_mcp_tool()
        with pytest.raises(ValueError, match="task_id"):
            await tool.create(create_kwargs={})

    @pytest.mark.asyncio
    async def test_calc_reward_missing_ground_truth_keys(self):
        """calc_reward returns 0.0 when ground_truth lacks required keys."""
        tool = self._make_mcp_tool()
        tool._instances["inst-1"] = {
            "ground_truth": {"seed": "abc"},  # missing qseed, session
        }
        reward = await tool.calc_reward("inst-1")
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_calc_reward_calls_validation_api(self):
        """calc_reward should call check_task and return the reward."""
        tool = self._make_mcp_tool()
        tool._instances["inst-1"] = {
            "ground_truth": {
                "base_url": "http://mock",
                "seed": "s1",
                "qseed": "q1",
                "session": "sess1",
                "mock_date": "2025-01-01",
            },
        }
        with patch("recipe.gui_agent.mcp_desktop_env_tool._check_task", return_value={"score": 75, "reward": 0.75}) as mock_check, \
             patch("recipe.gui_agent.mcp_desktop_env_tool._reset_task") as mock_reset:
            reward = await tool.calc_reward("inst-1")

        assert reward == 0.75
        mock_check.assert_called_once_with("http://mock", "s1", "q1", "sess1", "2025-01-01")
        mock_reset.assert_called_once_with("http://mock", "s1", "q1", "sess1", "2025-01-01")

    @pytest.mark.asyncio
    async def test_release_always_releases_address(self):
        """release should release the address even if mcp_client.close() fails."""
        tool = self._make_mcp_tool()
        mock_client = AsyncMock()
        mock_client.close.side_effect = RuntimeError("close failed")
        tool._instances["inst-1"] = {"mcp_client": mock_client}
        tool._address_client = AsyncMock()

        await tool.release("inst-1")

        mock_client.close.assert_called_once()
        tool._address_client.release.assert_called_once_with("inst-1")

    @pytest.mark.asyncio
    async def test_create_full_lifecycle(self):
        """create() should allocate, reboot, screenshot, and register instance."""
        import base64

        from recipe.gui_agent.mcp_desktop_env_tool import MCPDesktopEnvTool

        tool = self._make_mcp_tool()

        # Build a valid base64 screenshot
        img = _make_image("green", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        # Mock address client
        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        mock_addr.release = AsyncMock()
        tool._address_client = mock_addr

        # Mock _MCPClient construction and take_screenshot
        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)

        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, response = await tool.create(
                create_kwargs={
                    "task_id": "task-42",
                    "ground_truth": {"seed": "s1", "qseed": "q1"},
                }
            )

        # Verify address allocation & reboot
        mock_addr.allocate.assert_called_once_with(instance_id)
        mock_addr.reboot.assert_called_once()

        # Verify response has screenshot
        assert response.image is not None
        assert len(response.image) == 1
        assert isinstance(response.image[0], str)
        assert response.image[0].startswith("data:image/")

        # Verify instance registered
        assert instance_id in tool._instances
        assert tool._instances[instance_id]["task_id"] == "task-42"
        assert tool._instances[instance_id]["ground_truth"]["seed"] == "s1"
        # Session should be auto-generated
        assert isinstance(tool._instances[instance_id]["ground_truth"]["session"], str)
        assert len(tool._instances[instance_id]["ground_truth"]["session"]) > 0

    @pytest.mark.asyncio
    async def test_create_releases_on_reboot_failure(self):
        """If reboot fails, the address should be released."""
        tool = self._make_mcp_tool()

        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock(side_effect=TimeoutError("reboot failed"))
        mock_addr.release = AsyncMock()
        tool._address_client = mock_addr

        with pytest.raises(TimeoutError, match="reboot failed"):
            await tool.create(create_kwargs={"task_id": "task-1"})

        mock_addr.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_click_action(self):
        """execute() should call computer_use via MCP and return screenshot."""
        import base64

        tool = self._make_mcp_tool()

        # Build a screenshot response
        img = _make_image("blue", (50, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        mock_mcp = AsyncMock()
        mock_mcp.call_tool = AsyncMock(return_value=None)
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)

        tool._instances["inst-1"] = {
            "address": "10.0.0.1:5000",
            "mcp_client": mock_mcp,
            "task_id": "task-1",
            "ground_truth": {},
        }

        response, reward, metrics = await tool.execute(
            "inst-1", {"action": "left_click", "coordinate": [500, 300]}
        )

        # Verify MCP call
        mock_mcp.call_tool.assert_called_once_with(
            "computer_use", {"action": "left_click", "coordinate": [500, 300]}
        )
        # Verify screenshot taken
        mock_mcp.take_screenshot.assert_called_once()

        # Verify response
        assert response.image is not None
        assert len(response.image) == 1
        assert "left_click" in response.text
        assert "[500, 300]" in response.text
        assert reward == 0.1  # step_reward from config
        assert metrics["action"] == "left_click"

    @pytest.mark.asyncio
    async def test_execute_terminate_action(self):
        """execute() with terminate should return text only, no MCP call."""
        tool = self._make_mcp_tool()

        mock_mcp = AsyncMock()
        tool._instances["inst-1"] = {
            "address": "10.0.0.1:5000",
            "mcp_client": mock_mcp,
            "task_id": "task-1",
            "ground_truth": {},
        }

        response, reward, metrics = await tool.execute(
            "inst-1", {"action": "terminate", "status": "success"}
        )

        assert "terminated" in response.text.lower()
        assert reward == 0.0
        mock_mcp.call_tool.assert_not_called()
        mock_mcp.take_screenshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_screenshot_unavailable(self):
        """execute() should handle screenshot failure gracefully."""
        tool = self._make_mcp_tool()

        mock_mcp = AsyncMock()
        mock_mcp.call_tool = AsyncMock(return_value=None)
        mock_mcp.take_screenshot = AsyncMock(return_value=None)  # screenshot fails

        tool._instances["inst-1"] = {
            "address": "10.0.0.1:5000",
            "mcp_client": mock_mcp,
            "task_id": "task-1",
            "ground_truth": {},
        }

        response, reward, metrics = await tool.execute(
            "inst-1", {"action": "scroll", "coordinate": [500, 300], "direction": "down", "amount": 3}
        )

        assert response.image is None
        assert "screenshot unavailable" in response.text
        assert reward == 0.1  # step_reward still returned

    @pytest.mark.asyncio
    async def test_full_lifecycle_create_execute_reward_release(self):
        """End-to-end: create → execute → calc_reward → release."""
        import base64

        tool = self._make_mcp_tool()

        # Screenshot data
        img = _make_image("red", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        # Mock address client
        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        mock_addr.release = AsyncMock()
        tool._address_client = mock_addr

        # Mock MCP client
        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)
        mock_mcp.call_tool = AsyncMock(return_value=None)
        mock_mcp.close = AsyncMock()

        ground_truth = {
            "base_url": "http://mock-app",
            "seed": "s1",
            "qseed": "q1",
        }

        # 1. CREATE
        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, resp = await tool.create(
                create_kwargs={"task_id": "task-99", "ground_truth": ground_truth}
            )
        assert resp.image is not None
        # Session should be auto-generated
        stored_gt = tool._instances[instance_id]["ground_truth"]
        assert isinstance(stored_gt["session"], str)
        assert len(stored_gt["session"]) > 0

        # 2. EXECUTE (click)
        resp, reward, metrics = await tool.execute(
            instance_id, {"action": "left_click", "coordinate": [100, 200]}
        )
        assert resp.image is not None
        assert reward == 0.1

        # 3. EXECUTE (terminate)
        resp, reward, metrics = await tool.execute(
            instance_id, {"action": "terminate", "status": "success"}
        )
        assert reward == 0.0

        # 4. CALC_REWARD
        with patch("recipe.gui_agent.mcp_desktop_env_tool._check_task", return_value={"score": 100, "reward": 1.0}), \
             patch("recipe.gui_agent.mcp_desktop_env_tool._reset_task"):
            reward = await tool.calc_reward(instance_id)
        assert reward == 1.0

        # 5. RELEASE
        await tool.release(instance_id)
        mock_mcp.close.assert_called_once()
        mock_addr.release.assert_called_once_with(instance_id)
        assert instance_id not in tool._instances

    @pytest.mark.asyncio
    async def test_execute_unknown_instance_raises(self):
        """execute() with unknown instance_id should raise ValueError."""
        tool = self._make_mcp_tool()
        with pytest.raises(ValueError, match="Unknown instance_id"):
            await tool.execute("nonexistent", {"action": "left_click"})

    @pytest.mark.asyncio
    async def test_create_generates_session(self):
        """create() should generate a session UUID and inject it into ground_truth."""
        import base64

        tool = self._make_mcp_tool()

        img = _make_image("green", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        mock_addr.release = AsyncMock()
        tool._address_client = mock_addr

        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)

        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, resp = await tool.create(
                create_kwargs={
                    "task_id": "task-1",
                    "ground_truth": {"seed": "s1", "qseed": "q1"},
                }
            )

        gt = tool._instances[instance_id]["ground_truth"]
        assert "session" in gt
        assert isinstance(gt["session"], str)
        assert len(gt["session"]) > 0
        # Original keys preserved
        assert gt["seed"] == "s1"
        assert gt["qseed"] == "q1"

    @pytest.mark.asyncio
    async def test_create_navigates_with_url_rewrite(self):
        """create() should call browser_navigate with correct URL when url_rewrite is configured."""
        import base64

        tool = self._make_mcp_tool()

        img = _make_image("green", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        tool._address_client = mock_addr

        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)
        mock_mcp.call_tool = AsyncMock(return_value=None)

        url_rewrite = {
            "target": "http://localhost:3000/app",
            "args": [
                {"key": "seed", "param": "seed"},
                {"key": "session", "param": "_session_"},
                {"key": "mock_date", "param": "_mockdate_"},
            ],
        }

        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, resp = await tool.create(
                create_kwargs={
                    "task_id": "task-1",
                    "ground_truth": {"seed": "abc", "mock_date": "2025-01-01"},
                    "url_rewrite": url_rewrite,
                }
            )

        # browser_navigate should have been called
        mock_mcp.call_tool.assert_called_once()
        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "browser_navigate"
        nav_url = call_args[0][1]["url"]
        assert "http://localhost:3000/app/?" in nav_url
        assert "seed=abc" in nav_url
        assert "_session_=" in nav_url
        assert "_mockdate_=2025-01-01" in nav_url

    @pytest.mark.asyncio
    async def test_create_skips_navigation_without_url_rewrite(self):
        """create() should not call browser_navigate when url_rewrite is absent."""
        import base64

        tool = self._make_mcp_tool()

        img = _make_image("green", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        tool._address_client = mock_addr

        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)
        mock_mcp.call_tool = AsyncMock(return_value=None)

        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, resp = await tool.create(
                create_kwargs={
                    "task_id": "task-1",
                    "ground_truth": {"seed": "s1"},
                }
            )

        # browser_navigate should NOT have been called
        mock_mcp.call_tool.assert_not_called()


class TestBuildInitialUrl:
    """Tests for the _build_initial_url helper function."""

    def test_basic_url_construction(self):
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        ground_truth = {"seed": "abc", "session": "sess-1", "mock_date": "2025-01-01"}
        url_rewrite = {
            "target": "http://localhost:3000/app",
            "args": [
                {"key": "seed", "param": "seed"},
                {"key": "session", "param": "_session_"},
                {"key": "mock_date", "param": "_mockdate_"},
            ],
        }
        result = _build_initial_url(ground_truth, url_rewrite)
        assert result == "http://localhost:3000/app/?seed=abc&_session_=sess-1&_mockdate_=2025-01-01"

    def test_returns_none_without_url_rewrite(self):
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        assert _build_initial_url({"seed": "abc"}, None) is None

    def test_returns_none_without_target(self):
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        assert _build_initial_url({"seed": "abc"}, {"args": []}) is None
        assert _build_initial_url({"seed": "abc"}, {"target": "", "args": []}) is None

    def test_skips_missing_ground_truth_keys(self):
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        ground_truth = {"seed": "abc"}  # no session or mock_date
        url_rewrite = {
            "target": "http://localhost:3000/app",
            "args": [
                {"key": "seed", "param": "seed"},
                {"key": "session", "param": "_session_"},
            ],
        }
        result = _build_initial_url(ground_truth, url_rewrite)
        assert result == "http://localhost:3000/app/?seed=abc"

    def test_no_args_returns_bare_url(self):
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        result = _build_initial_url({}, {"target": "http://localhost:3000/app", "args": []})
        assert result == "http://localhost:3000/app/"

    def test_trailing_slash_stripped_from_target(self):
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        result = _build_initial_url(
            {"seed": "x"}, {"target": "http://localhost:3000/app/", "args": [{"key": "seed", "param": "seed"}]}
        )
        assert result == "http://localhost:3000/app/?seed=x"

    def test_real_meeting_data(self):
        """Test with real production meeting booking data format."""
        from recipe.gui_agent.mcp_desktop_env_tool import _build_initial_url

        # Simulates ground_truth after session injection in create()
        ground_truth = {
            "base_url": "http://ns008-cu-meeting-svc",
            "beg": "19:00",
            "building_id": 436,
            "building_name": "光启未来中心大厦A塔",
            "city_name": "深圳",
            "date": "大后天",
            "end": "20:00",
            "floor_id": 839,
            "floor_name": "20",
            "mock_date": "2026-03-19",
            "people": "andyliu;",
            "qseed": "974eb91672bb6db1",
            "seed": "f41863987dbec585",
            "title": "半年度客户沟通沟通",
            "valid": True,
            "session": "test-session-uuid",  # injected by create()
        }
        url_rewrite = {
            "args": [
                {"key": "seed", "param": "seed"},
                {"key": "mock_date", "param": "_mockdate_"},
                {"key": "session", "param": "_session_"},
            ],
            "source": "https://meeting.woa.com",
            "target": "http://ns008-cu-meeting-svc",
        }
        result = _build_initial_url(ground_truth, url_rewrite)
        assert result == (
            "http://ns008-cu-meeting-svc/"
            "?seed=f41863987dbec585&_mockdate_=2026-03-19&_session_=test-session-uuid"
        )


class TestMCPDesktopEnvToolRealData:
    """End-to-end tests using real production data format (meeting booking)."""

    def _make_mcp_tool(self):
        from recipe.gui_agent.mcp_desktop_env_tool import MCPDesktopEnvTool

        config = {
            "allocator_base_url": "http://ns008-cu-manager-svc",
            "allocator_env": "a4861800",
            "allocator_namespace": "Development",
            "expire_min": 60,
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 10,
        }
        MCPDesktopEnvTool._address_client = None
        return MCPDesktopEnvTool(config=config)

    @pytest.mark.asyncio
    async def test_meeting_create_injects_session_and_navigates(self):
        """Full create() flow with real meeting data: session injected, browser navigated."""
        import base64

        tool = self._make_mcp_tool()

        img = _make_image("green", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        mock_addr.release = AsyncMock()
        tool._address_client = mock_addr

        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)
        mock_mcp.call_tool = AsyncMock(return_value=None)

        create_kwargs = {
            "task_id": "meeting_f41863987dbec585_974eb91672bb6db1",
            "ground_truth": {
                "base_url": "http://ns008-cu-meeting-svc",
                "beg": "19:00",
                "building_id": 436,
                "building_name": "光启未来中心大厦A塔",
                "city_name": "深圳",
                "date": "大后天",
                "end": "20:00",
                "floor_id": 839,
                "floor_name": "20",
                "mock_date": "2026-03-19",
                "people": "andyliu;",
                "qseed": "974eb91672bb6db1",
                "seed": "f41863987dbec585",
                "title": "半年度客户沟通沟通",
                "valid": True,
            },
            "url_rewrite": {
                "args": [
                    {"key": "seed", "param": "seed"},
                    {"key": "mock_date", "param": "_mockdate_"},
                    {"key": "session", "param": "_session_"},
                ],
                "source": "https://meeting.woa.com",
                "target": "http://ns008-cu-meeting-svc",
            },
        }

        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, resp = await tool.create(create_kwargs=create_kwargs)

        # 1. Session should be auto-generated (UUID format)
        gt = tool._instances[instance_id]["ground_truth"]
        assert "session" in gt
        session = gt["session"]
        assert isinstance(session, str)
        assert len(session) == 36  # UUID format: 8-4-4-4-12

        # 2. Original ground_truth should NOT be mutated
        assert "session" not in create_kwargs["ground_truth"]

        # 3. browser_navigate should have been called with correct URL
        mock_mcp.call_tool.assert_called_once()
        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "browser_navigate"
        nav_url = call_args[0][1]["url"]
        assert nav_url == (
            f"http://ns008-cu-meeting-svc/?seed=f41863987dbec585"
            f"&_mockdate_=2026-03-19&_session_={session}"
        )

        # 4. Screenshot should be returned
        assert resp.image is not None
        assert len(resp.image) == 1

    @pytest.mark.asyncio
    async def test_meeting_full_lifecycle_with_reward(self):
        """Full lifecycle: create → calc_reward with real meeting data.

        Verifies that the auto-generated session flows through to the
        validation API call in calc_reward().
        """
        import base64

        tool = self._make_mcp_tool()

        img = _make_image("red", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_screenshot = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        mock_addr = AsyncMock()
        mock_addr.allocate = AsyncMock(return_value="10.0.0.1:5000")
        mock_addr.reboot = AsyncMock()
        mock_addr.release = AsyncMock()
        tool._address_client = mock_addr

        mock_mcp = AsyncMock()
        mock_mcp.take_screenshot = AsyncMock(return_value=b64_screenshot)
        mock_mcp.call_tool = AsyncMock(return_value=None)
        mock_mcp.close = AsyncMock()

        create_kwargs = {
            "task_id": "meeting_f41863987dbec585_974eb91672bb6db1",
            "ground_truth": {
                "base_url": "http://ns008-cu-meeting-svc",
                "mock_date": "2026-03-19",
                "qseed": "974eb91672bb6db1",
                "seed": "f41863987dbec585",
            },
            "url_rewrite": {
                "args": [
                    {"key": "seed", "param": "seed"},
                    {"key": "mock_date", "param": "_mockdate_"},
                    {"key": "session", "param": "_session_"},
                ],
                "target": "http://ns008-cu-meeting-svc",
            },
        }

        # CREATE
        with patch("recipe.gui_agent.mcp_desktop_env_tool._MCPClient", return_value=mock_mcp):
            instance_id, resp = await tool.create(create_kwargs=create_kwargs)

        session = tool._instances[instance_id]["ground_truth"]["session"]

        # CALC_REWARD — verify session is passed to the validation API
        with patch("recipe.gui_agent.mcp_desktop_env_tool._check_task", return_value={"score": 100, "reward": 1.0}) as mock_check, \
             patch("recipe.gui_agent.mcp_desktop_env_tool._reset_task") as mock_reset:
            reward = await tool.calc_reward(instance_id)

        assert reward == 1.0
        # Session should be the auto-generated one, not empty
        mock_check.assert_called_once_with(
            "http://ns008-cu-meeting-svc", "f41863987dbec585", "974eb91672bb6db1", session, "2026-03-19"
        )
        mock_reset.assert_called_once_with(
            "http://ns008-cu-meeting-svc", "f41863987dbec585", "974eb91672bb6db1", session, "2026-03-19"
        )

        # RELEASE
        await tool.release(instance_id)
        assert instance_id not in tool._instances


# ===========================================================================
# Tests for _base64_to_pil helper
# ===========================================================================


class TestBase64ToPil:
    """Tests for the _base64_to_pil helper function."""

    def test_raw_base64(self):
        """Should decode raw base64 string to PIL Image."""
        import base64

        from recipe.gui_agent.mcp_desktop_env_tool import _base64_to_pil

        img = _make_image("red", (5, 5))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = _base64_to_pil(b64)
        assert isinstance(result, Image.Image)
        assert result.size == (5, 5)

    def test_data_uri(self):
        """Should decode data URI formatted base64 to PIL Image."""
        import base64

        from recipe.gui_agent.mcp_desktop_env_tool import _base64_to_pil

        img = _make_image("blue", (8, 8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        result = _base64_to_pil(data_uri)
        assert isinstance(result, Image.Image)
        assert result.size == (8, 8)
