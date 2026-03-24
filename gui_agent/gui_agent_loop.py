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
"""GUI Agent Loop for Computer-Use Agent (CUA) training.

This agent loop drives a VLM to interact with a remote desktop environment
via ``computer_use`` tool calls.  Each *turn* (LLM generation + action
execution) becomes a separate trajectory because the prompt context changes
every turn (older screenshots are replaced with text placeholders to stay
within the vision context window).

The final binary task-completion reward is shared across all turn-level
trajectories via :class:`AgentLoopGroupOutput`.

Architecture
------------

::

    GUIAgentLoop.run()
    ├─ tool.create()          → acquire env, reset, initial screenshot
    ├─ for turn in 1..max_turns:
    │   ├─ keep_last_k_images()  → prune old screenshots from messages
    │   ├─ apply_chat_template() → prompt_ids for THIS turn
    │   ├─ server_manager.generate() → response_ids, logprobs
    │   ├─ save AgentLoopOutput  (this turn's prompt + response)
    │   ├─ parse tool call from response
    │   ├─ tool.execute()        → new screenshot
    │   └─ append messages to history
    ├─ tool.calc_reward()     → binary task completion
    └─ return AgentLoopGroupOutput(trajectories=..., shared_reward=...)

Integration with Fully-Async Training
--------------------------------------

This loop works out-of-the-box with ``FullyAsyncAgentLoopWorker`` because
it uses ``self.server_manager.generate()`` which is overridden by
``FullyAsyncLLMServerManager`` to handle partial rollout resume.  The
``extra_fields["min_global_steps"]`` / ``max_global_steps`` from async mode
track staleness across multi-turn trajectories automatically.
"""

import json
import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopGroupOutput,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from recipe.gui_agent.context_manager import (
    BaseContextStrategy,
    KeepLastKImagesStrategy,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class GUIAgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_ACTION = "processing_action"
    TERMINATED = "terminated"


# ---------------------------------------------------------------------------
# GUI Agent Loop
# ---------------------------------------------------------------------------


@register("gui_agent")
class GUIAgentLoop(AgentLoopBase):
    """Multi-turn GUI agent loop for desktop computer-use tasks.

    Each LLM turn produces a separate trajectory (because the prompt changes
    due to image pruning).  All trajectories share the final binary reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Config: multi_turn settings
        self.max_turns = self.rollout_config.multi_turn.max_assistant_turns or 20
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns or 20
        self.keep_last_k = 3  # default, can be overridden per-task from data
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_execute_retries = 2  # retry transient env errors before telling the model

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize tool(s) from config
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        # We expect exactly one tool: computer_use
        self.desktop_tool = self.tools.get("computer_use")
        if self.desktop_tool is None and tool_list:
            # Fallback: use the first tool
            self.desktop_tool = tool_list[0]

        # Tool parser
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopGroupOutput:
        """Run the GUI agent loop.

        Args:
            sampling_params: LLM sampling parameters.
            **kwargs: Dataset fields. Must include ``extra_info`` with ``task_id``
                and ``raw_prompt`` (initial messages).

        Returns:
            AgentLoopGroupOutput with one trajectory per turn and shared reward.
        """
        # raw_prompt contains only the system prompt (if any); the task query
        # comes from extra_info["question"] and is included in every user turn.
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {})
        task_id = extra_info.get("task_id", "unknown")
        task_query = extra_info.get("question", "")
        tools_kwargs = kwargs.get("tools_kwargs", {})

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}
        trajectories: list[AgentLoopOutput] = []

        logger.info("[GUI-%s] Starting agent loop", task_id)

        # --- Build context strategy from per-task data ---
        desktop_kwargs = tools_kwargs.get("computer_use", {})
        create_kwargs = desktop_kwargs.get("create_kwargs", {})
        keep_last_k = create_kwargs.get("keep_last_k_images", self.keep_last_k)
        context_strategy: BaseContextStrategy = KeepLastKImagesStrategy(k=keep_last_k)

        # --- Create env and get initial screenshot ---
        create_kwargs.setdefault("task_id", task_id)

        instance_id, initial_response = await self.desktop_tool.create(
            create_kwargs=create_kwargs,
        )

        try:
            # Add initial screenshot to messages and image_data
            image_data: list[Image.Image] = []
            if initial_response.image:
                for img in initial_response.image:
                    if img is not None:
                        image_data.append(img)
                # Append initial observation as a user message with screenshot
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"{task_query}\nPlease continue"},
                        ],
                    }
                )

            # --- Multi-turn loop ---
            turn = 0
            terminated = False
            fatal_error = False

            while turn < self.max_turns and not terminated and not fatal_error:
                turn += 1

                # 1. Prune old images using the pluggable context strategy
                messages, image_data = context_strategy.prepare_context(messages, image_data)

                # 2. Tokenize prompt for THIS turn
                prompt_ids = await self.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    images=image_data if image_data else None,
                )

                # Truncate prompt if too long
                if len(prompt_ids) > self.prompt_length:
                    prompt_ids = prompt_ids[-self.prompt_length:]

                # 3. Generate LLM response
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=image_data if image_data else None,
                    )

                # Track preemption metrics (for async training)
                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
                else:
                    metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

                response_ids = output.token_ids[: self.response_length]
                response_mask = [1] * len(response_ids)
                response_logprobs = output.log_probs[: len(response_ids)] if output.log_probs else None

                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                logger.info("[GUI-%s] Turn %d/%d: %d prompt tokens, %d response tokens\n%s", task_id, turn, self.max_turns, len(prompt_ids), len(response_ids), response_text)

                # Build extra_fields from async training metadata
                extra_fields: dict[str, Any] = {}
                if output.extra_fields:
                    extra_fields.update(output.extra_fields)
                extra_fields["trajectory_role"] = "intermediate"
                extra_fields["turn_scores"] = []
                extra_fields["tool_rewards"] = []
                extra_fields["turn_number"] = turn

                # Build multi_modal_data for this trajectory
                multi_modal_data: dict[str, Any] = {}
                if image_data:
                    multi_modal_data["images"] = list(image_data)

                # 4. Save this turn as a trajectory
                agent_metrics = AgentLoopMetrics(**{k: v for k, v in metrics.items() if k in AgentLoopMetrics.model_fields})
                traj = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    routed_experts=output.routed_experts,
                    multi_modal_data=multi_modal_data,
                    num_turns=turn * 2,  # each turn has user+assistant
                    metrics=agent_metrics,
                    extra_fields=extra_fields,
                )
                trajectories.append(traj)

                # 5. Parse tool call from response
                tools = [tool.tool_schema for tool in self.tools.values()]
                _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids, tools)

                if not tool_calls:
                    # No tool call → model chose to stop
                    terminated = True
                    continue

                # 6. Execute the first tool call
                tool_call = tool_calls[0]
                try:
                    tool_args = json.loads(tool_call.arguments)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse tool arguments: {tool_call.arguments}")
                    terminated = True
                    continue

                action = tool_args.get("action", "")

                # Check for terminate action
                if action == "terminate":
                    terminated = True
                    # Decode assistant response and add to messages
                    assistant_text = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    )
                    messages.append({"role": "assistant", "content": assistant_text})
                    continue

                # Execute action on desktop with retry for transient env errors
                tool_response = None
                tool_error = None
                for attempt in range(1, self.tool_execute_retries + 1):
                    try:
                        tool_response, tool_reward, tool_metrics = await self.desktop_tool.execute(
                            instance_id, tool_args
                        )
                        extra_fields["tool_rewards"].append(tool_reward)
                        tool_error = None
                        break
                    except TimeoutError:
                        # Fatal: env is broken (MCP timeout, env crash, etc.)
                        logger.error(f"Fatal: timeout executing tool for {task_id}, aborting rollout")
                        fatal_error = True
                        break
                    except Exception as e:
                        tool_error = e
                        if attempt < self.tool_execute_retries:
                            logger.warning(f"Tool execution failed (attempt {attempt}), retrying: {e}")
                        else:
                            logger.warning(f"Tool execution failed after {self.tool_execute_retries} attempts: {e}")
                            extra_fields["tool_rewards"].append(0.0)

                if fatal_error:
                    break

                # 7. Update message history with assistant response + tool result
                assistant_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
                )
                messages.append({"role": "assistant", "content": assistant_text})

                if tool_error is not None:
                    # All retries exhausted — tell the model what went wrong
                    error_text = f"Action failed: {tool_error}\n{task_query}\nPlease continue"
                    if image_data:
                        # Re-use the last known screenshot so the model has visual context
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": error_text},
                                ],
                            }
                        )
                        image_data.append(image_data[-1])
                    else:
                        messages.append({"role": "user", "content": error_text})
                elif tool_response.image:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": f"{task_query}\nPlease continue"},
                            ],
                        }
                    )

                    # Add new screenshot to image_data
                    for img in tool_response.image:
                        if img is not None:
                            image_data.append(img)
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{task_query}\nPlease continue",
                        }
                    )

            # --- Compute final reward ---
            if fatal_error:
                reward = 0.0
            else:
                reward = await self.desktop_tool.calc_reward(instance_id)
                logger.info("[GUI-%s] Final reward = %.4f", task_id, reward)

            # Mark the last trajectory as "final"
            # Set reward_score on ALL trajectories so the async reward loop
            # (RewardLoopWorker) is not invoked — it doesn't know data_source='meeting'.
            if trajectories:
                trajectories[-1].extra_fields["trajectory_role"] = "final"
                for traj in trajectories:
                    traj.reward_score = reward

            logger.info(
                "[GUI-%s] Done: %d turns, reward=%.4f", task_id, len(trajectories), reward
            )

            return AgentLoopGroupOutput(
                trajectories=trajectories,
                shared_reward=reward,
            )

        finally:
            # ALWAYS release the env back to the pool
            await self.desktop_tool.release(instance_id)
