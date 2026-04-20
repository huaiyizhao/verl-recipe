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
"""GUI Agent Loop for Computer-Use Agent (CUA) training under fully-async PPO.

Multi-turn VLM + remote-desktop interaction. Each turn yields an independent
trajectory because screenshot pruning makes each turn's prompt unique. The
terminal binary reward is shared across all trajectories.

The loop packs intermediate trajectories using
:class:`MultiTrajectoryAgentLoop` so that:

* ``AgentLoopWorker``, ``MessageQueue`` and ``FullyAsyncRollouter`` are all
  unmodified (each rollout still returns a single ``AgentLoopOutput``).
* The Trainer-side ``expand_intermediate_trajectories`` utility expands the
  packed trajectories into independent DataProto rows during batch assembly,
  keeping the GRPO grouping invariant (n × turns rows per prompt).
* Fully-async ``FullyAsyncLLMServerManager`` supplies partial-rollout resume
  transparently via ``self.server_manager.generate()``.
"""

import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.multi_trajectory_agent_loop import (
    MultiTrajectoryAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from recipe.fully_async_gui_agent.context_manager import (
    BaseContextStrategy,
    KeepLastKImagesStrategy,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register("gui_agent")
class GUIAgentLoop(MultiTrajectoryAgentLoop):
    """Multi-turn GUI agent loop for desktop computer-use tasks.

    Each LLM turn becomes an independent trajectory (because the prompt
    changes due to screenshot pruning). All trajectories share the final
    reward assigned at the end of the rollout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Multi-turn config.
        self.max_turns = self.rollout_config.multi_turn.max_assistant_turns or 20
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns or 20
        self.keep_last_k = 3  # default; can be overridden per-task via create_kwargs

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Tool registration (expects computer_use / DesktopEnvTool).
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
        ]

        self.desktop_tool = self.tools.get("computer_use")
        if self.desktop_tool is None and tool_list:
            # Fallback to the first registered tool.
            self.desktop_tool = tool_list[0]

        # Tool-call parser for extracting tool calls from LLM responses.
        self.tool_parser = ToolParser.get_tool_parser(
            self.rollout_config.multi_turn.format, self.tokenizer
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metrics(self, metrics: dict[str, Any]) -> AgentLoopMetrics:
        """Build AgentLoopMetrics, keeping only known fields."""
        return AgentLoopMetrics(
            **{k: v for k, v in metrics.items() if k in AgentLoopMetrics.model_fields}
        )

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | None:
        """Run the GUI agent multi-turn rollout.

        Args:
            sampling_params: LLM sampling params.
            **kwargs: dataset fields; must include ``raw_prompt`` and
                ``extra_info`` with ``task_id`` / ``question``.

        Returns:
            A single ``AgentLoopOutput`` for the final turn, with intermediate
            turns packed into ``extra_fields["intermediate_trajectories"]``.
            Returns ``None`` if the rollout must be discarded (env creation
            failed or a fatal error occurred mid-rollout).
        """
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        task_id = extra_info.get("task_id", "unknown")
        task_query = extra_info.get("question", "")
        tools_kwargs = kwargs.get("tools_kwargs", {}) or {}

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        logger.info(
            "[GUI-%s] Starting agent loop (request_id=%s, max_turns=%d, initial_messages=%d, query=%r)",
            task_id,
            request_id,
            self.max_turns,
            len(messages),
            (task_query[:80] + "...") if len(task_query) > 80 else task_query,
        )

        # Context management strategy (per-task, overridable via create_kwargs).
        desktop_kwargs = tools_kwargs.get("computer_use", {})
        create_kwargs = dict(desktop_kwargs.get("create_kwargs", {}))
        keep_last_k = create_kwargs.get("keep_last_k_images", self.keep_last_k)
        context_strategy: BaseContextStrategy = KeepLastKImagesStrategy(k=keep_last_k)

        create_kwargs.setdefault("task_id", task_id)
        logger.debug(
            "[GUI-%s] Context strategy=KeepLastKImagesStrategy(k=%d), create_kwargs=%s",
            task_id,
            keep_last_k,
            create_kwargs,
        )

        # Acquire a desktop session.
        try:
            instance_id, initial_response = await self.desktop_tool.create(
                create_kwargs=create_kwargs,
            )
        except Exception as exc:
            logger.error(
                "[GUIAgentLoop] Failed to create env for %s, discarding rollout: %s",
                task_id,
                exc,
            )
            return None
        logger.info(
            "[GUI-%s] Env session created: instance_id=%s, initial_images=%d",
            task_id,
            instance_id,
            len(initial_response.image or []),
        )

        try:
            # Seed messages with the initial screenshot.
            if initial_response.image:
                img_content: list[dict[str, Any]] = []
                for img in initial_response.image:
                    if img is not None:
                        img_content.append({"type": "image", "image": img})
                if img_content:
                    img_content.append(
                        {"type": "text", "text": f"{task_query}\nPlease continue"}
                    )
                    messages.append({"role": "user", "content": img_content})

            turn = 0
            terminated = False
            fatal_error = False

            # Last turn's snapshot, retained so we can either register it as
            # the *final* AgentLoopOutput (loop ends here) or promote it to an
            # intermediate trajectory (another turn follows).
            last_turn_ctx: dict[str, Any] | None = None

            while turn < self.max_turns and not terminated and not fatal_error:
                turn += 1
                logger.info(
                    "[GUI-%s][turn=%d] --- Begin turn (messages=%d) ---",
                    task_id,
                    turn,
                    len(messages),
                )

                # 1. Prune old images.
                messages = context_strategy.prepare_context(messages)

                # 2. Tokenize prompt for this turn.
                multi_modal_data = await self.process_vision_info(messages)
                image_data = multi_modal_data.get("images")
                prompt_ids = await self.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    images=image_data if image_data else None,
                )
                original_prompt_len = len(prompt_ids)
                truncated = False
                if len(prompt_ids) > self.prompt_length:
                    prompt_ids = prompt_ids[-self.prompt_length :]
                    truncated = True
                logger.debug(
                    "[GUI-%s][turn=%d] prompt_ids=%d (orig=%d, truncated=%s), images=%d",
                    task_id,
                    turn,
                    len(prompt_ids),
                    original_prompt_len,
                    truncated,
                    len(image_data) if image_data else 0,
                )

                # 3. LLM generation (partial-rollout-resume enabled upstream).
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=image_data if image_data else None,
                    )
                logger.debug(
                    "[GUI-%s][turn=%d] LLM generated response_ids=%d, num_preempted=%s",
                    task_id,
                    turn,
                    len(output.token_ids),
                    output.num_preempted,
                )

                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = (
                        output.num_preempted if output.num_preempted is not None else -1
                    )
                else:
                    metrics["num_preempted"] += (
                        output.num_preempted if output.num_preempted is not None else 0
                    )

                response_ids = output.token_ids[: self.response_length]
                response_mask = [1] * len(response_ids)
                response_logprobs = (
                    output.log_probs[: len(response_ids)] if output.log_probs else None
                )

                # Per-turn extra_fields (including async-training metadata).
                extra_fields: dict[str, Any] = {}
                if output.extra_fields:
                    extra_fields.update(output.extra_fields)
                extra_fields["trajectory_role"] = "intermediate"
                extra_fields["turn_number"] = turn

                last_turn_ctx = {
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response_mask": response_mask,
                    "response_logprobs": response_logprobs,
                    "multi_modal_data": multi_modal_data,
                    "routed_experts": output.routed_experts,
                    "num_turns": turn * 2,  # user+assistant per turn
                    "extra_fields": extra_fields,
                }

                # 4. Parse tool call.
                tool_schemas_all = [tool.tool_schema for tool in self.tools.values()]
                _, tool_calls = await self.tool_parser.extract_tool_calls(
                    response_ids, tool_schemas_all
                )

                if not tool_calls:
                    # No tool call → model stopped; this turn is the final one.
                    logger.info(
                        "[GUI-%s][turn=%d] No tool call parsed -> treat as final turn",
                        task_id,
                        turn,
                    )
                    terminated = True
                    continue

                # 5. Execute the first tool call.
                tool_call = tool_calls[0]
                try:
                    tool_args = json.loads(tool_call.arguments)
                except Exception as parse_exc:
                    logger.warning(
                        "[GUI-%s][turn=%d] Failed to parse tool arguments: %s (error: %s)",
                        task_id,
                        turn,
                        tool_call,
                        parse_exc,
                    )
                    terminated = True
                    continue

                action = tool_args.get("action", "")
                logger.info(
                    "[GUI-%s][turn=%d] Tool call: action=%s, args=%s",
                    task_id,
                    turn,
                    action,
                    {k: v for k, v in tool_args.items() if k != "action"},
                )

                if action == "terminate":
                    logger.info(
                        "[GUI-%s][turn=%d] Model issued terminate (status=%s)",
                        task_id,
                        turn,
                        tool_args.get("status"),
                    )
                    terminated = True
                    assistant_text = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True),
                    )
                    messages.append({"role": "assistant", "content": assistant_text})
                    continue

                try:
                    tool_response, _, _ = await self.desktop_tool.execute(
                        instance_id, tool_args
                    )
                except Exception as exec_exc:
                    logger.error(
                        "[GUIAgentLoop] Tool execution failed for %s, aborting rollout: %s",
                        task_id,
                        exec_exc,
                    )
                    fatal_error = True
                    break
                logger.debug(
                    "[GUI-%s][turn=%d] Tool executed OK: got_image=%s",
                    task_id,
                    turn,
                    bool(tool_response and tool_response.image),
                )

                # 6. The current turn is not the final turn; register it as
                #    an intermediate trajectory.
                self.append_intermediate_trajectory(
                    prompt_ids=last_turn_ctx["prompt_ids"],
                    response_ids=last_turn_ctx["response_ids"],
                    response_mask=last_turn_ctx["response_mask"],
                    response_logprobs=last_turn_ctx["response_logprobs"],
                    multi_modal_data=last_turn_ctx["multi_modal_data"],
                    num_turns=last_turn_ctx["num_turns"],
                    **last_turn_ctx["extra_fields"],
                )
                last_turn_ctx = None  # consumed

                # 7. Update message history.
                assistant_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True),
                )
                messages.append({"role": "assistant", "content": assistant_text})

                if tool_response and tool_response.image:
                    img_content = []
                    for img in tool_response.image:
                        if img is not None:
                            img_content.append({"type": "image", "image": img})
                    img_content.append(
                        {"type": "text", "text": f"{task_query}\nPlease continue"}
                    )
                    messages.append({"role": "user", "content": img_content})
                else:
                    messages.append(
                        {"role": "user", "content": f"{task_query}\nPlease continue"}
                    )

            if fatal_error:
                logger.warning("[GUIAgentLoop] Fatal error for %s at turn=%d, discarding rollout", task_id, turn)
                return None

            if last_turn_ctx is None:
                logger.warning("[GUIAgentLoop] No turns produced for %s, discarding rollout", task_id)
                return None

            if turn >= self.max_turns and not terminated:
                logger.info(
                    "[GUI-%s] Reached max_turns=%d without model terminate",
                    task_id,
                    self.max_turns,
                )

            # Compute terminal reward.
            shared_reward = await self.desktop_tool.calc_reward(instance_id)
            logger.info("[GUI-%s] Final reward = %.4f (turns=%d)", task_id, shared_reward, turn)

            # Build the final AgentLoopOutput from the last turn's snapshot.
            final_extra_fields = dict(last_turn_ctx["extra_fields"])
            final_extra_fields["trajectory_role"] = "final"

            final_output = AgentLoopOutput(
                prompt_ids=last_turn_ctx["prompt_ids"],
                response_ids=last_turn_ctx["response_ids"],
                response_mask=last_turn_ctx["response_mask"],
                response_logprobs=last_turn_ctx["response_logprobs"],
                routed_experts=last_turn_ctx["routed_experts"],
                multi_modal_data=last_turn_ctx["multi_modal_data"],
                num_turns=last_turn_ctx["num_turns"],
                metrics=self._build_metrics(metrics),
                extra_fields=final_extra_fields,
            )

            final_output = self.build_final_output(final_output, shared_reward)

            num_intermediate = len(
                final_output.extra_fields.get("intermediate_trajectories", [])
            )
            logger.info(
                "[GUI-%s] Done: %d trajectories (1 final + %d intermediate), reward=%.4f",
                task_id,
                num_intermediate + 1,
                num_intermediate,
                shared_reward,
            )
            return final_output

        finally:
            try:
                await self.desktop_tool.release(instance_id)
            except Exception:
                logger.warning(
                    "[GUIAgentLoop] Failed to release env for %s", task_id, exc_info=True
                )
