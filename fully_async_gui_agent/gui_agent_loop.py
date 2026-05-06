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

import asyncio
import json
import os
import sys
import time
import traceback
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
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from recipe.fully_async_gui_agent.context_manager import (
    BaseContextStrategy,
    KeepLastKImagesStrategy,
)
from recipe.fully_async_gui_agent.data_flow_logger import log_dataproto, log_message

# We bypass the ``logging`` framework entirely here because verl's global
# ``basicConfig(WARNING)`` plus Ray's early-attached handlers silently drop
# INFO/DEBUG records regardless of per-logger levels. ``print`` goes straight
# to stdout (picked up by Ray's log forwarder) and is unaffected by any of
# that. ``VERL_LOGGING_LEVEL=DEBUG`` still gates debug-only messages.
_DEBUG_ENABLED = os.getenv("VERL_LOGGING_LEVEL", "INFO").upper() == "DEBUG"


def _ts() -> str:
    """Short millisecond timestamp, e.g. ``2026-05-02 16:49:00.136``.

    Ray's log forwarder does not prepend timestamps to stderr lines, so we
    prepend our own to make it easy to correlate rollout activity with
    external events (e.g. desktop-env crashes, SIGABRT, trainer steps).
    """
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', lt)}.{ms:03d}"


def _log(msg: str, *, debug: bool = False) -> None:
    """Print-based logger. ``flush=True`` + ``stderr`` survives worker crashes."""
    if debug and not _DEBUG_ENABLED:
        return
    print(f"[{_ts()}] {msg}", file=sys.stderr, flush=True)


def _log_error(msg: str) -> None:
    """Always-on error logger.

    Bypasses ``_DEBUG_ENABLED`` so anything we deem "diagnostically must
    see" still surfaces in the default (non-DEBUG) configuration. Use for
    rollout failures, env-level errors, and anything we'd grep for during
    incident response. Lines start with ``[POTENTIAL ERROR]`` so they sort
    next to the structured error markers we emit elsewhere.
    """
    print(f"[{_ts()}] [POTENTIAL ERROR] {msg}", file=sys.stderr, flush=True)


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
        trajectory_info = kwargs.get("trajectory_info", {}) or {}
        sample_index = trajectory_info.get("sample_index", "?")
        rollout_n = trajectory_info.get("rollout_n", "?")
        global_step = trajectory_info.get("step", "?")
        validate = trajectory_info.get("validate", False)
        base_log_tag = (
            f"[GUI-{task_id}][rid={request_id[:8]}][sample={sample_index}]"
            f"[rollout={rollout_n}][step={global_step}][validate={validate}]"
        )
        log_tag = base_log_tag

        _log(
            f"[GUIAgentLoop][RUN_START] task_id={task_id} request_id={request_id} "
            f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
            f"validate={validate} max_turns={self.max_turns}"
        )
        _log(
            f"{log_tag} Starting agent loop (request_id={request_id}, "
            f"max_turns={self.max_turns}, initial_messages={len(messages)}, "
            f"query={(task_query[:80] + '...') if len(task_query) > 80 else task_query!r})"
        )

        # Context management strategy (per-task, overridable via create_kwargs).
        desktop_kwargs = tools_kwargs.get("computer_use", {})
        create_kwargs = dict(desktop_kwargs.get("create_kwargs", {}))
        keep_last_k = create_kwargs.get("keep_last_k_images", self.keep_last_k)
        context_strategy: BaseContextStrategy = KeepLastKImagesStrategy(k=keep_last_k)

        create_kwargs.setdefault("task_id", task_id)
        _log(
            f"{log_tag} Context strategy=KeepLastKImagesStrategy(k={keep_last_k}), "
            f"create_kwargs={create_kwargs}",
            debug=True,
        )

        # Acquire a desktop session.
        try:
            instance_id, initial_response = await self.desktop_tool.create(
                create_kwargs=create_kwargs,
            )
        except Exception as exc:
            # Traceback is already printed by desktop_env_tool._post via
            # logger.error(..., exc_info=True); keep a single summary line
            # here to avoid duplicating the full stack.
            _log_error(
                f"[GUIAgentLoop] Failed to create env for {task_id} {base_log_tag}, "
                f"discarding rollout: {exc!r}"
            )
            _log_error(
                f"[GUIAgentLoop][RETURN_NONE][create_failed] "
                f"task_id={task_id} request_id={request_id} sample_index={sample_index} "
                f"rollout_n={rollout_n} step={global_step} err={exc!r}"
            )
            return None
        log_tag = f"{base_log_tag}[iid={instance_id[:8]}]"
        _log(
            f"{log_tag} Env session created: instance_id={instance_id}, "
            f"initial_images={len(initial_response.image or [])}"
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
            fatal_error = False
            stop_reason = ""  # why the loop exited (for logging / debugging)

            # Last turn's snapshot; always retained as the *final*
            # AgentLoopOutput once the loop exits.
            last_turn_ctx: dict[str, Any] | None = None

            # Consecutive-failure counter for tool execution. A single tool
            # failure is surfaced to the model as a user message (so it can
            # retry with a different action); only when we see repeated
            # failures in a row do we treat the env as truly broken and
            # abort. This keeps single-action mistakes cheap: the rollout
            # still produces a usable final trajectory with whatever reward
            # the env eventually grades.
            consecutive_tool_failures = 0
            max_consecutive_tool_failures = int(
                getattr(self, "max_consecutive_tool_failures", 3)
            )

            while True:
                turn += 1

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
                _log(
                    f"{log_tag}[turn={turn}] prompt_ids={len(prompt_ids)} "
                    f"(orig={original_prompt_len}, truncated={truncated}), "
                    f"images={len(image_data) if image_data else 0}",
                    debug=True,
                )

                # 3. LLM generation (partial-rollout-resume enabled upstream).
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=image_data if image_data else None,
                    )
                _log(
                    f"{log_tag}[turn={turn}] LLM generated response_ids={len(output.token_ids)}, "
                    f"num_preempted={output.num_preempted}",
                    debug=True,
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

                tool_args: dict[str, Any] | None = None
                action: str = ""
                if tool_calls:
                    try:
                        tool_args = json.loads(tool_calls[0].arguments)
                        action = tool_args.get("action", "")
                    except Exception as parse_exc:
                        _log(
                            f"{log_tag}[turn={turn}] Failed to parse tool arguments: "
                            f"{tool_calls[0]} (error: {parse_exc!r})"
                        )
                        tool_args = None
                        action = ""

                if tool_args is not None:
                    _log(
                        f"{log_tag}[turn={turn}] Tool call: action={action}, "
                        f"args={ {k: v for k, v in tool_args.items() if k != 'action'} }"
                    )

                # 5. Decide whether this turn ends the rollout.
                #    The rollout ends when any of:
                #      (a) model emitted no parseable tool call;
                #      (b) model issued action=terminate;
                #      (c) we reached max_turns (this turn is the last one).
                #    In all three cases the current turn becomes the *final*
                #    trajectory (kept in ``last_turn_ctx``), not an
                #    intermediate one.
                is_final_turn = (
                    tool_args is None
                    or action == "terminate"
                    or turn >= self.max_turns
                )

                # 6. Execute the tool when we have a real action to run.
                #    ``terminate`` is a purely virtual action (model announces
                #    completion) so we skip the env step for it. For every
                #    other action we execute, even on the final turn, so the
                #    desktop reflects the full action sequence when reward
                #    is computed via ``/evaluate``.
                if tool_args is not None and action != "terminate":
                    try:
                        with simple_timer("tool_calls", metrics):
                            tool_response, _, _ = await self.desktop_tool.execute(
                                instance_id, tool_args
                            )
                        # Successful tool execution resets the failure streak.
                        consecutive_tool_failures = 0
                    except Exception as exec_exc:
                        consecutive_tool_failures += 1
                        _log_error(
                            f"[GUIAgentLoop] Tool execution failed for {task_id} {log_tag} "
                            f"(streak={consecutive_tool_failures}/"
                            f"{max_consecutive_tool_failures}): "
                            f"{exec_exc!r}\n{traceback.format_exc()}"
                        )
                        _log_error(
                            f"[GUIAgentLoop][SOFT_ERROR][tool_execute] "
                            f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                            f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                            f"turn={turn} action={action} err={exec_exc!r} "
                            f"streak={consecutive_tool_failures}/"
                            f"{max_consecutive_tool_failures}"
                        )

                        if consecutive_tool_failures >= max_consecutive_tool_failures:
                            # Repeated failures almost always mean the env
                            # session itself is dead (crashed VM, closed
                            # socket, auth lost). Further turns cannot make
                            # progress; exit the loop and let the post-loop
                            # logic keep whatever ``last_turn_ctx`` we have.
                            _log_error(
                                f"[GUIAgentLoop][FATAL_ERROR][tool_execute_repeated] "
                                f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                                f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                                f"turn={turn} action={action} err={exec_exc!r}"
                            )
                            fatal_error = True
                            break

                        # Take a fresh screenshot so the next turn always has
                        # visual context, even though the action itself failed.
                        # Without this, the user message would be pure text,
                        # and after context pruning all images could be lost,
                        # causing multi_modal_data to be empty for subsequent
                        # intermediate trajectories.
                        error_images = await self.desktop_tool.screenshot(instance_id)
                        _log(
                            f"{log_tag}[turn={turn}] Error recovery screenshot: "
                            f"got_image={bool(error_images)}"
                        )

                        # Build a synthetic tool_response so the downstream
                        # "append intermediate + extend messages" code path
                        # is reused verbatim.
                        tool_response = ToolResponse(
                            image=error_images,
                            text=(
                                f"Error executing action {action!r}: "
                                f"{type(exec_exc).__name__}: {exec_exc}. "
                                f"The screen was not changed. "
                                f"Please try a different action."
                            )
                        )
                    else:
                        _log(
                            f"{log_tag}[turn={turn}] Tool executed OK: "
                            f"got_image={bool(tool_response and tool_response.image)}",
                            debug=True,
                        )
                else:
                    tool_response = None

                # 7. If this turn is the final one, exit loop now without
                #    appending an intermediate or extending ``messages``
                #    (the next user turn will never be consumed).
                if is_final_turn:
                    if tool_args is None:
                        stop_reason = "no_tool_call"
                    elif action == "terminate":
                        stop_reason = "model_terminate"
                    else:
                        stop_reason = "max_turns"
                    _log(
                        f"{log_tag}[turn={turn}] Rollout ends: reason={stop_reason}"
                    )
                    break

                # 8. Not final: register this turn as an intermediate
                #    trajectory and update messages for the next turn.
                self.append_intermediate_trajectory(
                    prompt_ids=last_turn_ctx["prompt_ids"],
                    response_ids=last_turn_ctx["response_ids"],
                    response_mask=last_turn_ctx["response_mask"],
                    response_logprobs=last_turn_ctx["response_logprobs"],
                    routed_experts=last_turn_ctx["routed_experts"],
                    multi_modal_data=last_turn_ctx["multi_modal_data"],
                    num_turns=last_turn_ctx["num_turns"],
                    **last_turn_ctx["extra_fields"],
                )
                last_turn_ctx = None  # consumed as intermediate

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

            _log(
                f"[GUIAgentLoop][LOOP_EXIT] task_id={task_id} request_id={request_id} "
                f"instance_id={instance_id} sample_index={sample_index} "
                f"rollout_n={rollout_n} step={global_step} turn={turn} "
                f"stop_reason={stop_reason or ('fatal_error' if fatal_error else 'unknown')} "
                f"fatal_error={fatal_error} "
                f"last_turn_ctx_is_none={last_turn_ctx is None}"
            )

            # Environment-level failures (repeated tool exec errors, no
            # turns produced) are NOT the model's fault. Turning them into
            # reward=0 training samples would incorrectly punish the model
            # for env instability. Discard the rollout instead.
            if fatal_error:
                _log_error(
                    f"[GUIAgentLoop] Fatal error for {task_id} {log_tag} at turn={turn}, "
                    f"discarding rollout (env-level failure, not model's fault)"
                )
                _log_error(
                    f"[GUIAgentLoop][RETURN_NONE][fatal_error] "
                    f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                    f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                    f"turn={turn}"
                )
                return None

            if last_turn_ctx is None:
                _log_error(
                    f"[GUIAgentLoop] No turns produced for {task_id} {log_tag}, discarding rollout"
                )
                _log_error(
                    f"[GUIAgentLoop][RETURN_NONE][no_turns_produced] "
                    f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                    f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                    f"turn={turn}"
                )
                return None

            if turn >= self.max_turns and stop_reason == "max_turns":
                _log(
                    f"{log_tag} Reached max_turns={self.max_turns} without model terminate"
                )

            # Compute terminal reward. If the reward service itself fails,
            # that's an env-level problem (we cannot grade this rollout
            # honestly), so discard rather than defaulting to 0.
            try:
                shared_reward = await self.desktop_tool.calc_reward(instance_id)
            except Exception as reward_exc:
                _log_error(
                    f"[GUIAgentLoop] calc_reward failed for {task_id} {log_tag}: {reward_exc!r}; "
                    f"discarding rollout (env-level failure, not model's fault)"
                )
                _log_error(
                    f"[GUIAgentLoop][RETURN_NONE][calc_reward_failed] "
                    f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                    f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                    f"turn={turn} err={reward_exc!r}"
                )
                return None
            _log(
                f"{log_tag} Final reward = {shared_reward:.4f} "
                f"(turns={turn}, stop_reason={stop_reason})"
            )

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
            _log(
                f"{log_tag} Done: {num_intermediate + 1} trajectories "
                f"(1 final + {num_intermediate} intermediate), reward={shared_reward:.4f}"
            )

            # --- Data flow log: agent loop output ---
            log_message(
                "gui_agent_loop.build_final_output",
                f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                f"turns={turn} stop_reason={stop_reason} "
                f"reward={shared_reward:.4f} "
                f"num_intermediate={num_intermediate} "
                f"final_prompt_len={len(last_turn_ctx['prompt_ids'])} "
                f"final_response_len={len(last_turn_ctx['response_ids'])} "
                f"extra_fields_keys={list(final_output.extra_fields.keys())}",
            )

            return final_output

        finally:
            # ``release`` must run to completion even if the surrounding
            # coroutine is being cancelled (e.g. Ray actor restart, parameter
            # sync dropping rollouts mid-flight). A bare ``await release()``
            # inside ``finally`` is NOT cancel-safe: if the cancellation
            # arrives while ``release`` is itself awaiting the HTTP close
            # request, that inner await is cancelled too, the HTTP request
            # is never sent, and the desktop-env slot leaks server-side.
            #
            # ``asyncio.shield`` protects the inner coroutine from outer
            # cancellation. We still re-raise CancelledError after shielding
            # so that the framework can propagate cancellation upstream.
            try:
                await asyncio.shield(self.desktop_tool.release(instance_id))
            except asyncio.CancelledError:
                _log_error(
                    f"[GUIAgentLoop] release shielded but coroutine cancelled "
                    f"for {task_id} {log_tag}"
                )
                raise
            except Exception:
                _log_error(
                    f"[GUIAgentLoop] Failed to release env for {task_id} {log_tag}\n{traceback.format_exc()}"
                )
            _log(
                f"[GUIAgentLoop][RUN_END] task_id={task_id} request_id={request_id} "
                f"instance_id={instance_id} sample_index={sample_index} "
                f"rollout_n={rollout_n} step={global_step}"
            )
