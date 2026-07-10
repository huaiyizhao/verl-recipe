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
"""GUI Agent Loop for Computer-Use Agent (CUA) training under the V1 trainer.

Multi-turn VLM + remote-desktop interaction. Each turn yields an independent
trajectory because screenshot pruning makes each turn's prompt unique. The
terminal binary reward is shared across all trajectories.

``run()`` returns a **list** of ``AgentLoopOutput`` — one equivalent trajectory
per turn, differing only by metadata (``trajectory_role`` / ``turn_number``).
The V1 ``AgentLoopWorkerTQ._agent_loop_postprocess`` accepts a list natively,
writes each turn as its own TransferQueue row keyed ``{uid}_{session_id}_{index}``,
and broadcasts the final-turn reward to the earlier turns. No intermediate
packing, no trainer-side expansion: every turn is first-class from the start.

Each output carries per-turn ``multi_modal_data`` (the raw screenshots visible
in that turn's prompt window); the V1 worker recomputes the processor tensors
(``multi_modal_inputs``) from it. ``FullyAsyncLLMServerManager`` supplies
partial-rollout resume transparently via ``self.server_manager.generate()``.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from typing import Any
from uuid import uuid4

from recipe.fully_async_gui_agent.context_manager import Qwen3VLHistoryStrategy, TurnRecord
from recipe.fully_async_gui_agent.data_flow_logger import log_message
from recipe.fully_async_gui_agent.desktop_env_tool import DesktopEnvStepError

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.tool_registry import initialize_tools_from_config
from verl.utils.executor_guard import guard_stop_iteration
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

# We bypass the ``logging`` framework entirely here because verl's global
# ``basicConfig(WARNING)`` plus Ray's early-attached handlers silently drop
# INFO/DEBUG records regardless of per-logger levels. ``print`` goes straight
# to stdout (picked up by Ray's log forwarder) and is unaffected by any of
# that. ``GUI_AGENT_LOGGING_LEVEL`` controls GUI/Desktop rollout traces.
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_LOG_LEVEL = os.getenv("GUI_AGENT_LOGGING_LEVEL", "ERROR").upper()
_LOG_THRESHOLD = _LOG_LEVELS.get(_LOG_LEVEL, _LOG_LEVELS["ERROR"])


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


def _log(msg: str, *, level: str = "DEBUG", debug: bool | None = None) -> None:
    """Print-based logger controlled by ``GUI_AGENT_LOGGING_LEVEL``."""
    if debug:
        level = "DEBUG"
    level = level.upper()
    if _LOG_LEVELS.get(level, _LOG_LEVELS["INFO"]) < _LOG_THRESHOLD:
        return
    prefix = " [POTENTIAL ERROR]" if level == "ERROR" else ""
    print(f"[{_ts()}]{prefix} {msg}", file=sys.stderr, flush=True)


@register("gui_agent")
class GUIAgentLoop(AgentLoopBase):
    """Multi-turn GUI agent loop for desktop computer-use tasks.

    Each LLM turn becomes an independent trajectory (because the prompt
    changes due to screenshot pruning). All trajectories share the final
    reward assigned at the end of the rollout. ``run()`` returns the full
    per-turn trajectory list; the V1 worker postprocesses each into its own
    TransferQueue row.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Per-instance buffer of intermediate-turn trajectories, flushed at the
        # end of ``run()`` together with the final turn.
        self._trajectories: list[AgentLoopOutput] = []

        # Multi-turn config.
        self.max_turns = self.rollout_config.multi_turn.max_assistant_turns or 20
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns or 20
        self.turn_penalty_coef = float(self.rollout_config.agent.get("turn_penalty_coef", 0.0) or 0.0)
        self.keep_last_k = 3  # default; can be overridden per-task via create_kwargs
        self.history_n = 4  # default; can be overridden per-task via create_kwargs

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_env_reruns = int(self.rollout_config.agent.get("max_env_reruns", 1) or 0)
        self.use_chat_template_tools = bool(self.rollout_config.agent.get("use_chat_template_tools", False))
        self._rerun_released_instances: set[str] = set()

        # Tool registration (expects computer_use / DesktopEnvTool).
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        self.desktop_tool = self.tools.get("computer_use")
        if self.desktop_tool is None and tool_list:
            # Fallback to the first registered tool.
            self.desktop_tool = tool_list[0]

        # Tool-call parser for extracting tool calls from LLM responses.
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metrics(self, metrics: dict[str, Any]) -> AgentLoopMetrics:
        """Build AgentLoopMetrics, keeping only known fields."""
        return AgentLoopMetrics(**{k: v for k, v in metrics.items() if k in AgentLoopMetrics.model_fields})

    @staticmethod
    def _apply_turn_penalty(
        base_reward: float,
        *,
        turn: int,
        max_turns: int,
        turn_penalty_coef: float,
    ) -> tuple[float, float, float]:
        """Apply an efficiency penalty scaled by base reward in [0, 1]."""
        raw_turn_penalty = turn_penalty_coef * max(0, turn - 1) / max(1, max_turns)
        effective_turn_penalty = raw_turn_penalty * max(0.0, min(1.0, base_reward))
        adjusted_reward = base_reward - effective_turn_penalty
        return adjusted_reward, raw_turn_penalty, effective_turn_penalty

    def _make_turn_output(
        self,
        ctx: dict[str, Any],
        *,
        role: str,
        metrics: AgentLoopMetrics | None = None,
    ) -> AgentLoopOutput:
        """Build one per-turn ``AgentLoopOutput`` from a turn snapshot.

        Intermediate and final turns are identical in schema; they differ only
        by ``extra_fields["trajectory_role"]`` and (for the final turn) the
        carried rollout-level ``metrics``. ``multi_modal_data`` (the raw
        screenshots for this turn) is carried so the V1 worker can recompute
        the processor tensors; ``reward_score`` is stamped later by ``run()``.
        """
        extra = dict(ctx["extra_fields"])
        extra["trajectory_role"] = role
        # The V1 worker broadcasts ``extra_fields["reward_extra_info"]`` from the
        # final turn to the earlier turns. Because our reward is computed in-loop
        # (``reward_score`` is pre-set), the worker's reward path is skipped and
        # would never populate this key — so provide it here to avoid a KeyError.
        extra.setdefault("reward_extra_info", {})
        return AgentLoopOutput(
            prompt_ids=ctx["prompt_ids"],
            response_ids=ctx["response_ids"],
            response_mask=ctx["response_mask"],
            response_logprobs=ctx["response_logprobs"],
            routed_experts=ctx["routed_experts"],
            multi_modal_data=ctx["multi_modal_data"],
            num_turns=ctx["num_turns"],
            metrics=metrics if metrics is not None else AgentLoopMetrics(),
            extra_fields=extra,
        )

    @staticmethod
    def _extract_low_level_instruction(response: str, fallback_action: str | None = None) -> str:
        """Extract the Action: line from a model response (qwen3vl_agent style).

        Scans lines for one starting with ``Action:`` and returns the text
        after the prefix. Falls back to ``"Performing {action} action"`` when
        no Action line is found.
        """
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                result = stripped.split(":", 1)[1].strip()
                if result:
                    return result
        if fallback_action:
            return f"Performing {fallback_action} action"
        return ""

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    async def _rerun_after_env_failure(
        self,
        sampling_params: dict[str, Any],
        kwargs: dict[str, Any],
        *,
        reason: str,
        attempt: int,
        task_id: str,
        request_id: str,
        sample_index: Any,
        rollout_n: Any,
        global_step: Any,
        instance_id: str | None = None,
        turn: int | None = None,
        error: BaseException | None = None,
    ) -> AgentLoopOutput | None:
        if attempt >= self.max_env_reruns:
            return None

        if instance_id is not None:
            try:
                await asyncio.shield(self.desktop_tool.release(instance_id))
                self._rerun_released_instances.add(instance_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._rerun_released_instances.add(instance_id)
                _log(
                    f"[GUIAgentLoop][ENV_RERUN_RELEASE_FAILED] task_id={task_id} "
                    f"request_id={request_id} instance_id={instance_id} "
                    f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                    f"reason={reason}\n{traceback.format_exc()}",
                    level="ERROR",
                )

        self._trajectories.clear()
        retry_kwargs = dict(kwargs)
        retry_kwargs["_gui_env_rerun_attempt"] = attempt + 1
        _log(
            f"[GUIAgentLoop][ENV_RERUN] task_id={task_id} request_id={request_id} "
            f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
            f"attempt={attempt + 1}/{self.max_env_reruns} reason={reason} "
            f"turn={turn if turn is not None else '?'} err={error!r}",
            level="ERROR",
        )
        return await self.run(sampling_params, **retry_kwargs)

    def _log_return_none(
        self,
        *,
        reason: str,
        task_id: str,
        request_id: str,
        sample_index: Any,
        rollout_n: Any,
        global_step: Any,
        instance_id: str | None = None,
        turn: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        _log(
            f"[GUIAgentLoop][RETURN_NONE][{reason}] "
            f"task_id={task_id} request_id={request_id} "
            f"instance_id={instance_id or '<none>'} sample_index={sample_index} "
            f"rollout_n={rollout_n} step={global_step} "
            f"turn={turn if turn is not None else '?'} err={error!r}",
            level="ERROR",
        )

    async def _rerun_or_discard(
        self,
        sampling_params: dict[str, Any],
        kwargs: dict[str, Any],
        *,
        reason: str,
        attempt: int,
        task_id: str,
        request_id: str,
        sample_index: Any,
        rollout_n: Any,
        global_step: Any,
        instance_id: str | None = None,
        turn: int | None = None,
        error: BaseException | None = None,
    ) -> AgentLoopOutput | None:
        rerun_output = await self._rerun_after_env_failure(
            sampling_params,
            kwargs,
            reason=reason,
            attempt=attempt,
            task_id=task_id,
            request_id=request_id,
            sample_index=sample_index,
            rollout_n=rollout_n,
            global_step=global_step,
            instance_id=instance_id,
            turn=turn,
            error=error,
        )
        if rerun_output is not None:
            return rerun_output
        if attempt >= self.max_env_reruns:
            self._log_return_none(
                reason=reason,
                task_id=task_id,
                request_id=request_id,
                sample_index=sample_index,
                rollout_n=rollout_n,
                global_step=global_step,
                instance_id=instance_id,
                turn=turn,
                error=error,
            )
        return None

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput] | None:
        """Run the GUI agent multi-turn rollout.

        Args:
            sampling_params: LLM sampling params.
            **kwargs: dataset fields; must include ``raw_prompt`` and
                ``extra_info`` with ``task_id`` / ``question``.

        Returns:
            A ``list[AgentLoopOutput]`` — one equivalent trajectory per turn
            (intermediate turns first, final turn last), each carrying the
            shared episode reward. Returns ``None`` if the rollout must be
            discarded (env creation failed or a fatal error occurred
            mid-rollout).
        """
        rerun_attempt = int(kwargs.pop("_gui_env_rerun_attempt", 0) or 0)
        self._trajectories.clear()

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
        history_n = create_kwargs.get("history_n", self.history_n)
        context_strategy = Qwen3VLHistoryStrategy(history_n=history_n)

        create_kwargs.setdefault("task_id", task_id)
        _log(
            f"{log_tag} Context strategy=Qwen3VLHistoryStrategy(history_n={history_n}), create_kwargs={create_kwargs}",
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
            _log(
                f"[GUIAgentLoop] Failed to create env for {task_id} {base_log_tag}: {exc!r}",
                level="ERROR",
            )
            return await self._rerun_or_discard(
                sampling_params,
                kwargs,
                reason="create_failed",
                attempt=rerun_attempt,
                task_id=task_id,
                request_id=request_id,
                sample_index=sample_index,
                rollout_n=rollout_n,
                global_step=global_step,
                error=exc,
            )
        log_tag = f"{base_log_tag}[iid={instance_id[:8]}]"
        _log(
            f"{log_tag} Env session created: instance_id={instance_id}, "
            f"initial_images={len(initial_response.image or [])}"
        )

        try:
            # Extract the system message and initial screenshot.
            system_message = messages[0]  # {"role": "system", ...} from dataset
            current_screenshot = None
            if initial_response.image:
                for img in initial_response.image:
                    if img is not None:
                        current_screenshot = img
                        break

            if current_screenshot is None:
                _log(f"[GUIAgentLoop] No initial screenshot for {task_id} {log_tag}", level="ERROR")
                return await self._rerun_or_discard(
                    sampling_params,
                    kwargs,
                    reason="missing_initial_screenshot",
                    attempt=rerun_attempt,
                    task_id=task_id,
                    request_id=request_id,
                    sample_index=sample_index,
                    rollout_n=rollout_n,
                    global_step=global_step,
                    instance_id=instance_id,
                )

            turn = 0
            fatal_error = False
            fatal_error_exc: BaseException | None = None
            stop_reason = ""

            last_turn_ctx: dict[str, Any] | None = None

            # Persistent turn records for history strategy.
            turn_records: list[TurnRecord] = []

            while True:
                turn += 1

                # 1. Build messages from turn_records + current_screenshot.
                messages = context_strategy.build_messages(
                    system_message=system_message,
                    turn_records=turn_records,
                    current_screenshot=current_screenshot,
                    instruction=task_query,
                )

                # 2. Tokenize prompt for this turn.
                multi_modal_data = await self.process_vision_info(messages)
                image_data = multi_modal_data.get("images")
                template_tools = self.tool_schemas if self.use_chat_template_tools else None
                prompt_ids = await self.apply_chat_template(
                    messages,
                    tools=template_tools,
                    images=image_data if image_data else None,
                )
                original_prompt_len = len(prompt_ids)
                image_count = len(image_data) if image_data else 0
                if original_prompt_len > self.prompt_length:
                    _log(
                        f"{log_tag}[turn={turn}] [OVERLONG_PROMPT] "
                        f"prompt_ids={original_prompt_len} > max_prompt_length={self.prompt_length}, "
                        f"images={image_count}. Discarding rollout.",
                        level="ERROR",
                    )
                    return None
                _log(
                    f"{log_tag}[turn={turn}] prompt_ids={len(prompt_ids)} "
                    f"(orig={original_prompt_len}, truncated=False), "
                    f"images={image_count}",
                    debug=True,
                )

                # 3. LLM generation.
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

                if len(output.token_ids) == 0:
                    error_msg = (
                        f"{log_tag}[turn={turn}] Empty LLM response in rollout: "
                        f"prompt_ids={len(prompt_ids)}, images={image_count}, "
                        f"response_length={self.response_length}, stop_reason={output.stop_reason!r}, "
                        f"num_preempted={output.num_preempted}, extra_fields={output.extra_fields}"
                    )
                    _log(f"[GUIAgentLoop][EMPTY_RESPONSE] {error_msg}", level="ERROR")
                    raise RuntimeError(error_msg)

                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
                else:
                    metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

                response_ids = output.token_ids[: self.response_length]
                response_mask = [1] * len(response_ids)
                response_logprobs = output.log_probs[: len(response_ids)] if output.log_probs else None

                # Per-turn extra_fields.
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
                    "num_turns": turn * 2,
                    "extra_fields": extra_fields,
                }

                # 4. Parse tool calls. A single model response may contain
                # multiple computer_use calls; execute them sequentially below.
                tool_args_list: list[dict[str, Any]] = []
                actions: list[str] = []
                parse_error_text: str | None = None
                tool_schemas_all = [tool.tool_schema for tool in self.tools.values()]
                try:
                    _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids, tool_schemas_all)
                except Exception as parse_exc:
                    _log(
                        f"{log_tag}[turn={turn}] Failed to extract tool calls "
                        f"from model output; continuing without environment step "
                        f"(error: {parse_exc!r})",
                        level="ERROR",
                    )
                    tool_calls = []
                    parse_error_text = (
                        "Error: invalid tool call format. Please emit exactly one valid computer_use tool call."
                    )
                if tool_calls:
                    for tool_call_idx, tool_call in enumerate(tool_calls, start=1):
                        try:
                            parsed_args = json.loads(tool_call.arguments)
                            if not isinstance(parsed_args, dict):
                                raise ValueError("tool arguments must be a JSON object")
                            tool_args_list.append(parsed_args)
                            actions.append(str(parsed_args.get("action", "")))
                        except Exception as parse_exc:
                            _log(
                                f"{log_tag}[turn={turn}] Failed to parse tool arguments; "
                                f"skipping tool_call_index={tool_call_idx}: {tool_call} "
                                f"(error: {parse_exc!r})"
                            )
                            parse_error_text = (
                                "Error: invalid tool call format. "
                                "The <tool_call> content must be a valid JSON object. "
                                "Please emit exactly one valid computer_use tool call."
                            )
                if not tool_args_list:
                    _log(
                        f"{log_tag}[turn={turn}] No valid tool call parsed; continuing without environment step",
                        debug=True,
                    )

                if tool_args_list:
                    tool_call_summary = [
                        (args.get("action", ""), {k: v for k, v in args.items() if k != "action"})
                        for args in tool_args_list
                    ]
                    _log(f"{log_tag}[turn={turn}] Tool calls: {tool_call_summary}")

                # Decode assistant text and extract low_level_instruction.
                assistant_text = await self.loop.run_in_executor(
                    None,
                    guard_stop_iteration(lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=True)),
                )
                low_level_instruction = self._extract_low_level_instruction(
                    assistant_text, fallback_action=actions[0] if actions else None
                )

                # 5. Decide whether this turn ends the rollout.
                # Missing or malformed tool calls match eval behavior: no
                # environment step, no explicit error feedback, continue.
                is_final_turn = turn >= self.max_turns
                terminated_by_model = False
                backend_done = False

                # 6. Execute parsed tool calls sequentially.
                error_text: str | None = parse_error_text
                tool_response = None
                for tool_call_idx, tool_args in enumerate(tool_args_list, start=1):
                    action = tool_args.get("action", "")

                    try:
                        with simple_timer("tool_calls", metrics):
                            tool_response, _, tool_info = await self.desktop_tool.execute(instance_id, tool_args)
                        if tool_info.get("invalid_action") and tool_response.text:
                            error_text = tool_response.text
                            break
                        if action == "terminate":
                            terminated_by_model = True
                            is_final_turn = True
                            _log(
                                f"{log_tag}[turn={turn}] Model requested terminate "
                                f"at tool_call_index={tool_call_idx}; "
                                f"sent OSWorld step={tool_info.get('code')}"
                            )
                            break
                        if tool_info.get("done"):
                            backend_done = True
                            is_final_turn = True
                            break
                    except Exception as exec_exc:
                        if isinstance(exec_exc, DesktopEnvStepError):
                            fatal_error_exc = exec_exc
                            _log(
                                f"[GUIAgentLoop][FATAL_ERROR][desktop_step_failed] "
                                f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                                f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                                f"turn={turn} tool_call_index={tool_call_idx} action={action} err={exec_exc!r}",
                                level="ERROR",
                            )
                            fatal_error = True
                            break
                        fatal_error_exc = exec_exc
                        _log(
                            f"[GUIAgentLoop][FATAL_ERROR][tool_execute_failed] "
                            f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                            f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                            f"turn={turn} tool_call_index={tool_call_idx} action={action} err={exec_exc!r}\n"
                            f"{traceback.format_exc()}",
                            level="ERROR",
                        )
                        fatal_error = True
                        break
                    else:
                        _log(
                            f"{log_tag}[turn={turn}] Tool executed OK: "
                            f"tool_call_index={tool_call_idx}/{len(tool_args_list)} "
                            f"action={action} got_image={bool(tool_response and tool_response.image)}",
                            debug=True,
                        )

                if fatal_error:
                    break

                # 7. If this turn is the final one, record it and exit.
                if is_final_turn:
                    if terminated_by_model:
                        stop_reason = "model_terminate"
                    elif backend_done:
                        stop_reason = "env_done"
                    else:
                        stop_reason = "max_turns"
                    _log(f"{log_tag}[turn={turn}] Rollout ends: reason={stop_reason}")
                    break

                # 8. Not final: buffer this turn as an intermediate trajectory
                #    and record it for history.
                self._trajectories.append(self._make_turn_output(last_turn_ctx, role="intermediate"))
                last_turn_ctx = None  # consumed as intermediate

                # Record this turn for the history strategy.
                turn_records.append(
                    TurnRecord(
                        screenshot_image=current_screenshot,
                        assistant_raw_text=assistant_text,
                        low_level_instruction=low_level_instruction,
                        error_text=error_text,
                    )
                )

                # Update current_screenshot for the next turn.
                next_screenshot = None
                if tool_response and tool_response.image:
                    for img in tool_response.image:
                        if img is not None:
                            next_screenshot = img
                            break
                current_screenshot = next_screenshot if next_screenshot else current_screenshot

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
                _log(
                    f"[GUIAgentLoop] Fatal error for {task_id} {log_tag} at turn={turn}, "
                    f"env-level failure, not model's fault",
                    level="ERROR",
                )
                return await self._rerun_or_discard(
                    sampling_params,
                    kwargs,
                    reason="fatal_error",
                    attempt=rerun_attempt,
                    task_id=task_id,
                    request_id=request_id,
                    sample_index=sample_index,
                    rollout_n=rollout_n,
                    global_step=global_step,
                    instance_id=instance_id,
                    turn=turn,
                    error=fatal_error_exc,
                )

            if last_turn_ctx is None:
                _log(f"[GUIAgentLoop] No turns produced for {task_id} {log_tag}, discarding rollout", level="ERROR")
                self._log_return_none(
                    reason="no_turns_produced",
                    task_id=task_id,
                    request_id=request_id,
                    sample_index=sample_index,
                    rollout_n=rollout_n,
                    global_step=global_step,
                    instance_id=instance_id,
                    turn=turn,
                )
                return None

            if turn >= self.max_turns and stop_reason == "max_turns":
                _log(f"{log_tag} Reached max_turns={self.max_turns} without model terminate")

            # Compute terminal reward. If evaluation keeps failing after the
            # env tool's retries, it returns reward=0 rather than discarding
            # the rollout.
            try:
                base_reward = await self.desktop_tool.calc_reward(instance_id)
            except Exception as reward_exc:
                _log(
                    f"[GUIAgentLoop] calc_reward failed for {task_id} {log_tag}: {reward_exc!r}; "
                    f"discarding rollout (env-level failure, not model's fault)",
                    level="ERROR",
                )
                self._log_return_none(
                    reason="calc_reward_failed",
                    task_id=task_id,
                    request_id=request_id,
                    sample_index=sample_index,
                    rollout_n=rollout_n,
                    global_step=global_step,
                    instance_id=instance_id,
                    turn=turn,
                    error=reward_exc,
                )
                return None
            shared_reward, turn_penalty, effective_turn_penalty = self._apply_turn_penalty(
                base_reward,
                turn=turn,
                max_turns=self.max_turns,
                turn_penalty_coef=self.turn_penalty_coef,
            )
            _log(
                f"{log_tag} Final reward = {shared_reward:.4f} "
                f"(base={base_reward:.4f}, turn_penalty={turn_penalty:.4f}, "
                f"effective_turn_penalty={effective_turn_penalty:.4f}, "
                f"turns={turn}, stop_reason={stop_reason})"
            )

            # Build the final-turn output and assemble the full trajectory list.
            # Every turn shares the episode reward; the worker writes each as its
            # own row, differing only by ``trajectory_role`` / ``turn_number``.
            final_output = self._make_turn_output(last_turn_ctx, role="final", metrics=self._build_metrics(metrics))
            final_output.reward_score = shared_reward
            reward_extra_info = {
                "reward": shared_reward,
                "base_reward": base_reward,
                "turn_penalty": turn_penalty,
                "effective_turn_penalty": effective_turn_penalty,
            }
            final_output.extra_fields["reward"] = shared_reward
            final_output.extra_fields["reward_extra_info"] = reward_extra_info
            for traj in self._trajectories:
                traj.reward_score = shared_reward
                traj.extra_fields["reward"] = shared_reward
                traj.extra_fields["reward_extra_info"] = reward_extra_info
            # Screenshots/text now render in the mlflow trace via rollout_trace_op's OpenAI-chat
            # serialization of run()'s output (each AgentLoopOutput.multi_modal_data image ->
            # image_url part), so no separate conversation span is emitted here.

            # Chronological order: intermediate turns first, final turn last.
            trajectories = [*self._trajectories, final_output]
            self._trajectories = []

            num_intermediate = len(trajectories) - 1
            _log(
                f"{log_tag} Done: {len(trajectories)} trajectories "
                f"(1 final + {num_intermediate} intermediate), reward={shared_reward:.4f}"
            )

            # --- Data flow log: agent loop output ---
            log_message(
                "gui_agent_loop.run",
                f"task_id={task_id} request_id={request_id} instance_id={instance_id} "
                f"sample_index={sample_index} rollout_n={rollout_n} step={global_step} "
                f"turns={turn} stop_reason={stop_reason} "
                f"reward={shared_reward:.4f} base_reward={base_reward:.4f} "
                f"turn_penalty={turn_penalty:.4f} "
                f"effective_turn_penalty={effective_turn_penalty:.4f} "
                f"num_intermediate={num_intermediate} "
                f"final_prompt_len={len(last_turn_ctx['prompt_ids'])} "
                f"final_response_len={len(last_turn_ctx['response_ids'])}",
            )

            return trajectories

        except Exception as run_exc:
            _log(
                f"[GUIAgentLoop][RUN_EXCEPTION] task_id={task_id} request_id={request_id} "
                f"instance_id={instance_id} sample_index={sample_index} "
                f"rollout_n={rollout_n} step={global_step} err={run_exc!r}\n"
                f"{traceback.format_exc()}",
                level="ERROR",
            )
            rerun_output = await self._rerun_or_discard(
                sampling_params,
                kwargs,
                reason="run_exception",
                attempt=rerun_attempt,
                task_id=task_id,
                request_id=request_id,
                sample_index=sample_index,
                rollout_n=rollout_n,
                global_step=global_step,
                instance_id=instance_id,
                error=run_exc,
            )
            if rerun_output is not None:
                return rerun_output
            raise

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
            if instance_id in self._rerun_released_instances:
                self._rerun_released_instances.discard(instance_id)
            else:
                try:
                    await asyncio.shield(self.desktop_tool.release(instance_id))
                except asyncio.CancelledError:
                    _log(
                        f"[GUIAgentLoop] release shielded but coroutine cancelled for {task_id} {log_tag}",
                        level="ERROR",
                    )
                    raise
                except Exception:
                    _log(
                        f"[GUIAgentLoop] Failed to release env for {task_id} {log_tag}\n{traceback.format_exc()}",
                        level="ERROR",
                    )
            _log(
                f"[GUIAgentLoop][RUN_END] task_id={task_id} request_id={request_id} "
                f"instance_id={instance_id} sample_index={sample_index} "
                f"rollout_n={rollout_n} step={global_step}"
            )
