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

import numpy as np
from PIL import Image
from recipe.fully_async_gui_agent.context_manager import Qwen3VLHistoryStrategy, TurnRecord
from recipe.fully_async_gui_agent.data_flow_logger import log_message
from recipe.fully_async_gui_agent.desktop_env_tool import DesktopEnvStepError

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

# One-shot rollout-side probe: dump raw images + tokens + vLLM logprobs (the ONLY place these are
# together) so the microbench can recompute a FRESH vLLM (D) and FRESH HF (C) on the SAME data and
# compare to the recorded vLLM logprob (A). Gated by VERL_ROLLOUT_PROBE_DUMP=1.
_ROLLOUT_PROBE_DONE = [0]

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
        self.turn_penalty_coef = float(self.rollout_config.agent.get("turn_penalty_coef", 0.0) or 0.0)
        self.keep_last_k = 3  # default; can be overridden per-task via create_kwargs
        self.history_n = 4  # default; can be overridden per-task via create_kwargs
        self.loop_no_change_repeat_min = max(
            2,
            int(self.rollout_config.agent.get("loop_no_change_repeat_min", 5) or 5),
        )
        self.loop_no_change_diff_threshold = max(
            0.0,
            float(self.rollout_config.agent.get("loop_no_change_diff_threshold", 2.0) or 2.0),
        )

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_env_reruns = int(self.rollout_config.agent.get("max_env_reruns", 1) or 0)
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

    @staticmethod
    def _coord_bin(coord: Any, *, bin_size: int = 20) -> tuple[int, int] | None:
        if not isinstance(coord, list | tuple) or len(coord) != 2:
            return None
        try:
            x = float(coord[0])
            y = float(coord[1])
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return round(x / bin_size), round(y / bin_size)

    @staticmethod
    def _action_signature(tool_args: dict[str, Any], tool_info: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
        """Coarse action signature for conservative repeated-action detection."""
        tool_info = tool_info or {}
        action = str(tool_info.get("action") or tool_args.get("action") or "")
        if action in {"", "terminate", "answer"}:
            return None

        keys = tuple(str(key).lower() for key in (tool_args.get("keys") or []))
        coord = tool_info.get("actual_coordinate") or tool_info.get("raw_coordinate") or tool_args.get("coordinate")
        coord_bin = GUIAgentLoop._coord_bin(coord)

        if action in {
            "mouse_move",
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "left_click_drag",
        }:
            return action, coord_bin, keys
        if action == "type":
            text = str(tool_args.get("text", ""))
            return action, len(text), text[:64]
        if action == "key":
            return action, keys
        if action in {"scroll", "hscroll"}:
            pixels = tool_args.get("pixels", 0) or 0
            direction = 0
            try:
                direction = 1 if float(pixels) > 0 else -1 if float(pixels) < 0 else 0
            except (TypeError, ValueError):
                pass
            return action, direction, coord_bin, keys
        if action == "wait":
            return (action,)
        return action, coord_bin, keys

    @staticmethod
    def _screenshot_fingerprint(image: Image.Image | None) -> np.ndarray | None:
        if image is None:
            return None
        try:
            resample = getattr(Image, "Resampling", Image).BILINEAR
            small = image.convert("L").resize((32, 18), resample=resample)
            return np.asarray(small, dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def _screenshot_diff(before: Image.Image | None, after: Image.Image | None) -> float | None:
        before_fp = GUIAgentLoop._screenshot_fingerprint(before)
        after_fp = GUIAgentLoop._screenshot_fingerprint(after)
        if before_fp is None or after_fp is None:
            return None
        return float(np.mean(np.abs(before_fp - after_fp)))

    def _new_loop_no_change_stats(self) -> dict[str, Any]:
        return {
            "checked_actions": 0,
            "count": 0,
            "max_streak": 0,
            "current_streak": 0,
            "last_signature": None,
            "diff_sum": 0.0,
            "diff_count": 0,
        }

    def _record_loop_no_change_action(
        self,
        stats: dict[str, Any],
        *,
        tool_args: dict[str, Any],
        tool_info: dict[str, Any] | None,
        before_screenshot: Image.Image | None,
        after_screenshot: Image.Image | None,
    ) -> None:
        signature = self._action_signature(tool_args, tool_info)
        if signature is None:
            return
        diff = self._screenshot_diff(before_screenshot, after_screenshot)
        if diff is None:
            return

        stats["checked_actions"] += 1
        stats["diff_sum"] += diff
        stats["diff_count"] += 1

        same_action = signature == stats["last_signature"]
        no_change = diff <= self.loop_no_change_diff_threshold
        if same_action and no_change:
            stats["current_streak"] += 1
        else:
            stats["current_streak"] = 1
        stats["last_signature"] = signature
        stats["max_streak"] = max(int(stats["max_streak"]), int(stats["current_streak"]))

        if same_action and no_change and stats["current_streak"] >= self.loop_no_change_repeat_min:
            stats["count"] += 1

    @staticmethod
    def _loop_no_change_reward_info(stats: dict[str, Any]) -> dict[str, float | int]:
        checked_actions = int(stats["checked_actions"])
        count = int(stats["count"])
        diff_count = int(stats["diff_count"])
        mean_diff = float(stats["diff_sum"] / diff_count) if diff_count > 0 else 0.0
        return {
            "loop_no_change_score": float(count / max(checked_actions, 1)),
            "loop_no_change_count": count,
            "loop_no_change_checked_actions": checked_actions,
            "loop_no_change_max_streak": int(stats["max_streak"]),
            "loop_no_change_mean_diff": mean_diff,
        }

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

        self._intermediate_trajectories.clear()
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
        rerun_attempt = int(kwargs.pop("_gui_env_rerun_attempt", 0) or 0)
        self._intermediate_trajectories.clear()

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
            loop_no_change_stats = self._new_loop_no_change_stats()

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
                prompt_ids = await self.apply_chat_template(
                    messages,
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

                # One-shot rollout probe: dump raw images + tokens + vLLM logprobs so the microbench can
                # recompute FRESH vLLM (D) and FRESH HF (C) on the same data and 3-way compare to the
                # recorded vLLM logprob (A). Only a turn WITH images and WITH logprobs is useful.
                if (
                    os.getenv("VERL_ROLLOUT_PROBE_DUMP", "0") == "1"
                    and response_logprobs is not None
                    and multi_modal_data.get("images")
                    and _ROLLOUT_PROBE_DONE[0] < int(os.getenv("VERL_ROLLOUT_PROBE_MAX", "1"))
                ):
                    try:
                        import torch as _torch

                        _rp_dir = os.getenv("VERL_ROLLOUT_PROBE_DIR", "/tmp/rollout_probe")
                        os.makedirs(_rp_dir, exist_ok=True)
                        # Unique per worker (many AgentLoopWorkers run concurrently); otherwise they all
                        # write the same file and clobber / corrupt each other. Pick ANY one for the microbench.
                        _rp_path = os.path.join(_rp_dir, f"rollout_probe_pid{os.getpid()}_{_ROLLOUT_PROBE_DONE[0]}.pt")
                        _torch.save(
                            {
                                "prompt_ids": list(prompt_ids),
                                "response_ids": list(response_ids),
                                "response_logprobs": list(response_logprobs),  # A = vLLM logp (as recorded)
                                "multi_modal_data": multi_modal_data,  # raw PIL images sent to vLLM (for D & C)
                                "turn": turn,
                                "model_path": os.getenv("VERL_LOGPROB_DEBUG_TOKENIZER", ""),
                            },
                            _rp_path,
                        )
                        _ROLLOUT_PROBE_DONE[0] += 1
                        print(
                            f"[ROLLOUT_PROBE] dumped raw images + tokens + vLLM logp "
                            f"(n_img={len(multi_modal_data.get('images', []))}, resp={len(response_ids)}) -> {_rp_path}",
                            flush=True,
                        )
                    except Exception as _e:  # noqa: BLE001 - probe must never break rollout
                        print(f"[ROLLOUT_PROBE] dump failed: {_e!r}", flush=True)

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
                    lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=True),
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
                latest_tool_screenshot = None
                for tool_call_idx, tool_args in enumerate(tool_args_list, start=1):
                    action = tool_args.get("action", "")
                    before_tool_screenshot = latest_tool_screenshot or current_screenshot

                    try:
                        with simple_timer("tool_calls", metrics):
                            tool_response, _, tool_info = await self.desktop_tool.execute(instance_id, tool_args)
                        after_tool_screenshot = None
                        if tool_response and tool_response.image:
                            for img in tool_response.image:
                                if img is not None:
                                    after_tool_screenshot = img
                                    break
                        if after_tool_screenshot is not None:
                            self._record_loop_no_change_action(
                                loop_no_change_stats,
                                tool_args=tool_args,
                                tool_info=tool_info,
                                before_screenshot=before_tool_screenshot,
                                after_screenshot=after_tool_screenshot,
                            )
                            latest_tool_screenshot = after_tool_screenshot
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

                # 8. Not final: register this turn as intermediate and
                #    record it for history.
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
                current_screenshot = latest_tool_screenshot if latest_tool_screenshot else current_screenshot

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

            # Build the final AgentLoopOutput from the last turn's snapshot.
            reward_extra_info = {
                "base_reward": base_reward,
                "turn_penalty": turn_penalty,
                "effective_turn_penalty": effective_turn_penalty,
                **self._loop_no_change_reward_info(loop_no_change_stats),
            }
            for trajectory in self._intermediate_trajectories:
                trajectory.extra_fields["reward_extra_info"] = reward_extra_info

            final_extra_fields = dict(last_turn_ctx["extra_fields"])
            final_extra_fields["trajectory_role"] = "final"
            final_extra_fields["reward_extra_info"] = reward_extra_info

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

            num_intermediate = len(final_output.extra_fields.get("intermediate_trajectories", []))
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
                f"reward={shared_reward:.4f} base_reward={base_reward:.4f} "
                f"turn_penalty={turn_penalty:.4f} "
                f"effective_turn_penalty={effective_turn_penalty:.4f} "
                f"num_intermediate={num_intermediate} "
                f"final_prompt_len={len(last_turn_ctx['prompt_ids'])} "
                f"final_response_len={len(last_turn_ctx['response_ids'])} "
                f"extra_fields_keys={list(final_output.extra_fields.keys())}",
            )

            return final_output

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
