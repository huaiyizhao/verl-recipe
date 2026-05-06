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
"""Pluggable context management strategies for multi-turn agent loops.

Context strategies control how message history and image data are pruned
before each LLM generation turn. This keeps the prompt within the model's
vision context window while preserving important conversational context.

Available strategies:

* :class:`KeepLastKImagesStrategy` — keep the last *K* screenshots, replacing
  older ones with ``[screenshot omitted]``.
* :class:`SlidingWindowStrategy` — keep the last *N* conversation rounds and
  the last *M* image-bearing rounds (inspired by Galileo).

To add a new strategy, subclass :class:`BaseContextStrategy` and implement
:meth:`prepare_context`.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def keep_last_k_images(
    messages: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Replace old image blocks in ``messages`` with text placeholders.

    Walks the message list in reverse order, counting ``{"type": "image"}``
    blocks. Beyond the k-th most recent image the block is replaced with
    ``{"type": "text", "text": "[screenshot omitted]"}``. Messages are
    modified in-place and returned.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    total_images = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "image"
                    and "image" in block
                ):
                    total_images += 1

    if total_images <= k:
        logger.debug(
            "[ContextMgr/KeepLastK] no-op (total_images=%d <= k=%d)", total_images, k
        )
        return messages

    dropped = total_images - k
    logger.info(
        "[ContextMgr/KeepLastK] pruning %d old screenshots (keep last %d of %d)",
        dropped,
        k,
        total_images,
    )

    images_seen_from_end = 0
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if (
                isinstance(block, dict)
                and block.get("type") == "image"
                and "image" in block
            ):
                images_seen_from_end += 1
                if images_seen_from_end > k:
                    content[i] = {"type": "text", "text": "[screenshot omitted]"}

    return messages


class BaseContextStrategy(ABC):
    """Abstract base class for context management strategies."""

    @abstractmethod
    def prepare_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return the pruned messages for the next generation turn."""
        ...


class KeepLastKImagesStrategy(BaseContextStrategy):
    """Keep only the last K images; replace older ones with placeholders."""

    def __init__(self, k: int = 3):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def prepare_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return keep_last_k_images(messages, self.k)


class SlidingWindowStrategy(BaseContextStrategy):
    """Keep the last N conversation rounds and the last M image-bearing rounds.

    A *round* is a consecutive ``(assistant, tool/user)`` message group. The
    strategy first drops the oldest rounds beyond ``max_conversation_rounds``,
    then replaces images in rounds older than the last ``max_image_rounds``
    with ``[screenshot omitted]`` placeholders.

    The system message and the very first user message are always preserved.
    """

    def __init__(
        self,
        max_conversation_rounds: int = 10,
        max_image_rounds: int = 5,
    ):
        if max_conversation_rounds <= 0:
            raise ValueError("max_conversation_rounds must be positive")
        if max_image_rounds <= 0:
            raise ValueError("max_image_rounds must be positive")
        self.max_conversation_rounds = max_conversation_rounds
        self.max_image_rounds = max_image_rounds

    def prepare_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Step 1: identify prefix (system + first user) vs rounds.
        prefix: list[dict[str, Any]] = []
        rounds_start = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "system":
                prefix.append(msg)
                rounds_start = i + 1
            elif (role == "user" and not prefix) or (
                prefix and prefix[-1].get("role") == "system"
            ):
                # First user message after optional system prompt.
                prefix.append(msg)
                rounds_start = i + 1
                break
            else:
                break

        remaining = messages[rounds_start:]

        # Step 2: group remaining messages into rounds.
        rounds: list[list[dict[str, Any]]] = []
        current_round: list[dict[str, Any]] = []

        for msg in remaining:
            role = msg.get("role", "")
            if role == "assistant" and current_round:
                rounds.append(current_round)
                current_round = [msg]
            else:
                current_round.append(msg)

        if current_round:
            rounds.append(current_round)

        # Step 3: truncate to max_conversation_rounds.
        total_rounds = len(rounds)
        if total_rounds > self.max_conversation_rounds:
            dropped_rounds = total_rounds - self.max_conversation_rounds
            rounds = rounds[-self.max_conversation_rounds :]
            logger.info(
                "[ContextMgr/SlidingWindow] dropped %d oldest rounds (keep last %d of %d)",
                dropped_rounds,
                self.max_conversation_rounds,
                total_rounds,
            )

        # Step 4: replace images in rounds beyond max_image_rounds.
        image_round_indices: list[int] = []
        for idx, rnd in enumerate(rounds):
            has_image = False
            for msg in rnd:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "image"
                            and "image" in block
                        ):
                            has_image = True
                            break
                if has_image:
                    break
            if has_image:
                image_round_indices.append(idx)

        if len(image_round_indices) > self.max_image_rounds:
            rounds_to_strip = image_round_indices[
                : len(image_round_indices) - self.max_image_rounds
            ]
            logger.info(
                "[ContextMgr/SlidingWindow] stripping images in %d old image-rounds "
                "(keep last %d of %d image-rounds)",
                len(rounds_to_strip),
                self.max_image_rounds,
                len(image_round_indices),
            )
            for idx in rounds_to_strip:
                for msg in rounds[idx]:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for i in range(len(content)):
                            block = content[i]
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "image"
                                and "image" in block
                            ):
                                content[i] = {
                                    "type": "text",
                                    "text": "[screenshot omitted]",
                                }

        result_messages = prefix[:]
        for rnd in rounds:
            result_messages.extend(rnd)
        return result_messages


# ---------------------------------------------------------------------------
# Qwen3VL-style history strategy (aligned with OSWorld/mm_agents/qwen3vl_agent)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class TurnRecord:
    """Persistent record for one agent turn.

    Stores the raw facts of each turn independently from the messages
    sent to the model, enabling the strategy to rebuild messages from
    scratch every turn (with sliding-window truncation and Previous
    actions summarization).
    """

    # The screenshot observed *before* the model generated this turn's response.
    screenshot_image: Optional[Image.Image] = None

    # The model's raw decoded response text for this turn.
    assistant_raw_text: str = ""

    # A short imperative extracted from the response (the ``Action:`` line)
    # or a fallback like ``"Performing left_click action"``.
    low_level_instruction: str = ""

    # If the tool execution failed this turn, the error description;
    # otherwise ``None``.  When non-None the strategy inserts a pseudo
    # user text message after this turn's assistant message.
    error_text: Optional[str] = None


class Qwen3VLHistoryStrategy(BaseContextStrategy):
    """History strategy aligned with qwen3vl_agent's message assembly.

    Core semantics:
    * Only the most recent ``history_n`` turns are kept as (user image,
      assistant text) pairs in the messages.
    * Turns older than the window are summarized into a ``Previous actions:``
      text block inside the first retained user message.
    * The first retained user message always carries the instruction prompt
      (screenshot + ``Instruction: ... / Previous actions: ...``).
    * Subsequent user messages within the window carry **only** a screenshot.
    * When a turn has ``error_text`` set and falls within the retained window,
      a pseudo user text message is inserted after its assistant message.

    This class does NOT modify the system message (that comes from the
    dataset prompt[0] and is left untouched).

    Parameters
    ----------
    history_n : int
        Number of most-recent turns to keep as explicit (image, assistant)
        pairs.  Default is 4 (matching qwen3vl_agent).
    instruction_prompt_template : str
        A format-string with ``{instruction}`` and ``{previous_actions_str}``
        placeholders.  Default matches qwen3vl_agent.
    """

    _DEFAULT_INSTRUCTION_TEMPLATE = (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        "Instruction: {instruction}\n\n"
        "Previous actions:\n{previous_actions_str}"
    )

    def __init__(
        self,
        history_n: int = 4,
        instruction_prompt_template: str | None = None,
    ):
        if history_n <= 0:
            raise ValueError("history_n must be positive")
        self.history_n = history_n
        self.instruction_prompt_template = (
            instruction_prompt_template or self._DEFAULT_INSTRUCTION_TEMPLATE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Not used directly; prefer :meth:`build_messages`."""
        return messages

    def build_messages(
        self,
        system_message: dict[str, Any],
        turn_records: list[TurnRecord],
        current_screenshot: Image.Image,
        instruction: str,
    ) -> list[dict[str, Any]]:
        """Build the full messages list for the current generation turn.

        Parameters
        ----------
        system_message : dict
            The ``{"role":"system", ...}`` message from the dataset prompt.
        turn_records : list[TurnRecord]
            All completed turns so far (newest last).  A "turn" means the
            model has already responded and the tool has been executed.
        current_screenshot : PIL.Image.Image
            The screenshot for the *current* (not-yet-responded) turn.
        instruction : str
            The user's task instruction (``extra_info.question``).

        Returns
        -------
        list[dict]
            Ready-to-tokenize messages list (system + user/assistant pairs).
        """
        current_step = len(turn_records)
        history_start_idx = max(0, current_step - self.history_n)

        # --- Build Previous actions string (only summarizes early turns) ---
        previous_actions_parts: list[str] = []
        for i in range(history_start_idx):
            desc = turn_records[i].low_level_instruction or "unknown action"
            previous_actions_parts.append(f"Step {i + 1}: {desc}")
        previous_actions_str = "\n".join(previous_actions_parts) if previous_actions_parts else "None"

        instruction_prompt = self.instruction_prompt_template.format(
            instruction=instruction,
            previous_actions_str=previous_actions_str,
        )

        # --- Assemble messages ---
        messages: list[dict[str, Any]] = [system_message]

        history_len = min(self.history_n, current_step)
        if history_len > 0:
            # Windowed turns: turn_records[history_start_idx : current_step]
            windowed = turn_records[history_start_idx:current_step]

            for idx, record in enumerate(windowed):
                # User message (screenshot ± instruction_prompt)
                user_content: list[dict[str, Any]] = []
                if record.screenshot_image is not None:
                    user_content.append({"type": "image", "image": record.screenshot_image})
                if idx == 0:
                    # First retained user carries instruction prompt
                    user_content.append({"type": "text", "text": instruction_prompt})
                messages.append({"role": "user", "content": user_content})

                # Assistant message (pure text)
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": record.assistant_raw_text}],
                })

                # Error feedback (pseudo user text, C-scheme)
                if record.error_text:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": record.error_text}],
                    })

            # Current turn: user message with only the current screenshot
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": current_screenshot}],
            })
        else:
            # First turn ever: current screenshot + instruction prompt
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": current_screenshot},
                    {"type": "text", "text": instruction_prompt},
                ],
            })

        return messages
