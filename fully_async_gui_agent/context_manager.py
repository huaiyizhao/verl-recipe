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
