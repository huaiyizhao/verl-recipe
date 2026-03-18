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
before each LLM generation turn.  This keeps the prompt within the model's
vision context window while preserving important conversational context.

Available strategies:

* :class:`KeepLastKImagesStrategy` — keep the last *K* screenshots, replacing
  older ones with ``[screenshot omitted]``.  This is the default strategy and
  wraps the existing :func:`keep_last_k_images` helper.

* :class:`SlidingWindowStrategy` — keep the last *N* conversation rounds and
  the last *M* image-bearing rounds.  Inspired by Galileo's
  ``SlidingWindowStrategy``.

To add a new strategy, subclass :class:`BaseContextStrategy` and implement
:meth:`prepare_context`.
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Image-pruning helper (moved here to avoid circular imports)
# ---------------------------------------------------------------------------


def keep_last_k_images(
    messages: list[dict[str, Any]],
    image_data: list[Image.Image],
    k: int,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Replace old screenshots in *messages* with text placeholders.

    Walks the message list in reverse order, counting ``{"type": "image"}``
    content blocks.  Beyond the *k*-th most recent image the content block is
    replaced with ``{"type": "text", "text": "[screenshot omitted]"}``.

    The *image_data* list is trimmed to contain only the last *k* images so
    that it stays in sync with the messages that still reference images.

    Args:
        messages: Chat messages (modified **in-place** and returned).
        image_data: Ordered list of PIL images referenced by the messages.
        k: Number of recent screenshots to keep.

    Returns:
        (messages, pruned_image_data)
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # Count total images in messages
    total_images = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    total_images += 1

    if total_images <= k:
        return messages, image_data

    # Walk in reverse and keep only the last k
    images_seen_from_end = 0
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") == "image":
                images_seen_from_end += 1
                if images_seen_from_end > k:
                    # Replace with placeholder
                    content[i] = {"type": "text", "text": "[screenshot omitted]"}

    # Keep only the last k images in image_data
    if len(image_data) > k:
        pruned_image_data = image_data[-k:]
    else:
        pruned_image_data = image_data

    return messages, pruned_image_data


class BaseContextStrategy(ABC):
    """Abstract base class for context management strategies.

    A context strategy receives the full message history and associated image
    data, and returns a (possibly pruned) copy suitable for the next LLM
    generation turn.
    """

    @abstractmethod
    def prepare_context(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Prepare messages and images for the next generation turn.

        Implementations may modify *messages* in-place (the caller is expected
        to pass a mutable list) and should return the (messages, image_data)
        tuple.

        Args:
            messages: Chat message history (may be modified in-place).
            image_data: Ordered list of PIL images referenced by messages.

        Returns:
            (processed_messages, processed_image_data)
        """
        ...


class KeepLastKImagesStrategy(BaseContextStrategy):
    """Keep only the last *K* images, replacing older ones with text placeholders.

    This strategy wraps the existing :func:`keep_last_k_images` helper so that
    current behaviour is preserved when using the pluggable interface.

    Args:
        k: Number of recent screenshots to keep (default 3).
    """

    def __init__(self, k: int = 3):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def prepare_context(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        return keep_last_k_images(messages, image_data, self.k)


class SlidingWindowStrategy(BaseContextStrategy):
    """Keep the last N conversation rounds and last M image-bearing rounds.

    A *round* is a consecutive (assistant, tool/user) message pair.  This
    strategy first drops the oldest rounds beyond *max_conversation_rounds*,
    then replaces images in rounds older than the last *max_image_rounds*
    with ``[screenshot omitted]`` placeholders — similar to Galileo's
    ``SlidingWindowStrategy``.

    The system message (role ``"system"``) and the very first user message
    are always preserved regardless of window size.

    Args:
        max_conversation_rounds: Maximum number of assistant+tool rounds to
            keep in the message history (default 10).
        max_image_rounds: Maximum number of recent rounds whose images are
            kept as-is (default 5).  Older rounds within the conversation
            window have their images replaced with placeholders.
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
        image_data: list[Image.Image],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        # --- Step 1: identify prefix (system + first user) vs rounds ---
        prefix: list[dict[str, Any]] = []
        rounds_start = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "system":
                prefix.append(msg)
                rounds_start = i + 1
            elif role == "user" and not prefix or (prefix and prefix[-1].get("role") == "system"):
                # Keep the first user message after system (or the first msg)
                prefix.append(msg)
                rounds_start = i + 1
                break
            else:
                break

        remaining = messages[rounds_start:]

        # --- Step 2: group remaining messages into rounds ---
        # A round starts with an assistant message and includes subsequent
        # tool/user messages until the next assistant message.
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

        # --- Step 3: truncate to max_conversation_rounds ---
        if len(rounds) > self.max_conversation_rounds:
            # Count images in dropped rounds so we can trim image_data
            dropped_rounds = rounds[: len(rounds) - self.max_conversation_rounds]
            dropped_image_count = 0
            for rnd in dropped_rounds:
                for msg in rnd:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "image":
                                dropped_image_count += 1

            rounds = rounds[-self.max_conversation_rounds:]
            # Also trim image_data for fully dropped rounds
            if dropped_image_count > 0 and len(image_data) > dropped_image_count:
                image_data = image_data[dropped_image_count:]
            elif dropped_image_count > 0:
                image_data = []

        # --- Step 4: replace images in rounds beyond max_image_rounds ---
        # Count image-bearing rounds from the end
        image_round_indices: list[int] = []
        for idx, rnd in enumerate(rounds):
            has_image = False
            for msg in rnd:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image":
                            has_image = True
                            break
                if has_image:
                    break
            if has_image:
                image_round_indices.append(idx)

        if len(image_round_indices) > self.max_image_rounds:
            rounds_to_strip = image_round_indices[: len(image_round_indices) - self.max_image_rounds]
            images_stripped = 0
            for idx in rounds_to_strip:
                for msg in rounds[idx]:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for i in range(len(content)):
                            block = content[i]
                            if isinstance(block, dict) and block.get("type") == "image":
                                content[i] = {"type": "text", "text": "[screenshot omitted]"}
                                images_stripped += 1

            # Trim image_data to remove stripped images
            if images_stripped > 0 and len(image_data) > images_stripped:
                image_data = image_data[images_stripped:]
            elif images_stripped > 0:
                image_data = []

        # --- Step 5: also handle images in prefix ---
        # Count images in prefix
        prefix_images = 0
        for msg in prefix:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        prefix_images += 1

        # Reconstruct messages
        result_messages = prefix[:]
        for rnd in rounds:
            result_messages.extend(rnd)

        return result_messages, image_data
