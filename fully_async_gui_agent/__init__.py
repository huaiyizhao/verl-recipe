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
"""Fully-async GUI Agent (Computer-Use Agent) recipe for verl PPO training.

The generic multi-trajectory agent loop protocol lives in verl-async at
``verl.experimental.agent_loop.multi_trajectory_agent_loop``; this recipe
contains only the GUI-specific business logic (``GUIAgentLoop``,
``DesktopEnvTool``, context strategies).
"""

from recipe.fully_async_gui_agent.context_manager import (
    BaseContextStrategy,
    KeepLastKImagesStrategy,
    SlidingWindowStrategy,
)
from recipe.fully_async_gui_agent.desktop_env_tool import DesktopEnvTool
from recipe.fully_async_gui_agent.gui_agent_loop import GUIAgentLoop

__all__ = [
    "BaseContextStrategy",
    "DesktopEnvTool",
    "GUIAgentLoop",
    "KeepLastKImagesStrategy",
    "SlidingWindowStrategy",
]
