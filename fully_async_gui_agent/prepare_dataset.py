#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generate a parquet dataset for GUI Agent training from the desktop env's /tasks API.

The desktop environment service exposes ``GET /tasks`` which returns the full
catalog of available OSWorld tasks::

    {"tasks": [{"task_id": "...", "domain": "chrome", "instruction": "...",
                "related_apps": [...]}, ...], "total": N}

This script:
  1. Pulls the task list (optionally filtered by ``--domain``).
  2. Shuffles and splits into train / test parquet files.
  3. Writes rows in the schema expected by verl's ``RLHFDataset`` with
     ``data.return_raw_chat=True`` and consumed by
     ``recipe.fully_async_gui_agent.gui_agent_loop.GUIAgentLoop``:

        prompt     : list[dict]    # chat messages (system + first user turn)
        extra_info : dict          # {task_id, question, domain, index}

Example usage
-------------
Full dataset, 90/10 split, default URL::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --output-dir /efs/data/cua/rl

Single-domain smoke dataset::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --api-base-url http://10.192.64.33:2354 \\
        --domain chrome \\
        --limit 20 \\
        --train-ratio 0.8 \\
        --output-dir /tmp/cua_smoke

Stable task-file dataset::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --api-base-url http://10.192.64.33:2354 \\
        --output-dir /tmp/cua_stable

By default, ``--task-file`` is ``test_stable.json``; the rl server resolves it
under its configured task examples directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_TASK_FILE = "test_stable.json"


# Default system prompt used when no ``--system-prompt`` is passed.
# Keep this aligned with ``OSWorld-local/mm_agents/qwen3vl_agent_original.py``.
_DESCRIPTION_PROMPT = "\n".join(
    [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
        "* The screen's resolution is 1000x1000.",
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]
)

_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """

_TOOLS_DEF = {
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": _DESCRIPTION_PROMPT,
        "parameters": {
            "properties": {
                "action": {
                    "description": _ACTION_DESCRIPTION_PROMPT,
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "scroll",
                        "wait",
                        "terminate",
                    ],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type`.", "type": "string"},
                "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                "pixels": {"description": "The amount of scrolling.", "type": "number"},
                "time": {"description": "The seconds to wait.", "type": "number"},
                "status": {
                    "description": "The status of the task.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
            "type": "object",
        },
        "args_format": "Format the arguments as a JSON object.",
    },
}

DEFAULT_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(_TOOLS_DEF) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; no extra dependency)
# ---------------------------------------------------------------------------


def fetch_tasks(api_base_url: str, domain: str | None, task_file: str | None, timeout: int) -> list[dict[str, Any]]:
    url = api_base_url.rstrip("/") + "/tasks"
    params = {}
    if domain:
        params["domain"] = domain
    if task_file:
        params["task_file"] = task_file
    if params:
        url += "?" + urllib.parse.urlencode(params)

    print(f"[prepare_dataset] GET {url}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")

    payload = json.loads(body)
    tasks = payload.get("tasks") or []
    total = payload.get("total", len(tasks))
    print(f"[prepare_dataset] Received {len(tasks)} tasks (server total={total})")
    return tasks


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


#: ``data_source`` identifier written into every row. It flows through verl's
#: RLHFDataset into ``non_tensor_batch["data_source"]`` and is consulted by
#: ``reward_score.default_compute_score``. For GUI agent tasks the true reward
#: is produced inside the agent loop (via the desktop ``/evaluate`` API), so
#: the reward-manager path only needs to return a no-op fallback score.
DATA_SOURCE = "osworld"

#: ``agent_name`` selects which agent loop handles the rollout. Must match
#: ``@register("gui_agent")`` in ``recipe.fully_async_gui_agent.gui_agent_loop``.
#: Without this field (or a matching CLI override of ``default_agent_loop``),
#: verl falls back to ``single_turn_agent`` and ``GUIAgentLoop.run()`` never
#: executes — which silently returns ``reward_score=None`` and triggers the
#: reward manager path that we do not want for GUI agent training.
AGENT_NAME = "gui_agent"


def build_row(task: dict[str, Any], index: int, system_prompt: str) -> dict[str, Any]:
    """Build one parquet row for a single OSWorld task.

    Field consumers (keep this in sync with the source):

    * ``RLHFDataset`` (verl/utils/dataset/rl_dataset.py):
      ``data_source`` / ``prompt`` / ``extra_info`` / ``reward_model``.
    * ``AgentLoopWorker`` (verl/experimental/agent_loop/agent_loop.py): routes
      on ``agent_name`` → falls back to ``default_agent_loop`` when absent.
    * ``GUIAgentLoop.run()`` (recipe/fully_async_gui_agent/gui_agent_loop.py):
      reads ``extra_info.task_id`` / ``extra_info.question`` /
      ``extra_info.tools_kwargs.computer_use.create_kwargs``.
    * ``DesktopEnvTool.create()`` (recipe/fully_async_gui_agent/desktop_env_tool.py):
      reads ``create_kwargs.task_id`` to start the remote session.
    * ``NaiveRewardManager.run_single``: reads ``data_source`` and
      ``reward_model.ground_truth`` as a fallback — never takes effect for
      GUI agent because ``AgentLoopOutput.reward_score`` is set from the
      desktop ``/evaluate`` API.
    """
    task_id = task["task_id"]
    instruction = task.get("instruction") or ""
    domain = task.get("domain") or "unknown"

    # Initial chat: system prompt only. The GUI agent loop constructs all
    # subsequent messages itself (first user turn from ``extra_info.question``
    # + initial screenshot from the desktop env).
    prompt: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    # ``create_kwargs`` forwarded to ``DesktopEnvTool.create()``. ``task_id``
    # uniquely selects the task. Extra knobs (``keep_last_k_images``) are read
    # by the agent loop / context strategy.
    create_kwargs = {
        "task_id": task_id,
        "keep_last_k_images": 3,
    }

    extra_info: dict[str, Any] = {
        "split": "unknown",  # overwritten by caller after train/test split
        "index": index,
        "task_id": task_id,
        "question": instruction,
        "domain": domain,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "computer_use": {
                "create_kwargs": create_kwargs,
            },
        },
    }

    # ``reward_model`` is framework-required. Its ``ground_truth`` is only
    # consulted if the reward-manager fallback is hit; the real terminal
    # reward is emitted by the agent loop from the desktop ``/evaluate`` API.
    reward_model = {
        "style": "rule",
        "ground_truth": task_id,
    }

    return {
        "data_source": DATA_SOURCE,
        "agent_name": AGENT_NAME,
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def split_train_test(
    rows: list[dict[str, Any]], train_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    # Guarantee at least one test sample when ratio < 1.0 and we have ≥2 rows.
    if train_ratio < 1.0 and n_train == len(shuffled) and len(shuffled) >= 2:
        n_train -= 1

    return shuffled[:n_train], shuffled[n_train:]


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_parquet(rows: list[dict[str, Any]], path: str) -> None:
    """Write rows to parquet. Uses pandas for schema inference."""
    try:
        import pandas as pd  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to write parquet; install pandas + pyarrow"
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[prepare_dataset] Wrote {len(df)} rows → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("DESKTOP_API_BASE_URL", "http://10.192.64.238:2354"),
        help="Desktop env service base URL (default: $DESKTOP_API_BASE_URL or http://10.192.64.238:2354).",
    )
    parser.add_argument("--domain", default=None, help="Filter tasks by domain (chrome/gimp/...).")
    parser.add_argument(
        "--task-file",
        default=DEFAULT_TASK_FILE,
        help=(
            "Local task subset JSON path or file:// URL passed to the desktop service /tasks API "
            f"(default: {DEFAULT_TASK_FILE})."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N tasks (after filter). 0 = all.")
    parser.add_argument("--train-ratio", type=float, default=0.95, help="Fraction of tasks for train split (default 0.9).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to embed in every row.",
    )
    parser.add_argument(
        "--output-dir",
        default="/efs/data/cua/rl/osworld",
        help="Directory to write train.parquet / test.parquet.",
    )
    parser.add_argument(
        "--train-name", default="train.parquet", help="Train parquet filename inside --output-dir."
    )
    parser.add_argument(
        "--test-name", default="test.parquet", help="Test parquet filename inside --output-dir."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tasks = fetch_tasks(args.api_base_url, args.domain, args.task_file, args.timeout)
    if not tasks:
        print("[prepare_dataset] ERROR: /tasks returned zero tasks", file=sys.stderr)
        return 1

    if args.limit and args.limit < len(tasks):
        # Stable subset: sort by task_id so repeated runs produce the same slice.
        tasks = sorted(tasks, key=lambda t: t.get("task_id", ""))[: args.limit]
        print(f"[prepare_dataset] Truncated to first {len(tasks)} tasks (--limit)")

    rows = [
        build_row(task, index=i, system_prompt=args.system_prompt)
        for i, task in enumerate(tasks)
    ]

    train_rows, test_rows = split_train_test(rows, args.train_ratio, args.seed)
    # Stamp each row's split label now that we know train vs test.
    for row in train_rows:
        row["extra_info"]["split"] = "train"
    for row in test_rows:
        row["extra_info"]["split"] = "test"
    print(
        f"[prepare_dataset] Split: train={len(train_rows)} "
        f"test={len(test_rows)} (ratio={args.train_ratio})"
    )

    train_path = os.path.join(args.output_dir, args.train_name)
    test_path = os.path.join(args.output_dir, args.test_name)

    write_parquet(train_rows, train_path)
    write_parquet(test_rows, test_path)

    # Quick sanity preview of row 0.
    if train_rows:
        sample = train_rows[0]
        print("[prepare_dataset] Sample train row:")
        print(f"  data_source            = {sample['data_source']}")
        print(f"  agent_name             = {sample['agent_name']}")
        print(f"  prompt (roles)         = {[m['role'] for m in sample['prompt']]}")
        print(f"  extra_info.task_id     = {sample['extra_info']['task_id']}")
        print(f"  extra_info.domain      = {sample['extra_info']['domain']}")
        question = sample['extra_info']['question']
        print(f"  extra_info.question    = {question[:80]!r}{'...' if len(question) > 80 else ''}")
        ck = sample['extra_info']['tools_kwargs']['computer_use']['create_kwargs']
        print(f"  create_kwargs keys     = {sorted(ck.keys())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
