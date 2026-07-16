#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generate parquet datasets for GUI Agent training from the desktop env /tasks API.

The desktop environment service exposes ``GET /tasks`` which returns the full
catalog of available OSWorld tasks::

    {"tasks": [{"task_id": "...", "domain": "chrome", "instruction": "...",
                "related_apps": [...]}, ...], "total": N}

This script supports two modes:

  * Default local-split mode:
      1. Pull one task list (optionally filtered by ``--domain``).
      2. Shuffle and split it into train / test parquet files.
  * RL-proxy split mode (``--from-proxy-splits``):
      1. Pull ``/tasks?split=train`` and ``/tasks?split=test`` separately.
      2. Write them directly to train / test parquet files.
  3. Writes rows in the schema expected by verl's ``RLHFDataset`` with
     ``data.return_raw_chat=True`` and consumed by
     ``recipe.fully_async_gui_agent.gui_agent_loop.GUIAgentLoop``:

        prompt     : list[dict]    # chat messages (system + first user turn)
        extra_info : dict          # {task_id, question, domain, index}

Example usage
-------------
Full dataset, 95/5 split, default URL::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --output-dir /efs/data/cua/rl

Single-domain smoke dataset::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --api-base-url http://172.31.13.38:2354 \\
        --domain chrome \\
        --limit 20 \\
        --train-ratio 0.8 \\
        --output-dir /tmp/cua_smoke

Stable task-file dataset::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --api-base-url http://172.31.13.38:2354 \\
        --output-dir /tmp/cua_stable

By default, ``--task-file`` is ``test_stable.json``; the rl server resolves it
under its configured task examples directory.

RL-proxy train/eval split dataset::

    uv run python recipe/fully_async_gui_agent/prepare_dataset.py \\
        --api-base-url http://172.31.13.38:2354 \\
        --from-proxy-splits \\
        --output-dir /efs/data/cua/rl/osworld

In this mode the proxy resolves ``split=train`` from ``RL_PROXY_TRAIN_TASK_FILE``
and ``split=test``/``eval`` from ``RL_PROXY_EVAL_TASK_FILE``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_TASK_FILE = "test_stable.json"


def _load_computer_use_schema_module():
    try:
        from recipe.fully_async_gui_agent import computer_use_schema

        return computer_use_schema
    except ModuleNotFoundError:
        module_path = os.path.join(os.path.dirname(__file__), "computer_use_schema.py")
        spec = importlib.util.spec_from_file_location("_gui_agent_computer_use_schema", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load computer_use_schema.py from {module_path}") from None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_COMPUTER_USE_SCHEMA = _load_computer_use_schema_module()
DEFAULT_SYSTEM_PROMPT = _COMPUTER_USE_SCHEMA.build_computer_use_system_prompt()
CHAT_TEMPLATE_TOOLS_SYSTEM_PROMPT = _COMPUTER_USE_SCHEMA.build_computer_use_behavior_prompt()


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; no extra dependency)
# ---------------------------------------------------------------------------


def fetch_tasks(
    api_base_url: str,
    domain: str | None,
    task_file: str | None,
    timeout: int,
    *,
    split: str | None = None,
    auth_token: str | None = None,
) -> list[dict[str, Any]]:
    url = api_base_url.rstrip("/") + "/tasks"
    params = {}
    if domain:
        params["domain"] = domain
    if task_file:
        params["task_file"] = task_file
    elif split:
        params["split"] = split
    if params:
        url += "?" + urllib.parse.urlencode(params)

    print(f"[prepare_dataset] GET {url}")
    headers = {"Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(
            f"[prepare_dataset] HTTP error from /tasks: "
            f"status={exc.code} reason={exc.reason!r} url={url} body={error_body!r}",
            file=sys.stderr,
        )
        raise
    except urllib.error.URLError as exc:
        print(
            f"[prepare_dataset] URL error from /tasks: url={url} reason={exc.reason!r}",
            file=sys.stderr,
        )
        raise

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
    # uniquely selects the task. ``history_n`` controls how many historical
    # screenshots the agent loop retains in addition to the current screenshot.
    create_kwargs = {
        "task_id": task_id,
        "history_n": 2,
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


def build_rows(tasks: list[dict[str, Any]], *, split: str, system_prompt: str) -> list[dict[str, Any]]:
    rows = [build_row(task, index=i, system_prompt=system_prompt) for i, task in enumerate(tasks)]
    for row in rows:
        row["extra_info"]["split"] = split
    return rows


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


def limit_tasks(tasks: list[dict[str, Any]], limit: int, *, label: str) -> list[dict[str, Any]]:
    if limit and limit < len(tasks):
        # Stable subset: sort by task_id so repeated runs produce the same slice.
        limited = sorted(tasks, key=lambda t: t.get("task_id", ""))[:limit]
        print(f"[prepare_dataset] Truncated {label} to first {len(limited)} tasks (--limit)")
        return limited
    return tasks


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_parquet(rows: list[dict[str, Any]], path: str) -> None:
    """Write rows to parquet. Uses pandas for schema inference."""
    try:
        import pandas as pd  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError("pandas is required to write parquet; install pandas + pyarrow") from exc

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
        default=os.environ.get("DESKTOP_API_BASE_URL", "http://172.31.13.38:2354"),
        help="Desktop env service base URL (default: $DESKTOP_API_BASE_URL or http://172.31.13.38:2354).",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("RL_PROXY_AUTH_TOKEN"),
        help="Optional Bearer token for rl-proxy /tasks (default: $RL_PROXY_AUTH_TOKEN).",
    )
    parser.add_argument("--domain", default=None, help="Filter tasks by domain (chrome/gimp/...).")
    parser.add_argument(
        "--task-file",
        default=DEFAULT_TASK_FILE,
        help=(
            "Local task subset JSON path or file:// URL passed to the desktop service /tasks API "
            f"(default: {DEFAULT_TASK_FILE}). In --from-proxy-splits mode, the default value is ignored; "
            "an explicitly supplied --task-file is used for both train and test unless overridden."
        ),
    )
    parser.add_argument(
        "--from-proxy-splits",
        action="store_true",
        help=(
            "Fetch train and test rows separately from the rl-proxy /tasks split API "
            "instead of locally splitting one task list."
        ),
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Proxy split name for train rows in --from-proxy-splits mode (default: train).",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Proxy split name for test rows in --from-proxy-splits mode (default: test; proxy aliases it to eval).",
    )
    parser.add_argument(
        "--train-task-file",
        default=None,
        help="Explicit task_file for train rows. Overrides --train-split when set.",
    )
    parser.add_argument(
        "--test-task-file",
        default=None,
        help="Explicit task_file for test rows. Overrides --test-split when set.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N tasks (after filter). 0 = all.")
    parser.add_argument(
        "--train-limit", type=int, default=0, help="Train limit in --from-proxy-splits mode. 0 falls back to --limit."
    )
    parser.add_argument(
        "--test-limit", type=int, default=0, help="Test limit in --from-proxy-splits mode. 0 falls back to --limit."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.95, help="Fraction of tasks for train split (default 0.95)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to embed in every row.",
    )
    parser.add_argument(
        "--use-chat-template-tools",
        action="store_true",
        help=(
            "Use a behavior-only system prompt and let the model chat template render tool definitions "
            "from the tools= argument. Use this for Qwen3.5/qwen3_coder style tool calls."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/efs/data/cua/rl/osworld",
        help="Directory to write train.parquet / test.parquet.",
    )
    parser.add_argument("--train-name", default="train.parquet", help="Train parquet filename inside --output-dir.")
    parser.add_argument("--test-name", default="test.parquet", help="Test parquet filename inside --output-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_file_was_explicit = any(arg == "--task-file" or arg.startswith("--task-file=") for arg in sys.argv[1:])
    system_prompt = CHAT_TEMPLATE_TOOLS_SYSTEM_PROMPT if args.use_chat_template_tools else args.system_prompt

    if args.from_proxy_splits:
        train_task_file = args.train_task_file
        test_task_file = args.test_task_file
        if args.task_file and task_file_was_explicit:
            if train_task_file is None:
                train_task_file = args.task_file
            if test_task_file is None:
                test_task_file = args.task_file
        train_tasks = fetch_tasks(
            args.api_base_url,
            args.domain,
            train_task_file,
            args.timeout,
            split=args.train_split,
            auth_token=args.auth_token,
        )
        test_tasks = fetch_tasks(
            args.api_base_url,
            args.domain,
            test_task_file,
            args.timeout,
            split=args.test_split,
            auth_token=args.auth_token,
        )
        if not train_tasks:
            print("[prepare_dataset] ERROR: proxy train split returned zero tasks", file=sys.stderr)
            return 1
        if not test_tasks:
            print("[prepare_dataset] ERROR: proxy test split returned zero tasks", file=sys.stderr)
            return 1

        train_limit = args.train_limit or args.limit
        test_limit = args.test_limit or args.limit
        train_tasks = limit_tasks(train_tasks, train_limit, label="train")
        test_tasks = limit_tasks(test_tasks, test_limit, label="test")
        train_rows = build_rows(train_tasks, split="train", system_prompt=system_prompt)
        test_rows = build_rows(test_tasks, split="test", system_prompt=system_prompt)
        print(
            f"[prepare_dataset] Proxy splits: train={len(train_rows)} "
            f"test={len(test_rows)} (train_split={args.train_split!r}, test_split={args.test_split!r})"
        )
    else:
        tasks = fetch_tasks(args.api_base_url, args.domain, args.task_file, args.timeout, auth_token=args.auth_token)
        if not tasks:
            print("[prepare_dataset] ERROR: /tasks returned zero tasks", file=sys.stderr)
            return 1

        tasks = limit_tasks(tasks, args.limit, label="tasks")
        rows = [build_row(task, index=i, system_prompt=system_prompt) for i, task in enumerate(tasks)]

        train_rows, test_rows = split_train_test(rows, args.train_ratio, args.seed)
        # Stamp each row's split label now that we know train vs test.
        for row in train_rows:
            row["extra_info"]["split"] = "train"
        for row in test_rows:
            row["extra_info"]["split"] = "test"
        print(
            f"[prepare_dataset] Local split: train={len(train_rows)} test={len(test_rows)} (ratio={args.train_ratio})"
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
        question = sample["extra_info"]["question"]
        print(f"  extra_info.question    = {question[:80]!r}{'...' if len(question) > 80 else ''}")
        ck = sample["extra_info"]["tools_kwargs"]["computer_use"]["create_kwargs"]
        print(f"  create_kwargs keys     = {sorted(ck.keys())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
