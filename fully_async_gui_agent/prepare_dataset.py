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


DEFAULT_SYSTEM_PROMPT = (
    "You are a GUI agent controlling a Linux desktop. You will be shown a "
    "screenshot of the current screen at each step and must call the "
    "`computer_use` tool to interact with the desktop (click, type, scroll, "
    "keyboard shortcuts, etc.). When the task is complete, call `computer_use` "
    "with ``action=terminate`` and ``status=success`` or ``status=failure``."
)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; no extra dependency)
# ---------------------------------------------------------------------------


def fetch_tasks(api_base_url: str, domain: str | None, timeout: int) -> list[dict[str, Any]]:
    url = api_base_url.rstrip("/") + "/tasks"
    if domain:
        url += "?" + urllib.parse.urlencode({"domain": domain})

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


def build_row(task: dict[str, Any], index: int, system_prompt: str) -> dict[str, Any]:
    """Build one parquet row for a single task."""
    task_id = task["task_id"]
    instruction = task.get("instruction") or ""
    domain = task.get("domain") or "unknown"

    # Initial chat: system + first user turn. The agent loop will append the
    # initial screenshot (as a user message) after it creates the desktop
    # session, so the dataset only needs the textual kickoff.
    prompt: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    extra_info: dict[str, Any] = {
        "task_id": task_id,
        "question": instruction,
        "domain": domain,
        "index": index,
    }

    return {"prompt": prompt, "extra_info": extra_info}


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
        default=os.environ.get("DESKTOP_API_BASE_URL", "http://10.192.64.33:2354"),
        help="Desktop env service base URL (default: $DESKTOP_API_BASE_URL or http://10.192.64.33:2354).",
    )
    parser.add_argument("--domain", default=None, help="Filter tasks by domain (chrome/gimp/...).")
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N tasks (after filter). 0 = all.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of tasks for train split (default 0.9).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to embed in every row.",
    )
    parser.add_argument(
        "--output-dir",
        default="/efs/data/cua/rl",
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

    tasks = fetch_tasks(args.api_base_url, args.domain, args.timeout)
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
        print(f"  prompt[0].role     = {sample['prompt'][0]['role']}")
        print(f"  prompt[1].content  = {sample['prompt'][1]['content'][:80]!r}...")
        print(f"  extra_info.task_id = {sample['extra_info']['task_id']}")
        print(f"  extra_info.domain  = {sample['extra_info']['domain']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
