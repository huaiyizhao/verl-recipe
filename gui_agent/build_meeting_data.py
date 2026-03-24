"""
Preprocess the CU Meeting dataset to parquet format for end-to-end GUI agent training.

Produces training data compatible with verl's GUIAgentLoop + MCPDesktopEnvTool.
The agent is trained end-to-end — the task instruction is embedded directly in
the prompt so the model learns to complete it autonomously (no observer needed).

Usage:
    python -m recipe.gui_agent.build_meeting_data \
        --base_url http://your-meeting-app \
        --address_base_url http://your-mcp-manager

    python -m recipe.gui_agent.build_meeting_data \
        --max_count 256 --test_ratio 0.1 --local_save_dir /efs/data/cua/rl
"""

import argparse
import asyncio
import os
import random
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import aiohttp
import datasets


DEFAULT_SOURCE_URL = "https://meeting.woa.com"
DEFAULT_BASE_URL = "http://ns008-cu-meeting-svc"
DEFAULT_ADDRESS_BASE_URL = "http://ns008-cu-manager-svc"

GROUND_TRUTH_KEYS = (
    "building_id", "building_name", "city_name", "query_date",
    "floor_id", "floor_name", "room_id", "room_name",
    "beg", "end", "valid", "people", "date", "title", "query",
)


# ---------------------------------------------------------------------------
# Data collection from the meeting API
# ---------------------------------------------------------------------------


async def _fetch_one(
    session: aiohttp.ClientSession,
    api_url: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Send a single API request, return parsed sample or None."""
    async with sem:
        try:
            async with session.get(
                api_url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get("code") != 0:
                    return None
                result = data.get("result")
                if not result:
                    return None
                if result.get("building_id") != 436:
                    return None
                if result.get("date") == "今天":
                    return None
                if result.get("valid") is not True:
                    return None
                params = parse_qs(urlparse(str(resp.url)).query)
                return {
                    "result": result,
                    "seed": params.get("seed", [None])[0],
                    "qseed": params.get("qseed", [None])[0],
                    "mock_date": parse_qs(urlparse(api_url).query).get("_mockdate_", [None])[0],
                }
        except Exception:
            return None


async def poll_cu_api(
    base_url: str = DEFAULT_BASE_URL,
    max_count: int = 128,
    max_attempts: int = 10000,
    concurrency: int = 32,
) -> list[dict]:
    """Poll the CU API concurrently to collect meeting room booking tasks."""
    mock_date = datetime.now().strftime("%Y-%m-%d")
    api_url = f"{base_url}/api/__task__?_mockdate_={mock_date}"
    valid_samples: list[dict] = []
    total_attempts = 0
    batch_no = 0

    print(f"Polling {api_url}  target={max_count}, concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while len(valid_samples) < max_count and total_attempts < max_attempts:
            batch = min(concurrency, max_attempts - total_attempts)
            results = await asyncio.gather(*[_fetch_one(session, api_url, sem) for _ in range(batch)])
            total_attempts += batch
            batch_no += 1

            ok = sum(1 for r in results if r is not None)
            for r in results:
                if r is not None and len(valid_samples) < max_count:
                    valid_samples.append(r)

            print(
                f"[Batch {batch_no}] success={ok}, fail={batch - ok}, "
                f"total_valid={len(valid_samples)}/{max_count}"
            )

    print(f"Done: {len(valid_samples)} samples in {total_attempts} attempts")
    return valid_samples


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _build_ground_truth(sample: dict, base_url: str) -> dict:
    """Extract ground truth fields from a sample.

    These fields are used by MCPDesktopEnvTool.calc_reward() to validate
    task completion via ``GET <base_url>/api/__task__?seed=...&check=true``.
    """
    result = sample["result"]
    gt = {k: result.get(k) for k in GROUND_TRUTH_KEYS}
    gt.update(
        seed=sample.get("seed"),
        qseed=sample.get("qseed"),
        mock_date=sample.get("mock_date"),
        base_url=base_url,
        # `session` is injected at runtime by the MCP environment via url_rewrite
    )
    return gt


def _build_dataset_sample(
    idx: int,
    sample: dict,
    base_url: str,
    address_config: dict,
    split: str = "train",
) -> dict:
    """Convert a raw API sample to the verl dataset schema.

    Field consumers:
        - ``RLHFDataset``: data_source, prompt, extra_info, reward_model
        - ``AgentLoopWorker``: agent_name (selects @register'd loop)
        - ``GUIAgentLoop.run()``: raw_prompt (built from prompt), extra_info.task_id,
          tools_kwargs.computer_use.create_kwargs.ground_truth
        - ``MCPDesktopEnvTool.create()``: create_kwargs.task_id
        - ``MCPDesktopEnvTool.calc_reward()``: ground_truth.{seed, qseed, session, base_url}
    """
    query = sample["result"]["query"]
    gt = _build_ground_truth(sample, base_url)

    create_kwargs = {
        "ground_truth": gt,
        "address_config": address_config,
        "keep_last_k_images": 3,
        "url_rewrite": {
            "source": DEFAULT_SOURCE_URL,
            "target": base_url,
            "args": [
                {"key": "seed", "param": "seed"},
                {"key": "mock_date", "param": "_mockdate_"},
                {"key": "session", "param": "_session_"},
            ],
        },
    }

    return {
        # --- verl core fields ---
        "data_source": "meeting",
        "agent_name": "gui_agent",  # matches @register("gui_agent") in gui_agent_loop.py
        "prompt": [],  # GUI agent loop constructs all messages; task query is in extra_info["question"]
        "ability": "cu",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "seed": gt["seed"],
                "qseed": gt["qseed"],
                "mock_date": gt["mock_date"],
                "base_url": base_url,
            },
        },
        # --- extra_info: consumed by RLHFDataset + GUIAgentLoop ---
        "extra_info": {
            "split": split,
            "index": idx,
            "task_id": f"meeting_{gt['seed']}_{gt['qseed']}",
            "question": query,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "computer_use": {
                    "create_kwargs": create_kwargs,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poll CU Meeting API and build parquet dataset for end-to-end GUI agent training"
    )
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--address_base_url", default=DEFAULT_ADDRESS_BASE_URL)
    parser.add_argument("--address_env", default="a4861800")
    parser.add_argument("--address_namespace", default="Development")
    parser.add_argument("--address_expire_min", type=int, default=60)
    parser.add_argument("--address_retry_interval", type=float, default=1.0)
    parser.add_argument("--address_max_retries", type=int, default=60)
    parser.add_argument("--local_save_dir", default="/efs/data/cua/rl")
    parser.add_argument("--max_count", type=int, default=128,
                        help="Total number of samples to collect (split into train + test)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of samples to use for test split (default: 0.1)")
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    valid_samples = asyncio.run(poll_cu_api(
        base_url=args.base_url,
        max_count=args.max_count,
        concurrency=args.concurrency,
    ))

    if not valid_samples:
        print("No valid samples collected. Exiting.")
        exit(1)

    address_config = {
        "base_url": args.address_base_url,
        "env": args.address_env,
        "namespace": args.address_namespace,
        "expire_min": args.address_expire_min,
        "retry_interval": args.address_retry_interval,
        "max_retries": args.address_max_retries,
    }

    # Split into train / test
    random.seed(42)
    random.shuffle(valid_samples)
    n_test = max(1, int(len(valid_samples) * args.test_ratio))
    test_samples = valid_samples[:n_test]
    train_samples = valid_samples[n_test:]

    for split_name, samples in [("train", train_samples), ("test", test_samples)]:
        dataset_rows = [
            _build_dataset_sample(i, s, args.base_url, address_config, split=split_name)
            for i, s in enumerate(samples)
        ]
        ds = datasets.Dataset.from_list(dataset_rows)
        output_path = os.path.join(save_dir, f"{split_name}.parquet")
        ds.to_parquet(output_path)
        print(f"\n{split_name} dataset saved to: {output_path}  ({len(dataset_rows)} samples)")

    # Print a sample for verification
    print("\n--- Sample train row ---")
    s = _build_dataset_sample(0, train_samples[0], args.base_url, address_config, split="train")
    print(f"  data_source: {s['data_source']}")
    print(f"  agent_name:  {s['agent_name']}")
    print(f"  prompt:      {s['prompt']}")
    print(f"  task_id:     {s['extra_info']['task_id']}")
    gt_keys = list(s["extra_info"]["tools_kwargs"]["computer_use"]["create_kwargs"]["ground_truth"].keys())
    print(f"  ground_truth keys: {gt_keys}")
