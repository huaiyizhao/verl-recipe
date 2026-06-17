from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import random
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlsplit


@dataclass
class TimedResult:
    ok: bool
    latency: float
    payload: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1


@dataclass(frozen=True)
class TaskChoice:
    task_id: str
    domain: str
    task_config_path: str


class Client:
    def __init__(self, base_url: str, *, auth_token: str | None = None, timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

    def get(self, path: str, *, timeout: float | None = None) -> TimedResult:
        return self._request("GET", path, None, timeout=timeout)

    def post(self, path: str, payload: dict[str, Any] | None = None, *, timeout: float | None = None) -> TimedResult:
        return self._request("POST", path, payload or {}, timeout=timeout)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None, *, timeout: float | None = None) -> TimedResult:
        url = f"{self.base_url}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        req = request.Request(url, data=body, headers=headers, method=method)
        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout if timeout is None else timeout) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return TimedResult(False, time.perf_counter() - start, error=f"HTTP {exc.code}: {raw}")
        except Exception as exc:
            return TimedResult(False, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
        return TimedResult(True, time.perf_counter() - start, data)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def summarize(name: str, results: list[TimedResult], wall_seconds: float | None = None) -> dict[str, Any]:
    latencies = [item.latency for item in results]
    ok = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    summary = {
        "name": name,
        "total": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "latency_seconds": {
            "min": min(latencies) if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "sample_errors": [item.error for item in failed[:5]],
    }
    if wall_seconds is not None:
        summary["wall_seconds"] = wall_seconds
        summary["throughput_per_second"] = len(results) / wall_seconds if wall_seconds > 0 else 0.0
    summary["total_attempts"] = sum(item.attempts for item in results)
    summary["retried_requests"] = sum(1 for item in results if item.attempts > 1)
    return summary


def is_timeout_result(result: TimedResult) -> bool:
    text = (result.error or "").lower()
    return "timeout" in text or "timed out" in text


def with_timeout_retries(call, *, retries: int) -> TimedResult:
    attempts = 0
    total_latency = 0.0
    last = TimedResult(False, 0.0, error="not attempted", attempts=0)
    for _ in range(max(0, retries) + 1):
        attempts += 1
        last = call()
        total_latency += last.latency
        if last.ok or not is_timeout_result(last):
            last.attempts = attempts
            last.latency = total_latency
            return last
    last.attempts = attempts
    last.latency = total_latency
    return last


# ---------------------------------------------------------------------------
# Phase-resolved HTTP timing (to pinpoint WHERE a request fails/stalls).
#   connect  : TCP handshake (connect refused/timeout -> network / worker accept)
#   ttfb     : request sent -> first response byte (server-side env.step work)
#   download : reading the response body (large 1080p screenshot transfer)
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    ok: bool
    phase_failed: str | None  # connect | send | ttfb | download | http_status | None
    error_kind: str | None    # refused | timeout | reset | broken_pipe | http | other | None
    t_connect: float = 0.0
    t_ttfb: float = 0.0
    t_download: float = 0.0
    total: float = 0.0
    status: int | None = None
    nbytes: int = 0
    attempts: int = 1
    target: str = ""
    error: str | None = None


def classify_error(exc: BaseException) -> str:
    s = f"{type(exc).__name__}: {exc}".lower()
    if "refused" in s:
        return "refused"
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "reset" in s:
        return "reset"
    if "broken pipe" in s:
        return "broken_pipe"
    if "not connected" in s or "cannot assign" in s:
        return "exhausted"
    return "other"


def phased_post(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    connect_timeout: float,
    total_timeout: float,
    auth_token: str | None = None,
) -> PhaseResult:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    conn = http.client.HTTPConnection(host, port, timeout=connect_timeout)
    target = f"{host}:{port}"
    start = time.perf_counter()
    phase = "connect"
    t_connect = t_ttfb = t_download = 0.0
    try:
        conn.connect()  # raises on refused / connect timeout / no route
        t_connect = time.perf_counter() - start
        if conn.sock is not None:
            conn.sock.settimeout(total_timeout)  # subsequent read/write deadline
        phase = "send"
        conn.request("POST", path, body=body, headers=headers)
        phase = "ttfb"
        m = time.perf_counter()
        resp = conn.getresponse()  # blocks until status line + headers (server work done)
        t_ttfb = time.perf_counter() - m
        status = resp.status
        phase = "download"
        d = time.perf_counter()
        data = resp.read()  # full body (screenshot) transfer
        t_download = time.perf_counter() - d
        total = time.perf_counter() - start
        ok = 200 <= status < 300
        return PhaseResult(
            ok=ok,
            phase_failed=None if ok else "http_status",
            error_kind=None if ok else "http",
            t_connect=t_connect, t_ttfb=t_ttfb, t_download=t_download, total=total,
            status=status, nbytes=len(data), target=target,
            error=None if ok else f"HTTP {status}: {data[:200]!r}",
        )
    except Exception as exc:
        return PhaseResult(
            ok=False, phase_failed=phase, error_kind=classify_error(exc),
            t_connect=t_connect, t_ttfb=t_ttfb, t_download=t_download,
            total=time.perf_counter() - start, status=None, nbytes=0, target=target,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def phased_post_with_retries(
    host: str, port: int, path: str, payload: dict[str, Any], *,
    connect_timeout: float, total_timeout: float, auth_token: str | None,
    connect_retries: int, retry_backoff: float = 0.5,
) -> PhaseResult:
    attempts = 0
    last: PhaseResult | None = None
    for _ in range(max(0, connect_retries) + 1):
        attempts += 1
        r = phased_post(host, port, path, payload, connect_timeout=connect_timeout,
                        total_timeout=total_timeout, auth_token=auth_token)
        # Only retry true connect-phase failures (refused/timeout), mirroring the
        # real client's connect-retry policy. A slow/failed response is NOT retried.
        if r.ok or r.phase_failed != "connect":
            r.attempts = attempts
            return r
        last = r
        time.sleep(retry_backoff)
    if last is not None:
        last.attempts = attempts
    return last  # type: ignore[return-value]


def summarize_phased(name: str, results: list[PhaseResult], wall_seconds: float | None = None) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    def pstats(vals: list[float]) -> dict[str, float]:
        return {
            "min": min(vals) if vals else 0.0,
            "mean": statistics.fmean(vals) if vals else 0.0,
            "p50": percentile(vals, 50), "p95": percentile(vals, 95),
            "p99": percentile(vals, 99), "max": max(vals) if vals else 0.0,
        }

    # failure breakdown by (phase, error_kind)
    fail_breakdown: dict[str, int] = {}
    for r in failed:
        key = f"{r.phase_failed}:{r.error_kind}"
        fail_breakdown[key] = fail_breakdown.get(key, 0) + 1

    summary: dict[str, Any] = {
        "name": name,
        "total": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "failed_by_phase": dict(sorted(fail_breakdown.items(), key=lambda kv: -kv[1])),
        "phase_latency_ok": {
            "connect": pstats([r.t_connect for r in ok]),
            "ttfb": pstats([r.t_ttfb for r in ok]),
            "download": pstats([r.t_download for r in ok]),
            "total": pstats([r.total for r in ok]),
        },
        # how long the FAILED ones ran before giving up, per phase
        "phase_latency_failed": {
            "connect_fail_total": pstats([r.total for r in failed if r.phase_failed == "connect"]),
            "ttfb_fail_total": pstats([r.total for r in failed if r.phase_failed == "ttfb"]),
            "download_fail_total": pstats([r.total for r in failed if r.phase_failed == "download"]),
        },
        "response_bytes": pstats([float(r.nbytes) for r in ok]),
        "retried_requests": sum(1 for r in results if r.attempts > 1),
        "total_attempts": sum(r.attempts for r in results),
        "sample_errors": [f"{r.target} {r.phase_failed}/{r.error_kind}: {r.error}" for r in failed[:8]],
    }
    if wall_seconds is not None and wall_seconds > 0:
        summary["wall_seconds"] = wall_seconds
        summary["throughput_per_second"] = len(results) / wall_seconds
    return summary


def load_task_choices(examples_dir: Path, *, domain: str | None = None) -> list[TaskChoice]:
    root = examples_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"examples dir not found: {root}")
    pattern = f"{domain}/*.json" if domain else "*/*.json"
    choices: list[TaskChoice] = []
    for path in sorted(root.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_id = str(payload.get("id") or path.stem)
        choices.append(TaskChoice(task_id=task_id, domain=path.parent.name, task_config_path=str(path)))
    if not choices:
        raise ValueError(f"no task json files found in {root} domain={domain!r}")
    return choices


def load_task_choices_from_split(split_file: Path, examples_dir: Path, *, domain: str | None = None) -> list[TaskChoice]:
    """Load benchmark task choices restricted to the ids listed in a split file.

    The split file is a JSON mapping of ``domain -> [task_id, ...]`` (e.g.
    ``evaluation_examples/test_stable.json``). Each task config is resolved to
    ``<examples_dir>/<domain>/<task_id>.json``.
    """
    split_path = split_file.expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"split file not found: {split_path}")
    examples_root = examples_dir.expanduser().resolve()
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"split file must be a mapping of domain -> task ids: {split_path}")
    choices: list[TaskChoice] = []
    for split_domain, task_ids in data.items():
        if domain and split_domain != domain:
            continue
        if not isinstance(task_ids, list):
            continue
        for raw_task_id in task_ids:
            task_id = str(raw_task_id)
            config_path = examples_root / str(split_domain) / f"{task_id}.json"
            choices.append(TaskChoice(task_id=task_id, domain=str(split_domain), task_config_path=str(config_path)))
    if not choices:
        raise ValueError(f"no tasks found in split {split_path} domain={domain!r}")
    return choices


def random_action(rng: random.Random) -> str:
    actions = [
        lambda: f"pyautogui.sleep({rng.uniform(0.03, 0.2):.3f})",
        lambda: f"pyautogui.moveRel({rng.randint(-80, 80)}, {rng.randint(-80, 80)}, duration=0)",
        lambda: f"pyautogui.scroll({rng.choice([-3, -2, -1, 1, 2, 3])})",
        lambda: f"pyautogui.press('{rng.choice(['esc', 'tab', 'shift', 'ctrl', 'alt'])}')",
        lambda: f"pyautogui.hotkey('{rng.choice(['ctrl', 'alt'])}', '{rng.choice(['tab', 'esc'])}')",
    ]
    return rng.choice(actions)()


def wait_for_ready_pool(client: Client, *, min_ready: int, timeout: float, interval: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        result = client.get("/admin/pool")
        if result.ok:
            last = result.payload or {}
            pool = last.get("pool") or {}
            if int(pool.get("ready") or 0) >= min_ready:
                return last
        time.sleep(interval)
    raise TimeoutError(f"pool ready did not reach {min_ready} within {timeout}s; last={last}")


def run_parallel(items, workers: int, fn):
    results = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark advanced AWS RL server create/step/evaluate throughput and latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:2354")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--task-id", default=None, help="Use one fixed task id. If omitted, choose random task ids from the split file.")
    parser.add_argument("--examples-dir", default="evaluation_examples/examples")
    parser.add_argument("--split-file", default="evaluation_examples/test_stable.json", help="JSON mapping of domain -> task ids; create tasks are restricted to these.")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--sessions", type=int, default=160)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--create-workers", type=int, default=160)
    parser.add_argument("--step-workers", type=int, default=160)
    parser.add_argument("--evaluate-workers", type=int, default=160)
    parser.add_argument("--min-ready", type=int, default=11)
    parser.add_argument("--pool-wait-timeout", type=float, default=7200.0)
    parser.add_argument("--pool-poll-interval", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--create-timeout", type=float, default=450.0)
    parser.add_argument("--step-timeout", type=float, default=15.0)
    parser.add_argument("--evaluate-timeout", type=float, default=15.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="Per-attempt TCP connect timeout for phased step/evaluate.")
    parser.add_argument("--worker-route", dest="worker_route", action="store_true", default=True,
                        help="B2: route step/evaluate to the per-session worker_port returned by /session/create (default on).")
    parser.add_argument("--no-worker-route", dest="worker_route", action="store_false",
                        help="Send step/evaluate to the central proxy instead of the worker port.")
    parser.add_argument("--step-retries", type=int, default=2)
    parser.add_argument("--evaluate-retries", type=int, default=2)
    parser.add_argument("--step-interval-min", type=float, default=5.0, help="Min random delay (seconds) between consecutive steps within a session.")
    parser.add_argument("--step-interval-max", type=float, default=10.0, help="Max random delay (seconds) between consecutive steps within a session.")
    parser.add_argument("--evaluate-client-sleep-min", type=float, default=0.0)
    parser.add_argument("--evaluate-client-sleep-max", type=float, default=2.0)
    parser.add_argument("--action", default=None, help="Use one fixed step action. If omitted, generate random pyautogui actions.")
    parser.add_argument("--no-close", action="store_true")
    args = parser.parse_args()

    if args.evaluate_client_sleep_min < 0 or args.evaluate_client_sleep_max < 0:
        raise ValueError("evaluate client sleep bounds must be non-negative")
    if args.evaluate_client_sleep_max < args.evaluate_client_sleep_min:
        raise ValueError("evaluate client sleep max must be >= min")
    if args.step_interval_min < 0 or args.step_interval_max < 0:
        raise ValueError("step interval bounds must be non-negative")
    if args.step_interval_max < args.step_interval_min:
        raise ValueError("step interval max must be >= min")

    rng = random.Random(args.random_seed)
    task_choices = load_task_choices_from_split(Path(args.split_file), Path(args.examples_dir), domain=args.domain)

    client = Client(args.base_url, auth_token=args.auth_token, timeout=args.request_timeout)
    scenario_start = time.perf_counter()
    pool_before = wait_for_ready_pool(client, min_ready=args.min_ready, timeout=args.pool_wait_timeout, interval=args.pool_poll_interval)
    print(json.dumps({"event": "pool_ready", "pool": pool_before.get("pool")}, indent=2, ensure_ascii=False), flush=True)

    session_ids = [f"bench-{uuid.uuid4()}" for _ in range(args.sessions)]
    session_tasks: dict[str, TaskChoice] = {}
    for session_id in session_ids:
        if args.task_id:
            matching = [item for item in task_choices if item.task_id == args.task_id]
            task = rng.choice(matching) if matching else TaskChoice(task_id=args.task_id, domain=args.domain or "", task_config_path="")
        else:
            task = rng.choice(task_choices)
        session_tasks[session_id] = task

    def create_one(session_id: str) -> TimedResult:
        task = session_tasks[session_id]
        payload = {"session_id": session_id, "task_id": task.task_id}
        if task.domain:
            payload["domain"] = task.domain
        if task.task_config_path:
            payload["task_config_path"] = task.task_config_path
        return client.post("/session/create", payload, timeout=args.create_timeout)

    create_start = time.perf_counter()
    create_results = run_parallel(session_ids, args.create_workers, create_one)
    create_wall = time.perf_counter() - create_start
    created_sessions = [item.payload["session_id"] for item in create_results if item.ok and item.payload and item.payload.get("session_id")]

    # B2: map each session to the (host, port) its hot-path step/evaluate should
    # target. The worker_port (if any) is on the SAME host as the central proxy.
    proxy_host = urlsplit(args.base_url).hostname or "127.0.0.1"
    proxy_port = urlsplit(args.base_url).port or 2354
    session_target: dict[str, tuple[str, int]] = {}
    worker_routed = 0
    for item in create_results:
        if not (item.ok and item.payload):
            continue
        sid = item.payload.get("session_id")
        if not sid:
            continue
        wp = item.payload.get("worker_port")
        if args.worker_route and wp:
            session_target[sid] = (proxy_host, int(wp))
            worker_routed += 1
        else:
            session_target[sid] = (proxy_host, proxy_port)
    print(json.dumps({
        "event": "routing",
        "worker_route_enabled": args.worker_route,
        "created": len(created_sessions),
        "worker_routed": worker_routed,
        "central_routed": len(created_sessions) - worker_routed,
        "sample_targets": [f"{s}->{h}:{p}" for s, (h, p) in list(session_target.items())[:5]],
    }, indent=2, ensure_ascii=False), flush=True)

    def run_session_steps(session_id: str) -> list[PhaseResult]:
        # Steps within a session run sequentially, with a random delay between
        # consecutive steps to emulate an agent thinking between actions.
        local_rng = random.Random(session_id if args.random_seed is None else f"{args.random_seed}:{session_id}")
        host, port = session_target.get(session_id, (proxy_host, proxy_port))
        results: list[PhaseResult] = []
        for step_index in range(args.steps):
            if step_index > 0 and args.step_interval_max > 0:
                time.sleep(local_rng.uniform(args.step_interval_min, args.step_interval_max))
            action = args.action or random_action(local_rng)
            request_id = f"{session_id}-step-{step_index}"
            result = phased_post_with_retries(
                host, port, f"/session/{session_id}/step",
                {"request_id": request_id, "action": action, "pause": 0},
                connect_timeout=args.connect_timeout, total_timeout=args.step_timeout,
                auth_token=args.auth_token, connect_retries=args.step_retries,
            )
            results.append(result)
        return results

    step_start = time.perf_counter()
    session_step_results = run_parallel(created_sessions, args.step_workers, run_session_steps)
    step_results = [result for session_results in session_step_results for result in session_results]
    step_wall = time.perf_counter() - step_start

    def evaluate_one(item: tuple[str, float]) -> PhaseResult:
        session_id, sleep_seconds = item
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        host, port = session_target.get(session_id, (proxy_host, proxy_port))
        return phased_post_with_retries(
            host, port, f"/session/{session_id}/evaluate",
            {"settle_seconds": 0},
            connect_timeout=args.connect_timeout, total_timeout=args.evaluate_timeout,
            auth_token=args.auth_token, connect_retries=args.evaluate_retries,
        )

    evaluate_items = [
        (session_id, rng.uniform(args.evaluate_client_sleep_min, args.evaluate_client_sleep_max))
        for session_id in created_sessions
    ]
    evaluate_start = time.perf_counter()
    evaluate_results = run_parallel(evaluate_items, args.evaluate_workers, evaluate_one)
    evaluate_wall = time.perf_counter() - evaluate_start

    close_results: list[TimedResult] = []
    close_wall = 0.0
    if not args.no_close:
        def close_one(session_id: str) -> TimedResult:
            return client.post(f"/session/{session_id}/close", {})

        close_start = time.perf_counter()
        close_results = run_parallel(created_sessions, args.evaluate_workers, close_one)
        close_wall = time.perf_counter() - close_start

    scenario_wall = time.perf_counter() - scenario_start
    report = {
        "scenario": {
            "sessions_requested": args.sessions,
            "sessions_created": len(created_sessions),
            "steps_per_session": args.steps,
            "total_step_requests": len(step_results),
            "random_seed": args.random_seed,
            "task_selection": "fixed" if args.task_id else "random_from_split",
            "split_file": None if args.task_id else args.split_file,
            "unique_task_ids": len({task.task_id for task in session_tasks.values()}),
            "sample_task_ids": sorted({task.task_id for task in session_tasks.values()})[:10],
            "action_mode": "fixed" if args.action else "random",
            "sample_actions": [args.action] if args.action else [random_action(random.Random(index)) for index in range(min(10, max(1, args.steps)))],
            "step_interval_seconds": {"min": args.step_interval_min, "max": args.step_interval_max},
            "timeouts_seconds": {"create": args.create_timeout, "step": args.step_timeout, "evaluate": args.evaluate_timeout},
            "retries": {"step": args.step_retries, "evaluate": args.evaluate_retries},
            "evaluate_client_sleep_seconds": {
                "min": args.evaluate_client_sleep_min,
                "max": args.evaluate_client_sleep_max,
                "sample": [round(item[1], 3) for item in evaluate_items[:10]],
            },
            "scenario_wall_seconds": scenario_wall,
        },
        "create": summarize("create", create_results, create_wall),
        "step": summarize_phased("step", step_results, step_wall),
        "evaluate": summarize_phased("evaluate", evaluate_results, evaluate_wall),
        "close": summarize("close", close_results, close_wall) if close_results else None,
        "pool_after": client.get("/admin/pool").payload,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if len(created_sessions) == args.sessions and all(item.ok for item in step_results) and all(item.ok for item in evaluate_results) else 2


if __name__ == "__main__":
    sys.exit(main())
