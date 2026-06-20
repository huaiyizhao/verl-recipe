import asyncio
import importlib.util
import json
import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

from io import BytesIO
from types import SimpleNamespace

import aiohttp
from yarl import URL


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_ROOT = REPO_ROOT / "fully_async_gui_agent"
VERL_ASYNC_ROOT = REPO_ROOT.parent / "verl-async"
if str(VERL_ASYNC_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ASYNC_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_dataset = _load_module("prepare_dataset_under_test", RECIPE_ROOT / "prepare_dataset.py")
desktop_env_tool = _load_module("desktop_env_tool_under_test", RECIPE_ROOT / "desktop_env_tool.py")

_ONE_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _observation_with_screenshot() -> dict:
    return {"observation": {"screenshot": _ONE_PIXEL_PNG_B64}}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_fetch_tasks_uses_proxy_split_and_auth_header():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse(
            {
                "tasks": [
                    {"task_id": "task-1", "domain": "chrome", "instruction": "do it"},
                ],
                "total": 1,
            }
        )

    with patch.object(prepare_dataset.urllib.request, "urlopen", fake_urlopen):
        tasks = prepare_dataset.fetch_tasks(
            "http://proxy.example:2354",
            "chrome",
            None,
            17,
            split="train",
            auth_token="secret",
        )

    assert tasks[0]["task_id"] == "task-1"
    assert captured["timeout"] == 17
    assert captured["authorization"] == "Bearer secret"
    assert captured["url"] == "http://proxy.example:2354/tasks?domain=chrome&split=train"


def test_fetch_tasks_prints_http_error_body(capsys):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=BytesIO(b'{"detail":"no task file configured"}'),
        )

    with patch.object(prepare_dataset.urllib.request, "urlopen", fake_urlopen):
        try:
            prepare_dataset.fetch_tasks("http://proxy.example:2354", None, None, 17, split="eval")
        except urllib.error.HTTPError:
            pass
        else:
            raise AssertionError("fetch_tasks should re-raise HTTPError")

    captured = capsys.readouterr()
    assert "HTTP error from /tasks" in captured.err
    assert "status=404" in captured.err
    assert "no task file configured" in captured.err


def test_build_rows_stamps_proxy_split():
    rows = prepare_dataset.build_rows(
        [{"task_id": "task-1", "domain": "gimp", "instruction": "paint"}],
        split="test",
        system_prompt="system",
    )

    assert rows[0]["extra_info"]["split"] == "test"
    assert rows[0]["extra_info"]["index"] == 0
    assert rows[0]["extra_info"]["tools_kwargs"]["computer_use"]["create_kwargs"]["task_id"] == "task-1"


def test_step_payload_forwards_server_timeout_seconds():
    async def run_step():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}
        calls = []

        async def fake_post(path, payload=None, timeout=None, **kwargs):
            calls.append((path, payload, timeout, kwargs))
            return {**_observation_with_screenshot(), "done": False, "step_count": 1}

        tool._post_with_retries = fake_post
        await tool.execute("instance-1", {"action": "wait"})
        return calls

    calls = asyncio.run(run_step())
    path, payload, timeout, kwargs = calls[0]

    assert path == "/session/session-1/step"
    assert payload["action"] == "WAIT"
    assert payload["timeout_seconds"] == 360
    assert timeout.total == 400
    assert "retry_timeout_only" not in kwargs


def test_slow_step_logs_successful_step_over_threshold():
    async def run_step():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
                "slow_step_log_threshold": 30,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}

        async def fake_post(*args, **kwargs):
            return {**_observation_with_screenshot(), "done": False, "step_count": 7}

        logs = []

        def fake_log(msg, *, level="DEBUG", debug=None):
            logs.append((msg, level, debug))

        tool._post_with_retries = fake_post
        with (
            patch.object(desktop_env_tool.time, "monotonic", side_effect=[100.0, 131.25]),
            patch.object(desktop_env_tool, "_log", fake_log),
        ):
            await tool.execute("instance-1", {"action": "wait"})
        return logs

    logs = asyncio.run(run_step())
    slow_logs = [entry for entry in logs if "[DesktopEnvTool][SLOW_STEP]" in entry[0]]

    assert len(slow_logs) == 1
    assert slow_logs[0][1] == "ERROR"
    assert "elapsed=31.250s" in slow_logs[0][0]
    assert "action='wait'" in slow_logs[0][0]
    assert "actual_action='WAIT'" in slow_logs[0][0]
    assert "step_count=7" in slow_logs[0][0]


def test_step_failure_is_reported_as_desktop_env_step_error():
    async def run_step():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}

        async def fake_post(*args, **kwargs):
            raise RuntimeError("worker step failed")

        tool._post_with_retries = fake_post
        try:
            await tool.execute("instance-1", {"action": "wait"})
        except desktop_env_tool.DesktopEnvStepError as exc:
            return exc
        raise AssertionError("execute should raise DesktopEnvStepError")

    exc = asyncio.run(run_step())

    assert "/step failed for action='wait'" in str(exc)


def test_step_without_screenshot_recovers_with_wait_step():
    async def run_step():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}
        calls = []

        async def fake_post(path, payload=None, timeout=None, **kwargs):
            calls.append((path, payload, timeout, kwargs))
            if len(calls) == 1:
                return {"observation": {}, "done": False, "step_count": 1}
            return {**_observation_with_screenshot(), "done": False, "step_count": 2}

        tool._post_with_retries = fake_post
        response, reward, meta = await tool.execute("instance-1", {"action": "wait"})
        return response, reward, meta, calls

    response, reward, meta, calls = asyncio.run(run_step())

    assert len(response.image) == 1
    assert reward == 0.0
    assert meta["step_count"] == 1
    assert len(calls) == 2
    assert calls[0][1]["action"] == "WAIT"
    assert calls[1][1]["action"] == "WAIT"
    assert calls[1][1]["pause"] == 2.0


def test_create_without_screenshot_recovers_with_wait_step():
    async def run_create():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid:2354",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        calls = []

        async def fake_post(path, payload=None, timeout=None, **kwargs):
            calls.append((path, payload, timeout, kwargs))
            if path == "/session/create":
                return {
                    "session_id": "instance-1",
                    "worker_port": 12345,
                    "observation": {},
                }
            return {**_observation_with_screenshot(), "done": False, "step_count": 1}

        tool._post_with_retries = fake_post
        instance_id, response = await tool.create("instance-1", {"task_id": "task-1"})
        return instance_id, response, calls

    instance_id, response, calls = asyncio.run(run_create())

    assert instance_id == "instance-1"
    assert len(response.image) == 1
    assert len(calls) == 2
    assert calls[0][0] == "/session/create"
    assert calls[1][0] == "/session/instance-1/step"
    assert calls[1][1]["action"] == "WAIT"
    assert calls[1][1]["pause"] == 2.0


def test_step_without_screenshot_after_recovery_is_reported_as_desktop_env_step_error():
    async def run_step():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}

        async def fake_post(*args, **kwargs):
            return {"observation": {}, "done": False, "step_count": 1}

        tool._post_with_retries = fake_post
        try:
            await tool.execute("instance-1", {"action": "wait"})
        except desktop_env_tool.DesktopEnvStepError as exc:
            return exc
        raise AssertionError("execute should raise DesktopEnvStepError")

    exc = asyncio.run(run_step())

    assert "recovery WAIT returned no valid screenshot" in str(exc)


def test_terminate_without_screenshot_does_not_fail():
    async def run_step():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}

        async def fake_post(*args, **kwargs):
            return {"observation": {}, "done": True, "step_count": 1}

        tool._post_with_retries = fake_post
        return await tool.execute("instance-1", {"action": "terminate", "status": "success"})

    response, reward, meta = asyncio.run(run_step())

    assert response.image == []
    assert reward == 0.0
    assert meta["action"] == "terminate"
    assert meta["done"] is True


def test_http_error_detail_includes_proxy_detail():
    exc = aiohttp.ClientResponseError(
        request_info=SimpleNamespace(url=URL("http://proxy.example/session/s1/step")),
        history=(),
        status=500,
        message='{"detail":"worker command timed out"}',
        headers=None,
    )

    detail = desktop_env_tool.DesktopEnvTool._error_detail(exc)

    assert "status=500" in detail
    assert "worker command timed out" in detail


def test_screenshot_failure_returns_empty_images():
    async def run_screenshot():
        tool = desktop_env_tool.DesktopEnvTool(
            {
                "api_base_url": "http://desktop.invalid",
                "step_timeout": 400,
                "step_server_timeout": 360,
            }
        )
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}

        async def fake_post(*args, **kwargs):
            raise RuntimeError("screenshot step failed")

        tool._post_with_retries = fake_post
        return await tool.screenshot("instance-1")

    assert asyncio.run(run_screenshot()) == []


def test_desktop_tool_reads_proxy_auth_token_env():
    with patch.dict(os.environ, {"RL_PROXY_AUTH_TOKEN": "proxy-secret"}, clear=True):
        tool = desktop_env_tool.DesktopEnvTool({"api_base_url": "http://desktop.invalid"})

    assert tool.auth_token == "proxy-secret"
