import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


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
            return {"observation": {}, "done": False, "step_count": 1}

        tool._post_with_retries = fake_post
        await tool.execute("instance-1", {"action": "wait"})
        return calls

    calls = asyncio.run(run_step())
    path, payload, timeout, kwargs = calls[0]

    assert path == "/session/session-1/step"
    assert payload["action"] == "WAIT"
    assert payload["timeout_seconds"] == 360
    assert timeout.total == 400
    assert kwargs["retry_timeout_only"] is True


def test_desktop_tool_reads_proxy_auth_token_env():
    with patch.dict(os.environ, {"RL_PROXY_AUTH_TOKEN": "proxy-secret"}, clear=True):
        tool = desktop_env_tool.DesktopEnvTool({"api_base_url": "http://desktop.invalid"})

    assert tool.auth_token == "proxy-secret"
