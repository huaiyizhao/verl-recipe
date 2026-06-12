import asyncio
import importlib.util
import json
import sys
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[2]
VERL_ASYNC_ROOT = RECIPE_ROOT.parent / "verl-async"
sys.path.insert(0, str(VERL_ASYNC_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


desktop_env_tool = _load_module(
    "desktop_env_tool_under_test",
    RECIPE_ROOT / "fully_async_gui_agent" / "desktop_env_tool.py",
)
prepare_dataset = _load_module(
    "prepare_dataset_under_test",
    RECIPE_ROOT / "fully_async_gui_agent" / "prepare_dataset.py",
)


def _tool_json_from_prompt(prompt: str) -> dict:
    tool_text = prompt.rsplit("<tools>", 1)[1].split("</tools>", 1)[0]
    return json.loads(tool_text)


def test_prepare_dataset_prompt_uses_runtime_tool_schema():
    assert _tool_json_from_prompt(
        prepare_dataset.DEFAULT_SYSTEM_PROMPT
    ) == desktop_env_tool.build_computer_use_tool_dict()


def test_click_and_scroll_modifier_keys_are_held_during_action():
    click_code = desktop_env_tool._translate_action_to_pyautogui(
        {"action": "left_click", "coordinate": [500, 500], "keys": ["ctrl", "shift"]},
        1000,
        1000,
        1920,
        1080,
    )
    assert click_code == "\n".join(
        [
            "pyautogui.keyDown('ctrl')",
            "pyautogui.keyDown('shift')",
            "pyautogui.click(960, 540)",
            "pyautogui.keyUp('shift')",
            "pyautogui.keyUp('ctrl')",
        ]
    )

    scroll_code = desktop_env_tool._translate_action_to_pyautogui(
        {"action": "scroll", "coordinate": [250, 250], "pixels": -3, "keys": ["shift"]},
        1000,
        1000,
        1920,
        1080,
    )
    assert scroll_code == "\n".join(
        [
            "pyautogui.moveTo(480, 270)",
            "pyautogui.keyDown('shift')",
            "pyautogui.scroll(-3)",
            "pyautogui.keyUp('shift')",
        ]
    )


def test_hscroll_executes_pyautogui_hscroll():
    code = desktop_env_tool._translate_action_to_pyautogui(
        {"action": "hscroll", "pixels": 4, "keys": ["ctrl"]},
        1000,
        1000,
        1920,
        1080,
    )
    assert code == "\n".join(
        [
            "pyautogui.keyDown('ctrl')",
            "pyautogui.hscroll(4)",
            "pyautogui.keyUp('ctrl')",
        ]
    )


def test_owl_action_aliases_are_accepted():
    click_params = {"action": "click", "coordinate": [500, 500]}
    drag_params = {"action": "drag", "coordinate": [250, 250]}

    assert desktop_env_tool._validate_action_parameters(click_params) is None
    assert desktop_env_tool._validate_action_parameters(drag_params) is None

    click_code = desktop_env_tool._translate_action_to_pyautogui(
        click_params,
        1000,
        1000,
        1920,
        1080,
    )
    drag_code = desktop_env_tool._translate_action_to_pyautogui(
        drag_params,
        1000,
        1000,
        1920,
        1080,
    )

    assert click_code == "pyautogui.click(960, 540)"
    assert drag_code == "pyautogui.dragTo(480, 270, duration=0.5)"


def test_keys_must_be_an_array_for_modifier_actions():
    error = desktop_env_tool._validate_action_parameters(
        {"action": "left_click", "keys": "ctrl"}
    )
    assert error == "action 'left_click' requires keys as an array when provided"


def test_answer_action_marks_rollout_done_without_backend_step():
    async def run_answer():
        tool = desktop_env_tool.DesktopEnvTool({"api_base_url": "http://desktop.invalid"})
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}
        return await tool.execute(
            "instance-1",
            {"action": "answer", "text": "42"},
        )

    response, reward, info = asyncio.run(run_answer())

    assert response.text == "Answer: 42"
    assert reward == 0.0
    assert info["action"] == "answer"
    assert info["answer"] == "42"
    assert info["code"] == "DONE"
    assert info["done"] is True
    assert info["status"] == "success"


def test_answer_action_accepts_status_without_text():
    async def run_answer_status_only():
        tool = desktop_env_tool.DesktopEnvTool({"api_base_url": "http://desktop.invalid"})
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}
        return await tool.execute(
            "instance-1",
            {"action": "answer", "status": "failure"},
        )

    async def run_answer_without_args():
        tool = desktop_env_tool.DesktopEnvTool({"api_base_url": "http://desktop.invalid"})
        tool._instances["instance-1"] = {"session_id": "session-1", "task_id": "task-1"}
        return await tool.execute("instance-1", {"action": "answer"})

    assert desktop_env_tool._validate_action_parameters({"action": "answer", "status": "success"}) is None
    assert desktop_env_tool._validate_action_parameters({"action": "answer", "status": "failure"}) is None
    assert (
        desktop_env_tool._validate_action_parameters({"action": "answer", "status": "unknown"})
        == "action 'answer' requires status to be either 'success' or 'failure' when provided"
    )
    assert desktop_env_tool._validate_action_parameters({"action": "answer"}) is None

    response, reward, info = asyncio.run(run_answer_status_only())

    assert response.text == "Answer status: failure; code: FAIL"
    assert reward == 0.0
    assert info["action"] == "answer"
    assert info["answer"] == ""
    assert info["code"] == "FAIL"
    assert info["done"] is True
    assert info["status"] == "failure"

    response, reward, info = asyncio.run(run_answer_without_args())

    assert response.text == "Answer status: success; code: DONE"
    assert reward == 0.0
    assert info["action"] == "answer"
    assert info["answer"] == ""
    assert info["code"] == "DONE"
    assert info["done"] is True
    assert info["status"] == "success"
