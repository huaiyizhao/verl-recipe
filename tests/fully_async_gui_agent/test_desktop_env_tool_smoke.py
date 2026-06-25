import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


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


def _load_gui_agent_loop_module():
    recipe_pkg = types.ModuleType("recipe")
    recipe_pkg.__path__ = [str(RECIPE_ROOT)]
    gui_pkg = types.ModuleType("recipe.fully_async_gui_agent")
    gui_pkg.__path__ = [str(RECIPE_ROOT / "fully_async_gui_agent")]
    sys.modules["recipe"] = recipe_pkg
    sys.modules["recipe.fully_async_gui_agent"] = gui_pkg

    for module_name in ("context_manager", "data_flow_logger", "desktop_env_tool"):
        _load_module(
            f"recipe.fully_async_gui_agent.{module_name}",
            RECIPE_ROOT / "fully_async_gui_agent" / f"{module_name}.py",
        )

    return _load_module(
        "recipe.fully_async_gui_agent.gui_agent_loop",
        RECIPE_ROOT / "fully_async_gui_agent" / "gui_agent_loop.py",
    )


def _tool_json_from_prompt(prompt: str) -> dict:
    tool_text = prompt.rsplit("<tools>", 1)[1].split("</tools>", 1)[0]
    return json.loads(tool_text)


def test_prepare_dataset_prompt_uses_runtime_tool_schema():
    assert _tool_json_from_prompt(
        prepare_dataset.DEFAULT_SYSTEM_PROMPT
    ) == desktop_env_tool.build_computer_use_tool_dict()


def test_turn_penalty_scales_with_base_reward():
    gui_agent_loop = _load_gui_agent_loop_module()

    assert gui_agent_loop.GUIAgentLoop._apply_turn_penalty(
        0.0, turn=50, max_turns=50, turn_penalty_coef=0.1
    ) == pytest.approx((0.0, 0.098, 0.0))
    assert gui_agent_loop.GUIAgentLoop._apply_turn_penalty(
        0.5, turn=26, max_turns=50, turn_penalty_coef=0.1
    ) == pytest.approx((0.475, 0.05, 0.025))
    assert gui_agent_loop.GUIAgentLoop._apply_turn_penalty(
        1.0, turn=26, max_turns=50, turn_penalty_coef=0.1
    ) == pytest.approx((0.95, 0.05, 0.05))


def test_desktop_env_concurrency_limiter_try_acquire_and_release():
    async def run_limiter():
        limiter = desktop_env_tool.DesktopEnvConcurrencyLimiter(2)

        assert await limiter.try_acquire("a") == {
            "acquired": True,
            "max_sessions": 2,
            "in_use": 1,
        }
        assert await limiter.try_acquire("b") == {
            "acquired": True,
            "max_sessions": 2,
            "in_use": 2,
        }
        assert await limiter.try_acquire("c") == {
            "acquired": False,
            "max_sessions": 2,
            "in_use": 2,
        }
        assert await limiter.release("a") == {"max_sessions": 2, "in_use": 1}
        assert await limiter.try_acquire("c") == {
            "acquired": True,
            "max_sessions": 2,
            "in_use": 2,
        }

    asyncio.run(run_limiter())


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


def test_type_action_uses_unicode_input_for_non_ascii_text():
    ascii_code = desktop_env_tool._translate_action_to_pyautogui(
        {"action": "type", "text": "hello\nworld"},
        1000,
        1000,
        1920,
        1080,
    )
    assert ascii_code == "\n".join(
        [
            "pyautogui.typewrite('hello', interval=0.01)",
            "pyautogui.press('enter')",
            "pyautogui.typewrite('world', interval=0.01)",
        ]
    )

    mixed_code = desktop_env_tool._translate_action_to_pyautogui(
        {"action": "type", "text": "abc\u4e2d\u6587def\n\u03a9"},
        1000,
        1000,
        1920,
        1080,
    )
    assert mixed_code == "\n".join(
        [
            "pyautogui.typewrite('abc', interval=0.01)",
            "for _unicode_hex in ['4e2d', '6587']:",
            "    pyautogui.hotkey('ctrl', 'shift', 'u')",
            "    pyautogui.typewrite(_unicode_hex, interval=0.01)",
            "    pyautogui.press('enter')",
            "pyautogui.typewrite('def', interval=0.01)",
            "pyautogui.press('enter')",
            "pyautogui.hotkey('ctrl', 'shift', 'u')",
            "pyautogui.typewrite('3a9', interval=0.01)",
            "pyautogui.press('enter')",
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
