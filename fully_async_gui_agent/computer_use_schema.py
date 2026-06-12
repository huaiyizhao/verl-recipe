import copy
import json
from typing import Any


_COMPUTER_USE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or "
            "applications menu. You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait "
            "and take successive screenshots to see the results of your actions. E.g. if you click "
            "on Firefox and a window doesn't open, try wait and taking another screenshot.\n"
            "* The screen's resolution is {screen_width}x{screen_height}.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should "
            "consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load even after waiting, "
            "try adjusting your cursor position so that the tip of the cursor visually falls on the "
            "element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of "
            "the element. Don't click boxes on their edges unless asked."
        ),
        "parameters": {
            "properties": {
                "action": {
                    "description": (
                        "\n"
                        "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).\n"
                        "* `scroll`: Performs a scroll of the mouse scroll wheel. If `coordinate` is provided, first move the mouse to that coordinate, then scroll.\n"
                        "* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status.\n"
                        "* `answer`: Answer a question.\n"
                    ),
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "triple_click",
                        "scroll",
                        "wait",
                        "terminate",
                        "answer",
                    ],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type`.", "type": "string"},
                "coordinate": {
                    "description": "The x,y coordinates for mouse actions.",
                    "type": "array",
                },
                "pixels": {
                    "description": "The amount of scrolling.",
                    "type": "number",
                },
                "time": {"description": "The seconds to wait.", "type": "number"},
                "status": {
                    "description": "The status of the task. Required by `action=terminate` or `action=answer`.",
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


def build_computer_use_tool_dict(screen_width: int = 1000, screen_height: int = 1000) -> dict[str, Any]:
    """Build the prompt/runtime computer_use tool schema from one source."""
    tool = copy.deepcopy(_COMPUTER_USE_TOOL)
    tool["function"]["description"] = tool["function"]["description"].format(
        screen_width=screen_width,
        screen_height=screen_height,
    )
    return tool


def build_computer_use_system_prompt(screen_width: int = 1000, screen_height: int = 1000) -> str:
    """Build the Qwen-style XML tool prompt used by dataset rows."""
    tools_def = build_computer_use_tool_dict(screen_width, screen_height)
    return """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
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
