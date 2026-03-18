"""Live integration test: DesktopEnvTool against mock desktop server.

Usage:
    # Terminal 1: start mock server
    uvicorn recipe.gui_agent.tests.mock_desktop_server:app --port 18923

    # Terminal 2: run this test
    python recipe/gui_agent/tests/test_desktop_tool_live.py
"""

import asyncio
import sys

from recipe.gui_agent.desktop_env_tool import DesktopEnvTool


async def main(api_base_url: str):
    tool = DesktopEnvTool(
        config={
            "api_base_url": api_base_url,
            "screen_width": 1000,
            "screen_height": 1000,
            "timeout": 10,
        }
    )

    print("=" * 60)
    print("DesktopEnvTool live integration test")
    print(f"API: {api_base_url}")
    print("=" * 60)

    # --- create ---
    print("\n[1] create(task_id='test-task-1') ...")
    instance_id, resp = await tool.create(create_kwargs={"task_id": "test-task-1"})
    assert resp.image is not None and len(resp.image) == 1
    img = resp.image[0]
    print(f"    OK  instance_id={instance_id}, screenshot={img.size}, mode={img.mode}")

    # --- execute several actions ---
    actions = [
        {"action": "left_click", "coordinate": [500, 300]},
        {"action": "type", "text": "hello world"},
        {"action": "scroll", "pixels": -200},
        {"action": "key", "keys": ["ctrl", "s"]},
        {"action": "wait", "time": 1},
    ]
    for i, params in enumerate(actions):
        print(f"\n[{i + 2}] execute({params}) ...")
        resp, reward, metrics = await tool.execute(instance_id, params)
        assert resp.image is not None and len(resp.image) == 1
        print(f"    OK  reward={reward}, screenshot={resp.image[0].size}, text={resp.text}")

    # --- calc_reward (should be 1.0 after 5 steps) ---
    print(f"\n[{len(actions) + 2}] calc_reward() ...")
    reward = await tool.calc_reward(instance_id)
    print(f"    OK  reward={reward}")
    assert reward == 1.0, f"Expected 1.0 after 5 steps, got {reward}"

    # --- terminate action (does not hit env) ---
    print(f"\n[{len(actions) + 3}] execute(terminate) ...")
    resp, reward, metrics = await tool.execute(instance_id, {"action": "terminate", "status": "success"})
    assert resp.image is None
    print(f"    OK  text={resp.text}")

    # --- release ---
    print(f"\n[{len(actions) + 4}] release() ...")
    await tool.release(instance_id)
    print("    OK  env released")

    # --- verify release is idempotent ---
    print(f"\n[{len(actions) + 5}] release() again (idempotent) ...")
    await tool.release(instance_id)
    print("    OK  no error")

    # --- calc_reward on released instance ---
    reward = await tool.calc_reward(instance_id)
    assert reward == 0.0
    print(f"\n    calc_reward after release = {reward} (expected 0.0)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18923"
    asyncio.run(main(url))
