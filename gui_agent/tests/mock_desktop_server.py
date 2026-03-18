"""Minimal mock desktop environment pool server for testing."""

import io
import random

from fastapi import FastAPI
from fastapi.responses import Response
from PIL import Image

app = FastAPI()
envs: dict = {}


@app.post("/envs/acquire")
async def acquire():
    env_id = f"env-{random.randint(1000, 9999)}"
    envs[env_id] = {"task_id": None, "steps": 0}
    return {"env_id": env_id}


@app.post("/envs/{env_id}/reset")
async def reset(env_id: str, body: dict):
    envs[env_id] = {"task_id": body["task_id"], "steps": 0}
    return {"status": "ok"}


@app.get("/envs/{env_id}/screenshot")
async def screenshot(env_id: str):
    img = Image.new(
        "RGB",
        (1000, 1000),
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/envs/{env_id}/action")
async def action(env_id: str, body: dict):
    envs[env_id]["steps"] += 1
    return {"status": "ok"}


@app.post("/envs/{env_id}/task_status")
async def task_status(env_id: str):
    completed = envs.get(env_id, {}).get("steps", 0) >= 5
    return {"completed": completed}


@app.post("/envs/{env_id}/release")
async def release(env_id: str):
    envs.pop(env_id, None)
    return {"status": "ok"}
