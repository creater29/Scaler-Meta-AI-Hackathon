"""
FastAPI server exposing the SoftwareDevEnvironment via HTTP.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from server.models import SoftwareDevAction, StepResult, SoftwareDevObservation, EpisodeState
from server.environments.software_dev_env import SoftwareDevEnvironment
from server.tasks.catalog import TASK_MAP

app = FastAPI(title="OpenEnv Software Dev Environment", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# One shared environment instance per server process
_env = SoftwareDevEnvironment(
    enable_llm=bool(os.getenv("API_BASE_URL")),
    llm_endpoint=os.getenv("API_BASE_URL"),
)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/reset", response_model=SoftwareDevObservation)
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment, optionally selecting a specific task."""
    try:
        return _env.reset(task_id=req.task_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {e}")


@app.post("/step", response_model=StepResult)
def step(action: SoftwareDevAction):
    """Take one action in the environment."""
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EpisodeState)
def state():
    """Return current episode metadata."""
    return _env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return [
        {"task_id": t.task_id, "title": t.title,
         "difficulty": t.difficulty, "category": t.category}
        for t in TASK_MAP.values()
    ]


@app.get("/")
def root():
    """Root redirect — points visitors to the interactive API docs."""
    return {
        "name": "OpenEnv Software Dev Environment",
        "version": "0.1.0",
        "status": "running",
        "docs": "http://0.0.0.0:8000/docs",
        "endpoints": [
            "POST /reset  — start a new episode",
            "POST /step   — take an action",
            "GET  /state  — current episode metadata",
            "GET  /tasks  — list available tasks",
            "GET  /health — health check",
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
