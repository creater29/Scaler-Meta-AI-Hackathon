"""
Python client for the SoftwareDevEnvironment server.
Supports both sync and async usage matching the OpenEnv spec.
"""
from __future__ import annotations
import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    truncated: bool
    info: dict


class SoftwareDevEnv:
    """Sync client for the SoftwareDevEnvironment HTTP server."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, task_id: Optional[str] = None) -> dict:
        payload = {"task_id": task_id} if task_id else {}
        r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: int, target_file: Optional[str] = None,
             text_input: Optional[str] = None) -> StepResult:
        payload = {"action_type": action_type,
                   "target_file": target_file,
                   "text_input": text_input}
        r = self._session.post(f"{self.base_url}/step", json=payload, timeout=30)
        r.raise_for_status()
        d = r.json()
        return StepResult(
            observation=d["observation"],
            reward=d["reward"],
            done=d["done"],
            truncated=d["truncated"],
            info=d.get("info", {}),
        )

    def state(self) -> dict:
        r = self._session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

    def list_tasks(self) -> list[dict]:
        r = self._session.get(f"{self.base_url}/tasks", timeout=10)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._session.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
