"""
Core environment logic — the OpenEnv Environment base implementation.
step(), reset(), state() pattern compatible with OpenEnv framework.
"""
from __future__ import annotations
import copy, uuid, time
from typing import Any, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import (
    SoftwareDevAction, SoftwareDevObservation, EpisodeState,
    StepResult, ActionType
)
from server.tasks.base import Task
from server.tasks.catalog import TASK_MAP, ALL_TASKS
from server.sandbox.filesystem import VirtualFilesystem
from server.sandbox.executor import SandboxedExecutor
from server.graders.graders import CompositeGrader


class SoftwareDevEnvironment:
    """
    OpenEnv-compatible environment for software development tasks.
    Exposes reset() / step() / state() per the OpenEnv spec.
    """
    MAX_STEPS = 30

    def __init__(self, task_id: Optional[str] = None,
                 enable_llm: bool = False,
                 llm_endpoint: Optional[str] = None) -> None:
        self.task_id = task_id
        self.enable_llm = enable_llm
        self.llm_endpoint = llm_endpoint
        self._executor = SandboxedExecutor()
        self._grader = CompositeGrader(enable_llm=enable_llm, llm_endpoint=llm_endpoint)
        self._task: Optional[Task] = None
        self._fs: Optional[VirtualFilesystem] = None
        self._episode_id: str = ""
        self._step: int = 0
        self._done: bool = False
        self._action_history: list[dict] = []
        self._cumulative_reward: float = 0.0

    # ── OpenEnv API ──────────────────────────────────────────────────────
    def reset(self, task_id: Optional[str] = None) -> SoftwareDevObservation:
        tid = task_id or self.task_id or ALL_TASKS[0].task_id
        self._task = copy.deepcopy(TASK_MAP[tid])
        self._fs = VirtualFilesystem()
        for path, content in self._task.starter_files.items():
            self._fs.write(path, content)
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._done = False
        self._action_history = []
        self._cumulative_reward = 0.0
        return self._build_obs()

    def step(self, action: SoftwareDevAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")
        self._step += 1
        result = self._dispatch_action(action)
        self._action_history.append({"step": self._step, "action": action.model_dump(),
                                     "result_keys": list(result.keys())})
        is_submit = action.action_type == ActionType.SUBMIT
        is_final = is_submit or self._step >= self.MAX_STEPS
        grade = self._grader.grade(
            self._task, self._fs, self._executor,
            self._action_history, is_final=is_final
        )
        reward = self._shaped_reward(result, grade, is_final)
        self._cumulative_reward += reward
        self._done = is_submit or self._step >= self.MAX_STEPS
        obs = self._build_obs(last_action_result=result,
                              grading_score=grade["score"],
                              done=self._done,
                              truncated=(self._step >= self.MAX_STEPS and not is_submit))
        return StepResult(observation=obs, reward=reward,
                          done=self._done,
                          truncated=(self._step >= self.MAX_STEPS and not is_submit),
                          info={"grading": grade, "action_result": result})

    def state(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self._episode_id,
            task_id=self._task.task_id if self._task else "",
            step_count=self._step, max_steps=self.MAX_STEPS,
            cumulative_reward=self._cumulative_reward,
            done=self._done, truncated=self._step >= self.MAX_STEPS,
            metadata={"task_title": self._task.title if self._task else "",
                      "difficulty": self._task.difficulty if self._task else ""},
        )

    # ── Action dispatch ──────────────────────────────────────────────────
    def _dispatch_action(self, a: SoftwareDevAction) -> dict[str, Any]:
        t = a.action_type
        if t == ActionType.NO_OP:
            return {"status": "ok", "msg": "no-op"}
        if t == ActionType.READ_FILE:
            content = self._fs.read(a.target_file or "")
            return {"status": "ok" if content is not None else "error",
                    "content": content or "File not found."}
        if t == ActionType.WRITE_FILE:
            self._fs.write(a.target_file or "solution.py", a.text_input or "")
            return {"status": "ok", "msg": f"Wrote {a.target_file}"}
        if t == ActionType.EDIT_FILE:
            ok = self._fs.patch(a.target_file or "solution.py",
                                a.text_input or "", "")
            return {"status": "ok" if ok else "error"}
        if t == ActionType.DELETE_FILE:
            ok = self._fs.delete(a.target_file or "")
            return {"status": "ok" if ok else "error"}
        if t == ActionType.RUN_TESTS:
            test_snap = {**self._fs.snapshot(), **self._task.test_files}
            tfs = VirtualFilesystem(); tfs.restore(test_snap)
            return self._executor.run_tests(tfs)
        if t == ActionType.RUN_LINTER:
            return self._executor.run_linter(self._fs)
        if t == ActionType.BUILD:
            return self._executor.build(self._fs)
        if t == ActionType.SUBMIT:
            return {"status": "submitted", "msg": "Episode ending."}
        if t == ActionType.ASK_QUESTION:
            return {"status": "ok", "hint": self._task.get_hint(self._step)}
        return {"status": "error", "msg": f"Unknown action type {t}"}

    # ── Reward shaping ───────────────────────────────────────────────────
    def _shaped_reward(self, result: dict, grade: dict, is_final: bool) -> float:
        r = -0.02  # step penalty
        r += grade["score"] * 0.5
        if result.get("all_passed"):
            r += 1.0
        if result.get("clean"):
            r += 0.2
        if is_final:
            r += grade["score"] * 5.0
            if grade.get("accepted"):
                r += 3.0
        return round(r, 4)

    # ── Observation builder ──────────────────────────────────────────────
    def _build_obs(self, last_action_result: dict = None,
                   grading_score: float = 0.0,
                   done: bool = False, truncated: bool = False) -> SoftwareDevObservation:
        files = self._fs.snapshot() if self._fs else {}
        hint = self._task.get_hint(self._step) if self._task else ""
        text_obs = (
            f"Task: {self._task.title}\n"
            f"Description: {self._task.description}\n"
            f"Step: {self._step}/{self.MAX_STEPS} | Score: {grading_score:.2f}\n"
            f"Files: {list(files.keys())}\n"
            f"Hint: {hint}"
        )
        return SoftwareDevObservation(
            task_id=self._task.task_id if self._task else "",
            task_title=self._task.title if self._task else "",
            task_description=self._task.description if self._task else "",
            task_category=self._task.category if self._task else "",
            difficulty=self._task.difficulty if self._task else "",
            step=self._step, max_steps=self.MAX_STEPS,
            files=files,
            last_action_result=last_action_result or {},
            grading_score=grading_score,
            hint=hint, done=done, truncated=truncated,
            text_observation=text_obs,
        )
