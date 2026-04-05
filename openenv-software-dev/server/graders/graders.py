"""
Graders: programmatic (deterministic) + LLM (semantic) + composite.
"""
from __future__ import annotations
import os, re
from typing import Any, Optional
from server.tasks.base import Task
from server.sandbox.filesystem import VirtualFilesystem
from server.sandbox.executor import SandboxedExecutor


# ── Programmatic Grader ──────────────────────────────────────────────────
class ProgrammaticGrader:
    WEIGHTS = {"test_score": 0.40, "file_score": 0.10,
               "pattern_score": 0.20, "acceptance": 0.30}

    def grade(self, task: Task, fs: VirtualFilesystem,
              executor: SandboxedExecutor, is_final: bool = False) -> dict[str, Any]:
        snapshot = fs.snapshot()
        # ── acceptance check ─────────────────────────────────────────────
        acceptance = task.acceptance_check(snapshot)
        checks = acceptance.get("checks", [])
        # ── tests ────────────────────────────────────────────────────────
        # inject test files into fs snapshot temporarily
        test_snap = dict(snapshot)
        test_snap.update(task.test_files)
        test_fs = VirtualFilesystem(); test_fs.restore(test_snap)
        test_res = executor.run_tests(test_fs)
        total = max(test_res["total"], 1)
        test_score = test_res["passed"] / total
        # ── required files ───────────────────────────────────────────────
        req = task.metrics.required_files
        file_score = sum(fs.exists(f) for f in req) / max(len(req), 1)
        # ── patterns ─────────────────────────────────────────────────────
        combined = " ".join(snapshot.values()).lower()
        req_p = task.metrics.required_patterns
        forb_p = task.metrics.forbidden_patterns
        pattern_ok = all(p.lower() in combined for p in req_p)
        pattern_clean = not any(p.lower() in combined for p in forb_p)
        pattern_score = (0.5 if pattern_ok else 0.0) + (0.5 if pattern_clean else 0.0)

        breakdown = {
            "test_score": round(test_score, 4),
            "file_score": round(file_score, 4),
            "pattern_score": round(pattern_score, 4),
            "acceptance": 1.0 if acceptance.get("accepted") else 0.0,
        }
        composite = sum(breakdown[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return {"score": round(min(composite, 1.0), 4),
                "breakdown": breakdown, "checks": checks,
                "test_results": test_res}


# ── LLM Grader ──────────────────────────────────────────────────────────
DIMENSION_WEIGHTS = {"correctness": 0.40, "code_quality": 0.20,
                     "completeness": 0.25, "efficiency": 0.15}

class LLMGrader:
    def __init__(self, endpoint: Optional[str] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None) -> None:
        self.endpoint = endpoint or os.getenv("API_BASE_URL", "")
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("HF_TOKEN", "")

    def grade(self, task: Task, fs: VirtualFilesystem) -> dict[str, Any]:
        try:
            import httpx, json
            solution = "\n\n".join(
                f"# {p}\n{c}" for p, c in fs.snapshot().items()
                if not p.startswith("test_")
            )
            prompt = (
                f"Task: {task.title}\n"
                f"Description: {task.description}\n"
                f"Reference solution:\n{task.metrics.reference_solution or 'N/A'}\n\n"
                f"Agent solution:\n{solution}\n\n"
                "Score the agent solution on: correctness(0-1), code_quality(0-1), "
                "completeness(0-1), efficiency(0-1). Reply ONLY with JSON: "
                '{"correctness":0.9,"code_quality":0.8,"completeness":1.0,"efficiency":0.7,"explanation":"..."}'
            )
            resp = httpx.post(
                f"{self.endpoint}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0},
                timeout=30.0,
            )
            raw = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(raw)
            dims = {k: float(data.get(k, 0.5)) for k in DIMENSION_WEIGHTS}
            score = sum(dims[k] * DIMENSION_WEIGHTS[k] for k in DIMENSION_WEIGHTS)
            return {"score": round(min(score, 1.0), 4), "dimensions": dims,
                    "explanation": data.get("explanation", ""), "raw_response": raw}
        except Exception as e:
            return {"score": 0.5, "dimensions": {k: 0.5 for k in DIMENSION_WEIGHTS},
                    "explanation": f"LLM grader unavailable: {e}", "raw_response": ""}


# ── Composite Grader ─────────────────────────────────────────────────────
class CompositeGrader:
    def __init__(self, enable_llm: bool = False, llm_endpoint: Optional[str] = None,
                 prog_weight: float = 0.6, llm_weight: float = 0.4) -> None:
        self.prog_grader = ProgrammaticGrader()
        self.llm_grader = LLMGrader(endpoint=llm_endpoint) if enable_llm else None
        self.enable_llm = enable_llm
        self.prog_weight = prog_weight
        self.llm_weight = llm_weight

    def grade(self, task: Task, fs: VirtualFilesystem, executor: SandboxedExecutor,
              action_history: list, is_final: bool = False) -> dict[str, Any]:
        prog = self.prog_grader.grade(task, fs, executor, is_final)
        llm_result = None
        if self.enable_llm and self.llm_grader and is_final:
            llm_result = self.llm_grader.grade(task, fs)
        if llm_result:
            score = self.prog_weight * prog["score"] + self.llm_weight * llm_result["score"]
        else:
            score = prog["score"]
        efficiency_factor = max(0.0, 1.0 - len(action_history) / 100.0)
        score = min(1.0, score + 0.05 * efficiency_factor)
        return {"score": round(score, 4), "programmatic": prog,
                "llm": llm_result, "accepted": score >= 0.8}
