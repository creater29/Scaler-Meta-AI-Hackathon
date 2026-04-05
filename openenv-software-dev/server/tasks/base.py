"""
Task definitions: bug_fix, feature_impl, code_review.
Each task carries starter files, test files, metrics, and acceptance criteria.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskMetrics:
    required_files: list[str] = field(default_factory=list)
    required_functions: list[str] = field(default_factory=list)
    expected_test_pass_count: int = 0
    forbidden_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    max_cyclomatic_complexity: int = 10
    reference_solution: Optional[str] = None


@dataclass
class Task(abc.ABC):
    task_id: str
    title: str
    description: str
    difficulty: str   # easy | medium | hard
    category: str     # bug_fix | feature_impl | code_review
    starter_files: dict[str, str]
    test_files: dict[str, str]
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    tags: list[str] = field(default_factory=list)
    time_limit_seconds: float = 300.0

    @abc.abstractmethod
    def get_hint(self, step: int) -> str: ...

    @abc.abstractmethod
    def acceptance_check(self, snapshot: dict[str, str]) -> dict[str, Any]: ...
