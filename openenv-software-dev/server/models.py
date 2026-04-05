"""
Pydantic models for actions, observations, and state.
These are the core data structures shared between server and client.
"""
from __future__ import annotations
from enum import IntEnum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ActionType(IntEnum):
    NO_OP = 0
    READ_FILE = 1
    WRITE_FILE = 2
    EDIT_FILE = 3
    DELETE_FILE = 4
    RUN_TESTS = 5
    RUN_LINTER = 6
    BUILD = 7
    SUBMIT = 8
    ASK_QUESTION = 9


class SoftwareDevAction(BaseModel):
    action_type: int = Field(..., description="ActionType enum value")
    target_file: Optional[str] = Field(None, description="File path to operate on")
    text_input: Optional[str] = Field(None, description="Text content (for write/edit/ask)")
    line_start: Optional[int] = Field(None, description="Start line for EDIT_FILE")
    line_end: Optional[int] = Field(None, description="End line for EDIT_FILE")

class GradingResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    accepted: bool = False


class SoftwareDevObservation(BaseModel):
    task_id: str
    task_title: str
    task_description: str
    task_category: str          # bug_fix | feature_impl | code_review
    difficulty: str             # easy | medium | hard
    step: int
    max_steps: int
    files: dict[str, str]       # filename -> content
    last_action_result: dict[str, Any] = Field(default_factory=dict)
    test_results: dict[str, Any] = Field(default_factory=dict)
    lint_results: dict[str, Any] = Field(default_factory=dict)
    grading_score: float = 0.0
    hint: str = ""
    done: bool = False
    truncated: bool = False
    # Text obs for LLM agents
    text_observation: str = ""


class EpisodeState(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool
    truncated: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: SoftwareDevObservation
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)
