from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionName(str, Enum):
    learn_skill = "learn_skill"
    build_project = "build_project"
    gain_experience = "gain_experience"
    apply_job = "apply_job"


class JobSpec(BaseModel):
    """Target role requirements and compensation."""

    id: str
    title: str
    company_tier: str = Field(description="e.g. startup, mid, enterprise")
    required_skills: List[str] = Field(default_factory=list)
    min_experience_months: int = 0
    min_projects: int = 0
    salary_band_low: int = 0
    salary_band_high: int = 0


class CareerState(BaseModel):
    """Full environment state returned by /state and used by /grader."""

    episode_id: str
    task: Literal["easy", "medium", "hard"]
    skills: List[str] = Field(default_factory=list)
    projects: int = 0
    experience: int = Field(0, description="Total experience in months.")
    days_left: int = 0
    budget: int = 0
    job: JobSpec
    step_count: int = 0
    max_steps: int = 0
    initial_days: int = 0
    initial_budget: int = 0
    applied: bool = False
    apply_success: Optional[bool] = None
    readiness_at_apply: Optional[float] = None
    last_reward: float = 0.0
    last_info: str = ""
    done: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "episode_id": "00000000-0000-0000-0000-000000000001",
                "task": "medium",
                "skills": ["python", "sql"],
                "projects": 1,
                "experience": 6,
                "days_left": 10,
                "budget": 2000,
                "job": {
                    "id": "job-1",
                    "title": "Backend Engineer",
                    "company_tier": "mid",
                    "required_skills": ["python", "sql"],
                    "min_experience_months": 0,
                    "min_projects": 2,
                    "salary_band_low": 90000,
                    "salary_band_high": 130000,
                },
                "step_count": 3,
                "max_steps": 50,
                "initial_days": 14,
                "initial_budget": 4000,
                "applied": False,
                "apply_success": None,
                "readiness_at_apply": None,
                "last_reward": 0.15,
                "last_info": "",
                "done": False,
            }
        }
    )


class StepAction(BaseModel):
    """Typed action for /step."""

    action: ActionName
    skill: Optional[str] = Field(
        default=None,
        description="Required for learn_skill: one of the curriculum skills.",
    )


class StepResponse(BaseModel):
    """Result of a single environment step."""

    state: CareerState
    reward: float
    done: bool
    info: str = ""


class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = "easy"
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed for episode_id derivation (deterministic episodes).",
    )


class ResetResponse(BaseModel):
    state: CareerState
    observation: CareerState
    success: bool = True


class TaskInfo(BaseModel):
    id: Literal["easy", "medium", "hard"]
    name: str
    description: str
    difficulty: int
    constraints: dict[str, Any]


class TasksResponse(BaseModel):
    environment: str
    tasks: List[TaskInfo]


class GraderRequest(BaseModel):
    """Grader accepts a full state snapshot (deterministic scoring)."""

    state: CareerState


class GraderResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float]
    early_apply_penalty_applied: bool
    notes: str = ""


class BaselineResponse(BaseModel):
    task: Literal["easy", "medium", "hard"]
    final_score: float
    trajectory_length: int
    applied: bool
    apply_success: Optional[bool]
    actions: List[str]


class ReadinessSummary(BaseModel):
    """Deterministic readiness context (no randomness)."""

    tier_optimal_gate: float = Field(
        default=0.0,
        description="Recommended readiness before applying for this task tier.",
    )
    readiness_at_apply: Optional[float] = Field(
        default=None, description="Internal readiness when apply_job ran; null if never applied."
    )
    gap_below_gate: Optional[float] = Field(
        default=None,
        description="If applied early: gate minus readiness at apply (positive = how far under bar).",
    )
    applied_before_recommended_bar: bool = False


class PreparationGapDetail(BaseModel):
    """Concrete numbers recruiters care about in this simulation."""

    missing_skills: List[str] = Field(default_factory=list)
    projects_have: int = 0
    projects_required: int = 0
    projects_shortfall: int = 0
    experience_months_have: int = 0
    experience_months_required: int = 0
    experience_months_shortfall: int = 0


class AnalysisResponse(BaseModel):
    """Structured strategy feedback after an episode (GET /analysis)."""

    strategy_quality: Literal["Poor", "Average", "Good", "Excellent"]
    summary: str = Field(
        default="",
        description="Short, realistic narrative tying together outcome, role bar, and next moves.",
    )
    mistakes: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(
        default_factory=list,
        description="What already meets or exceeds the posting (deterministic checklist).",
    )
    priority_actions: List[str] = Field(
        default_factory=list,
        description="Ordered next moves (highest leverage first).",
    )
    readiness: ReadinessSummary = Field(default_factory=ReadinessSummary)
    gaps: PreparationGapDetail = Field(default_factory=PreparationGapDetail)
    grader_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Same 0–1 estimate as /grader for this state, if computable.",
    )
    rubric: dict[str, float] = Field(
        default_factory=dict,
        description="Subset of grader breakdown (skill_match, projects, experience, etc.).",
    )
