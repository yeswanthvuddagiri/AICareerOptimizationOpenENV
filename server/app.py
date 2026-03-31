from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException

from analysis import analyze_episode
from environment import CareerEnvironment
from grader import compute_grader_score
from models import (
    ActionName,
    AnalysisResponse,
    BaselineResponse,
    CareerState,
    GraderRequest,
    GraderResponse,
    ResetRequest,
    ResetResponse,
    StepAction,
    StepResponse,
    TaskInfo,
    TasksResponse,
)

app = FastAPI(
    title="AI Career Optimization Environment",
    version="1.0.0",
    description="Job preparation simulation with skills, projects, experience, time, and budget.",
)

_env = CareerEnvironment()


@app.get("/")
def root() -> dict:
    """Root URL — browsers opening http://localhost:8000/ land here."""
    return {
        "service": "AI Career Optimization Environment",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /reset": "Start an episode (optional body: task, seed).",
            "POST /step": "Step with JSON body: action, optional skill.",
            "GET /state": "Current state (call /reset first).",
            "GET /tasks": "List easy / medium / hard tasks.",
            "POST /grader": "Score a CareerState JSON body.",
            "GET /grader": "Score the current server episode state (after /step).",
            "GET /baseline": "Fixed policy demo (?task=easy|medium|hard).",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-career-optimization-env"}


@app.post("/reset", response_model=ResetResponse)
def reset(body: Optional[ResetRequest] = None) -> ResetResponse:
    req = body or ResetRequest()
    st = _env.reset(task=req.task, seed=req.seed)
    return ResetResponse(state=st, observation=st, success=True)


@app.post("/step", response_model=StepResponse)
def step(body: StepAction) -> StepResponse:
    try:
        return _env.step(body)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/state", response_model=ResetResponse)
def get_state() -> ResetResponse:
    try:
        st = _env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ResetResponse(state=st, observation=st, success=True)


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    task_list: List[TaskInfo] = [
        TaskInfo(
            id="easy",
            name="Foundations — skills-first screening",
            description=(
                "Screening-style loop: role emphasizes core tool skills. "
                "Generous calendar and budget; no project or tenure bar."
            ),
            difficulty=1,
            constraints={
                "focus": "skills_only",
                "initial_days": 28,
                "initial_budget": 8000,
                "job_requires_experience": False,
                "job_requires_projects": False,
            },
        ),
        TaskInfo(
            id="medium",
            name="Product engineering loop — portfolio + deadline",
            description=(
                "Mid-level platform role: must demonstrate shipped work under a two-week sprint. "
                "Budget is tighter; multiple projects required."
            ),
            difficulty=2,
            constraints={
                "focus": "skills_and_projects",
                "initial_days": 14,
                "initial_budget": 4500,
                "job_requires_experience": False,
                "job_requires_projects": True,
            },
        ),
        TaskInfo(
            id="hard",
            name="Staff track — capital, experience, and calendar risk",
            description=(
                "Senior ML role: long experience bar, several shipped projects, broad skill coverage, "
                "and strict time/budget trade-offs."
            ),
            difficulty=3,
            constraints={
                "focus": "skills_experience_projects_budget_time",
                "initial_days": 28,
                "initial_budget": 7000,
                "job_requires_experience": True,
                "job_requires_projects": True,
            },
        ),
    ]
    return TasksResponse(environment="AI Career Optimization Environment", tasks=task_list)


def _grader_from_state(st: CareerState) -> GraderResponse:
    score, breakdown, early = compute_grader_score(st)
    notes = (
        "Early-application decision penalty applied (readiness below optimal gate)."
        if early
        else "No early-application penalty."
    )
    return GraderResponse(
        score=score,
        breakdown=breakdown,
        early_apply_penalty_applied=early,
        notes=notes,
    )


@app.post("/grader", response_model=GraderResponse)
def grader(body: GraderRequest) -> GraderResponse:
    return _grader_from_state(body.state)
def grader(body: GraderRequest):
    print("DEBUG SKILLS:", body.state.skills)
    return _grader_from_state(body.state)
@app.get("/auto_grader")
def auto_grader():
    try:
        st = _env.state

        # DEBUG (optional)
        print("AUTO GRADER SKILLS:", st.skills)

        return _grader_from_state(st)

    except Exception as e:
        return {"error": str(e)}
@app.get("/analysis", response_model=AnalysisResponse)
def analysis() -> AnalysisResponse:
    """Structured coaching feedback from the current episode state (GET /analysis)."""
    try:
        st = _env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        return analyze_episode(st)
    except Exception:
        return AnalysisResponse(
            strategy_quality="Poor",
            summary="Could not derive structured feedback from the current session state.",
            mistakes=["Analysis failed — state may be incomplete."],
            suggestions=["Call POST /reset, take steps with POST /step, then retry GET /analysis."],
            strengths=[],
            priority_actions=["Reset the environment and replay a full episode before requesting analysis."],
            grader_score=None,
            rubric={},
        )


def _baseline_sequence(task: str) -> List[StepAction]:
    """Fixed policy used only for /baseline reproducibility."""
    if task == "easy":
        return [
            StepAction(action=ActionName.learn_skill, skill="python"),
            StepAction(action=ActionName.learn_skill, skill="sql"),
            StepAction(action=ActionName.learn_skill, skill="communication"),
            StepAction(action=ActionName.apply_job),
        ]
    if task == "medium":
        return [
            StepAction(action=ActionName.learn_skill, skill="python"),
            StepAction(action=ActionName.learn_skill, skill="system_design"),
            StepAction(action=ActionName.learn_skill, skill="sql"),
            StepAction(action=ActionName.learn_skill, skill="kubernetes"),
            StepAction(action=ActionName.build_project),
            StepAction(action=ActionName.build_project),
            StepAction(action=ActionName.build_project),
            StepAction(action=ActionName.apply_job),
        ]
    # hard — fixed order: skills (5d) + experience to meet tenure bar (12d) + projects (8d) + apply
    return [
        StepAction(action=ActionName.learn_skill, skill="python"),
        StepAction(action=ActionName.learn_skill, skill="ml"),
        StepAction(action=ActionName.learn_skill, skill="system_design"),
        StepAction(action=ActionName.learn_skill, skill="data_engineering"),
        StepAction(action=ActionName.learn_skill, skill="communication"),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.gain_experience),
        StepAction(action=ActionName.build_project),
        StepAction(action=ActionName.build_project),
        StepAction(action=ActionName.build_project),
        StepAction(action=ActionName.build_project),
        StepAction(action=ActionName.apply_job),
    ]


@app.get("/baseline", response_model=BaselineResponse)
def baseline(task: str = "easy") -> BaselineResponse:
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be easy, medium, or hard")
    seq = _baseline_sequence(task)
    actions: List[str] = []
    st = _env.reset(task=task, seed=424242)
    for a in seq:
        if _env.state.done:
            break
        actions.append(a.action.value + (f":{a.skill}" if a.skill else ""))
        _env.step(a)
    score, _, _ = compute_grader_score(_env.state)
    return BaselineResponse(
        task=task,  # type: ignore[arg-type]
        final_score=score,
        trajectory_length=len(actions),
        applied=_env.state.applied,
        apply_success=_env.state.apply_success,
        actions=actions,
    )
    
def main():
    """
    OpenEnv entrypoint — DO NOT start uvicorn here
    """
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
