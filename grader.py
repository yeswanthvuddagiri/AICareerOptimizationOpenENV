from __future__ import annotations

from typing import Dict, Tuple

from models import CareerState


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def optimal_readiness_gate(task: str) -> float:
    """Applying below this readiness hurts the efficiency of the outcome (decision quality)."""
    return {"easy": 0.72, "medium": 0.80, "hard": 0.86}[task]


def compute_grader_score(state: CareerState) -> Tuple[float, Dict[str, float], bool]:
    """
    Weighted rubric:
    - skill match 40%
    - experience 20%
    - projects 20%
    - days_left efficiency 10%
    - budget remaining 10%
    Early apply: extra penalty if applied when readiness was below the task gate.
    """
    job = state.job
    req = set(job.required_skills)
    have = set(state.skills)
    skill_match = 1.0 if not req else len(req & have) / len(req)

    if job.min_experience_months <= 0:
        exp_score = 1.0
    else:
        exp_score = min(1.0, state.experience / job.min_experience_months)

    if job.min_projects <= 0:
        proj_score = 1.0
    else:
        proj_score = min(1.0, state.projects / job.min_projects)

    eff = state.days_left / max(1, state.initial_days)
    bud = state.budget / max(1, state.initial_budget)

    base = (
        0.40 * skill_match
        + 0.20 * exp_score
        + 0.20 * proj_score
        + 0.10 * eff
        + 0.10 * bud
    )

    early_apply_penalty_applied = False
    if state.applied and state.readiness_at_apply is not None:
        gate = optimal_readiness_gate(state.task)
        if state.readiness_at_apply < gate:
            early_apply_penalty_applied = True
            gap = gate - state.readiness_at_apply
            base *= max(0.08, 1.0 - 0.65 * gap)

    if state.applied:
        if state.apply_success is True:
            base = min(1.0, base + 0.04)
        elif state.apply_success is False:
            base *= 0.82

    score = _clamp01(base)
    breakdown = {
        "skill_match": round(skill_match, 4),
        "experience": round(exp_score, 4),
        "projects": round(proj_score, 4),
        "time_efficiency": round(eff, 4),
        "budget_efficiency": round(bud, 4),
        "weighted_base": round(
            0.40 * skill_match
            + 0.20 * exp_score
            + 0.20 * proj_score
            + 0.10 * eff
            + 0.10 * bud,
            4,
        ),
    }
    return score, breakdown, early_apply_penalty_applied
