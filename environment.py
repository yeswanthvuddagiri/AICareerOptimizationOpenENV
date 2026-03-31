from __future__ import annotations

import math
import uuid
from typing import List, Optional, Tuple

from models import ActionName, CareerState, JobSpec, StepAction, StepResponse

# --- Curriculum & economics -------------------------------------------------
SKILL_CATALOG = [
    "python",
    "system_design",
    "sql",
    "ml",
    "communication",
    "kubernetes",
    "data_engineering",
]

SKILL_LEARN_COST = 180  # budget per course / certification
PROJECT_DURATION_DAYS = 2  # projects consume calendar time
EXPERIENCE_INCREMENT_MONTHS = 3  # each internship / contract block


def _job_for_task(task: str, episode_tag: str) -> JobSpec:
    """Distinct jobs per task so tasks are clearly different (not toy duplicates)."""
    if task == "easy":
        return JobSpec(
            id=f"easy-{episode_tag}",
            title="Junior Data Analyst",
            company_tier="mid",
            required_skills=["python", "sql", "communication"],
            min_experience_months=0,
            min_projects=0,
            salary_band_low=65000,
            salary_band_high=85000,
        )
    if task == "medium":
        return JobSpec(
            id=f"medium-{episode_tag}",
            title="Software Engineer II (Platform)",
            company_tier="enterprise",
            required_skills=["python", "system_design", "sql", "kubernetes"],
            min_experience_months=0,
            min_projects=3,
            salary_band_low=120000,
            salary_band_high=155000,
        )
    return JobSpec(
        id=f"hard-{episode_tag}",
        title="Senior ML Engineer",
        company_tier="enterprise",
        required_skills=["python", "ml", "system_design", "data_engineering", "communication"],
        min_experience_months=36,
        min_projects=4,
        salary_band_low=175000,
        salary_band_high=230000,
    )


def _task_params(task: str) -> Tuple[int, int, int]:
    """initial_days, initial_budget, max_steps."""
    if task == "easy":
        return 28, 8000, 64
    if task == "medium":
        return 14, 4500, 48
    # Strict vs easy/medium but feasible: senior bar needs many months + shipped work.
    return 28, 7000, 60


def readiness_score(state: CareerState) -> float:
    """Deterministic [0,1] preparation signal used for apply success and grader."""
    job = state.job
    req = set(job.required_skills)
    have = set(state.skills)
    if not req:
        skill_part = 1.0
    else:
        skill_part = len(req & have) / len(req)
    if job.min_experience_months <= 0:
        exp_part = 1.0
    else:
        exp_part = min(1.0, state.experience / job.min_experience_months)
    if job.min_projects <= 0:
        proj_part = 1.0
    else:
        proj_part = min(1.0, state.projects / job.min_projects)
    return 0.45 * skill_part + 0.30 * exp_part + 0.25 * proj_part


def apply_success_probability(state: CareerState) -> float:
    """Logistic mapping of readiness — interpretable 'offer probability' (deterministic)."""
    r = readiness_score(state)
    x = 14.0 * (r - 0.68)
    return 1.0 / (1.0 + math.exp(-x))


def apply_offer_threshold(task: str) -> float:
    """Stricter bar on harder tasks (still smooth via probability mapping)."""
    return {"easy": 0.58, "medium": 0.70, "hard": 0.80}[task]


def apply_success_deterministic(state: CareerState) -> bool:
    """Offer if deterministic probability meets task bar."""
    return apply_success_probability(state) >= apply_offer_threshold(state.task)


def time_penalty() -> float:
    return -0.05


class CareerEnvironment:
    def __init__(self) -> None:
        self._state: Optional[CareerState] = None

    @property
    def state(self) -> CareerState:
        if self._state is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        return self._state

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> CareerState:
        if seed is not None:
            episode_id = str(uuid.UUID(int=seed & ((1 << 128) - 1)))
        else:
            episode_id = str(uuid.uuid4())
        days, budget, max_steps = _task_params(task)
        tag = episode_id[:8]
        job = _job_for_task(task, tag)
        self._state = CareerState(
            episode_id=episode_id,
            task=task,  # type: ignore[arg-type]
            skills=[],
            projects=0,
            experience=0,
            days_left=days,
            budget=budget,
            job=job,
            step_count=0,
            max_steps=max_steps,
            initial_days=days,
            initial_budget=budget,
            applied=False,
            apply_success=None,
            readiness_at_apply=None,
            last_reward=0.0,
            last_info="Episode started.",
            done=False,
        )
        return self._state.model_copy(deep=True)

    def step(self, action: StepAction) -> StepResponse:
        s = self.state
        if s.done:
            return StepResponse(
                state=s.model_copy(deep=True),
                reward=0.0,
                done=True,
                info="Episode already finished.",
            )

        reward = time_penalty()
        info_parts: List[str] = []

        def finish_step(msg: str) -> StepResponse:
            s.last_reward = reward
            s.last_info = msg
            s.step_count += 1
            if s.days_left <= 0 or s.step_count >= s.max_steps:
                s.done = True
                if s.days_left <= 0:
                    s.last_info += " Out of time."
                else:
                    s.last_info += " Max steps reached."
            return StepResponse(state=s.model_copy(deep=True), reward=reward, done=s.done, info=msg)

        if action.action == ActionName.learn_skill:
            skill = (action.skill or "").strip().lower()
            if not skill:
                reward += -0.1
                return finish_step("learn_skill requires a skill name.")
            if skill not in SKILL_CATALOG:
                reward += -0.1
                return finish_step(f"Unknown skill '{skill}'.")
            if skill in s.skills:
                reward += -0.05
                return finish_step(f"Skill '{skill}' already acquired.")
            if s.budget < SKILL_LEARN_COST:
                reward += -0.15
                return finish_step("Insufficient budget for training.")
            if s.days_left < 1:
                reward += -0.1
                return finish_step("No time left to train.")
            s.budget -= SKILL_LEARN_COST
            s.days_left -= 1
            s.skills = sorted(set(s.skills + [skill]))
            reward += 0.2
            info_parts.append(
                f"Learned {skill} (-{SKILL_LEARN_COST} budget, -1 day). "
                f"Estimated offer probability if applying now: {apply_success_probability(s):.2f}."
            )
            return finish_step(" ".join(info_parts))

        if action.action == ActionName.build_project:
            if s.days_left < PROJECT_DURATION_DAYS:
                reward += -0.1
                return finish_step("Not enough days to complete a project.")
            s.days_left -= PROJECT_DURATION_DAYS
            s.projects += 1
            reward += 0.2
            info_parts.append(
                f"Shipped project #{s.projects} (-{PROJECT_DURATION_DAYS} days). "
                f"Estimated offer probability if applying now: {apply_success_probability(s):.2f}."
            )
            return finish_step(" ".join(info_parts))

        if action.action == ActionName.gain_experience:
            if s.days_left < 1:
                reward += -0.1
                return finish_step("No time left for experience.")
            s.days_left -= 1
            s.experience += EXPERIENCE_INCREMENT_MONTHS
            reward += 0.3
            info_parts.append(
                f"Gained experience (+{EXPERIENCE_INCREMENT_MONTHS} mo, -1 day). "
                f"Estimated offer probability if applying now: {apply_success_probability(s):.2f}."
            )
            return finish_step(" ".join(info_parts))

        if action.action == ActionName.apply_job:
            r0 = readiness_score(s)
            p = apply_success_probability(s)
            s.readiness_at_apply = r0
            ok = apply_success_deterministic(s)
            s.applied = True
            s.apply_success = ok
            if ok:
                reward += 1.0
                info_parts.append(
                    f"Application outcome: offer extended (readiness={r0:.2f}, p_offer={p:.3f})."
                )
            else:
                reward += -0.5
                info_parts.append(
                    f"Application outcome: rejected (readiness={r0:.2f}, p_offer={p:.3f})."
                )
            s.done = True
            s.last_reward = reward
            s.last_info = " ".join(info_parts)
            s.step_count += 1
            return StepResponse(state=s.model_copy(deep=True), reward=reward, done=True, info=s.last_info)

        reward += -0.1
        return finish_step("Unknown action.")
