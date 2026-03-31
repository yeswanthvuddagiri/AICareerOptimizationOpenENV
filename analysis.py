"""Post-episode strategy analysis (deterministic, no randomness)."""

from __future__ import annotations

from typing import List, Set

from grader import compute_grader_score, optimal_readiness_gate
from models import AnalysisResponse, CareerState, PreparationGapDetail, ReadinessSummary


def _norm_skills_have(skills: object) -> Set[str]:
    if not skills:
        return set()
    out: Set[str] = set()
    for x in skills:
        try:
            out.add(str(x).lower())
        except Exception:
            continue
    return out


def _safe_int(val: object, default: int = 0) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except Exception:
        return default


def _safe_float(val: object) -> float | None:
    try:
        return float(val)  # type: ignore[arg-type]
    except Exception:
        return None


def analyze_episode(state: CareerState) -> AnalysisResponse:
    mistakes: List[str] = []
    missing_skill_names: List[str] = []
    strengths: List[str] = []
    priority_actions: List[str] = []

    task = "easy"
    try:
        t = state.task
        if t in ("easy", "medium", "hard"):
            task = t
    except Exception:
        pass

    gate = optimal_readiness_gate(task)
    job = None
    required_raw: List[str] = []
    job_title = "role"
    try:
        job = state.job
        job_title = getattr(job, "title", None) or "role"
        required_raw = list(job.required_skills or [])
    except Exception:
        job = None

    applied = False
    try:
        applied = bool(state.applied)
    except Exception:
        pass

    r_at_apply: float | None = None
    try:
        if state.readiness_at_apply is not None:
            r_at_apply = _safe_float(state.readiness_at_apply)
    except Exception:
        r_at_apply = None

    early_penalty = False
    grader_score: float | None = None
    rubric: dict[str, float] = {}
    try:
        grader_score, breakdown, early_penalty = compute_grader_score(state)
        rubric = {
            k: float(v)
            for k, v in (breakdown or {}).items()
            if isinstance(v, (int, float))
        }
    except Exception:
        grader_score = None
        rubric = {}
        early_penalty = False

    early_mistake = False
    if applied:
        if early_penalty:
            early_mistake = True
        elif r_at_apply is not None and r_at_apply < gate:
            early_mistake = True
    if early_mistake:
        mistakes.append("Applied too early")

    have = _norm_skills_have(getattr(state, "skills", None))
    if job is not None:
        for s in required_raw:
            if not isinstance(s, str) or not s.strip():
                continue
            if s.lower() not in have:
                missing_skill_names.append(s)

    if missing_skill_names:
        mistakes.append("Missing required skills")

    min_p, proj = 0, 0
    min_e, exp = 0, 0
    if job is not None:
        min_p = _safe_int(getattr(job, "min_projects", 0), 0)
        proj = _safe_int(getattr(state, "projects", 0), 0)
        min_e = _safe_int(getattr(job, "min_experience_months", 0), 0)
        exp = _safe_int(getattr(state, "experience", 0), 0)

    proj_short = max(0, min_p - proj) if min_p > 0 else 0
    exp_short = max(0, min_e - exp) if min_e > 0 else 0

    if min_p > 0 and proj < min_p:
        mistakes.append("Insufficient projects")
    if min_e > 0 and exp < min_e:
        mistakes.append("Not enough experience")

    gap_below: float | None = None
    if early_mistake and r_at_apply is not None:
        gap_below = max(0.0, gate - r_at_apply)

    readiness_block = ReadinessSummary(
        tier_optimal_gate=round(gate, 4),
        readiness_at_apply=round(r_at_apply, 4) if r_at_apply is not None else None,
        gap_below_gate=round(gap_below, 4) if gap_below is not None else None,
        applied_before_recommended_bar=early_mistake,
    )

    gaps = PreparationGapDetail(
        missing_skills=list(missing_skill_names),
        projects_have=proj,
        projects_required=min_p,
        projects_shortfall=proj_short,
        experience_months_have=exp,
        experience_months_required=min_e,
        experience_months_shortfall=exp_short,
    )

    # Strengths (what is already solid)
    if job is not None and required_raw and not missing_skill_names:
        strengths.append(
            f"Skill checklist for this {task}-tier posting is satisfied ({len(required_raw)} required areas)."
        )
    elif job is not None and required_raw and len(missing_skill_names) < len(required_raw):
        strengths.append(
            f"Partial skill coverage: {len(required_raw) - len(missing_skill_names)} of "
            f"{len(required_raw)} required areas already credentialled."
        )

    if min_p <= 0:
        strengths.append("This posting does not insist on a minimum project count; focus stays on skills and timing.")
    elif proj >= min_p:
        strengths.append(
            f"Portfolio depth meets the bar ({proj} shipped project(s); minimum {min_p})."
        )

    if min_e <= 0:
        strengths.append("No explicit tenure/months floor on this posting — seniority signal is mostly skills and delivery.")
    elif exp >= min_e:
        strengths.append(
            f"Experience signal is at or above the stated floor ({exp} months vs {min_e} required)."
        )

    try:
        days_left = _safe_int(getattr(state, "days_left", 0), 0)
        initial_days = _safe_int(getattr(state, "initial_days", 1), 1)
        if initial_days > 0 and days_left > 0:
            time_eff = days_left / initial_days
            if time_eff >= 0.35:
                strengths.append(
                    f"Calendar efficiency is reasonable: {days_left} day(s) left from {initial_days} — "
                    "room remained to iterate before the episode ended."
                )
    except Exception:
        pass

    apply_ok: bool | None = None
    try:
        apply_ok = state.apply_success
    except Exception:
        apply_ok = None

    suggestions: List[str] = []

    if missing_skill_names:
        suggestions.append(
            f"Prioritize courses or certifications that close: {', '.join(missing_skill_names)} — "
            "recruiters screen on this stack before reading narrative."
        )
    if proj_short > 0:
        suggestions.append(
            f"Add {proj_short} concrete project(s) with scope docs, metrics, and (simulated) deploy story — "
            f"you are at {proj} / {min_p}."
        )
    if exp_short > 0:
        suggestions.append(
            f"Target roughly {exp_short} more month(s) of relevant experience signal "
            f"(now {exp} / {min_e} months) before swinging at this bar."
        )
    if early_mistake:
        suggestions.append(
            f"Defer applications until readiness ≥ {gate:.2f} for this tier "
            "(yours at send time was "
            f"{(round(r_at_apply, 3) if r_at_apply is not None else 'unknown')}). "
            "Early sends burn finite recruiter attention."
        )

    # Priority actions (ordered)
    if missing_skill_names:
        priority_actions.append(
            "1) Skills: finish " + ", ".join(missing_skill_names) + " before more outreach."
        )
    if proj_short > 0:
        priority_actions.append(
            f"2) Delivery: ship {proj_short} portfolio-grade project(s) with measurable outcomes."
        )
    if exp_short > 0:
        priority_actions.append(
            f"3) Tenure: accumulate ~{exp_short} month(s) more relevant experience."
        )
    if early_mistake:
        priority_actions.append(
            "4) Timing: treat apply_job as a terminal decision — only pull when the checklist above is green."
        )
    elif applied and apply_ok is True:
        priority_actions.append(
            "4) Sustain: document what worked (ordering of learn → build → experience → apply) for the next loop."
        )
    elif not applied:
        priority_actions.append(
            "4) Closeout: when bars are green, submit once — avoid endless polishing past diminishing returns."
        )

    if not priority_actions:
        priority_actions.append(
            "1) Re-run with a written plan: skills → projects (if required) → experience (if required) → single apply."
        )

    # Strategy label
    if "Applied too early" in mistakes or len(mistakes) >= 2:
        strategy_quality = "Poor"
    elif len(mistakes) == 1:
        strategy_quality = "Average"
    elif applied and apply_ok is True:
        strategy_quality = "Excellent"
    elif applied:
        strategy_quality = "Average"
    else:
        strategy_quality = "Good"

    score_txt = f"{grader_score:.2f}" if grader_score is not None else "n/a"
    summary_parts: List[str] = []
    summary_parts.append(
        f"Coaching read on «{job_title}» ({task} difficulty): composite trajectory score ≈ {score_txt}."
    )
    if early_mistake:
        summary_parts.append(
            "You pressed the application while still under the recommended readiness band — "
            "that reads as premature in this simulation and mirrors weak signal-to-noise in real pipelines."
        )
    elif applied and apply_ok is True and not mistakes:
        summary_parts.append(
            "Posting requirements line up with your end state and the employer extended an offer — "
            "strong alignment between prep order and bar."
        )
    elif applied and apply_ok is False:
        summary_parts.append(
            "You applied, but the outcome was a pass — double-check whether gaps were structural "
            "(skills, portfolio, tenure) or timing (readiness vs gate)."
        )
    elif not applied and not mistakes:
        summary_parts.append(
            "You stopped short of applying despite a clean checklist — "
            "either you ran out of steps/time or held off; consider a single decisive apply next episode."
        )
    else:
        summary_parts.append(
            "Several preparation dimensions still miss the posting — fix the structured gaps before leaning on network or referrals."
        )

    summary = " ".join(summary_parts)

    if not suggestions:
        if strategy_quality == "Excellent":
            suggestions.append(
                "Keep using a gated pipeline: exhaust hard requirements, then one high-quality application burst."
            )
        else:
            suggestions.append(
                "Next run, log each step’s cost (days, budget) next to posting bullets so trade-offs stay visible."
            )

    return AnalysisResponse(
        strategy_quality=strategy_quality,
        summary=summary,
        mistakes=mistakes,
        suggestions=suggestions,
        strengths=strengths,
        priority_actions=priority_actions,
        readiness=readiness_block,
        gaps=gaps,
        grader_score=grader_score,
        rubric=rubric,
    )
