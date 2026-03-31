"""
Production inference loop for the AI Career Optimization OpenEnv FastAPI server.

Uses the OpenAI Python client against the Hugging Face OpenAI-compatible router,
with a rule-based fallback agent when the LLM is unavailable or fails.

Environment variables
---------------------
  HF_TOKEN       (optional) — If unset or invalid, only the rule-based agent runs.
  API_BASE_URL   (optional) — Default: https://router.huggingface.co/v1
  MODEL_NAME     (optional) — Default: mistralai/Mistral-7B-Instruct-v0.2
  ENV_BASE_URL   (optional) — Career FastAPI base URL. Default: http://localhost:8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import httpx
from openai import OpenAI

VALID_ACTIONS = frozenset({"learn_skill", "build_project", "gain_experience", "apply_job"})
MAX_STEPS = 10
HTTP_ATTEMPTS = 2  # initial + one retry

DEFAULT_ENV_BASE = "http://localhost:8000"
DEFAULT_HF_ROUTER_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

FALLBACK_SKILL_DEFAULT = "python"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("inference")


def _api_base_url() -> str:
    return os.environ.get("API_BASE_URL", DEFAULT_HF_ROUTER_BASE).rstrip("/")


def _model_name() -> str:
    name = os.environ.get("MODEL_NAME", "").strip()
    return name if name else DEFAULT_MODEL


def _env_base_url() -> str:
    return os.environ.get("ENV_BASE_URL", DEFAULT_ENV_BASE).rstrip("/")


def _try_build_llm_client() -> Optional[OpenAI]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        log.warning("HF_TOKEN not set; LLM disabled, using rule-based agent only.")
        return None
    try:
        return OpenAI(
            base_url=_api_base_url(),
            api_key=token,
            timeout=120.0,
            max_retries=0,
        )
    except Exception as e:
        log.warning("Could not create LLM client (%s); using rule-based agent only.", e)
        return None


def _http_post_json(client: httpx.Client, url: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = client.post(url, json=body if body is not None else {}, timeout=120.0)
    r.raise_for_status()
    return r.json()


def _http_get_json(client: httpx.Client, url: str) -> Dict[str, Any]:
    r = client.get(url, timeout=60.0)
    r.raise_for_status()
    return r.json()


T = TypeVar("T")


def _with_retry(label: str, fn: Callable[[], T], attempts: int = HTTP_ATTEMPTS) -> Optional[T]:
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 < attempts:
                log.warning("%s failed (%s); retrying once.", label, e)
    if last:
        log.warning("%s failed after retries: %s", label, last)
    return None


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_llm_action(raw: str) -> Tuple[str, Optional[str]]:
    text = _strip_code_fence(raw)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            action = str(obj.get("action", "")).strip()
            skill = obj.get("skill")
            skill_s = str(skill).strip().lower() if skill is not None else None
            if action in VALID_ACTIONS:
                return action, skill_s if action == "learn_skill" else None
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        text = line
        break

    if "|" in text and text.count("|") == 1:
        left, right = text.split("|", 1)
        action, skill = left.strip(), right.strip().lower()
        if action in VALID_ACTIONS:
            return action, skill if action == "learn_skill" and skill else None

    m = re.match(
        r"^(learn_skill|build_project|gain_experience|apply_job)(?::([\w]+))?$",
        text.strip(),
    )
    if m:
        action = m.group(1)
        skill = m.group(2).lower() if m.group(2) else None
        return action, skill if action == "learn_skill" else None

    raise ValueError(f"unparseable action: {raw!r}")


def rule_based_action(state: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Priority: missing required skills → learn_skill; then projects below min → build_project;
    then experience below min → gain_experience; else apply_job.
    """
    job = state.get("job") or {}
    required = [s for s in (job.get("required_skills") or []) if isinstance(s, str) and s.strip()]
    have = {str(x).lower() for x in (state.get("skills") or [])}

    for s in required:
        if s.lower() not in have:
            return "learn_skill", s.lower()

    min_projects = int(job.get("min_projects") or 0)
    projects = int(state.get("projects") or 0)
    if min_projects > 0 and projects < min_projects:
        return "build_project", None

    min_exp = int(job.get("min_experience_months") or 0)
    experience = int(state.get("experience") or 0)
    if min_exp > 0 and experience < min_exp:
        return "gain_experience", None

    return "apply_job", None


def build_messages(state: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You control a job candidate in a simulation. "
        "Goal: get hired efficiently — build skills, projects, and experience under time and "
        "budget constraints, then apply when you are competitive. "
        "Respond with ONLY a single action token. No explanation, no markdown, no extra text. "
        "Valid actions: learn_skill, build_project, gain_experience, apply_job. "
        "For learn_skill use the form learn_skill:skill_name (e.g. learn_skill:python). "
        "Allowed skill_name values: python, system_design, sql, ml, communication, "
        "kubernetes, data_engineering."
    )
    user = (
        "Current state (JSON):\n"
        f"{json.dumps(state, indent=2)}\n\n"
        "Output one line only (example: learn_skill:python or apply_job)."
    )
    return system, user


def _chat_completion(llm: OpenAI, model: str, system: str, user: str) -> str:
    completion = llm.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    msg = completion.choices[0].message
    return (msg.content or "").strip()


def _chat_or_empty(llm: Optional[OpenAI], model: str, system: str, user: str) -> str:
    if llm is None:
        return ""
    last_err: Optional[Exception] = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            return _chat_completion(llm, model, system, user)
        except Exception as e:
            last_err = e
            if attempt + 1 < HTTP_ATTEMPTS:
                log.warning("LLM attempt %d failed: %s; retrying once.", attempt + 1, e)
    if last_err:
        log.warning("LLM failed after %d attempts: %s", HTTP_ATTEMPTS, last_err)
    return ""


def _resolve_action(
    llm: Optional[OpenAI],
    model: str,
    state: Dict[str, Any],
    step_i: int,
) -> Tuple[str, Optional[str]]:
    """Prefer LLM; on failure or bad parse, rule-based agent (logged)."""
    system, user = build_messages(state)
    raw = _chat_or_empty(llm, model, system, user)

    if raw:
        try:
            action, skill = parse_llm_action(raw)
            if action == "learn_skill" and not skill:
                skill = FALLBACK_SKILL_DEFAULT
            return action, skill
        except ValueError:
            log.warning("Step %d: parse failed (%r).", step_i, raw[:200])

    log.info("Using fallback agent")
    return rule_based_action(state)


def _safe_grader(http: httpx.Client, env_url: str) -> Tuple[float, Optional[Dict[str, Any]]]:
    out = _with_retry("GET /grader", lambda: _http_get_json(http, f"{env_url}/grader"))
    if out is None:
        return 0.0, None
    score = out.get("score")
    try:
        return float(score) if score is not None else 0.0, out
    except (TypeError, ValueError):
        return 0.0, out


def main() -> None:
    env_url = _env_base_url()
    router_url = _api_base_url()
    model = _model_name()
    llm = _try_build_llm_client()

    task = os.environ.get("TASK", "easy").strip().lower()
    if task not in ("easy", "medium", "hard"):
        task = "easy"
        log.warning("Invalid TASK; defaulting to easy.")

    log.info("ENV_BASE_URL=%s", env_url)
    log.info("API_BASE_URL (router)=%s MODEL_NAME=%s", router_url, model)

    total_reward = 0.0
    state: Dict[str, Any] = {}
    done = False

    try:
        with httpx.Client() as http:
            reset_out = _with_retry(
                "POST /reset",
                lambda: _http_post_json(http, f"{env_url}/reset", {"task": task}),
            )
            if reset_out is None:
                log.error("POST /reset failed; cannot run episode.")
            else:
                s = reset_out.get("state")
                if isinstance(s, dict):
                    state = s
                    done = bool(state.get("done"))
                else:
                    log.error("Invalid /reset response.")

                for step_i in range(1, MAX_STEPS + 1):
                    if done:
                        break
                    try:
                        action, skill = _resolve_action(llm, model, state, step_i)
                        step_body: Dict[str, Any] = {"action": action}
                        if action == "learn_skill":
                            step_body["skill"] = skill or FALLBACK_SKILL_DEFAULT

                        step_out = _with_retry(
                            f"POST /step step={step_i}",
                            lambda: _http_post_json(
                                http, f"{env_url}/step", step_body
                            ),
                        )
                        if step_out is None:
                            log.error("POST /step failed; stopping steps.")
                            break
                        reward = float(step_out.get("reward", 0.0))
                        done = bool(step_out.get("done"))
                        new_state = step_out.get("state")
                        if isinstance(new_state, dict):
                            state = new_state
                        total_reward += reward
                        print(
                            f"step={step_i} action={action} skill={step_body.get('skill')} "
                            f"reward={reward:.4f} done={done} cumulative={total_reward:.4f}",
                            flush=True,
                        )
                    except Exception as e:
                        log.warning("Step %d error (%s); trying rule-based step.", step_i, e)
                        log.info("Using fallback agent")
                        try:
                            action, skill = rule_based_action(state)
                            step_body = {"action": action}
                            if action == "learn_skill":
                                step_body["skill"] = skill or FALLBACK_SKILL_DEFAULT
                            step_out = _with_retry(
                                f"POST /step recovery step={step_i}",
                                lambda: _http_post_json(
                                    http, f"{env_url}/step", step_body
                                ),
                            )
                            if step_out:
                                total_reward += float(step_out.get("reward", 0.0))
                                done = bool(step_out.get("done"))
                                ns = step_out.get("state")
                                if isinstance(ns, dict):
                                    state = ns
                        except Exception as e2:
                            log.warning("Recovery step failed: %s", e2)

            score, grader_resp = _safe_grader(http, env_url)

            print("--- done ---", flush=True)
            print(f"final_grader_score={score}", flush=True)
            print(f"total_step_reward_sum={total_reward:.4f}", flush=True)
            if grader_resp and isinstance(grader_resp.get("breakdown"), dict):
                print(f"breakdown={json.dumps(grader_resp['breakdown'])}", flush=True)
    except Exception as e:
        log.exception("Episode error (continuing to report score): %s", e)
        print("--- done ---", flush=True)
        print("final_grader_score=0.0", flush=True)
        print(f"total_step_reward_sum={total_reward:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted.")
        sys.exit(130)
    except Exception as e:
        log.exception("Unexpected: %s", e)
        print("--- done ---", flush=True)
        print("final_grader_score=0.0", flush=True)
        sys.exit(0)
