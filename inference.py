"""
CHROME Inference Script — Cognitive Human Resource Optimization & Market Engine
===================================
MANDATORY environment variables:
    OPENAI_API_KEY   Your OpenAI-compatible API key  (checked first)
    HF_TOKEN         Hugging Face token               (fallback, for HF router)
    API_KEY          Generic key fallback             (last resort)
    API_BASE_URL     The API endpoint for the LLM     (default: HF router)
    MODEL_NAME       The model identifier             (default: gemini-3.1-flash-lite-preview)

STDOUT FORMAT
The script emits exactly three line types to stdout:

    [START] task=<task_name> env=chrome model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

USAGE
    # Minimal — using OpenAI:
    OPENAI_API_KEY=sk-... API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o python -m hr.inference

    # HF router:
    HF_TOKEN=hf_... MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python -m hr.inference
"""

import json
import os
import re
import sys
import threading
import time
from typing import List, Optional

import uvicorn
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Guard: ensure package is importable ───────────────────────────────
try:
    from hr.server.app import app
    from hr import HREnv
except ImportError as exc:
    print(
        "[ERROR] Cannot import 'hr' package. "
        "Run: pip install -e '.[inference]' from the repo root first.\n"
        f"Details: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────
ENV_URL = "http://127.0.0.1:8321"

# OPENAI_API_KEY is the primary credential per submission guidelines;
# HF_TOKEN and API_KEY are fallbacks for alternative inference providers.
API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")

BENCHMARK = "chrome"
TASK_NAMES = {0: "easy", 1: "medium", 2: "hard"}
TASK_MAX_STEPS = {0: 40, 1: 100, 2: 150}
TEMPERATURE = 0.1
MAX_TOKENS = 2048
SUCCESS_THRESHOLD = 0.1

SYSTEM_PROMPT = """You are an expert HR recruiter AI optimizing hiring for a company.

## RULES
- You have a fixed budget and must hire candidates for teams.
- Each team has a target headcount, minimum intelligence threshold, and an ideal type composition.
- The market has a dynamic salary ledger: hiring from a bucket raises prices (scarcity).
- Adjacent salary buckets also feel pricing pressure (coupled scarcity in hard mode).
- Salary offered must be >= market minimum for the candidate's intelligence bucket.
- Goal: maximize revenue while satisfying all constraints within budget.

## KEY MECHANICS
- **Intelligence Buckets**: 0-20, 21-40, 41-60, 61-80, 81-100.
- **Team Chemistry**: ideal_mix of types (Junior/Mid/Senior/Lead). Revenue = `Base * (0.5 + 0.5 * Chemistry)`.
- **Diminishing Returns**: Nth hire → `1/√N` revenue factor.
- **Coupled Scarcity**: Hiring from one bucket raises adjacent buckets (Hard mode).
- **Market Shocks**: Sudden supply drops at fixed steps (Hard mode).

## TOOLS (respond with JSON)
1. hire_candidate(candidate_id, team_name, offered_salary) - Hire at offered salary (Lakhs)
2. get_team_summary() - Refresh status (costs 0 steps)
3. get_market_ledger() - View salary floors (costs 0 steps)

## STRATEGY
- Hire at or near market minimum (within 5% gets +0.20 bonus).
- FILLING a team gives +1.00 completion bonus.
- Fill all teams with avg intel >= threshold.
- AVOID over-budget hires (-0.55 penalty).
- Do NOT repeatedly request status without hiring.

## RESPONSE FORMAT
{"tool": "<name>", "args": {<arguments>}}
"""


# ── Stdout Logging (strict format) ────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error.replace('\n', ' ').strip() if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────

def parse_tool_call(text: str) -> tuple:
    clean = re.sub(r'<[^>]+>.*?</[^>]+>', '', text, flags=re.DOTALL)
    clean = re.sub(r'```\w*\n?', '', clean).strip()
    start = clean.find('{')
    try:
        end = clean.rfind('}')
        if start != -1 and end != -1:
            data = json.loads(clean[start:end + 1])
            return data.get("tool", "hire_candidate"), data.get("args", {})
    except Exception:
        pass
    return ("hire_candidate", {"candidate_id": 0, "team_name": "N/A", "offered_salary": 0.0})


def format_action(tool: str, args: dict) -> str:
    if tool == "hire_candidate":
        cid = args.get('candidate_id', '?')
        team = args.get('team_name', '?')
        sal = args.get('offered_salary', '?')
        return f"hire_candidate({cid},'{team}',{sal})"
    return f"{tool}()"


# ── Task Runner ────────────────────────────────────────────────────────

def run_task(env, client: OpenAI, task_id: int) -> float:
    task_name = TASK_NAMES[task_id]
    max_steps = TASK_MAX_STEPS[task_id]

    rewards: List[float] = []
    step_count = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env.reset(task_id=task_id, seed=42)
        consecutive_info = 0

        for _turn in range(max_steps * 5):
            # Free state query (not logged as a step)
            state = env.call_tool("get_team_summary")
            if state.get("done"):
                score = state.get("grader_score", 0.0)
                break

            env_step = state.get("action_count", 0)
            candidates = state.get("available_candidates", [])

            teams_json = json.dumps([
                {t["name"]: "%d/%d (Min:%d, Mix:%s)" % (
                    t["current_headcount"], t["target_headcount"],
                    t["required_intel_threshold"], t["ideal_mix"])}
                for t in state["teams"]
            ])
            cands_json = json.dumps([
                {c["candidate_id"]: [c["type"], c["intel_score"], c["current_min_salary"]]}
                for c in candidates
            ])
            context = (
                "Step %d/%d | Budget: %.2fL | Revenue: %.4f\n" % (
                    env_step + 1, max_steps, state["budget_remaining"], state["revenue_projection"])
                + "Teams: %s\n" % teams_json
                + "Candidates (ID:[Type,Intel,Salary]): %s\n" % cands_json
            )

            if consecutive_info >= 3:
                context += "\n[SYSTEM] You have requested status repeatedly. HIRE NOW."

            # LLM call
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            text = response.choices[0].message.content or ""
            tool_name, args = parse_tool_call(text)

            # Execute tool
            try:
                result = env.call_tool(tool_name, **args)
            except Exception as e:
                print(f"[DEBUG] Tool error: {e}", file=sys.stderr, flush=True)
                continue

            # Free actions — don't log as [STEP]
            if tool_name in ("get_team_summary", "get_market_ledger"):
                consecutive_info += 1
                if consecutive_info >= 15:
                    score = state.get("grader_score", 0.0)
                    break
                continue

            # Hire action — log as [STEP]
            consecutive_info = 0
            step_count += 1
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error_msg = result.get("message") if not result.get("success", True) else None

            rewards.append(reward)
            log_step(step=step_count, action=format_action(tool_name, args),
                     reward=reward, done=done, error=error_msg)

            if done:
                score = result.get("grader_score", 0.0)
                break

            # Budget exhaustion early exit
            if "Insufficient budget" in str(result.get("message", "")):
                score = result.get("grader_score", state.get("grader_score", 0.0))
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=step_count, score=score, rewards=rewards)

    return score


def main():
    if not API_KEY:
        print(
            "[ERROR] No API key found. Set OPENAI_API_KEY (or HF_TOKEN) before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Start environment server in background thread
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8321, log_level="error"),
        daemon=True,
    ).start()
    time.sleep(2)  # wait for server to be ready

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with HREnv(base_url=ENV_URL).sync() as env:
        for task_id in range(3):
            run_task(env, client, task_id)


if __name__ == "__main__":
    main()