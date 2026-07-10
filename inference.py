"""OpenAI-compatible inference runner for the CHROME Core benchmark."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import uvicorn
from openai import OpenAI

try:
    from dotenv import load_dotenv
    import os
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    load_dotenv()
except ImportError:
    pass

try:
    from hr.server.app import app
    from hr import HREnv
except ImportError as exc:
    print(
        "[ERROR] Cannot import 'hr'. Run: pip install -e '.[inference]'\n"
        f"Details: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


ENV_URL = "http://127.0.0.1:8321"
API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")

BENCHMARK = "chrome-core"
TASK_NAMES = {0: "easy", 1: "medium", 2: "hard"}
TASK_MAX_STEPS = {0: 40, 1: 100, 2: 150}
TEMPERATURE = 0.1
MAX_TOKENS = 8192
SUCCESS_THRESHOLD = 0.1
import pydantic
from typing import Dict, Any

class ActionSchema(pydantic.BaseModel):
    tool: str
    args: Dict[str, Any]

VALID_TOOLS = {"hire_candidate", "get_team_summary", "get_market_ledger"}
CORE_REQUEST_OPTIONS = {
    "response_format": {"type": "json_object"},
    "extra_body": {"reasoning": {"effort": "low", "exclude": True}},
}

SYSTEM_PROMPT = """You are an expert HR recruiter AI optimizing hiring for a company.

## RULES
- You have a fixed budget and must hire candidates for teams.
- Each team has a target headcount, minimum intelligence threshold, and an ideal type composition.
- Hiring from a bucket raises its price; in Hard, every hire also cumulatively raises adjacent buckets.
- Salary offered must be >= the current market minimum for the candidate's intelligence bucket.
- Goal: maximize revenue while satisfying all constraints within budget.

## KEY MECHANICS
- Intelligence buckets: 0-20, 21-40, 41-60, 61-80, 81-100.
- Team chemistry: Revenue = Base * (0.5 + 0.5 * Chemistry).
- Diminishing returns: Nth hire has a 1/sqrt(N) revenue factor.
- Hard mode includes cumulative coupled scarcity and deterministic market shocks.

## TOOLS
1. hire_candidate(candidate_id, team_name, offered_salary)
2. get_team_summary()
3. get_market_ledger()

## RESPONSE FORMAT
Return exactly one JSON object and no other text:
{"tool": "hire_candidate", "args": {"candidate_id": 1, "team_name": "Engineering", "offered_salary": 8.0}}

`candidate_id` must be a JSON integer, not a quoted string. For observation
tools, return an empty args object, for example:
{"tool": "get_team_summary", "args": {}}
"""


class ActionParseError(ValueError):
    """The model response did not satisfy the Core action protocol."""


@dataclass
class TaskResult:
    model: str
    seed: int
    task: str
    grader_score: float = 0.0
    environment_score_at_failure: float = 0.0
    step_count: int = 0
    success: bool = False
    rewards: list[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported_cost_usd: Optional[float] = None
    failure_reason: Optional[str] = None
    retries: int = 0
    temperature_fallback: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return round(sum(self.rewards), 4)

    def to_dict(self, include_trace: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["total_reward"] = self.total_reward
        if not include_trace:
            result.pop("trace", None)
        return result


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err = error.replace("\n", " ").strip() if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(result: TaskResult) -> None:
    rewards = ",".join(f"{reward:.2f}" for reward in result.rewards)
    print(
        f"[END] success={str(result.success).lower()} steps={result.step_count} "
        f"score={result.grader_score:.3f} rewards={rewards}",
        flush=True,
    )


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    clean = text.strip()
    
    data = None
    end = clean.rfind('}')
    if end != -1:
        start = clean.rfind('{', 0, end)
        while start != -1:
            try:
                candidate = json.loads(clean[start:end+1])
                if isinstance(candidate, dict) and {"tool", "args"}.issubset(candidate.keys()):
                    data = candidate
                    break
            except json.JSONDecodeError:
                pass
            start = clean.rfind('{', 0, start)
            
    if data is None:
        try:
            data = json.loads(clean)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ActionParseError(f"malformed JSON: {exc}") from exc

    if not isinstance(data, dict) or set(data) != {"tool", "args"}:
        raise ActionParseError("action must contain exactly 'tool' and 'args'")
    
    tool = data["tool"]
    args = data["args"]
    if tool not in VALID_TOOLS:
        raise ActionParseError(f"unknown tool: {tool!r}")
    
    if not isinstance(args, dict):
        raise ActionParseError("args must be a JSON object")

    if tool in {"get_team_summary", "get_market_ledger"}:
        if args:
            raise ActionParseError(f"{tool} accepts no arguments")
        return tool, args

    expected = {"candidate_id", "team_name", "offered_salary"}
    if set(args) != expected:
        raise ActionParseError(
            "hire_candidate args must be candidate_id, team_name, offered_salary"
        )
    candidate_id = args["candidate_id"]
    if (
        isinstance(candidate_id, bool)
        or not isinstance(candidate_id, (int, float))
        or not math.isfinite(float(candidate_id))
        or not float(candidate_id).is_integer()
    ):
        raise ActionParseError("candidate_id must be an integer")
    args["candidate_id"] = int(candidate_id)
    if not isinstance(args["team_name"], str) or not args["team_name"]:
        raise ActionParseError("team_name must be a non-empty string")
    salary = args["offered_salary"]
    if isinstance(salary, bool) or not isinstance(salary, (int, float)):
        raise ActionParseError("offered_salary must be numeric")
    if not math.isfinite(float(salary)) or salary < 0:
        raise ActionParseError("offered_salary must be finite and non-negative")
    return tool, args


def format_action(tool: str, args: dict[str, Any]) -> str:
    if tool == "hire_candidate":
        return (
            f"hire_candidate({args['candidate_id']},"
            f"'{args['team_name']}',{args['offered_salary']})"
        )
    return f"{tool}()"


def build_context(state: dict[str, Any], max_steps: int,
                  consecutive_info: int, last_action_result: Optional[str] = None) -> str:
    teams = json.dumps([
        {team["name"]: "%d/%d (Min:%d, Mix:%s)" % (
            team["current_headcount"], team["target_headcount"],
            team["required_intel_threshold"], team["ideal_mix"],
        )}
        for team in state["teams"]
    ])
    candidates = json.dumps([
        {candidate["candidate_id"]: [
            candidate["type"], candidate["intel_score"],
            candidate["current_min_salary"],
        ]}
        for candidate in state.get("available_candidates", [])
    ])
    context = (
        "Step %d/%d | Budget: %.2fL | Revenue: %.4f\n" % (
            state.get("action_count", 0) + 1, max_steps,
            state["budget_remaining"], state["revenue_projection"],
        )
        + f"Teams: {teams}\nCandidates (ID:[Type,Intel,Salary]): {candidates}\n"
    )
    if consecutive_info >= 3:
        context += "\n[SYSTEM] You have requested status repeatedly. HIRE NOW."
    if last_action_result:
        context += f"\n[PREVIOUS ACTION RESULT]\n{last_action_result}\n"
    return context


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 504} or "timeout" in type(exc).__name__.lower()


def _usage(response: Any) -> tuple[int, int, Optional[float]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, None
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = getattr(usage, "cost", None)
    if cost is None and getattr(usage, "model_extra", None):
        cost = usage.model_extra.get("cost")
    return prompt, completion, float(cost) if cost is not None else None


def _completion(client: OpenAI, model: str, context: str,
                request_options: Optional[dict[str, Any]] = None) -> tuple[Any, int, bool]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if request_options:
        kwargs.update(request_options)
    retries = 0
    temperature_fallback = False
    while True:
        try:
            return client.chat.completions.create(**kwargs), retries, temperature_fallback
        except Exception as exc:
            message = str(exc).lower()
            status = getattr(exc, "status_code", None)
            if "temperature" in message and status in {400, 422} and "temperature" in kwargs:
                kwargs.pop("temperature")
                temperature_fallback = True
                continue
            if retries < 2 and _is_transient(exc):
                time.sleep(2 ** retries)
                retries += 1
                continue
            raise


def run_task(env: Any, client: OpenAI, task_id: int, model: str,
             seed: int = 42, emit_logs: bool = True,
             request_options: Optional[dict[str, Any]] = None) -> TaskResult:
    task_name = TASK_NAMES[task_id]
    max_steps = TASK_MAX_STEPS[task_id]
    result = TaskResult(model=model, seed=seed, task=task_name)
    started = time.perf_counter()
    if emit_logs:
        log_start(task_name, model)

    try:
        env.reset(task_id=task_id, seed=seed)
        consecutive_info = 0
        last_action_result = None

        for turn in range(max_steps * 5):
            state = env.call_tool("get_team_summary")
            current_score = float(state.get("grader_score", 0.0) or 0.0)
            if state.get("done"):
                result.grader_score = current_score
                break

            context = build_context(state, max_steps, consecutive_info, last_action_result)
            try:
                response, retries, fallback = _completion(
                    client, model, context, request_options=request_options
                )
            except Exception as exc:
                result.failure_reason = f"provider_error: {exc}"
                result.environment_score_at_failure = current_score
                break

            result.retries += retries
            result.temperature_fallback = result.temperature_fallback or fallback
            prompt_tokens, completion_tokens, cost = _usage(response)
            result.prompt_tokens += prompt_tokens
            result.completion_tokens += completion_tokens
            if cost is not None:
                result.reported_cost_usd = (result.reported_cost_usd or 0.0) + cost

            try:
                choice = response.choices[0]
                message = choice.message
                raw = message.content or ""
            except (AttributeError, TypeError, IndexError):
                result.success = False
                result.failure_reason = "api_error: received invalid response payload from provider"
                result.environment_score_at_failure = current_score
                break
            reasoning = getattr(message, "reasoning", None)
            if reasoning is None and getattr(message, "model_extra", None):
                reasoning = message.model_extra.get("reasoning")
            trace_row: dict[str, Any] = {
                "turn": turn + 1,
                "response": raw,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_chars": len(raw),
                "reasoning_chars": len(reasoning or ""),
            }
            try:
                tool_name, args = parse_tool_call(raw)
            except ActionParseError as exc:
                result.failure_reason = f"protocol_error: {exc}"
                result.environment_score_at_failure = current_score
                trace_row["error"] = result.failure_reason
                result.trace.append(trace_row)
                break

            trace_row["tool"] = tool_name
            trace_row["args"] = args
            try:
                tool_result = env.call_tool(tool_name, **args)
            except Exception as exc:
                result.failure_reason = f"tool_error: {exc}"
                result.environment_score_at_failure = current_score
                trace_row["error"] = result.failure_reason
                result.trace.append(trace_row)
                break

            trace_row["result"] = tool_result
            result.trace.append(trace_row)
            
            action_str = format_action(tool_name, args)
            
            if tool_name in {"get_team_summary", "get_market_ledger"}:
                consecutive_info += 1
                if consecutive_info >= 15:
                    result.failure_reason = "information_action_limit"
                    result.environment_score_at_failure = current_score
                    break
                last_action_result = f"You attempted: {action_str}\nResult: Retrieved {tool_name} successfully."
                continue

            consecutive_info = 0
            result.step_count += 1
            reward = float(tool_result.get("reward", 0.0) or 0.0)
            result.rewards.append(reward)
            error = tool_result.get("message") if not tool_result.get("success", True) else None
            
            if error:
                last_action_result = f"You attempted: {action_str}\nResult: FAILED - {error}. Please adjust your strategy."
            else:
                last_action_result = f"You attempted: {action_str}\nResult: SUCCESS. Reward: {reward:.4f}"

            if emit_logs:
                log_step(
                    result.step_count, format_action(tool_name, args), reward,
                    bool(tool_result.get("done", False)), error,
                )
            if tool_result.get("done"):
                result.grader_score = float(tool_result.get("grader_score", 0.0) or 0.0)
                break
        else:
            result.failure_reason = "turn_limit"

        if result.failure_reason is None and result.grader_score == 0.0:
            final_state = env.call_tool("get_team_summary")
            result.grader_score = float(final_state.get("grader_score", 0.0) or 0.0)
        if result.failure_reason is not None:
            result.grader_score = 0.0
        result.success = result.failure_reason is None and result.grader_score >= SUCCESS_THRESHOLD
    finally:
        result.runtime_seconds = round(time.perf_counter() - started, 3)
        if emit_logs:
            log_end(result)
    return result


def make_client() -> OpenAI:
    headers = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "CHROME Core")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY, default_headers=headers)


def main() -> None:
    if not API_KEY:
        print("[ERROR] Set OPENROUTER_API_KEY (or another compatible API key).", file=sys.stderr)
        sys.exit(1)
    if not MODEL_NAME:
        print("[ERROR] Set MODEL_NAME for a single-model run.", file=sys.stderr)
        sys.exit(1)

    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8321, log_level="error"),
        daemon=True,
    ).start()
    time.sleep(2)
    client = make_client()
    with HREnv(base_url=ENV_URL).sync() as env:
        for task_id in range(3):
            run_task(env, client, task_id, MODEL_NAME)


if __name__ == "__main__":
    main()
