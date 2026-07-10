"""Run the fixed CHROME Core four-model benchmark through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -- LOCAL OVERRIDE HACK --
import importlib.util, pathlib, sys
root = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location('hr', root/'__init__.py', submodule_search_locations=[str(root)])
hr = importlib.util.module_from_spec(spec)
sys.modules['hr'] = hr
spec.loader.exec_module(hr)
# -------------------------

import uvicorn
from openai import OpenAI

from hr import HREnv
from hr.inference import (
    CORE_REQUEST_OPTIONS,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TASK_NAMES,
    TEMPERATURE,
    run_task,
)
from hr.server.app import app

MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-3-flash-preview",
    "minimax/minimax-m3",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-flash",
    "google/gemini-3.1-flash-lite",
]
SEED = 42
OPENROUTER_URL = "https://openrouter.ai/api/v1"
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
README_PATH = PROJECT_ROOT / "README.md"
ENV_URL = "http://127.0.0.1:8322"
README_START = "<!-- CHROME_CORE_RESULTS_START -->"
README_END = "<!-- CHROME_CORE_RESULTS_END -->"


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def fetch_openrouter_models() -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(
        f"{OPENROUTER_URL}/models",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return {item["id"]: item for item in payload.get("data", [])}


def pricing_for(model_data: dict[str, Any]) -> dict[str, float]:
    pricing = model_data.get("pricing", {})
    return {
        "prompt_per_token": float(pricing.get("prompt", 0.0) or 0.0),
        "completion_per_token": float(pricing.get("completion", 0.0) or 0.0),
    }


def cost_for(row: dict[str, Any], pricing: dict[str, float]) -> tuple[float, str]:
    reported = row.get("reported_cost_usd")
    if reported is not None:
        return float(reported), "reported"
    estimate = (
        row["prompt_tokens"] * pricing["prompt_per_token"]
        + row["completion_tokens"] * pricing["completion_per_token"]
    )
    return estimate, "estimated"


def render_trace(model: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# CHROME Core Trace: {model}",
        "",
        f"Seed: `{SEED}`",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['task'].title()}",
            "",
            f"- Score: `{row['grader_score']:.4f}`",
            f"- Success: `{str(row['success']).lower()}`",
            f"- Steps: `{row['step_count']}`",
            f"- Total reward: `{row['total_reward']:.4f}`",
            f"- Runtime: `{row['runtime_seconds']:.3f}s`",
            f"- Tokens: `{row['prompt_tokens']}` prompt / `{row['completion_tokens']}` completion",
            f"- Cost: `${row['cost_usd']:.6f}` ({row['cost_source']})",
            f"- Failure: `{row['failure_reason'] or 'none'}`",
            "",
            "| Turn | Tool | Result |",
            "|---:|---|---|",
        ])
        for event in row.get("trace", []):
            tool = event.get("tool", "protocol failure")
            outcome = event.get("error")
            if outcome is None:
                result = event.get("result", {})
                outcome = result.get("message", "observation") if isinstance(result, dict) else "observation"
            outcome = str(outcome).replace("|", "\\|").replace("\n", " ")
            if event.get("error"):
                outcome += (
                    f" (finish={event.get('finish_reason')}, "
                    f"response_chars={event.get('response_chars', 0)}, "
                    f"reasoning_chars={event.get('reasoning_chars', 0)}, "
                    f"response={event.get('response', '')!r})"
                )
            lines.append(f"| {event['turn']} | `{tool}` | {outcome} |")
        lines.append("")
    return "\n".join(lines)


def aggregate(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    model_rows = [row for row in rows if row["model"] == model]
    scores = {row["task"]: (row["grader_score"] if row.get("success") else None) for row in model_rows}
    complete = len(model_rows) == 3
    valid_scores = [s for s in scores.values() if s is not None]
    avg = sum(valid_scores) / 3 if (complete and len(valid_scores) == 3) else None
    return {
        "model": model,
        "complete": complete,
        "average_score": avg,
        "easy": scores.get("easy"),
        "medium": scores.get("medium"),
        "hard": scores.get("hard"),
        "successes": sum(bool(row.get("success")) for row in model_rows),
        "total_cost_usd": sum(row.get("cost_usd", 0.0) for row in model_rows),
    }


def make_openrouter_client() -> OpenAI:
    headers = {"X-Title": os.getenv("OPENROUTER_APP_TITLE", "CHROME Core")}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    return OpenAI(
        base_url=OPENROUTER_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers=headers,
    )


def render_readme_results(summary: dict[str, Any]) -> str:
    lines = [
        README_START,
        "| Model | Avg | Easy | Medium | Hard | Cost | Trace |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    
    all_rows = []
    greedy_row = {
        "model": "Heuristic Greedy baseline",
        "average_score": 0.6918,
        "easy": 0.8433,
        "medium": 0.6889,
        "hard": 0.5433,
        "total_cost_usd": 0.0,
        "is_baseline": True
    }
    all_rows.append(greedy_row)
    for row in summary.get("aggregates", []):
        all_rows.append(row)
        
    all_rows.sort(key=lambda r: r.get("average_score") or -1.0, reverse=True)

    for row in all_rows:
        values = [row.get(name) for name in ("average_score", "easy", "medium", "hard")]
        scores = [f"{value:.3f}" if value is not None else "-" for value in values]
        if row.get("is_baseline"):
            slug_col = "`rollout_demo.py`"
            model_col = row["model"]
        else:
            slug = model_slug(row["model"])
            slug_col = f"[trace](benchmark_results/core/{slug}.md)"
            model_col = f"`{row['model']}`"
            
        lines.append(
            f"| {model_col} | {scores[0]} | {scores[1]} | {scores[2]} | "
            f"{scores[3]} | ${row.get('total_cost_usd', 0.0):.4f} | "
            f"{slug_col} |"
        )
    
    lines.extend([
        "",
        README_END,
    ])
    return "\n".join(lines)


def update_readme(summary: dict[str, Any]) -> None:
    content = README_PATH.read_text(encoding="utf-8")
    start = content.find(README_START)
    end = content.find(README_END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError("README benchmark markers are missing or invalid")
    end += len(README_END)
    updated = content[:start] + render_readme_results(summary) + content[end:]
    _atomic_write(README_PATH, updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run only the first model on Easy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[ERROR] OPENROUTER_API_KEY is required.", file=sys.stderr)
        raise SystemExit(1)

    results_dir = RESULTS_ROOT / ("smoke" if args.smoke else "core")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"

    available = fetch_openrouter_models()
    selected_models = [MODELS[0]] if args.smoke else MODELS
    missing = [model for model in selected_models if model not in available]
    if missing:
        print(f"[ERROR] OpenRouter model IDs unavailable: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)
    pricing = {model: pricing_for(available[model]) for model in selected_models}

    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8322, log_level="error"),
        daemon=True,
    ).start()
    time.sleep(2)

    rows: list[dict[str, Any]] = []
    spent = 0.0
    existing_models = []
    old_spent = 0.0
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            old_summary = json.load(f)
            rows = old_summary.get("results", [])
            spent = old_summary.get("total_cost_usd", 0.0)
            old_spent = spent
            existing_models = old_summary.get("models", [])
            if "pricing_snapshot" in old_summary:
                pricing.update(old_summary["pricing_snapshot"])
    client = make_openrouter_client()
    task_ids = [0] if args.smoke else list(TASK_NAMES)

    with HREnv(base_url=ENV_URL).sync() as env:
        for model in selected_models:
            model_rows: list[dict[str, Any]] = []
            
            for task_id in task_ids:
                task_name = TASK_NAMES[task_id]
                cached = [r for r in rows if r["model"] == model and r["task"] == task_name and r.get("success")]
                if cached:
                    print(f"    Skipping cached {model} - {task_name}")
                    model_rows.append(cached[0])
                    continue
                
                # Remove prior failed attempts from rows if we are retrying
                rows = [r for r in rows if not (r["model"] == model and r["task"] == task_name)]
                
                task_result = run_task(
                    env, client, task_id, model, seed=SEED,
                    request_options=CORE_REQUEST_OPTIONS,
                    emit_logs=not args.smoke
                )
                row = task_result.to_dict(include_trace=True)
                cost, source = cost_for(row, pricing[model])
                row["cost_usd"] = round(cost, 8)
                row["cost_source"] = source
                spent += cost
                rows.append(row)
                model_rows.append(row)
                print(f"    [Cost so far: ${spent:.4f} (New: ${(spent - old_spent):.4f})]")
                if (spent - old_spent) > 2.0:
                    print(f"    [BUDGET EXCEEDED] New spend ${(spent - old_spent):.4f} > $2.00. Aborting!")
                    break
            if model_rows:
                trace_path = results_dir / f"{model_slug(model)}.md"
                _atomic_write(trace_path, render_trace(model, model_rows))
            if (spent - old_spent) > 2.0:
                break

    all_models = list(dict.fromkeys(existing_models + selected_models))
    public_rows = [{key: value for key, value in row.items() if key != "trace"} for row in rows]
    summary = {
        "benchmark": "CHROME Core",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": OPENROUTER_URL,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": "json_object",
        "reasoning": {"effort": "low", "exclude": True},
        "models": all_models,
        "tasks": [TASK_NAMES[task_id] for task_id in task_ids],
        "total_cost_usd": round(spent, 6),
        "pricing_snapshot": pricing,
        "prompt_sha256": __import__("hashlib").sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "complete": len(rows) == len(all_models) * len(task_ids),
        "results": public_rows,
        "aggregates": [aggregate(rows, model) for model in all_models],
    }
    _atomic_write(summary_path, json.dumps(summary, indent=2) + "\n")
    if summary["complete"] and not args.smoke:
        update_readme(summary)
        print(f"README leaderboard updated: {README_PATH}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
