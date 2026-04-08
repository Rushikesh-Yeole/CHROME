---
title: CHROME — HR Optimization & Market Engine
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - rl
  - hr
  - recruiting
  - optimization
license: mit
---

# CHROME — Cognitive Human Resource Optimization & Market Engine

**Submission for [Scaler School of Technology × Meta PyTorch Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)**

---

CHROME is an [OpenEnv](https://openenv.dev)-compliant RL environment that frames autonomous talent acquisition as a sequential decision problem. An LLM or RL agent plays the role of a hiring manager — allocating budget, reading a dynamic salary market, and filling teams — under constraints that make greedy heuristics break down fast.

The environment is designed to be genuinely hard: not "hard because the prompt is vague," but hard because the combinatorics are NP-complete, the market state is endogenous to your own actions, and revenue is non-linear in ways that invalidate local reasoning.

---

## What makes it a real optimization challenge

The base problem — assign N candidates to K teams under a budget, subject to per-team headcount and intelligence constraints — is a variant of **multi-dimensional bin packing**. That's NP-hard in the static case. CHROME makes it harder in three compounding ways.

**The feasibility space shifts with every action.** This isn't a static combinatorial problem you solve once. Hiring candidate X changes the market price of every candidate in the same intelligence bucket, which changes the set of affordable sequences for all remaining decisions. The optimal plan at step 1 is not the optimal plan at step 10, because *you caused the change*. This puts it closer to **online optimization with endogenous costs** than classical NP-hard assignment — a regime where even an oracle that knows the future can't pre-compute the solution without simulating the execution.

**The value function is supermodular and non-separable.** Revenue from hiring candidate X depends on who's already on the team (chemistry) and how many people are already there (diminishing returns). You cannot evaluate candidates independently. The marginal value of a Senior engineer on a team of two Juniors is different from their marginal value on a team of two other Seniors — and that difference compounds across 20 teams and 74 hires in Hard mode. Local search and greedy both degrade badly in supermodular regimes because improving one team often worsens the global objective.

**Coupled scarcity creates cascading price effects.** In Hard mode, hiring from one salary bucket raises adjacent buckets by 3%. A sequence of hires that looks budget-safe in isolation can render an entire tier of candidates unaffordable later in the episode. The agent has to model second-order market effects to avoid painting itself into a corner — something a one-step lookahead cannot do.

**There is no dominant strategy.** Constraint satisfaction (30%), revenue ratio (60%), and cost efficiency (10%) pull in different directions. Maximising revenue means front-loading high-intel candidates from expensive buckets, which destroys cost efficiency and exhausts the budget before all teams are filled. No single policy dominates — the environment requires genuine multi-objective balancing.

The benchmark scores below make this concrete: a well-implemented revenue-per-rupee greedy scores 0.843 on Easy but degrades to 0.543 on Hard. A frontier LLM (Gemini) scores 0.809 on Easy but 0.516 on Hard. Both fail in the same way — they can't anticipate how their own actions reshape the cost landscape two or three moves out. That's the gap a learned policy is meant to close.

---

## Architecture

```
CHROME/
├── __init__.py                  # Public API surface
├── models.py                    # Pydantic contracts: HRAction, HRObservation
├── client.py                    # MCPToolClient (sync + async)
├── inference.py                 # Drop-in LLM baseline runner
├── pyproject.toml
├── openenv.yaml
├── Dockerfile                   # HF Spaces entry point
├── server/
│   ├── hr_environment.py        # All environment logic, market mechanics, MCP tools
│   ├── app.py                   # FastAPI server
│   └── Dockerfile               # Server-only container
├── examples/
│   └── rollout_demo.py          # Greedy vs random policy comparison
└── tests/
    ├── test_hr_env.py
    └── inference_gemini_3.1_flash_lite.md   # Reproducible baseline logs
```

---

## Action Space (MCP Tools)

| Tool | Parameters | Step Cost | Notes |
|------|-----------|-----------|-------|
| `hire_candidate` | `candidate_id`, `team_name`, `offered_salary` | 1 | Salary must meet current market minimum |
| `get_team_summary` | — | 0 | Returns all teams, open slots, current roster, revenue projection |
| `get_market_ledger` | — | 0 | Current prices per intelligence bucket |

Free observation tools mean agents can reason as much as they want before committing to a hire.

---

## Observation Space

Every tool call returns a unified JSON observation:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether the action went through |
| `message` | str | Feedback — what happened and why |
| `reward` | float | Dense per-step signal |
| `done` | bool | Episode terminated |
| `budget_remaining` | float | Remaining budget in ₹ Lakhs |
| `revenue_projection` | float | Cumulative revenue so far |
| `action_count` | int | Hire actions consumed |
| `available_candidates` | list[dict] | IDs, intel scores, types, current min salaries |
| `grader_score` | float | Live grader estimate `[0.0, 1.0]` |

---

## Core Mechanics

### Market Ledger

Five intelligence buckets, each with a base salary and scarcity-driven inflation:

| Bucket | Base (₹L) | Price after N hires from this bucket |
|--------|-----------|--------------------------------------|
| 0–20 | 3.0 | `3.0 × (1 + N × 0.08)` |
| 21–40 | 5.0 | same formula |
| 41–60 | 8.0 | same formula |
| 61–80 | 12.0 | same formula |
| 81–100 | 18.0 | same formula |

Hard mode adds cross-bucket coupling: every hire in bucket `k` raises buckets `k±1` by 3%.

### Team Chemistry

```python
chemistry = 1.0 - L1_distance(actual_type_mix, ideal_type_mix) / 2.0
effective_revenue = base_revenue * (0.5 + 0.5 * chemistry)
```

### Diminishing Returns

```python
revenue_contribution = base_revenue / sqrt(N)   # N = current team size after hire
```

### Reward Function

```
Successful hire:
  +0.50 × (intel / 100) × revenue_multiplier
  +0.40  if intel ≥ team threshold
  -0.30  if intel < team threshold
  +0.20  if offered_salary within 5% of market min
  +1.00  if hire completes the team
  -0.05  per action

Failed hire:
  -0.25  salary below market minimum
  -0.55  over budget
  -0.10  invalid candidate or team
```

### Episode Score (Grader)

```
score = 0.30 × constraint_satisfaction
      + 0.60 × revenue_ratio
      + 0.10 × cost_efficiency
```

Fully deterministic. Same seed, same score.

---

## Task Progression

| Task | Teams | Candidates | Budget | Hires | Steps | Active Mechanics |
|------|-------|-----------|--------|-------|-------|-----------------|
| Easy (0) | 5 | 100 | ₹130L | 12 | 40 | Static market |
| Medium (1) | 12 | 250 | ₹420L | 40 | 100 | Dynamic scarcity + chemistry + diminishing returns |
| Hard (2) | 20 | 500 | ₹740L | 74 | 150 | All of the above + coupled scarcity + 6 market shocks |

---

## Benchmarks

Three agents, all scores deterministic and reproducible.

| Task | Random | Greedy (revenue/₹) | Gemini 3.1 Flash Lite |
|------|--------|--------------------|-----------------------|
| Easy (0) | 0.154 | 0.843 | **0.809** |
| Medium (1) | 0.000 | 0.689 | 0.628 |
| Hard (2) | 0.075 | 0.543 | 0.516 |
| **Average** | **0.076** | **0.692** | **0.651** |

Random and Greedy: avg over 5 seeds from `rollout_demo.py`. Gemini: single run, seed=42, temperature=0.1.

The Easy → Hard degradation is steep for both the algorithmic greedy and the LLM, and the failure mode is the same in both cases. The greedy has perfect knowledge of the reward formula and uses exact chemistry and diminishing-return math in its scoring — yet still can't clear 0.55 on Hard. That ceiling is where the environment stops being solvable by reasoning about one step at a time.

### LLM trace — `gemini-3.1-flash-lite-preview` (seed=42, temperature=0.1)

```
[START] task=easy   env=chrome model=gemini-3.1-flash-lite-preview
[END]   success=true  steps=12  score=0.809  rewards=0.94,0.93,1.99,...

[START] task=medium env=chrome model=gemini-3.1-flash-lite-preview
[END]   success=true  steps=35  score=0.628  rewards=1.26,0.68,...,-0.55

[START] task=hard   env=chrome model=gemini-3.1-flash-lite-preview
[END]   success=true  steps=50  score=0.516  rewards=1.47,0.67,...,-0.55
```

Full log: [`tests/inference_gemini_3.1_flash_lite.md`](tests/inference_gemini_3.1_flash_lite.md)

### Heuristic trace — `rollout_demo.py` greedy (avg over 5 seeds)

```
Task Easy (0)
  Greedy (revenue/₹)    Grader: 0.8433   Reward/hire: +1.293   Hires: 12.0
  Random (lower bound)  Grader: 0.1540   Reward/hire: +0.941   Hires: 10.8

Task Medium (1)
  Greedy (revenue/₹)    Grader: 0.6889   Reward/hire: +1.024   Hires: 34.2
  Random (lower bound)  Grader: 0.0000   Reward/hire: +0.559   Hires: 27.6

Task Hard (2)
  Greedy (revenue/₹)    Grader: 0.5433   Reward/hire: +0.977   Hires: 49.6
  Random (lower bound)  Grader: 0.0752   Reward/hire: +0.504   Hires: 39.6
```

---

## Quick Start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/rushisyeole/chrome-env
cd chrome-env

pip install -e ".[dev]"

uvicorn hr.server.app:app --host 0.0.0.0 --port 8000 --reload
pytest tests/ -v
```

### API — connect directly to this Space

```python
from hr.client import HREnv

with HREnv(base_url="https://rushisyeole-chrome-env.hf.space").sync() as env:
    env.reset(task_id=2, seed=0)   # Hard mode
    state = env.call_tool("get_team_summary")
```

Or run the inference script against it without spinning up a local server:

```bash
OPENAI_API_KEY=sk-...  \
API_BASE_URL=https://rushisyeole-chrome-env.hf.space  \
MODEL_NAME=gpt-4o  \
python -m hr.inference
```

### Docker

```bash
docker build -t chrome-env .
docker run -p 8000:8000 chrome-env
```

---

## License

MIT