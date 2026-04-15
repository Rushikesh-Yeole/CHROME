# CHROME
### Cognitive Human Resource Optimization & Market Engine

An OpenEnv-compliant RL environment for autonomous talent acquisition. An agent acts as a hiring manager across K teams, allocating a fixed budget in a **scarcity-coupled salary market** where every hire changes the cost of future hires — making the value function non-separable, the feasibility surface endogenous, and greedy policies provably suboptimal.

```
pip install -e ".[inference]"
OPENAI_API_KEY=sk-... MODEL_NAME=gpt-4o python -m hr.inference
```

---

## Benchmarks

Three agents. All scores deterministic, seed=42.

| Task | Teams | Hires | Random | Greedy (revenue/₹) | Gemini 3.1 Flash Lite |
|------|-------|-------|--------|--------------------|----------------------|
| Easy | 5 | 12 | 0.154 | 0.843 | **0.809** |
| Medium | 12 | 40 | 0.000 | 0.689 | **0.628** |
| Hard | 20 | 74 | 0.075 | 0.543 | **0.516** |
| **Avg** | | | **0.076** | **0.692** | **0.651** |

The greedy baseline uses exact server-side chemistry and diminishing-return formulas, strict threshold enforcement, and a budget-reservation heuristic. It has perfect knowledge of the reward function. It still can't clear 0.55 on Hard. The Gemini traces show the same failure mode — both collapse because they can't model how their own hires inflate adjacent salary buckets two or three moves out. That gap is the point.

Greedy: avg over 5 seeds (`rollout_demo.py`). LLM: `gemini-3.1-flash-lite-preview`, seed=42, temperature=0.1. Full traces in [`tests/inference_gemini_3.1_flash_lite.md`](tests/inference_gemini_3.1_flash_lite.md).

---

## Why standard policies fail here

**The base problem is NP-hard.** Assigning N candidates to K teams under per-team headcount and intel thresholds within a budget is a variant of multi-dimensional bin packing. CHROME makes three structural changes that rule out even approximate greedy solutions.

**1 — Endogenous costs.**
The salary market is a function of your own action history. Hiring candidate X from bucket `k` inflates the price of every remaining candidate in that bucket by `8%` (and adjacent buckets by `3%` in Hard mode). The optimal plan at step 1 is not the optimal plan at step 10, because you changed the landscape. This puts it closer to **online combinatorial optimization with endogenous costs** than classical NP-hard assignment — a regime where even an oracle with full lookahead must simulate execution to plan.

**2 — Non-separable value function.**
Revenue from a hire depends on the current team composition (chemistry) and team size (diminishing returns). You cannot score candidates independently. The marginal value of a Senior hire is entirely different depending on who's already on the team — and this dependency compounds across 20 teams simultaneously. Local search and greedy degrade badly in supermodular regimes; improving one team frequently worsens the global objective.

```python
chemistry = 1.0 - L1_distance(actual_type_mix, ideal_type_mix) / 2.0
effective_revenue = base_revenue * (0.5 + 0.5 * chemistry)
revenue_contribution = effective_revenue / sqrt(team_size)   # diminishing returns
```

**3 — No dominant strategy.**
The grader weights constraint satisfaction (30%), revenue ratio (60%), and cost efficiency (10%). These objectives genuinely conflict. Front-loading high-intel candidates maximises revenue but exhausts the budget before all teams are filled. Filling every team regardless of composition satisfies constraints but leaves revenue on the table. The hard task has 6 deterministic market shocks at fixed steps — at which point a policy that hasn't modelled market state is just reacting.

---

## Environment

### Action space (MCP tools)

| Tool | Parameters | Step cost |
|------|-----------|-----------|
| `hire_candidate` | `candidate_id`, `team_name`, `offered_salary` | 1 |
| `get_team_summary` | — | 0 |
| `get_market_ledger` | — | 0 |

Free observation tools mean an agent can reason arbitrarily before committing. Only hires consume the step budget.

### Observation space

Every tool call returns a unified JSON observation:

| Field | Type |
|-------|------|
| `success` | bool |
| `reward` | float — dense per-step signal |
| `done` | bool |
| `budget_remaining` | float (₹ Lakhs) |
| `revenue_projection` | float |
| `action_count` | int |
| `available_candidates` | list[dict] — id, type, intel, current_min_salary |
| `grader_score` | float `[0.0, 1.0]` — live episode estimate |

### Market dynamics

Five intelligence buckets. Price after N hires from bucket k:

| Bucket | Base (₹L) | Price formula |
|--------|-----------|--------------|
| 0–20 | 3.0 | `3.0 × (1 + N × 0.08)` |
| 21–40 | 5.0 | same |
| 41–60 | 8.0 | same |
| 61–80 | 12.0 | same |
| 81–100 | 18.0 | same |

Hard mode adds cross-bucket coupling: every hire in bucket k raises `k±1` by 3%.

### Reward (dense, per-step)

```
Successful hire:
  +0.50 × (intel / 100) × chemistry_multiplier
  +0.40  intel ≥ team threshold
  -0.30  intel < team threshold
  +0.20  salary within 5% of market minimum
  +1.00  hire completes the team
  -0.05  per action (time pressure)

Failed hire:
  -0.25  salary below market minimum
  -0.55  over budget
  -0.10  invalid target
```

### Grader (deterministic)

```
score = 0.30 × constraint_satisfaction
      + 0.60 × (actual_revenue / oracle_revenue)
      + 0.10 × (hiring_completeness × spend_efficiency)
```

The oracle ceiling is computed at static prices — deliberately unreachable under dynamic scarcity, consistent with standard RL benchmarking practice. Same seed, same actions → identical score, guaranteed.

### Task progression

| Task | Teams | Candidates | Budget | Max steps | Active mechanics |
|------|-------|-----------|--------|-----------|-----------------|
| Easy (0) | 5 | 100 | ₹130L | 40 | Static market |
| Medium (1) | 12 | 250 | ₹420L | 100 | Dynamic scarcity + chemistry + diminishing returns |
| Hard (2) | 20 | 500 | ₹740L | 150 | All above + coupled scarcity + 6 market shocks |

---

## Architecture

```
CHROME/
├── __init__.py              # HREnv, HRAction, HRObservation
├── models.py                # Pydantic contracts
├── client.py                # MCPToolClient (sync + async)
├── inference.py             # LLM baseline runner
├── openenv.yaml             # OpenEnv spec manifest
├── Dockerfile               # HF Spaces entry point
├── server/
│   ├── hr_environment.py    # Market mechanics, chemistry, grader
│   ├── app.py               # FastAPI / step / reset / state
│   └── Dockerfile
├── examples/
│   └── rollout_demo.py      # Greedy vs random policy comparison
└── tests/
    ├── test_hr_env.py
    └── inference_gemini_3.1_flash_lite.md
```

---

## Getting started

```bash
git clone https://github.com/rushisyeole/CHROME
cd CHROME
pip install -e ".[dev]"

# Server
uvicorn hr.server.app:app --host 0.0.0.0 --port 8000 --reload

# Tests
pytest tests/ -v

# Policy comparison (greedy vs random, 5 seeds)
python -m hr.examples.rollout_demo
```

### Client

```python
from hr.client import HREnv

with HREnv(base_url="http://localhost:8000").sync() as env:
    env.reset(task_id=2, seed=42)                    # Hard mode

    ledger = env.call_tool("get_market_ledger")      # free
    state  = env.call_tool("get_team_summary")       # free

    result = env.call_tool(
        "hire_candidate",
        candidate_id=42,
        team_name="Engineering",
        offered_salary=12.6,                         # must be >= market minimum
    )
    # {"success": True, "reward": 1.26, "done": False, "grader_score": 0.08, ...}
```

### LLM baseline

```bash
pip install -e ".[inference]"

# OpenAI-compatible
OPENAI_API_KEY=sk-...  API_BASE_URL=https://api.openai.com/v1  MODEL_NAME=gpt-4o \
python -m hr.inference

# HuggingFace router
HF_TOKEN=hf_...  MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
python -m hr.inference
```

Structured stdout:
```
[START] task=hard env=chrome model=gpt-4o
[STEP]  step=1 action=hire_candidate(42,'Engineering',12.6) reward=1.26 done=false error=null
[END]   success=true steps=50 score=0.516 rewards=1.26,0.68,...
```

### Docker

```bash
docker build -t chrome-env .
docker run -p 8000:8000 chrome-env
```

Live on HuggingFace Spaces: [huggingface.co/spaces/rushisyeole/chrome-env](https://huggingface.co/spaces/rushisyeole/chrome-env)

---

## Roadmap

**Multi-agent.** The current environment has one agent controlling all 20 teams. The natural extension is K specialized agents — one per team or per market tier — with a coordinator. This creates a non-cooperative game: each agent's hires inflate prices for the others, introducing competition over shared candidate pools that doesn't exist in the single-agent version. The market coupling already encodes the game-theoretic structure; adding per-agent budgets and independent episode scores is the primary thing needed.

**RL training loop.** The dense reward signal and fully deterministic resets are designed for policy gradient training. The obvious next step is a PPO or GRPO loop against all three difficulty levels and measuring whether a learned policy can outperform the greedy ceiling — specifically on Hard, where the greedy degrades most sharply. The oracle revenue gap on Hard (~0.46 headroom above greedy) is wide enough that a learned policy has real room to improve.

**Market calibration.** The current scarcity formula (`8% per hire from bucket`) is hand-tuned. Fitting these parameters to real compensation data (e.g., Glassdoor or Levels.fyi salary percentiles) would make CHROME a closer proxy for actual recruiting dynamics and more useful as an evaluation surface for applied HR automation research.

---