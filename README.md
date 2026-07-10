# CHROME: Cognitive Human Resource Optimization & Market Engine

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rushisyeole/chrome-env)
[![OpenEnv Compliance](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/openenv/openenv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

CHROME is an **OpenEnv-compliant Reinforcement Learning (RL) environment** and evaluation benchmark built to test how well AI agents can plan under real-world constraints.

In CHROME, an AI agent acts as a hiring manager tasked with building $K$ different teams from a pool of candidates under a strict budget. Unlike simple resource-matching puzzles, CHROME introduces **dynamic market scarcity** and **team chemistry dynamics**. Decisions made in the early steps directly alter the salary requirements and feasibility of all future steps, making simple "greedy" choice strategies fail.

---

## Core Leaderboard

The controlled **Core** track evaluates LLMs on a standardized, deterministic testbed. This benchmark measures multi-step planning, budget control, and how agents react to sudden market changes.

<!-- CHROME_CORE_RESULTS_START -->
| Model | Avg | Easy | Medium | Hard | Cost | Trace |
|---|---:|---:|---:|---:|---:|---|
| Heuristic Greedy baseline | 0.692 | 0.843 | 0.689 | 0.543 | $0.0000 | `rollout_demo.py` |
| `google/gemini-3-flash-preview` | 0.671 | 0.817 | 0.665 | 0.531 | $0.5854 | [trace](benchmark_results/core/google-gemini-3-flash-preview.md) |
| `google/gemini-2.5-flash` | 0.664 | 0.799 | 0.688 | 0.506 | $0.2723 | [trace](benchmark_results/core/google-gemini-2-5-flash.md) |
| `google/gemini-3.1-flash-lite` | 0.634 | 0.788 | 0.636 | 0.477 | $0.2016 | [trace](benchmark_results/core/google-gemini-3-1-flash-lite.md) |
| `meta-llama/llama-3.3-70b-instruct` | 0.613 | 0.841 | 0.603 | 0.395 | $0.5044 | [trace](benchmark_results/core/meta-llama-llama-3-3-70b-instruct.md) |
| `minimax/minimax-m3` | 0.534 | 0.655 | 0.618 | 0.329 | $1.4064 | [trace](benchmark_results/core/minimax-minimax-m3.md) |
| `google/gemini-3.5-flash` | FAIL | 0.772 | 0.746 | FAIL | $1.8079 | [trace](benchmark_results/core/google-gemini-3-5-flash.md) |
<!-- CHROME_CORE_RESULTS_END -->

> All core evals use standardized environment settings.

---

## Why Simple Strategies Fail Here

Standard hiring baselines or greedy models degrade quickly in CHROME because of three distinct environment mechanics:

### 1. Dynamic Salary Scarcity (Your actions change the market)
The salary market responds to the agent's choices. Hiring a candidate from a specific skill bucket depletes the supply, driving up the minimum wage requirement for all remaining candidates in that bucket (and adjacent buckets in Hard mode):

$$\text{Salary Floor} = \text{Base Salary} \times (1.0 + 0.08 \times \text{Hires} + 0.03 \times \text{Adjacent Hires})$$

* **Base Salary**: Base minimum salary for a candidate's skill tier (ranging from ₹3L to ₹18L).
* **Hires**: Total candidates hired out of that specific skill tier.
* **Adjacent Hires**: Hires made in neighboring skill tiers (reflecting cross-industry wage inflation).

Because hiring someone makes everyone else in their field more expensive, early high-value hires compress the remaining budget and inflate future candidate costs. Agents must balance spending early vs. saving for later.

### 2. Team Chemistry & Diminishing Returns (Candidates aren't isolated)
You cannot evaluate candidates in isolation. A hire's revenue contribution depends on who is already on the team (Chemistry) and experiences natural diminishing returns as the team grows:

$$\text{Revenue} = \frac{\text{Multiplier}}{\sqrt{\text{Team Size}}} \times \sum_{c} \left( \text{Intel}_c \times (0.5 + 0.5 \times \text{Chemistry}) \right)$$

* **Team Size**: Headcount currently hired to the team (enforcing diminishing returns via the square root).
* **Intel**: The candidate's capability score (0–100).
* **Chemistry**: Evaluated by comparing the hired roles to the ideal team mix:

$$\text{Chemistry} = 1.0 - \frac{1}{2} \sum_{\text{Role}} | \text{Actual}\% - \text{Ideal}\% |$$

Where `Actual%` is the proportion of Junior, Mid, Senior, or Lead engineers currently hired, and `Ideal%` is the target role mix. The value of adding a Senior engineer shifts depending on whether the team already has enough Juniors or Leads.

### 3. Exogenous Market Shocks (Surprise attrition)
In Hard Mode, the environment triggers competitor "poaching" campaigns at fixed steps (e.g., step 20, 40, 60...). These shocks remove high-scoring candidates directly from the pool, driving up scarcity costs without the agent's intervention. 

Agents that do not maintain budget buffers or fail to secure critical roles before these milestones run out of resources before filling their teams.

---

## Environment Interface (MCP Tools)

CHROME runs natively on the **Model Context Protocol (MCP)**, exposing environment controls as structured tools.

### Available Actions

| MCP Tool | Parameters | Step Cost | Description |
|:---|:---|:---:|:---|
| `hire_candidate` | `candidate_id` (int)<br>`team_name` (str)<br>`offered_salary` (float) | **1** | Hires the candidate for the team. Salary must meet the current market minimum. |
| `get_team_summary` | — | **0** | View headcount, roles, revenue projections, and available candidates. |
| `get_market_ledger` | — | **0** | Check the live salary floors for all skill tiers. |

*Note: Free observation tools allow agents to execute arbitrary planning, reasoning, or lookahead steps before committing to a permanent state change.*

---

## Episode Grader & Objective Function

At the end of an episode, the agent is scored on a scale of `[0.0, 1.0]` based on a balance of three goals:

$$\text{Grader Score} = 0.3 \times \text{Constraints} + 0.6 \times \text{Revenue Ratio} + 0.1 \times \text{Spend Efficiency}$$

1. **Constraints (30%)**: Percentage of teams fully staffed (headcount target met) with an average team intelligence score above the required threshold.
2. **Revenue Ratio (60%)**: Revenue generated compared to the **Aspirational Oracle Revenue** (a theoretical ceiling calculated at static prices).
3. **Spend Efficiency (10%)**: Incentivizes staffing the teams without overbidding:
$$\text{Spend Efficiency} = \text{Hiring Completeness} \times \text{Average } \left( \frac{\text{Market Min}}{\text{Paid Salary}} \right)$$

---

## Task Difficulty Levels

| Dimension | Easy (0) | Medium (1) | Hard (2) |
|:---|:---:|:---:|:---:|
| **Teams ($K$)** | 5 | 12 | 20 |
| **Candidates ($N$)** | 100 | 250 | 500 |
| **Budget** | ₹130L | ₹420L | ₹740L |
| **Max Steps** | 40 | 100 | 150 |
| **Dynamic Scarcity** | Static | Active (0.08) | Active (0.08) |
| **Coupled Scarcity** | Disabled | Disabled | Active (0.03) |
| **Team Chemistry** | Disabled | Enabled | Enabled |
| **Market Shocks** | None | None | 6 Deterministic Shocks |

---

## Getting Started

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/rushisyeole/CHROME.git
cd CHROME
pip install -e ".[dev]"
```

### Running the Environment Server
Start the FastAPI server, which exposes the MCP tools over HTTP:
```bash
uvicorn hr.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run Tests
```bash
pytest tests/ -v
```

### Baseline Rollouts
Compare a simple heuristic greedy policy against a random policy over 5 seeds:
```bash
python -m hr.examples.rollout_demo
```

---

## Evaluating LLM Agents on CHROME

To evaluate an LLM agent, run the inference runner. Ensure your `OPENROUTER_API_KEY` is exported:

```bash
pip install -e ".[inference]"

# Run a single evaluation run (e.g. Gemini 3.5 Flash)
OPENROUTER_API_KEY=sk-or-... MODEL_NAME=google/gemini-3.5-flash python -m hr.inference

# Run the complete CHROME-Core benchmark suite
OPENROUTER_API_KEY=sk-or-... python -m hr.benchmark
```
