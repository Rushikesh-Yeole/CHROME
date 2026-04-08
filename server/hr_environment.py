"""
CHROME — Cognitive Human Resource Optimization & Market Engine.

Server-side environment logic. Simulates a recruitment funnel where an
LLM agent hires candidates for teams under budget, intelligence-threshold,
and revenue-impact constraints, navigating a dynamic scarcity-based salary
market with team chemistry and diminishing returns.
"""

import math
import random
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

# ── Constants ────────────────────────────────────────────────────────────
BUCKETS = ["0-20", "21-40", "41-60", "61-80", "81-100"]
BASE_SALARIES = {"0-20": 3.0, "21-40": 5.0, "41-60": 8.0, "61-80": 12.0, "81-100": 18.0}
SCARCITY_RATE = 0.08
SPILLOVER_RATE = 0.03
ADJACENT_BUCKETS = {
    "0-20":   ["21-40"],
    "21-40":  ["0-20", "41-60"],
    "41-60":  ["21-40", "61-80"],
    "61-80":  ["41-60", "81-100"],
    "81-100": ["61-80"],
}
CANDIDATE_TYPES = ["Junior", "Mid", "Senior", "Lead"]

# ── Task Definitions ─────────────────────────────────────────────────────

TASKS = {
    0: {  # Easy — Chemistry off, static market, small scale
        "teams": [
            {"name": "Engineering", "required_intel_threshold": 50,
             "target_headcount": 3, "revenue_multiplier": 1.5,
             "ideal_mix": {"Junior": 0.33, "Mid": 0.34, "Senior": 0.33, "Lead": 0.0}},
            {"name": "Support", "required_intel_threshold": 30,
             "target_headcount": 3, "revenue_multiplier": 1.0,
             "ideal_mix": {"Junior": 0.50, "Mid": 0.50, "Senior": 0.0, "Lead": 0.0}},
            {"name": "Marketing", "required_intel_threshold": 40,
             "target_headcount": 2, "revenue_multiplier": 1.2,
             "ideal_mix": {"Junior": 0.50, "Mid": 0.25, "Senior": 0.25, "Lead": 0.0}},
            {"name": "Sales", "required_intel_threshold": 35,
             "target_headcount": 2, "revenue_multiplier": 1.1,
             "ideal_mix": {"Junior": 0.50, "Mid": 0.50, "Senior": 0.0, "Lead": 0.0}},
            {"name": "QA", "required_intel_threshold": 45,
             "target_headcount": 2, "revenue_multiplier": 1.3,
             "ideal_mix": {"Junior": 0.50, "Mid": 0.50, "Senior": 0.0, "Lead": 0.0}},
        ],
        "num_candidates": 100,
        "candidate_types": ["Junior", "Mid", "Senior"],
        "intel_range": (20, 80), "budget": 130.0,
        "dynamic_scarcity": False, "coupled_scarcity": False,
        "chemistry_enabled": False, "diminishing_returns": False,
        "shocks": {},
        "max_steps": 40,
    },
    1: {  # Medium — Chemistry on, scarcity on, coupled off, medium scale
        "teams": [
            {"name": "Engineering", "required_intel_threshold": 60,
             "target_headcount": 5, "revenue_multiplier": 2.0,
             "ideal_mix": {"Junior": 0.20, "Mid": 0.30, "Senior": 0.35, "Lead": 0.15}},
            {"name": "DataScience", "required_intel_threshold": 55,
             "target_headcount": 4, "revenue_multiplier": 1.8,
             "ideal_mix": {"Junior": 0.10, "Mid": 0.20, "Senior": 0.50, "Lead": 0.20}},
            {"name": "Marketing", "required_intel_threshold": 40,
             "target_headcount": 4, "revenue_multiplier": 1.3,
             "ideal_mix": {"Junior": 0.35, "Mid": 0.35, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Sales", "required_intel_threshold": 35,
             "target_headcount": 4, "revenue_multiplier": 1.2,
             "ideal_mix": {"Junior": 0.40, "Mid": 0.30, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Support", "required_intel_threshold": 30,
             "target_headcount": 4, "revenue_multiplier": 1.0,
             "ideal_mix": {"Junior": 0.50, "Mid": 0.30, "Senior": 0.15, "Lead": 0.05}},
            {"name": "QA", "required_intel_threshold": 50,
             "target_headcount": 3, "revenue_multiplier": 1.4,
             "ideal_mix": {"Junior": 0.25, "Mid": 0.40, "Senior": 0.25, "Lead": 0.10}},
            {"name": "DevOps", "required_intel_threshold": 55,
             "target_headcount": 3, "revenue_multiplier": 1.6,
             "ideal_mix": {"Junior": 0.15, "Mid": 0.35, "Senior": 0.35, "Lead": 0.15}},
            {"name": "ProductMgmt", "required_intel_threshold": 50,
             "target_headcount": 3, "revenue_multiplier": 1.5,
             "ideal_mix": {"Junior": 0.10, "Mid": 0.30, "Senior": 0.30, "Lead": 0.30}},
            {"name": "Design", "required_intel_threshold": 45,
             "target_headcount": 3, "revenue_multiplier": 1.3,
             "ideal_mix": {"Junior": 0.30, "Mid": 0.40, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Legal", "required_intel_threshold": 55,
             "target_headcount": 2, "revenue_multiplier": 1.1,
             "ideal_mix": {"Junior": 0.00, "Mid": 0.25, "Senior": 0.50, "Lead": 0.25}},
            {"name": "Finance", "required_intel_threshold": 50,
             "target_headcount": 3, "revenue_multiplier": 1.2,
             "ideal_mix": {"Junior": 0.20, "Mid": 0.40, "Senior": 0.30, "Lead": 0.10}},
            {"name": "HR", "required_intel_threshold": 40,
             "target_headcount": 2, "revenue_multiplier": 1.0,
             "ideal_mix": {"Junior": 0.30, "Mid": 0.40, "Senior": 0.20, "Lead": 0.10}},
        ],
        "num_candidates": 250,
        "candidate_types": ["Junior", "Mid", "Senior", "Lead"],
        "intel_range": (10, 95), "budget": 420.0,
        "dynamic_scarcity": True, "coupled_scarcity": False,
        "chemistry_enabled": True, "diminishing_returns": True,
        "shocks": {},
        "max_steps": 100,
    },
    2: {  # Hard — Everything on, large scale, shocks, coupled scarcity
        "teams": [
            {"name": "Engineering", "required_intel_threshold": 65,
             "target_headcount": 6, "revenue_multiplier": 2.2,
             "ideal_mix": {"Junior": 0.15, "Mid": 0.25, "Senior": 0.40, "Lead": 0.20}},
            {"name": "DataScience", "required_intel_threshold": 60,
             "target_headcount": 5, "revenue_multiplier": 2.0,
             "ideal_mix": {"Junior": 0.10, "Mid": 0.15, "Senior": 0.50, "Lead": 0.25}},
            {"name": "AI_Research", "required_intel_threshold": 70,
             "target_headcount": 4, "revenue_multiplier": 2.5,
             "ideal_mix": {"Junior": 0.05, "Mid": 0.15, "Senior": 0.45, "Lead": 0.35}},
            {"name": "DevOps", "required_intel_threshold": 55,
             "target_headcount": 4, "revenue_multiplier": 1.7,
             "ideal_mix": {"Junior": 0.20, "Mid": 0.35, "Senior": 0.30, "Lead": 0.15}},
            {"name": "Security", "required_intel_threshold": 60,
             "target_headcount": 3, "revenue_multiplier": 1.9,
             "ideal_mix": {"Junior": 0.10, "Mid": 0.25, "Senior": 0.40, "Lead": 0.25}},
            {"name": "ProductMgmt", "required_intel_threshold": 50,
             "target_headcount": 4, "revenue_multiplier": 1.6,
             "ideal_mix": {"Junior": 0.15, "Mid": 0.30, "Senior": 0.30, "Lead": 0.25}},
            {"name": "Marketing", "required_intel_threshold": 40,
             "target_headcount": 5, "revenue_multiplier": 1.4,
             "ideal_mix": {"Junior": 0.35, "Mid": 0.35, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Sales", "required_intel_threshold": 35,
             "target_headcount": 5, "revenue_multiplier": 1.3,
             "ideal_mix": {"Junior": 0.40, "Mid": 0.30, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Design", "required_intel_threshold": 45,
             "target_headcount": 4, "revenue_multiplier": 1.5,
             "ideal_mix": {"Junior": 0.30, "Mid": 0.35, "Senior": 0.25, "Lead": 0.10}},
            {"name": "QA", "required_intel_threshold": 50,
             "target_headcount": 4, "revenue_multiplier": 1.4,
             "ideal_mix": {"Junior": 0.25, "Mid": 0.40, "Senior": 0.25, "Lead": 0.10}},
            {"name": "Support", "required_intel_threshold": 30,
             "target_headcount": 5, "revenue_multiplier": 1.0,
             "ideal_mix": {"Junior": 0.50, "Mid": 0.30, "Senior": 0.15, "Lead": 0.05}},
            {"name": "Finance", "required_intel_threshold": 50,
             "target_headcount": 3, "revenue_multiplier": 1.3,
             "ideal_mix": {"Junior": 0.15, "Mid": 0.40, "Senior": 0.35, "Lead": 0.10}},
            {"name": "Legal", "required_intel_threshold": 55,
             "target_headcount": 3, "revenue_multiplier": 1.2,
             "ideal_mix": {"Junior": 0.00, "Mid": 0.25, "Senior": 0.50, "Lead": 0.25}},
            {"name": "HR", "required_intel_threshold": 40,
             "target_headcount": 3, "revenue_multiplier": 1.0,
             "ideal_mix": {"Junior": 0.30, "Mid": 0.40, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Logistics", "required_intel_threshold": 35,
             "target_headcount": 3, "revenue_multiplier": 1.1,
             "ideal_mix": {"Junior": 0.40, "Mid": 0.35, "Senior": 0.20, "Lead": 0.05}},
            {"name": "CustomerSuccess", "required_intel_threshold": 40,
             "target_headcount": 3, "revenue_multiplier": 1.2,
             "ideal_mix": {"Junior": 0.35, "Mid": 0.35, "Senior": 0.20, "Lead": 0.10}},
            {"name": "Analytics", "required_intel_threshold": 55,
             "target_headcount": 3, "revenue_multiplier": 1.6,
             "ideal_mix": {"Junior": 0.15, "Mid": 0.30, "Senior": 0.35, "Lead": 0.20}},
            {"name": "TechWriting", "required_intel_threshold": 45,
             "target_headcount": 2, "revenue_multiplier": 1.1,
             "ideal_mix": {"Junior": 0.25, "Mid": 0.50, "Senior": 0.25, "Lead": 0.00}},
            {"name": "Compliance", "required_intel_threshold": 50,
             "target_headcount": 2, "revenue_multiplier": 1.2,
             "ideal_mix": {"Junior": 0.10, "Mid": 0.30, "Senior": 0.40, "Lead": 0.20}},
            {"name": "BizDev", "required_intel_threshold": 45,
             "target_headcount": 3, "revenue_multiplier": 1.4,
             "ideal_mix": {"Junior": 0.20, "Mid": 0.35, "Senior": 0.30, "Lead": 0.15}},
        ],
        "num_candidates": 500,
        "candidate_types": ["Junior", "Mid", "Senior", "Lead"],
        "intel_range": (5, 98), "budget": 740.0,
        "dynamic_scarcity": True, "coupled_scarcity": True,
        "chemistry_enabled": True, "diminishing_returns": True,
        "shocks": {
            20: ("41-60", 3), 40: ("61-80", 2), 60: ("81-100", 2),
            80: ("21-40", 3), 100: ("61-80", 2), 120: ("41-60", 2),
        },
        "max_steps": 150,
    },
}


def _bucket_for(intel: int) -> str:
    if intel <= 20: return "0-20"
    if intel <= 40: return "21-40"
    if intel <= 60: return "41-60"
    if intel <= 80: return "61-80"
    return "81-100"


class HREnvironment(MCPEnvironment):
    """CHROME — Cognitive Human Resource Optimization & Market Engine."""

    def __init__(self) -> None:
        mcp = FastMCP("chrome_env")

        @mcp.tool
        def hire_candidate(candidate_id: int, team_name: str, offered_salary: float) -> dict:
            """Hire a candidate for a team at the offered salary (in Lakhs).
            Salary must meet or exceed the market minimum for the candidate's
            intelligence bucket. Deducts from budget, updates market ledger."""
            return self._handle_hire(candidate_id, team_name, offered_salary)

        @mcp.tool
        def get_team_summary() -> dict:
            """View all teams' hiring status, candidates, and revenue. Free (no step cost)."""
            return {
                "teams": deepcopy(self._teams),
                "available_candidates": self._candidate_summaries(),
                "revenue_projection": round(self._revenue, 4),
                "budget_remaining": round(self._budget, 2),
                "action_count": self._action_count,
                "max_possible_revenue": round(self._max_possible_revenue, 4) if hasattr(self, '_max_possible_revenue') else 0,
                "done": self._done,
                "grader_score": self._compute_grader(),
            }

        @mcp.tool
        def get_market_ledger() -> dict:
            """View the live market salary ledger. Free (no step cost)."""
            return deepcopy(self._ledger)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = True

    # ── OpenEnv interface ────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: int = 0,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        rng = random.Random(seed)
        cfg = TASKS[task_id]

        self._teams = deepcopy(cfg["teams"])
        for t in self._teams:
            t["current_headcount"] = 0

        self._initial_budget = cfg["budget"]
        self._budget = cfg["budget"]
        self._revenue = 0.0
        self._dynamic_scarcity = cfg["dynamic_scarcity"]
        self._coupled_scarcity = cfg.get("coupled_scarcity", False)
        self._chemistry_enabled = cfg.get("chemistry_enabled", False)
        self._diminishing_returns = cfg.get("diminishing_returns", False)
        self._shocks = dict(cfg["shocks"])

        # Generate candidates deterministically
        self._candidates = []
        for i in range(cfg["num_candidates"]):
            # Use realistic bell-curve distribution (centered at 55 std-dev 15)
            intel_raw = int(rng.gauss(55, 15))
            # Clamp between the task's allowed intel bounds
            intel = max(cfg["intel_range"][0], min(cfg["intel_range"][1], intel_raw))
            ctype = rng.choice(cfg["candidate_types"])
            self._candidates.append({
                "candidate_id": i, "intel_score": intel,
                "type": ctype, "intel_bucket": _bucket_for(intel),
            })

        # Build market ledger
        self._ledger = {}
        for b in BUCKETS:
            supply = sum(1 for c in self._candidates if c["intel_bucket"] == b)
            self._ledger[b] = {
                "base_min_salary": BASE_SALARIES[b],
                "initial_supply": supply,
                "remaining_supply": supply,
                "current_min_salary": BASE_SALARIES[b],
            }

        self._history: list[dict] = []
        self._action_count = 0
        self._max_steps = cfg.get("max_steps", 30)
        self._done = False
        self._task_id = task_id
        self._max_possible_revenue = self._compute_oracle()

        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0,
        )

        return Observation(done=False, reward=0.0, metadata={
            "message": (
                f"Episode started. Task {task_id} | Budget: ₹{self._budget}L | "
                f"{len(self._teams)} team(s) | {len(self._candidates)} candidates"
            ),
            "task_id": task_id,
            "budget": self._budget,
            "max_steps": self._max_steps,
            "teams": deepcopy(self._teams),
            "available_candidates": self._candidate_summaries(),
            "market_ledger": deepcopy(self._ledger),
            "max_possible_revenue": round(self._max_possible_revenue, 4),
        })

    def _step_impl(self, action: Action, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0, metadata={
            "error": "Use MCP tools: hire_candidate, get_team_summary, get_market_ledger"
        })

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state

    # ── Team Chemistry ───────────────────────────────────────────────────

    def _team_chemistry(self, team_name: str) -> float:
        """Returns 0.0–1.0 based on how well hires match team ideal_mix."""
        if not self._chemistry_enabled:
            return 1.0

        hires = [h for h in self._history if h["team"] == team_name]
        if not hires:
            return 1.0  # no penalty yet

        team = next(t for t in self._teams if t["name"] == team_name)
        ideal = team.get("ideal_mix", {})
        if not ideal:
            return 1.0

        # Compute actual type distribution
        type_counts = {ct: 0 for ct in CANDIDATE_TYPES}
        for h in hires:
            type_counts[h["type"]] = type_counts.get(h["type"], 0) + 1
        total = len(hires)
        actual = {k: v / total for k, v in type_counts.items()}

        # L1 distance (ranges 0–2), normalize to chemistry score 0–1
        distance = sum(abs(actual.get(k, 0) - ideal.get(k, 0)) for k in ideal)
        chemistry = max(0.0, 1.0 - distance / 2.0)
        return chemistry

    # ── Hire logic ───────────────────────────────────────────────────────

    def _handle_hire(self, candidate_id: int, team_name: str, offered_salary: float) -> dict:
        if self._done:
            return self._error("Episode over. Call reset().")

        self._action_count += 1
        self._apply_shocks()

        # Validate
        candidate = next((c for c in self._candidates if c["candidate_id"] == candidate_id), None)
        if not candidate:
            return self._result(False, "Invalid candidate_id", -0.1)

        team = next((t for t in self._teams if t["name"] == team_name), None)
        if not team:
            return self._result(False, "Invalid team_name", -0.1)

        if team["current_headcount"] >= team["target_headcount"]:
            return self._result(False, f"Team '{team_name}' already full", -0.1)

        bucket = candidate["intel_bucket"]
        min_sal = self._ledger[bucket]["current_min_salary"]

        if round(offered_salary, 2) < round(min_sal, 2):
            r = -0.20 - 0.05
            return self._result(False, f"Offer ₹{offered_salary}L < market min ₹{min_sal:.2f}L", r)

        if offered_salary > self._budget:
            r = -0.50 - 0.05
            return self._result(False, f"Insufficient budget (₹{self._budget:.2f}L left)", r)

        # ── Execute hire ──
        self._budget -= offered_salary
        team["current_headcount"] += 1

        # Update ledger scarcity (primary bucket)
        info = self._ledger[bucket]
        info["remaining_supply"] = max(0, info["remaining_supply"] - 1)
        if self._dynamic_scarcity:
            hired_count = info["initial_supply"] - info["remaining_supply"]
            info["current_min_salary"] = info["base_min_salary"] * (1 + hired_count * SCARCITY_RATE)

        # Coupled scarcity: adjacent buckets feel pressure
        if self._coupled_scarcity:
            for adj_bucket in ADJACENT_BUCKETS.get(bucket, []):
                adj_info = self._ledger[adj_bucket]
                adj_hired = adj_info["initial_supply"] - adj_info["remaining_supply"]
                adj_info["current_min_salary"] = adj_info["base_min_salary"] * (
                    1 + adj_hired * SCARCITY_RATE + SPILLOVER_RATE
                )

        # Revenue with diminishing returns and chemistry
        n = team["current_headcount"]  # already incremented
        base_rev = (candidate["intel_score"] / 100.0) * team["revenue_multiplier"]
        if self._diminishing_returns:
            base_rev *= (1.0 / math.sqrt(n))
        chemistry = self._team_chemistry(team_name)
        rev = base_rev * (0.5 + 0.5 * chemistry)
        self._revenue += rev

        # Remove from pool
        self._candidates = [c for c in self._candidates if c["candidate_id"] != candidate_id]

        self._history.append({
            "candidate_id": candidate_id, "team": team_name,
            "offered_salary": offered_salary, "intel_score": candidate["intel_score"],
            "type": candidate["type"],
            "revenue_contribution": rev,
            "market_min_at_hire": min_sal,
        })

        # Dense reward — flat time penalty
        reward = (candidate["intel_score"] / 100.0) * team["revenue_multiplier"] * 0.5
        if self._chemistry_enabled:
            reward *= chemistry
        if candidate["intel_score"] >= team["required_intel_threshold"]:
            reward += 0.40
        else:
            reward -= 0.30
        if min_sal > 0 and abs(offered_salary - min_sal) / min_sal <= 0.05:
            reward += 0.20
        # Team completion bonus
        if team["current_headcount"] >= team["target_headcount"]:
            reward += 1.0
        reward -= 0.05  # flat time penalty per action

        self._check_done()
        result = self._result(True, f"Hired #{candidate_id} → {team_name} @ ₹{offered_salary}L", reward)
        if self._done:
            result["grader_score"] = self._compute_grader()
        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    def _result(self, success: bool, message: str, reward: float) -> dict:
        self._check_done()
        return {
            "success": success, "message": message,
            "reward": round(reward, 4), "done": self._done,
            "budget_remaining": round(self._budget, 2),
            "revenue_projection": round(self._revenue, 4),
            "step_count": self._action_count,
            "available_candidates": self._candidate_summaries(),
            "grader_score": self._compute_grader(),
        }

    def _error(self, message: str) -> dict:
        return {"success": False, "message": message, "reward": 0.0, "done": True}

    def _candidate_summaries(self) -> list[dict]:
        return [
            {"candidate_id": c["candidate_id"], "intel_score": c["intel_score"],
             "type": c["type"], "intel_bucket": c["intel_bucket"],
             "current_min_salary": self._ledger[c["intel_bucket"]]["current_min_salary"]}
            for c in self._candidates
        ]

    def _check_done(self) -> None:
        if self._action_count >= self._max_steps:
            self._done = True
            return
        if self._budget <= 0:
            self._done = True
            return
        if self._budget < min(bucket["current_min_salary"] for bucket in self._ledger.values()):
            self._done = True
            return
        # All teams fully staffed
        if all(t["current_headcount"] >= t["target_headcount"] for t in self._teams):
            self._done = True

    def _apply_shocks(self) -> None:
        """Deterministic market shocks at fixed steps."""
        if self._action_count not in self._shocks:
            return
        bucket, count = self._shocks[self._action_count]
        info = self._ledger[bucket]
        removed = min(count, info["remaining_supply"])
        info["remaining_supply"] = max(0, info["remaining_supply"] - removed)
        if self._dynamic_scarcity:
            hired_count = info["initial_supply"] - info["remaining_supply"]
            info["current_min_salary"] = info["base_min_salary"] * (1 + hired_count * SCARCITY_RATE)

    # ── Greedy oracle ────────────────────────────────────────────────────

    def _compute_oracle(self) -> float:
        """Compute max possible revenue by greedy assignment of best candidates.
        Uses static base salaries as an aspirational ceiling benchmark."""
        sorted_teams = sorted(self._teams, key=lambda t: t["revenue_multiplier"], reverse=True)
        used: set[int] = set()
        total_rev = 0.0
        total_cost = 0.0
        team_hire_counts: dict[str, int] = {t["name"]: 0 for t in self._teams}

        for team in sorted_teams:
            valid = sorted(
                [c for c in self._candidates
                 if c["intel_score"] >= team["required_intel_threshold"]
                 and c["candidate_id"] not in used],
                key=lambda c: c["intel_score"], reverse=True,
            )
            hired = 0
            for c in valid:
                if hired >= team["target_headcount"]:
                    break
                cost = BASE_SALARIES[c["intel_bucket"]]
                if total_cost + cost <= self._initial_budget:
                    used.add(c["candidate_id"])
                    total_cost += cost
                    hired += 1
                    team_hire_counts[team["name"]] += 1
                    n = team_hire_counts[team["name"]]
                    rev = (c["intel_score"] / 100.0) * team["revenue_multiplier"]
                    if self._diminishing_returns:
                        rev *= (1.0 / math.sqrt(n))
                    total_rev += rev

        return max(total_rev, 0.001)

    # ── Deterministic grader ─────────────────────────────────────────────

    def _compute_grader(self) -> float:
        """Final episode score in [0.0, 1.0]."""
        # Constraint satisfaction: team filled + avg intel >= threshold
        satisfied = 0
        for team in self._teams:
            hires = [h for h in self._history if h["team"] == team["name"]]
            if len(hires) >= team["target_headcount"]:
                avg_intel = sum(h["intel_score"] for h in hires) / len(hires)
                if avg_intel >= team["required_intel_threshold"]:
                    satisfied += 1
        constraint_ratio = satisfied / len(self._teams) if self._teams else 0

        # Revenue ratio (vs aspirational oracle)
        rev_ratio = min(self._revenue / self._max_possible_revenue, 1.0)

        # Cost efficiency: rewards completing the job AND spending wisely
        total_target = sum(t["target_headcount"] for t in self._teams)
        if not self._history or total_target == 0:
            cost_ratio = 0.0
        else:
            hiring_completeness = min(len(self._history) / total_target, 1.0)
            spend_efficiency = sum(
                h["market_min_at_hire"] / h["offered_salary"]
                for h in self._history
            ) / len(self._history)
            cost_ratio = hiring_completeness * spend_efficiency

        score = 0.30 * constraint_ratio + 0.60 * rev_ratio + 0.10 * cost_ratio
        return round(max(0.0, min(1.0, score)), 4)
