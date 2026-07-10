"""
rollout_demo.py — Compare a chemistry-aware greedy policy across all three CHROME tasks.

    cd CHROME/
    python -m hr.examples.rollout_demo

The greedy policy is designed as a proper algorithmic baseline:
  - Strict intel-threshold enforcement (never incur the -0.30 penalty)
  - Exact market-min salary offers (always captures the +0.20 pricing bonus)
  - Marginal revenue scoring: (intel/100) × rev_multiplier × 1/√(N+1) × chemistry_after
  - Revenue-per-rupee ranking to stretch budget across all teams
  - Diminishing-returns-aware team balancing (avoids front-loading one team)
  - Budget safety margin: reserves a floor before committing to expensive hires
"""

import math
import random
import threading
import time

import uvicorn

from hr import HREnv
from hr.server.app import app
from hr.server.hr_environment import TASKS

SERVER_URL = "http://127.0.0.1:8766"

TASK_MAX_STEPS = {0: 40, 1: 100, 2: 150}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _intel_bucket(intel: int) -> str:
    if intel <= 20:  return "0-20"
    if intel <= 40:  return "21-40"
    if intel <= 60:  return "41-60"
    if intel <= 80:  return "61-80"
    return "81-100"


def _chemistry_after(current_roster: list[dict], candidate_type: str,
                     ideal_mix: dict, target_headcount: int) -> float:
    """
    Compute team chemistry if we were to add a candidate of this type.

    chemistry = 1 - L1_distance(actual_mix, ideal_mix) / 2
    Mirrors the server-side _team_chemistry() implementation exactly.
    """
    if not ideal_mix:
        return 1.0
    hypothetical = current_roster + [{"type": candidate_type}]
    n = len(hypothetical)
    type_counts: dict[str, float] = {}
    for h in hypothetical:
        type_counts[h["type"]] = type_counts.get(h["type"], 0) + 1
    actual_mix = {t: type_counts.get(t, 0) / n for t in ideal_mix}
    l1 = sum(abs(actual_mix.get(t, 0) - ideal_mix.get(t, 0)) for t in ideal_mix)
    return 1.0 - l1 / 2.0


def _marginal_revenue(candidate: dict, team: dict,
                      roster: list[dict], chemistry_enabled: bool,
                      diminishing_returns: bool) -> float:
    """
    Expected revenue contribution of hiring this candidate onto the team.

    Matches server formula:
        base = (intel / 100) * revenue_multiplier
        if diminishing_returns: base /= sqrt(N)
        if chemistry:           base *= (0.5 + 0.5 * chemistry_after)
    """
    n = team["current_headcount"] + 1          # headcount after this hire
    base = (candidate["intel_score"] / 100.0) * team["revenue_multiplier"]
    if diminishing_returns:
        base /= math.sqrt(n)
    if chemistry_enabled:
        chem = _chemistry_after(
            roster,
            candidate.get("type", "Mid"),
            team.get("ideal_mix", {}),
            team["target_headcount"],
        )
        base *= (0.5 + 0.5 * chem)
    return base


# ── Policies ──────────────────────────────────────────────────────────────────

class GreedyPolicy:
    """
    Revenue-per-rupee greedy with chemistry awareness and budget planning.

    Decision loop:
      1. Filter to teams still needing hires.
      2. For each (team, candidate) pair where intel >= threshold:
             score = marginal_revenue(candidate, team) / offered_salary
      3. Pick the highest-scoring pair; offer exactly market minimum.
      4. If no threshold-safe hire exists, try a below-threshold hire only
         when it is the sole option remaining for a team (better than nothing
         for constraint_satisfaction in the grader).
      5. Reserve a budget safety margin proportional to remaining unfilled slots
         to avoid spending-out before all teams are serviced.
    """
    name = "Greedy (revenue/₹)"

    def __init__(self):
        # Track rosters locally to compute chemistry deltas without extra API calls
        self._rosters: dict[str, list[dict]] = {}
        self._chemistry_on = False
        self._diminishing_returns_on = False

    def reset(self, task_id: int = 0):
        self._rosters = {}
        config = TASKS[task_id]
        self._chemistry_on = config["chemistry_enabled"]
        self._diminishing_returns_on = config["diminishing_returns"]

    def _reserve(self, teams: list[dict], ledger_min: float) -> float:
        """
        Minimum budget to keep in reserve so we can afford at least one hire
        per still-unfilled team.  Uses the cheapest ledger price as a floor.
        """
        unfilled_teams = sum(
            1 for t in teams
            if t["current_headcount"] < t["target_headcount"]
        )
        return max(0.0, (unfilled_teams - 1) * ledger_min)

    def act(self, candidates: list[dict], teams: list[dict],
            budget_remaining: float) -> tuple | None:

        unfilled = [t for t in teams
                    if t["current_headcount"] < t["target_headcount"]]
        if not unfilled or not candidates:
            return None

        # Sync local roster cache with server state
        for team in teams:
            if team["name"] not in self._rosters:
                self._rosters[team["name"]] = []

        cheapest_salary = min(c["current_min_salary"] for c in candidates) if candidates else 0.0
        reserve = self._reserve(teams, cheapest_salary)

        best_score = -float("inf")
        best_action = None

        for team in unfilled:
            roster = self._rosters.get(team["name"], [])

            # Candidates that meet the intel threshold for this team
            valid = [c for c in candidates
                     if c["intel_score"] >= team["required_intel_threshold"]
                     and c["current_min_salary"] <= budget_remaining - reserve]

            # Relax budget reserve if this is the last unfilled team
            if not valid and len(unfilled) == 1:
                valid = [c for c in candidates
                         if c["intel_score"] >= team["required_intel_threshold"]
                         and c["current_min_salary"] <= budget_remaining]

            for cand in valid:
                salary = cand["current_min_salary"]  # exact min → +0.20 reward bonus
                rev = _marginal_revenue(
                    cand, team, roster,
                    chemistry_enabled=self._chemistry_on,
                    diminishing_returns=self._diminishing_returns_on,
                )
                score = rev / salary if salary > 0 else rev

                # Tie-break: prefer candidates that improve chemistry most
                chem_delta = 0.0
                if self._chemistry_on:
                    chem_delta = _chemistry_after(
                        roster, cand.get("type", "Mid"),
                        team.get("ideal_mix", {}), team["target_headcount"]
                    )
                score += chem_delta * 0.001  # small nudge, doesn't override economics

                if score > best_score:
                    best_score = score
                    best_action = ("hire", cand["candidate_id"], team["name"], salary)

        if best_action:
            return best_action

        # Fallback: if absolutely no threshold-safe hire exists, accept a below-
        # threshold hire for the team with the highest revenue_multiplier to
        # partially satisfy constraint_satisfaction in the grader.
        unfilled_sorted = sorted(unfilled, key=lambda t: t["revenue_multiplier"], reverse=True)
        for team in unfilled_sorted:
            affordable = [c for c in candidates
                          if c["current_min_salary"] <= budget_remaining]
            if affordable:
                # Pick the highest-intel affordable candidate
                best_fb = max(affordable, key=lambda c: c["intel_score"])
                return ("hire", best_fb["candidate_id"], team["name"],
                        best_fb["current_min_salary"])

        return None

    def on_hire(self, team_name: str, candidate: dict) -> None:
        """Call this after a successful hire to keep the local roster up to date."""
        if team_name not in self._rosters:
            self._rosters[team_name] = []
        self._rosters[team_name].append({"type": candidate.get("type", "Mid")})


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(env, policy, task_id: int, seed: int) -> tuple[float, float, int]:
    """Run one full episode. Returns (total_reward, grader_score, hire_count)."""
    policy.reset(task_id)


if __name__ == "__main__":
    main()
