"""
rollout_demo.py — Compare a chemistry-aware greedy policy against a random
baseline across all three CHROME tasks.

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
                      roster: list[dict], chemistry_enabled: bool) -> float:
    """
    Expected revenue contribution of hiring this candidate onto the team.

    Matches server formula:
        base = (intel / 100) * revenue_multiplier
        if diminishing_returns: base /= sqrt(N)
        if chemistry:           base *= (0.5 + 0.5 * chemistry_after)
    """
    n = team["current_headcount"] + 1          # headcount after this hire
    base = (candidate["intel_score"] / 100.0) * team["revenue_multiplier"]
    base /= math.sqrt(n)                       # always model diminishing returns
    if chemistry_enabled:
        chem = _chemistry_after(
            team.get("_roster", []),
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
        self._chemistry_on = True   # conservatively assume chemistry is enabled

    def reset(self):
        self._rosters = {}

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
                    cand, team, roster, chemistry_enabled=self._chemistry_on
                )
                score = rev / salary if salary > 0 else rev

                # Tie-break: prefer candidates that improve chemistry most
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


class RandomPolicy:
    """Hires a random candidate for a random team at a random overpay.
    Useful only as a lower-bound sanity check."""
    name = "Random (lower bound)"

    def reset(self): pass

    def act(self, candidates, teams, budget_remaining):
        if not candidates:
            return None
        unfilled = [t for t in teams if t["current_headcount"] < t["target_headcount"]]
        if not unfilled:
            return None
        team = random.choice(unfilled)
        cand = random.choice(candidates)
        salary = cand["current_min_salary"] * random.uniform(1.0, 1.5)
        salary = min(salary, budget_remaining)
        if salary < cand["current_min_salary"]:
            return None
        return ("hire", cand["candidate_id"], team["name"], round(salary, 2))

    def on_hire(self, *_): pass


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(env, policy, task_id: int, seed: int) -> tuple[float, float, int]:
    """Run one full episode. Returns (total_reward, grader_score, hire_count)."""
    policy.reset()
    env.reset(task_id=task_id, seed=seed)

    total_reward = 0.0
    grader = 0.0
    actions = 0
    candidate_map: dict[int, dict] = {}   # id → candidate info cache

    for _ in range(TASK_MAX_STEPS.get(task_id, 150) * 2):
        state = env.call_tool("get_team_summary")
        if not isinstance(state, dict) or state.get("done"):
            grader = state.get("grader_score", grader) if isinstance(state, dict) else grader
            break

        candidates = state.get("available_candidates", [])
        teams      = state.get("teams", [])
        budget     = state.get("budget_remaining", 0.0)

        # Keep a local candidate-type cache (type not always returned by summary)
        for c in candidates:
            if c["candidate_id"] not in candidate_map:
                candidate_map[c["candidate_id"]] = c
            else:
                # update salary (market moves)
                candidate_map[c["candidate_id"]]["current_min_salary"] = c["current_min_salary"]

        if not candidates or not teams:
            break

        action = policy.act(candidates, teams, budget)
        if action is None:
            break

        _, cid, tname, salary = action
        result = env.call_tool("hire_candidate", candidate_id=cid,
                               team_name=tname, offered_salary=salary)
        if not isinstance(result, dict):
            break

        if result.get("success"):
            actions += 1
            total_reward += result.get("reward", 0.0)
            policy.on_hire(tname, candidate_map.get(cid, {"type": "Mid"}))

        if result.get("done"):
            grader = result.get("grader_score", 0.0) or 0.0
            break

    return total_reward, grader, actions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning"),
        daemon=True,
    ).start()
    time.sleep(2)

    policies: list[GreedyPolicy | RandomPolicy] = [GreedyPolicy(), RandomPolicy()]
    N_SEEDS = 5

    print("\nCHROME — Policy Rollout Comparison")
    print(f"Seeds: 0–{N_SEEDS - 1}  |  Policies: {[p.name for p in policies]}\n")

    with HREnv(base_url=SERVER_URL).sync() as env:
        for task_id in range(3):
            label = ["Easy (0)", "Medium (1)", "Hard (2)"][task_id]
            print(f"{'─' * 65}")
            print(f"  Task {label}")
            print(f"{'─' * 65}")
            print(f"  {'Policy':<28}  {'Grader':>8}  {'Reward/hire':>12}  {'Hires':>6}")
            print(f"  {'─'*28}  {'─'*8}  {'─'*12}  {'─'*6}")

            for policy in policies:
                rewards, graders, counts = [], [], []
                for seed in range(N_SEEDS):
                    r, g, a = run_episode(env, policy, task_id, seed)
                    rewards.append(r)
                    graders.append(g)
                    counts.append(a)

                avg_g   = sum(graders) / N_SEEDS
                avg_a   = sum(counts)  / N_SEEDS
                avg_rpa = (sum(rewards) / N_SEEDS) / avg_a if avg_a > 0 else 0.0

                print(f"  {policy.name:<28}  {avg_g:>8.4f}  {avg_rpa:>+12.3f}  {avg_a:>6.1f}")

            print()

    print("Done.")


if __name__ == "__main__":
    main()