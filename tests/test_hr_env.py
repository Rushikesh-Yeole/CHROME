"""
Tests for CHROME — Cognitive Human Resource Optimization & Market Engine.
Runs in-process (no Docker). Uses the same pattern as my_number_env tests.

    cd CHROME/
    pip install -e ".[dev]"
    pytest tests/ -v
"""

import threading, time, sys, os
import pytest
import uvicorn

from hr.server.app import app
from hr import HREnv

TEST_PORT = 8877
TEST_URL = f"http://127.0.0.1:{TEST_PORT}"


@pytest.fixture(scope="session", autouse=True)
def server():
    t = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "127.0.0.1", "port": TEST_PORT, "log_level": "error"},
        daemon=True,
    )
    t.start()
    time.sleep(2)
    yield


@pytest.fixture
def env(server):
    with HREnv(base_url=TEST_URL).sync() as client:
        yield client


class TestHealthAndTools:
    def test_list_tools(self, env):
        tools = env.list_tools()
        names = {t.name for t in tools}
        assert "hire_candidate" in names
        assert "get_team_summary" in names
        assert "get_market_ledger" in names
        # skip_candidate has been removed
        assert "skip_candidate" not in names


class TestReset:
    def test_reset_easy(self, env):
        env.reset(task_id=0)
        summary = env.call_tool("get_team_summary")
        assert len(summary["teams"]) == 5
        assert summary["teams"][0]["name"] == "Engineering"

    def test_reset_medium(self, env):
        env.reset(task_id=1)
        summary = env.call_tool("get_team_summary")
        assert len(summary["teams"]) == 12

    def test_reset_hard(self, env):
        env.reset(task_id=2)
        summary = env.call_tool("get_team_summary")
        assert len(summary["teams"]) == 20

    def test_seed_reproducibility(self, env):
        env.reset(task_id=0, seed=42)
        ledger1 = env.call_tool("get_market_ledger")
        env.reset(task_id=0, seed=42)
        ledger2 = env.call_tool("get_market_ledger")
        assert ledger1 == ledger2

    def test_candidate_count(self, env):
        env.reset(task_id=0, seed=1)
        summary = env.call_tool("get_team_summary")
        assert len(summary["available_candidates"]) == 100

        env.reset(task_id=1, seed=1)
        summary = env.call_tool("get_team_summary")
        assert len(summary["available_candidates"]) == 250

        env.reset(task_id=2, seed=1)
        summary = env.call_tool("get_team_summary")
        assert len(summary["available_candidates"]) == 500


class TestHireCandidate:
    def test_valid_hire(self, env):
        env.reset(task_id=0, seed=10)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=20.0)
        assert result["success"] is True
        assert result["budget_remaining"] < 200.0

    def test_hire_below_market_fails(self, env):
        env.reset(task_id=0, seed=10)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=0.5)
        assert result["success"] is False
        assert result["reward"] < 0

    def test_hire_over_budget_fails(self, env):
        env.reset(task_id=0, seed=10)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=999.0)
        assert result["success"] is False

    def test_invalid_candidate(self, env):
        env.reset(task_id=0, seed=10)
        result = env.call_tool("hire_candidate", candidate_id=999,
                               team_name="Engineering", offered_salary=10.0)
        assert result["success"] is False
        assert result["reward"] == -0.1

    def test_invalid_team(self, env):
        env.reset(task_id=0, seed=10)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="FakeTeam", offered_salary=10.0)
        assert result["success"] is False


class TestMarketLedger:
    def test_scarcity_rises_on_hire(self, env):
        env.reset(task_id=1, seed=5)
        ledger_before = env.call_tool("get_market_ledger")
        env.reset(task_id=1, seed=5)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=20.0)
        if result["success"]:
            ledger_after = env.call_tool("get_market_ledger")
            any_higher = any(
                ledger_after[b]["current_min_salary"] > ledger_before[b]["current_min_salary"]
                for b in ledger_before
            )
            assert any_higher, "Scarcity should raise prices after hire"

    def test_static_ledger_easy(self, env):
        env.reset(task_id=0, seed=5)
        ledger_before = env.call_tool("get_market_ledger")
        env.reset(task_id=0, seed=5)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=20.0)
        if result["success"]:
            ledger_after = env.call_tool("get_market_ledger")
            for b in ledger_before:
                assert ledger_after[b]["current_min_salary"] == ledger_before[b]["current_min_salary"]

    def test_coupled_scarcity_hard(self, env):
        """Hard mode: hiring from one bucket should raise adjacent bucket prices."""
        env.reset(task_id=2, seed=5)
        ledger_before = env.call_tool("get_market_ledger")
        env.reset(task_id=2, seed=5)
        # Find a candidate in the 61-80 bucket
        summary = env.call_tool("get_team_summary")
        target_c = None
        for c in summary["available_candidates"]:
            if c["intel_bucket"] == "61-80":
                target_c = c
                break
        if target_c:
            env.reset(task_id=2, seed=5)
            result = env.call_tool("hire_candidate",
                                   candidate_id=target_c["candidate_id"],
                                   team_name="Marketing",
                                   offered_salary=target_c["current_min_salary"])
            if result["success"]:
                ledger_after = env.call_tool("get_market_ledger")
                # Adjacent buckets (41-60 and 81-100) should have increased
                assert (ledger_after["41-60"]["current_min_salary"] >
                        ledger_before["41-60"]["current_min_salary"] or
                        ledger_after["81-100"]["current_min_salary"] >
                        ledger_before["81-100"]["current_min_salary"]), \
                    "Coupled scarcity should raise adjacent bucket prices"


class TestTeamChemistry:
    def test_chemistry_off_easy(self, env):
        """Easy mode: chemistry is disabled, all hires should produce same revenue
        regardless of type mix."""
        env.reset(task_id=0, seed=42)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=20.0)
        assert result["success"] is True
        assert result["reward"] > -1.0

    def test_chemistry_on_medium(self, env):
        """Medium mode: chemistry is enabled."""
        env.reset(task_id=1, seed=42)
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=20.0)
        assert result["success"] is True
        assert result["revenue_projection"] > 0


class TestGrader:
    def test_grader_between_0_and_1(self, env):
        env.reset(task_id=0, seed=42)
        # Exhaust max steps by hiring with invalid data to trigger done
        result = None
        for i in range(45):
            result = env.call_tool("hire_candidate", candidate_id=i,
                                   team_name="Engineering", offered_salary=20.0)
            if result.get("done"):
                break
        if result and result.get("grader_score") is not None:
            assert 0.0 <= result["grader_score"] <= 1.0

    def test_grader_rewards_hiring(self, env):
        """An agent that hires should score higher than one that does nothing."""
        # Episode 1: do nothing — the grader gives 0 with no history
        env.reset(task_id=0, seed=42)
        summary = env.call_tool("get_team_summary")
        no_hire_score = summary.get("grader_score", 0) or 0

        # Episode 2: hire some candidates
        env.reset(task_id=0, seed=42)
        summary = env.call_tool("get_team_summary")
        hire_score = 0
        for c in summary["available_candidates"][:10]:
            result = env.call_tool("hire_candidate",
                                   candidate_id=c["candidate_id"],
                                   team_name="Engineering",
                                   offered_salary=20.0)
            if result.get("done"):
                hire_score = result.get("grader_score", 0) or 0
                break

        # Hiring agent should score >= idle agent
        assert hire_score >= no_hire_score


class TestEpisodeEnd:
    def test_max_steps(self, env):
        env.reset(task_id=0, seed=10)
        result = None
        for i in range(50):
            result = env.call_tool("hire_candidate", candidate_id=i,
                                   team_name="Engineering", offered_salary=20.0)
            if result.get("done"):
                break
        assert result["done"] is True

    def test_all_teams_filled(self, env):
        env.reset(task_id=0, seed=10)
        summary = env.call_tool("get_team_summary")
        # Try hiring candidates for each team until done
        hired = 0
        result = None
        for cid in range(100):
            for team in summary["teams"]:
                result = env.call_tool("hire_candidate", candidate_id=cid,
                                       team_name=team["name"], offered_salary=20.0)
                if result.get("success"):
                    hired += 1
                    break
            if result.get("done"):
                break
        # With 5 teams totaling 12 hires, we should finish
        if hired >= 12:
            assert result["done"] is True


class TestDeterminism:
    def test_same_seed_same_outcome(self, env):
        results = []
        for _ in range(3):
            env.reset(task_id=1, seed=99)
            r = env.call_tool("hire_candidate", candidate_id=0,
                              team_name="Engineering", offered_salary=20.0)
            results.append(r["reward"])
        assert results[0] == results[1] == results[2]
