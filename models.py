"""
Data models for CHROME — Cognitive Human Resource Optimization & Market Engine.

Action and Observation Pydantic contracts for the wire protocol.
In practice, MCP tool calls are used instead of raw HRAction,
but these provide type documentation for the API.
"""

from typing import Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class HRAction(Action):
    """Action the agent can take in the HR environment."""
    action_type: str = Field(..., description="One of: hire_candidate, get_team_summary, get_market_ledger")
    candidate_id: Optional[int] = Field(default=None, description="ID of the candidate to act on")
    team_name: Optional[str] = Field(default=None, description="Target team name")
    offered_salary: Optional[float] = Field(default=None, description="Offered salary in Lakhs")


class HRObservation(Observation):
    """Observation returned after each action."""
    message: str = Field(default="", description="Human-readable feedback")
    available_candidates: List[Dict] = Field(default_factory=list, description="Current candidate queue")
    team_status: Dict[str, Dict] = Field(default_factory=dict, description="Team hiring status")
    market_ledger_snapshot: Dict[str, float] = Field(default_factory=dict, description="Current salary ledger")
    current_revenue_projection: float = Field(default=0.0, description="Accumulated revenue")
    step_count: int = Field(default=0, description="Actions taken so far")
    budget_remaining: float = Field(default=0.0, description="Remaining budget in Lakhs")
    max_possible_revenue: float = Field(default=0.0, description="Oracle-computed max revenue")
