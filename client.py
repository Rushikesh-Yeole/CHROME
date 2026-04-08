"""
Client for CHROME — Cognitive Human Resource Optimization & Market Engine.

Usage (sync):
    from hr.client import HREnv

    with HREnv(base_url="http://localhost:8000").sync() as env:
        env.reset(task_id=0)
        tools = env.list_tools()
        result = env.call_tool("hire_candidate", candidate_id=0,
                               team_name="Engineering", offered_salary=10.0)

Usage (async):
    async with HREnv(base_url="http://localhost:8000") as env:
        await env.reset(task_id=1)
        result = await env.call_tool("get_market_ledger")
"""

from openenv.core.mcp_client import MCPToolClient


class HREnv(MCPToolClient):
    """MCP client for CHROME — Cognitive Human Resource Optimization & Market Engine.

    Inherited methods: reset(), list_tools(), call_tool(), close().

    MCP tools:
      - hire_candidate(candidate_id, team_name, offered_salary)
      - get_team_summary()
      - get_market_ledger()
    """
    pass
