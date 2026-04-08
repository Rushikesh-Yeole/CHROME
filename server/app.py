"""FastAPI application for CHROME — Cognitive Human Resource Optimization & Market Engine."""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .hr_environment import HREnvironment
except (ImportError, ValueError):
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from hr.server.hr_environment import HREnvironment

app = create_app(
    HREnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="chrome_env",
)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
