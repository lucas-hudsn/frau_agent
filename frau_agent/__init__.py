import os

# Silence pydantic-ai's first-run telemetry banner; must be set before import.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from .agent import build_agent, build_model  # noqa: E402

__all__ = ["build_agent", "build_model"]
