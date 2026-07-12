"""Multi-agent workflow components."""

from scholar_agent.agents.prototype_loop import (
    PrototypeLoopConfig,
    run_prototype_loop,
)
from scholar_agent.agents.researcher import (
    ResearchAgent,
    ResearchAgentConfig,
    ResearchPassResult,
    ResearchRunResult,
)

__all__ = [
    "PrototypeLoopConfig",
    "ResearchAgent",
    "ResearchAgentConfig",
    "ResearchPassResult",
    "ResearchRunResult",
    "run_prototype_loop",
]
