"""Engine layer entry points."""

from .types import ActionType, AnalysisNodeStatus, FieldPriority, WorkflowStage

__all__ = [
    "ActionType",
    "AnalysisNodeStatus",
    "FieldPriority",
    "WorkflowStage",
]


def get_coordinator():
    """Lazy import to avoid circular dependency."""
    from .coordinator import AgentActionCoordinator

    return AgentActionCoordinator


def get_executor():
    """Lazy import to avoid circular dependency."""
    from .executor import LLMExecutor

    return LLMExecutor
