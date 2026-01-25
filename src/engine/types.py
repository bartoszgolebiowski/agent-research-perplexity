from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
    """Types of actions that the coordinator can request."""

    LLM_SKILL = "llm_skill"
    TOOL = "tool"
    COMPLETE = "complete"
    NOOP = "noop"


class FieldPriority(str, Enum):
    """Priority level for extraction fields."""

    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"


class AnalysisNodeStatus(str, Enum):
    """Lifecycle status of an analysis node."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStage(str, Enum):
    """Defines the stages of the agent workflow."""

    INITIAL = "INITIAL"
    COORDINATOR = "COORDINATOR"
    COMPLETED = "COMPLETED"

    # ---------------------------------------------------------------------------
    # ICP Intelligence System Workflow Stages
    # ---------------------------------------------------------------------------

    # Phase A: Ingestion
    ICP_INGESTION = "ICP_INGESTION"
    ICP_VALIDATION = "ICP_VALIDATION"
    ICP_QUEUE_READY = "ICP_QUEUE_READY"

    # Phase B: Research Loop
    ICP_NODE_START = "ICP_NODE_START"
    ICP_FORMULATE_QUERY = "ICP_FORMULATE_QUERY"
    ICP_EXECUTE_SEARCH = "ICP_EXECUTE_SEARCH"

    # Split processing (preferred for mid-sized LLMs)
    ICP_VERIFY_SOURCE = "ICP_VERIFY_SOURCE"
    ICP_EXTRACT_DATA = "ICP_EXTRACT_DATA"
    ICP_VALIDATE_EXTRACTION = "ICP_VALIDATE_EXTRACTION"
    ICP_DISCOVER_ENTITIES = "ICP_DISCOVER_ENTITIES"

    # Legacy merged processing (deprecated)
    ICP_PROCESS_RESULTS = "ICP_PROCESS_RESULTS"

    # Decision Gates
    ICP_VERIFY_DECISION = "ICP_VERIFY_DECISION"  # After source verification
    ICP_DECISION_GATE = "ICP_DECISION_GATE"  # After validation
    ICP_REFINE_QUERY = "ICP_REFINE_QUERY"  # Retry with refined query
    ICP_NODE_COMPLETE = "ICP_NODE_COMPLETE"
    ICP_NODE_FAILED = "ICP_NODE_FAILED"
    ICP_NODE_PARTIAL = "ICP_NODE_PARTIAL"
    ICP_NODE_SKIPPED = "ICP_NODE_SKIPPED"

    # Phase C: Dynamic Expansion
    ICP_EXPAND_TREE = "ICP_EXPAND_TREE"

    # Phase D: Report Generation
    ICP_ALL_NODES_DONE = "ICP_ALL_NODES_DONE"
    ICP_GENERATE_REPORT = "ICP_GENERATE_REPORT"
    ICP_REPORT_COMPLETE = "ICP_REPORT_COMPLETE"
