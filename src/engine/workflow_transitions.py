from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

from .types import ActionType, WorkflowStage
from ..skills.base import SkillName
from ..tools.models import ToolName


# For NOOP and COMPLETE actions, the skill/tool is not used but we need a placeholder
_NOOP_PLACEHOLDER: Optional[SkillName] = None


TRANSITIONS: Dict[
    WorkflowStage, Tuple[ActionType, Optional[Union[SkillName, ToolName]], str]
] = {
    # ---------------------------------------------------------------------------
    # ICP Intelligence System Workflow Transitions
    # ---------------------------------------------------------------------------
    # Phase A: Ingestion (handled specially in ICPCoordinator)
    WorkflowStage.ICP_INGESTION: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "File ingestion - handled externally",
    ),
    WorkflowStage.ICP_VALIDATION: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Input validation - handled externally",
    ),
    WorkflowStage.ICP_QUEUE_READY: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Queue initialized - proceeding to first node",
    ),
    # Phase B: Research Loop
    WorkflowStage.ICP_NODE_START: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Starting analysis of next node",
    ),
    WorkflowStage.ICP_FORMULATE_QUERY: (
        ActionType.LLM_SKILL,
        SkillName.FORMULATE_QUERY,
        "Formulating optimized search query for current node",
    ),
    WorkflowStage.ICP_EXECUTE_SEARCH: (
        ActionType.TOOL,
        ToolName.WEB_SEARCH,
        "Executing web search with formulated query",
    ),
    # Split processing for better reliability with mid-sized LLMs
    WorkflowStage.ICP_VERIFY_SOURCE: (
        ActionType.LLM_SKILL,
        SkillName.VERIFY_SOURCE,
        "Verifying search results are about target company",
    ),
    WorkflowStage.ICP_VERIFY_DECISION: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Decision gate - routing based on source verification",
    ),
    WorkflowStage.ICP_EXTRACT_DATA: (
        ActionType.LLM_SKILL,
        SkillName.EXTRACT_DATA,
        "Extracting structured data from verified results",
    ),
    WorkflowStage.ICP_VALIDATE_EXTRACTION: (
        ActionType.LLM_SKILL,
        SkillName.VALIDATE_EXTRACTION,
        "Validating extraction completeness",
    ),
    WorkflowStage.ICP_DISCOVER_ENTITIES: (
        ActionType.LLM_SKILL,
        SkillName.DISCOVER_ENTITIES,
        "Discovering new entities for investigation",
    ),
    # Decision Gates
    WorkflowStage.ICP_DECISION_GATE: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Decision gate - routing based on validation result",
    ),
    WorkflowStage.ICP_REFINE_QUERY: (
        ActionType.LLM_SKILL,
        SkillName.REFINE_QUERY,
        "Refining search query for retry attempt",
    ),
    WorkflowStage.ICP_NODE_COMPLETE: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Node completed successfully",
    ),
    WorkflowStage.ICP_NODE_FAILED: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Node failed after exhausting retries",
    ),
    WorkflowStage.ICP_NODE_PARTIAL: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Node partially completed",
    ),
    WorkflowStage.ICP_NODE_SKIPPED: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Node skipped (nice-to-have only)",
    ),
    # Phase C: Dynamic Expansion
    WorkflowStage.ICP_EXPAND_TREE: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Expanding analysis tree with discovered entities",
    ),
    # Phase D: Report Generation
    WorkflowStage.ICP_ALL_NODES_DONE: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "All nodes processed - preparing report",
    ),
    WorkflowStage.ICP_GENERATE_REPORT: (
        ActionType.NOOP,
        _NOOP_PLACEHOLDER,
        "Generating HTML report from collected data",
    ),
    WorkflowStage.ICP_REPORT_COMPLETE: (
        ActionType.COMPLETE,
        _NOOP_PLACEHOLDER,
        "ICP analysis complete - report generated",
    ),
}
