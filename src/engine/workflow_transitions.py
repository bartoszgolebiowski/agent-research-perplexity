from __future__ import annotations

from typing import Dict, Tuple, Union

from .types import ActionType, WorkflowStage
from ..skills.base import SkillName
from ..tools.models import ToolName


TRANSITIONS: Dict[WorkflowStage, Tuple[ActionType, Union[SkillName, ToolName], str]] = {
    # ---------------------------------------------------------------------------
    # Original Workflow Transitions
    # ---------------------------------------------------------------------------
    WorkflowStage.INITIAL: (
        ActionType.TOOL,
        ToolName.HELLO_WORLD,
        "Analyzing the user query and planning next steps",
    ),
    WorkflowStage.COORDINATOR: (
        ActionType.LLM_SKILL,
        SkillName.ANALYZE_AND_PLAN,
        "Coordinating the next actions based on the current state",
    ),
    WorkflowStage.COMPLETED: (
        ActionType.COMPLETE,
        SkillName.HELLO_WORLD,
        "Workflow completed successfully",
    ),
    
    # ---------------------------------------------------------------------------
    # ICP Intelligence System Workflow Transitions
    # ---------------------------------------------------------------------------
    
    # Phase A: Ingestion (handled specially in ICPCoordinator)
    WorkflowStage.ICP_INGESTION: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "File ingestion - handled externally",
    ),
    WorkflowStage.ICP_VALIDATION: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Input validation - handled externally",
    ),
    WorkflowStage.ICP_QUEUE_READY: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Queue initialized - proceeding to first node",
    ),
    
    # Phase B: Research Loop
    WorkflowStage.ICP_NODE_START: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
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
    WorkflowStage.ICP_EXTRACT_DATA: (
        ActionType.LLM_SKILL,
        SkillName.EXTRACT_DATA,
        "Extracting structured data from search results",
    ),
    WorkflowStage.ICP_VALIDATE_DATA: (
        ActionType.LLM_SKILL,
        SkillName.VALIDATE_DATA,
        "Validating extracted data completeness",
    ),
    
    # Decision Gates
    WorkflowStage.ICP_DECISION_GATE: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Decision gate - routing based on validation result",
    ),
    WorkflowStage.ICP_REFINE_QUERY: (
        ActionType.LLM_SKILL,
        SkillName.REFINE_QUERY,
        "Refining search query for retry attempt",
    ),
    WorkflowStage.ICP_NODE_COMPLETE: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Node completed successfully",
    ),
    WorkflowStage.ICP_NODE_FAILED: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Node failed after exhausting retries",
    ),
    WorkflowStage.ICP_NODE_PARTIAL: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Node partially completed",
    ),
    WorkflowStage.ICP_NODE_SKIPPED: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Node skipped (nice-to-have only)",
    ),
    
    # Phase C: Dynamic Expansion
    WorkflowStage.ICP_DISCOVER_ENTITIES: (
        ActionType.LLM_SKILL,
        SkillName.DISCOVER_ENTITIES,
        "Discovering new entities for dynamic expansion",
    ),
    WorkflowStage.ICP_EXPAND_TREE: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Expanding analysis tree with discovered entities",
    ),
    
    # Phase D: Report Generation
    WorkflowStage.ICP_ALL_NODES_DONE: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "All nodes processed - preparing report",
    ),
    WorkflowStage.ICP_GENERATE_REPORT: (
        ActionType.NOOP,
        SkillName.HELLO_WORLD,
        "Generating HTML report from collected data",
    ),
    WorkflowStage.ICP_REPORT_COMPLETE: (
        ActionType.COMPLETE,
        SkillName.HELLO_WORLD,
        "ICP analysis complete - report generated",
    ),
}
