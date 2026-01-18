from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from src.skills.base import SkillName
from src.skills.models import (
    DiscoverEntitiesOutput,
    ExtractDataOutput,
    FormulateQueryOutput,
    RefineQueryOutput,
    ValidateDataOutput,
)
from src.tools.models import ToolName, WebSearchResponse

from .models import (
    AgentState,
    ConstitutionalMemory,
    EpisodicMemory,
    ProceduralMemory,
    ResourceMemory,
    SemanticMemory,
    WorkflowMemory,
    WorkingMemory,
)
from src.engine.types import WorkflowStage

SkillHandler = Callable[[AgentState, BaseModel], AgentState]
ToolHandler = Callable[[AgentState, BaseModel], AgentState]


def create_initial_state(
    goal: Optional[str] = None,
    core: Optional[ConstitutionalMemory] = None,
    semantic: Optional[SemanticMemory] = None,
    episodic: Optional[EpisodicMemory] = None,
    workflow: Optional[WorkflowMemory] = None,
    working: Optional[WorkingMemory] = None,
    procedural: Optional[ProceduralMemory] = None,
    resource: Optional[ResourceMemory] = None,
) -> AgentState:
    """Initialize a fully-populated state tree."""
    if goal is None and workflow is None:
        raise ValueError("Either goal or workflow must be provided")

    core = core or ConstitutionalMemory()
    semantic = semantic or SemanticMemory()
    episodic = episodic or EpisodicMemory()
    workflow = workflow or WorkflowMemory(goal=goal)  # type: ignore
    working = working or WorkingMemory()
    procedural = procedural or ProceduralMemory()
    resource = resource or ResourceMemory()

    return AgentState(
        core=core,
        semantic=semantic,
        episodic=episodic,
        workflow=workflow,
        working=working,
        procedural=procedural,
        resource=resource,
    )


def create_icp_initial_state(
    analysis_input: Any,
    queue: Any,
) -> AgentState:
    """Initialize state for an ICP analysis workflow."""
    state = create_initial_state(
        goal=f"ICP Analysis: {analysis_input.name}",
        workflow=WorkflowMemory(
            goal=f"ICP Analysis: {analysis_input.name}",
            is_icp_workflow=True,
            current_stage=WorkflowStage.ICP_QUEUE_READY,
        ),
        semantic=SemanticMemory(
            global_constraints=analysis_input.global_constraints.model_dump(),
            analysis_input=analysis_input,
        ),
    )
    state.icp_queue = queue
    return state


def update_state_from_skill(
    state: AgentState, skill: SkillName, output: BaseModel
) -> AgentState:
    """Route a structured skill output to its handler."""

    handler = _SKILL_HANDLERS.get(skill)
    if handler is None:
        raise ValueError(f"No handler registered for skill {skill}")
    return handler(state, output)


# ---------------------------------------------------------------------------
# ICP Skill Handlers
# ---------------------------------------------------------------------------


def skill_formulate_query_handler(
    state: AgentState, output: FormulateQueryOutput
) -> AgentState:
    """Handler for query formulation skill."""
    new_state = deepcopy(state)
    # Store the formulated query in working memory
    new_state.working.current_search_query = output.search_query
    new_state.working.missing_fields = output.target_fields

    # Advance to search execution stage
    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_EXECUTE_SEARCH,
        reason=output.query_strategy,
    )

    return new_state


def skill_extract_data_handler(
    state: AgentState, output: ExtractDataOutput
) -> AgentState:
    """Handler for data extraction skill."""
    new_state = deepcopy(state)

    # Store extracted data in working memory
    new_state.working.current_extracted_data = output
    new_state.working.missing_fields = output.missing_fields

    # Log extraction in episodic memory
    new_state.episodic.extraction_history.append(
        {
            "node_id": new_state.working.current_node_id,
            "timestamp": datetime.now().isoformat(),
            "extracted_count": len(output.extracted_fields),
            "missing_count": len(output.missing_fields),
        }
    )

    # Advance to validation stage
    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_VALIDATE_DATA,
        reason=output.chain_of_thought,
    )

    return new_state


def skill_validate_data_handler(
    state: AgentState, output: ValidateDataOutput
) -> AgentState:
    """Handler for data validation skill."""
    new_state = deepcopy(state)

    # Store validation result
    new_state.working.current_validation_result = output
    new_state.working.missing_fields = output.missing_must_have

    # Route to appropriate next stage based on recommendation
    if output.recommended_action == "proceed":
        next_stage = WorkflowStage.ICP_DISCOVER_ENTITIES
    elif output.recommended_action == "retry":
        next_stage = WorkflowStage.ICP_REFINE_QUERY
        new_state.working.retry_reason = output.validation_notes
    elif output.recommended_action == "partial":
        next_stage = WorkflowStage.ICP_NODE_PARTIAL
    else:  # fail
        next_stage = WorkflowStage.ICP_NODE_FAILED

    new_state.workflow.record_transition(
        to_stage=next_stage,
        reason=output.validation_notes,
    )

    return new_state


def skill_refine_query_handler(
    state: AgentState, output: RefineQueryOutput
) -> AgentState:
    """Handler for query refinement skill (retry flow)."""
    new_state = deepcopy(state)

    # Update the search query with refined version
    new_state.working.current_search_query = output.refined_query
    new_state.working.missing_fields = output.focus_fields
    new_state.working.retry_reason = output.failure_analysis

    # Advance back to search execution for retry
    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_EXECUTE_SEARCH,
        reason=output.refinement_strategy,
    )

    return new_state


def skill_discover_entities_handler(
    state: AgentState, output: DiscoverEntitiesOutput
) -> AgentState:
    """Handler for entity discovery skill."""
    new_state = deepcopy(state)

    if output.should_expand and output.discovered_entities:
        # Store discovered entities for tree expansion
        new_state.working.query_analysis = {
            "discovered_entities": output.discovered_entities,
            "discovery_notes": output.discovery_notes,
        }
        new_state.workflow.record_transition(
            to_stage=WorkflowStage.ICP_EXPAND_TREE,
            reason=output.discovery_notes,
        )
    else:
        # No expansion needed, mark node complete
        new_state.workflow.record_transition(
            to_stage=WorkflowStage.ICP_NODE_COMPLETE,
            reason="No new entities discovered",
        )

    return new_state


_SKILL_HANDLERS: Dict[SkillName, SkillHandler] = {
    SkillName.FORMULATE_QUERY: skill_formulate_query_handler,  # type: ignore
    SkillName.EXTRACT_DATA: skill_extract_data_handler,  # type: ignore
    SkillName.VALIDATE_DATA: skill_validate_data_handler,  # type: ignore
    SkillName.REFINE_QUERY: skill_refine_query_handler,  # type: ignore
    SkillName.DISCOVER_ENTITIES: skill_discover_entities_handler,  # type: ignore
}


def update_state_from_tool(
    state: AgentState, tool: ToolName, output: BaseModel
) -> AgentState:
    """Route a structured tool output to its handler."""

    handler = _TOOL_HANDLERS.get(tool)
    if handler is None:
        raise ValueError(f"No handler registered for tool {tool}")
    return handler(state, output)


# ---------------------------------------------------------------------------
# ICP Tool Handlers
# ---------------------------------------------------------------------------


def tool_web_search_handler(state: AgentState, output: WebSearchResponse) -> AgentState:
    """Handler for web search tool results."""
    new_state = deepcopy(state)

    # Store search results in working memory
    new_state.working.current_search_results = output

    # Update resource tracking
    new_state.resource.api_calls_made += 1
    new_state.resource.last_api_call_time = output.executed_at

    # Log search in episodic memory
    new_state.episodic.search_history.append(
        {
            "node_id": output.node_id,
            "query": output.query,
            "timestamp": output.executed_at.isoformat(),
            "result_count": len(output.results),
            "success": output.success,
            "http_errors": output.http_errors,
        }
    )

    # Handle search errors
    if not output.success:
        # Log error
        new_state.episodic.error_log.append(
            {
                "node_id": output.node_id,
                "error_type": "search_failed",
                "message": output.error_message,
                "http_errors": output.http_errors,
                "timestamp": output.executed_at.isoformat(),
            }
        )

        # Check if we can retry
        current_node = new_state.get_current_node()
        if current_node and current_node.can_retry():
            new_state.workflow.record_transition(
                to_stage=WorkflowStage.ICP_REFINE_QUERY,
                reason=f"Search failed: {output.error_message}. Attempting retry.",
            )
        else:
            new_state.workflow.record_transition(
                to_stage=WorkflowStage.ICP_NODE_FAILED,
                reason=f"Search failed and no retries remaining: {output.error_message}",
            )
    else:
        # Search succeeded, proceed to extraction
        new_state.workflow.record_transition(
            to_stage=WorkflowStage.ICP_EXTRACT_DATA,
            reason=f"Search returned {len(output.results)} results",
        )

    return new_state


_TOOL_HANDLERS: Dict[ToolName, ToolHandler] = {
    ToolName.WEB_SEARCH: tool_web_search_handler,  # type: ignore
}


# ---------------------------------------------------------------------------
# ICP Stage Handlers (State Transitions)
# ---------------------------------------------------------------------------


def handle_queue_next(state: AgentState) -> AgentState:
    """Move to the next pending node or finish."""
    from .queue import AnalysisQueue

    new_state = deepcopy(state)
    queue: AnalysisQueue = new_state.icp_queue  # type: ignore

    # Clear working memory
    new_state.working.current_node_id = None
    new_state.working.current_search_query = None
    new_state.working.current_search_results = None
    new_state.working.current_extracted_data = None
    new_state.working.current_validation_result = None
    new_state.working.missing_fields = []
    new_state.working.retry_reason = None

    next_node = queue.get_next_pending()
    if next_node:
        new_state.working.current_node_id = next_node.id
        queue.mark_in_progress(next_node.id)
        new_state.workflow.record_transition(
            to_stage=WorkflowStage.ICP_FORMULATE_QUERY,
            reason=f"Processing node: {next_node.name}",
        )
    else:
        new_state.workflow.record_transition(
            to_stage=WorkflowStage.ICP_ALL_NODES_DONE,
            reason="No pending nodes",
        )

    return new_state


def handle_node_done(state: AgentState) -> AgentState:
    """Handle completed/failed/partial/skipped node and advance."""
    from .queue import AnalysisQueue
    from .models import AnalysisNodeResult, ExtractedDataPoint
    from src.engine.types import AnalysisNodeStatus

    new_state = deepcopy(state)
    queue: AnalysisQueue = new_state.icp_queue  # type: ignore
    node_id = new_state.working.current_node_id
    stage = state.workflow.current_stage

    if node_id:
        # Map stage to status and update queue
        status_map = {
            WorkflowStage.ICP_NODE_COMPLETE: AnalysisNodeStatus.COMPLETED,
            WorkflowStage.ICP_NODE_FAILED: AnalysisNodeStatus.FAILED,
            WorkflowStage.ICP_NODE_PARTIAL: AnalysisNodeStatus.PARTIAL,
            WorkflowStage.ICP_NODE_SKIPPED: AnalysisNodeStatus.SKIPPED,
        }
        status = status_map.get(stage, AnalysisNodeStatus.COMPLETED)

        # Store node result
        _store_node_result(new_state, node_id, status)

        if status == AnalysisNodeStatus.COMPLETED:
            queue.mark_completed(node_id)
            new_state.workflow.nodes_successful += 1
        elif status == AnalysisNodeStatus.FAILED:
            queue.mark_failed(node_id)
            new_state.workflow.nodes_failed += 1
        elif status == AnalysisNodeStatus.PARTIAL:
            queue.mark_partial(node_id)
        else:
            queue.mark_skipped(node_id)

        new_state.workflow.nodes_processed += 1

    # Transition to get next node
    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_QUEUE_READY,
        reason="Moving to next node",
    )
    return new_state


def handle_expand_tree(state: AgentState, enable_expansion: bool = True) -> AgentState:
    """Handle dynamic tree expansion."""
    from .queue import AnalysisQueue
    from .models import AnalysisNode

    new_state = deepcopy(state)

    if not enable_expansion:
        new_state.workflow.record_transition(
            to_stage=WorkflowStage.ICP_NODE_COMPLETE,
            reason="Dynamic expansion disabled",
        )
        return new_state

    discovery_data = new_state.working.query_analysis
    parent_id = new_state.working.current_node_id

    if discovery_data and "discovered_entities" in discovery_data and parent_id:
        queue: AnalysisQueue = new_state.icp_queue  # type: ignore

        for entity in discovery_data["discovered_entities"]:
            new_node = AnalysisNode(
                name=entity.get("name", "Discovered Entity"),
                description=entity.get("description", ""),
                parameters={"entity_type": entity.get("entity_type", "unknown")},
            )
            queue.add_dynamic_node(new_node, parent_id)

    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_NODE_COMPLETE,
        reason="Tree expansion complete",
    )
    return new_state


def handle_all_done(state: AgentState) -> AgentState:
    """Transition to report generation."""
    new_state = deepcopy(state)
    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_GENERATE_REPORT,
        reason="All nodes processed",
    )
    return new_state


def handle_generate_report(state: AgentState) -> AgentState:
    """Transition to completion."""
    new_state = deepcopy(state)
    new_state.workflow.record_transition(
        to_stage=WorkflowStage.ICP_REPORT_COMPLETE,
        reason="Report generation complete",
    )
    return new_state


def _store_node_result(
    state: AgentState, node_id: str, status: "AnalysisNodeStatus"
) -> None:
    """Store the result for a completed node (mutates state in place)."""
    from .queue import AnalysisQueue
    from .models import AnalysisNodeResult, ExtractedDataPoint
    from src.engine.types import AnalysisNodeStatus

    queue: AnalysisQueue = state.icp_queue  # type: ignore
    node = queue.get_node(node_id)

    if not node:
        return

    extracted_data = state.working.current_extracted_data
    search_query = state.working.current_search_query

    extracted_points: list[ExtractedDataPoint] = []
    missing_fields: list[str] = []
    queries_used: list[str] = []

    if search_query:
        queries_used.append(search_query)

    if extracted_data and hasattr(extracted_data, "extracted_fields"):
        for ef in extracted_data.extracted_fields:
            extracted_points.append(
                ExtractedDataPoint(
                    field_name=ef.field_name,
                    value=ef.value,
                    raw_value=ef.raw_value,
                    source_url=ef.source_url,
                    confidence=ef.confidence,
                    notes=ef.notes,
                )
            )
        missing_fields = extracted_data.missing_fields

    node.result = AnalysisNodeResult(
        extracted_data=extracted_points,
        missing_fields=missing_fields,
        search_queries_used=queries_used,
        completed_at=datetime.now(),
    )
    node.status = status

    state.semantic.extracted_knowledge[node_id] = {
        "status": status.value,
        "data": [dp.model_dump() for dp in extracted_points],
        "missing": missing_fields,
    }


# Stage handler registry for ICP workflow
ICP_STAGE_HANDLERS = {
    WorkflowStage.ICP_QUEUE_READY: handle_queue_next,
    WorkflowStage.ICP_NODE_COMPLETE: handle_node_done,
    WorkflowStage.ICP_NODE_FAILED: handle_node_done,
    WorkflowStage.ICP_NODE_PARTIAL: handle_node_done,
    WorkflowStage.ICP_NODE_SKIPPED: handle_node_done,
    WorkflowStage.ICP_EXPAND_TREE: handle_expand_tree,
    WorkflowStage.ICP_ALL_NODES_DONE: handle_all_done,
    WorkflowStage.ICP_GENERATE_REPORT: handle_generate_report,
}
