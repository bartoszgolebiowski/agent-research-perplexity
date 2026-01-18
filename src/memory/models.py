from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.tools.models import HelloWorldRequest, WebSearchRequest
from src.engine.types import AnalysisNodeStatus, FieldPriority, WorkflowStage


# ---------------------------------------------------------------------------
# ICP Analysis Data Models
# ---------------------------------------------------------------------------


class ExtractionField(BaseModel):
    """Defines a specific field to extract during analysis."""

    name: str = Field(..., description="Name of the field to extract")
    description: str = Field(..., description="Human-readable description of what to find")
    priority: FieldPriority = Field(
        default=FieldPriority.MUST_HAVE,
        description="Priority level of this field"
    )
    data_type: str = Field(
        default="string",
        description="Expected data type: string, number, date, currency, list"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Expected unit if applicable (e.g., USD, EUR)"
    )


class ExtractedDataPoint(BaseModel):
    """A single extracted data point with citation."""

    field_name: str = Field(..., description="Name of the extraction field this satisfies")
    value: Any = Field(..., description="Extracted value")
    raw_value: Optional[str] = Field(
        default=None,
        description="Original value before normalization"
    )
    source_url: str = Field(..., description="Citation URL for this data point")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this extraction"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes about conversion or normalization"
    )


class AnalysisNodeResult(BaseModel):
    """Results of processing an analysis node."""

    extracted_data: List[ExtractedDataPoint] = Field(
        default_factory=list,
        description="Successfully extracted data points"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Names of fields that could not be extracted"
    )
    search_queries_used: List[str] = Field(
        default_factory=list,
        description="Search queries that were executed"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when processing completed"
    )


class AnalysisNode(BaseModel):
    """
    A recursive analysis task node (Composite Pattern).
    
    Each node represents a research task that can contain sub-tasks.
    The system processes these nodes to extract structured data.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this node"
    )
    name: str = Field(..., description="Human-readable name for this analysis task")
    description: str = Field(
        default="",
        description="Context for the LLM - describes what to research"
    )
    extraction_fields: List[ExtractionField] = Field(
        default_factory=list,
        description="Specific fields to extract"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual modifiers (e.g., years, roles, regions)"
    )
    sub_tasks: List[AnalysisNode] = Field(
        default_factory=list,
        description="Child nodes for hierarchical analysis"
    )
    status: AnalysisNodeStatus = Field(
        default=AnalysisNodeStatus.PENDING,
        description="Current lifecycle status"
    )
    attempt_count: int = Field(
        default=0,
        description="Number of processing attempts"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for this node"
    )
    result: Optional[AnalysisNodeResult] = Field(
        default=None,
        description="Results after processing"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of parent node (for dynamic expansion)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Node creation timestamp"
    )

    def get_must_have_fields(self) -> List[ExtractionField]:
        """Return only the must-have priority fields."""
        return [f for f in self.extraction_fields if f.priority == FieldPriority.MUST_HAVE]

    def get_nice_to_have_fields(self) -> List[ExtractionField]:
        """Return only the nice-to-have priority fields."""
        return [f for f in self.extraction_fields if f.priority == FieldPriority.NICE_TO_HAVE]

    def is_data_complete(self) -> bool:
        """Check if all must-have fields have been extracted."""
        if self.result is None:
            return False
        must_have_names = {f.name for f in self.get_must_have_fields()}
        extracted_names = {dp.field_name for dp in self.result.extracted_data}
        return must_have_names.issubset(extracted_names)

    def can_retry(self) -> bool:
        """Check if retry is allowed based on attempt count."""
        return self.attempt_count < self.max_retries

    def flatten_tree(self) -> List[AnalysisNode]:
        """Flatten the node tree into a list (parent before children)."""
        result: List[AnalysisNode] = [self]
        for child in self.sub_tasks:
            result.extend(child.flatten_tree())
        return result


class GlobalConstraints(BaseModel):
    """Global constraints applied to all analysis tasks."""

    currency: str = Field(
        default="USD",
        description="Target currency for financial data normalization"
    )
    language: str = Field(
        default="en",
        description="Preferred language for search queries"
    )
    date_format: str = Field(
        default="%Y-%m-%d",
        description="Standard date format to use"
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2024, 2025],
        description="Years of interest for temporal data"
    )
    allowed_regions: List[str] = Field(
        default_factory=list,
        description="Geographic regions to focus on"
    )
    custom_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom constraints"
    )


class ICPAnalysisInput(BaseModel):
    """
    Root input schema for the ICP Intelligence System.
    
    This is the main entry point for defining analysis jobs.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this analysis job"
    )
    name: str = Field(..., description="Name of the analysis job")
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the overall analysis"
    )
    global_constraints: GlobalConstraints = Field(
        default_factory=GlobalConstraints,
        description="Global constraints applied to all nodes"
    )
    root_nodes: List[AnalysisNode] = Field(
        default_factory=list,
        description="Top-level analysis nodes"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Job creation timestamp"
    )
    input_file_path: Optional[str] = Field(
        default=None,
        description="Path to the source input file"
    )

    def get_all_nodes(self) -> List[AnalysisNode]:
        """Flatten all nodes into a single list."""
        result: List[AnalysisNode] = []
        for root in self.root_nodes:
            result.extend(root.flatten_tree())
        return result

    def get_pending_nodes(self) -> List[AnalysisNode]:
        """Get all nodes that are still pending."""
        return [n for n in self.get_all_nodes() if n.status == AnalysisNodeStatus.PENDING]

    def get_node_by_id(self, node_id: str) -> Optional[AnalysisNode]:
        """Find a node by its ID."""
        for node in self.get_all_nodes():
            if node.id == node_id:
                return node
        return None


class ICPAnalysisOutput(BaseModel):
    """
    Output schema containing all analysis results.
    
    This is serialized to JSON for report generation.
    """

    input: ICPAnalysisInput = Field(..., description="Original input specification")
    completed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when analysis completed"
    )
    total_nodes: int = Field(default=0, description="Total number of nodes processed")
    successful_nodes: int = Field(default=0, description="Number of successful extractions")
    partial_nodes: int = Field(default=0, description="Number of partial extractions")
    failed_nodes: int = Field(default=0, description="Number of failed extractions")
    skipped_nodes: int = Field(default=0, description="Number of skipped nodes")

    def compute_statistics(self) -> None:
        """Compute statistics from the input nodes."""
        all_nodes = self.input.get_all_nodes()
        self.total_nodes = len(all_nodes)
        self.successful_nodes = sum(
            1 for n in all_nodes if n.status == AnalysisNodeStatus.COMPLETED
        )
        self.partial_nodes = sum(
            1 for n in all_nodes if n.status == AnalysisNodeStatus.PARTIAL
        )
        self.failed_nodes = sum(
            1 for n in all_nodes if n.status == AnalysisNodeStatus.FAILED
        )
        self.skipped_nodes = sum(
            1 for n in all_nodes if n.status == AnalysisNodeStatus.SKIPPED
        )


# ---------------------------------------------------------------------------
# Agent Memory Layers
# ---------------------------------------------------------------------------


class ConstitutionalMemory(BaseModel):
    """The "agent's DNA." Security and ethical principles that the agent MUST NOT break. Guardrails."""

    require_citations: bool = Field(
        default=True,
        description="Whether all extracted data must have source citations"
    )
    allow_hallucination: bool = Field(
        default=False,
        description="Whether the agent is allowed to invent data without sources"
    )
    normalize_currency: bool = Field(
        default=True,
        description="Whether to normalize currency to target constraint"
    )


class WorkingMemory(BaseModel):
    """The context of the current session (RAM). What we're talking about "right now.\" """

    query_analysis: Optional[Any] = Field(
        default=None, description="Analysis and plan of the current query."
    )
    
    # ICP Working Context
    current_node_id: Optional[str] = Field(
        default=None,
        description="ID of the currently processing analysis node"
    )
    current_search_query: Optional[str] = Field(
        default=None,
        description="The current search query being executed"
    )
    current_search_results: Optional[Any] = Field(
        default=None,
        description="Results from the most recent search"
    )
    current_extracted_data: Optional[Any] = Field(
        default=None,
        description="Data extracted from the current search"
    )
    current_validation_result: Optional[Any] = Field(
        default=None,
        description="Result of the current data validation"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Fields that are still missing for the current node"
    )
    retry_reason: Optional[str] = Field(
        default=None,
        description="Reason for retry if in retry flow"
    )


class WorkflowTransition(BaseModel):
    """Record of a workflow stage transition."""

    from_stage: WorkflowStage
    to_stage: WorkflowStage
    timestamp: datetime = Field(default_factory=datetime.now)
    reason: Optional[str] = None


class WorkflowMemory(BaseModel):
    """The State Machine. Where am I in the business process?"""

    current_stage: WorkflowStage = Field(
        default=WorkflowStage.INITIAL, description="Current stage in the workflow"
    )
    goal: str = Field(..., description="The initial goal that started the workflow.")
    history: List[WorkflowTransition] = Field(
        default_factory=list, description="Historical record of stage transitions."
    )
    
    # ICP Workflow State
    is_icp_workflow: bool = Field(
        default=False,
        description="Whether this is an ICP analysis workflow"
    )
    nodes_processed: int = Field(
        default=0,
        description="Number of nodes that have been processed"
    )
    nodes_successful: int = Field(
        default=0,
        description="Number of successfully completed nodes"
    )
    nodes_failed: int = Field(
        default=0,
        description="Number of failed nodes"
    )

    def record_transition(
        self, to_stage: WorkflowStage, reason: Optional[str] = None
    ) -> None:
        """Helper to append a transition to history if the stage changed."""
        if self.current_stage != to_stage:
            self.history.append(
                WorkflowTransition(
                    from_stage=self.current_stage, to_stage=to_stage, reason=reason
                )
            )
            self.current_stage = to_stage


class EpisodicMemory(BaseModel):
    """What happened? Interaction history, event logs."""

    search_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of search queries and results"
    )
    extraction_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of data extractions"
    )
    error_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of errors encountered"
    )


class SemanticMemory(BaseModel):
    """What do I know? The knowledge base (RAG), facts about the world and the user."""
    
    # ICP Knowledge Store
    global_constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Global constraints for the ICP analysis"
    )
    analysis_input: Optional[Any] = Field(
        default=None,
        description="The full ICP analysis input specification"
    )
    extracted_knowledge: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated extracted data keyed by node ID"
    )


class ProceduralMemory(BaseModel):
    """How do I do it? Tool definitions, APIs, user manuals."""

    available_search_providers: List[str] = Field(
        default_factory=lambda: ["perplexity"],
        description="Available search providers"
    )


class ResourceMemory(BaseModel):
    """Do I have the resources? System status, API availability, limits."""

    api_calls_made: int = Field(
        default=0,
        description="Number of API calls made in this session"
    )
    api_call_limit: Optional[int] = Field(
        default=None,
        description="Maximum API calls allowed (None = unlimited)"
    )
    last_api_call_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last API call"
    )


class AgentState(BaseModel):
    """Full memory object available to the engine layer."""

    core: ConstitutionalMemory
    working: WorkingMemory
    workflow: WorkflowMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory
    procedural: ProceduralMemory
    resource: ResourceMemory
    
    # ICP-specific runtime data (not part of memory layers)
    icp_queue: Optional[Any] = Field(
        default=None,
        description="The analysis queue for ICP workflow"
    )

    def get_hello_world_request(self) -> HelloWorldRequest:
        """Constructs a HelloWorldRequest from the agent state."""
        return HelloWorldRequest(query="Hello, World!")

    def get_web_search_request(self) -> WebSearchRequest:
        """Constructs a WebSearchRequest from the agent state."""
        if not self.working.current_node_id or not self.working.current_search_query:
            raise ValueError("No current node or search query set")
        
        return WebSearchRequest(
            query=self.working.current_search_query,
            node_id=self.working.current_node_id,
            target_fields=self.working.missing_fields,
            attempt_number=self._get_current_attempt_number(),
        )

    def _get_current_attempt_number(self) -> int:
        """Get the current attempt number for the current node."""
        if self.icp_queue and self.working.current_node_id:
            node = self.icp_queue.get_node(self.working.current_node_id)
            if node:
                return node.attempt_count + 1
        return 1

    def get_current_node(self) -> Optional[Any]:
        """Get the current analysis node being processed."""
        if self.icp_queue and self.working.current_node_id:
            return self.icp_queue.get_node(self.working.current_node_id)
        return None
