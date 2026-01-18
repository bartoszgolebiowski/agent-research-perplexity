from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.engine.types import WorkflowStage


class AnalyzeAndPlanSkillOutput(BaseModel):
    """Structured fields returned by the analyze and plan skill."""

    chain_of_thought: str = Field(
        default="",
        description="Detailed reasoning about the user's query and the context.",
    )
    next_stage: WorkflowStage = Field(
        default=WorkflowStage.COORDINATOR,
        description="Recommended next workflow stage.",
    )


# ---------------------------------------------------------------------------
# ICP Intelligence System Skills
# ---------------------------------------------------------------------------


class FormulateQueryOutput(BaseModel):
    """Output from the query formulation skill."""

    search_query: str = Field(
        ..., description="Optimized search query for web search API"
    )
    query_strategy: str = Field(
        default="", description="Explanation of query formulation strategy"
    )
    target_fields: List[str] = Field(
        default_factory=list, description="List of field names this query targets"
    )
    alternative_queries: List[str] = Field(
        default_factory=list, description="Alternative query formulations for fallback"
    )


class ExtractedFieldValue(BaseModel):
    """A single extracted field value with metadata."""

    field_name: str = Field(..., description="Name of the extraction field")
    value: Any = Field(..., description="Extracted value")
    raw_value: Optional[str] = Field(
        default=None, description="Original value before normalization"
    )
    source_url: str = Field(..., description="Citation URL")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this extraction"
    )
    notes: Optional[str] = Field(
        default=None, description="Notes about conversion or normalization"
    )


class ExtractDataOutput(BaseModel):
    """Output from the data extraction skill."""

    chain_of_thought: str = Field(
        default="", description="Reasoning about the extraction process"
    )
    extracted_fields: List[ExtractedFieldValue] = Field(
        default_factory=list, description="Successfully extracted field values"
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Names of fields that could not be extracted"
    )
    partial_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Fields with partial/uncertain data - field name to notes",
    )


class ValidateDataOutput(BaseModel):
    """Output from the data validation skill."""

    is_complete: bool = Field(
        ..., description="Whether all must-have fields are satisfied"
    )
    missing_must_have: List[str] = Field(
        default_factory=list, description="Names of missing must-have fields"
    )
    missing_nice_to_have: List[str] = Field(
        default_factory=list, description="Names of missing nice-to-have fields"
    )
    validation_notes: str = Field(default="", description="Notes about the validation")
    recommended_action: str = Field(
        default="proceed", description="Recommended next action: proceed, retry, fail"
    )
    retry_focus: List[str] = Field(
        default_factory=list, description="Fields to focus on in retry query"
    )


class RefineQueryOutput(BaseModel):
    """Output from the query refinement skill (for retries)."""

    refined_query: str = Field(
        ..., description="New refined search query targeting missing data"
    )
    refinement_strategy: str = Field(
        default="", description="Explanation of why this query should work better"
    )
    focus_fields: List[str] = Field(
        default_factory=list, description="Specific fields this query focuses on"
    )
    failure_analysis: str = Field(
        default="", description="Analysis of why previous query failed"
    )


class DiscoverEntitiesOutput(BaseModel):
    """Output from entity discovery skill (for dynamic expansion)."""

    discovered_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of discovered entities to add as sub-tasks",
    )
    discovery_notes: str = Field(
        default="", description="Notes about the discovery process"
    )
    should_expand: bool = Field(
        default=False, description="Whether new sub-tasks should be created"
    )
