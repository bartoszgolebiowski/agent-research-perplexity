from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ICP Intelligence System Skills
# ---------------------------------------------------------------------------


class FormulateQueryOutput(BaseModel):
    """Output from the query formulation skill."""

    search_query: str = Field(
        default="", description="Optimized search query for web search API"
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

    field_name: str = Field(default="", description="Name of the extraction field")
    value: Any = Field(default=None, description="Extracted value")
    raw_value: Optional[str] = Field(
        default=None, description="Original value before normalization"
    )
    source_url: str = Field(default="", description="Citation URL")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this extraction"
    )
    notes: Optional[str] = Field(
        default=None, description="Notes about conversion or normalization"
    )


class RefineQueryOutput(BaseModel):
    """Output from the query refinement skill (for retries)."""

    refined_query: str = Field(
        default="", description="New refined search query targeting missing data"
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


class VerifySourceOutput(BaseModel):
    """Output from source verification skill."""

    is_valid: bool = Field(
        default=False, description="Whether search results are about the target company"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in verification"
    )
    verification_notes: str = Field(
        default="", description="Explanation of verification decision"
    )
    detected_company: Optional[str] = Field(
        default=None, description="Company name detected in results (if different)"
    )


class ExtractDataOutput(BaseModel):
    """Output from data extraction skill."""

    extracted_fields: List[ExtractedFieldValue] = Field(
        default_factory=list, description="Successfully extracted field values"
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Names of fields that could not be extracted"
    )
    extraction_notes: str = Field(
        default="", description="Notes about the extraction process"
    )


class ValidateExtractionOutput(BaseModel):
    """Output from extraction validation skill."""

    is_complete: bool = Field(
        default=False, description="Whether all must-have fields are satisfied"
    )
    recommended_action: str = Field(
        default="proceed",
        description="Recommended next action: proceed, retry, fail, partial",
    )
    retry_focus: List[str] = Field(
        default_factory=list, description="Fields to focus on in retry query"
    )
    validation_reasoning: str = Field(
        default="", description="Reasoning for the validation decision"
    )


class DiscoverEntitiesOutput(BaseModel):
    """Output from entity discovery skill."""

    discovered_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of discovered entities to add as sub-tasks",
    )
    should_expand: bool = Field(
        default=False, description="Whether new sub-tasks should be created"
    )
    discovery_reasoning: str = Field(
        default="", description="Reasoning for entity discovery decisions"
    )


class ProcessSearchResultsOutput(BaseModel):
    """
    DEPRECATED: Legacy merged output for Extraction, Validation, and Entity Discovery.

    This skill combines operations into a single LLM call.
    Prefer using the split skills for better reliability with smaller models.
    """

    # Extraction Section
    extracted_fields: List[ExtractedFieldValue] = Field(
        default_factory=list, description="Successfully extracted field values"
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Names of fields that could not be extracted"
    )

    # Validation Section
    is_complete: bool = Field(
        default=False, description="Whether all must-have fields are satisfied"
    )
    recommended_action: str = Field(
        default="proceed",
        description="Recommended next action: proceed, retry, fail, partial",
    )
    retry_focus: List[str] = Field(
        default_factory=list, description="Fields to focus on in retry query"
    )

    # Discovery Section
    discovered_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of discovered entities to add as sub-tasks",
    )
    should_expand: bool = Field(
        default=False, description="Whether new sub-tasks should be created"
    )

    # Reasoning
    chain_of_thought: str = Field(
        default="",
        description="Step-by-step reasoning for extraction, validation, and discovery",
    )
