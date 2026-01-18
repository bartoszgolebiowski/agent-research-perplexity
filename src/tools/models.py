from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    """Names of available tools."""

    HELLO_WORLD = "hello_world"
    WEB_SEARCH = "web_search"
    REPORT_GENERATOR = "report_generator"


class HelloWorldRequest(BaseModel):
    """A simple request model for testing connectivity."""

    query: str


class HelloWorldResponse(BaseModel):
    """A simple response model for testing connectivity."""

    message: str


# ---------------------------------------------------------------------------
# Web Search Tool Models
# ---------------------------------------------------------------------------


class SearchResultItem(BaseModel):
    """A single search result from the web search."""

    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Text snippet from the result")
    published_date: Optional[str] = Field(
        default=None,
        description="Published date if available"
    )


class WebSearchRequest(BaseModel):
    """Request model for web search tool."""

    query: str = Field(..., description="The search query to execute")
    node_id: str = Field(..., description="ID of the analysis node this search is for")
    target_fields: List[str] = Field(
        default_factory=list,
        description="Names of fields we're trying to extract"
    )
    attempt_number: int = Field(
        default=1,
        description="Which attempt this is (for retry logic)"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the search"
    )


class WebSearchResponse(BaseModel):
    """Response model from web search tool."""

    query: str = Field(..., description="The executed search query")
    node_id: str = Field(..., description="ID of the analysis node")
    results: List[SearchResultItem] = Field(
        default_factory=list,
        description="List of search results"
    )
    raw_content: str = Field(
        default="",
        description="Raw concatenated content from all results"
    )
    success: bool = Field(
        default=True,
        description="Whether the search was successful"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if search failed"
    )
    http_errors: List[str] = Field(
        default_factory=list,
        description="List of HTTP errors encountered (403, 404, etc.)"
    )
    executed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of search execution"
    )
