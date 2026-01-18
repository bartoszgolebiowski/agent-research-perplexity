from __future__ import annotations
from unittest import result

"""Web Search tool implementation using Perplexity SDK."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from perplexity import Perplexity
from perplexity.types import SearchCreateResponse

from .models import (
    SearchResultItem,
    WebSearchRequest,
    WebSearchResponse,
)


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """Configuration for the web search client."""

    api_key: str
    max_results: int = 3
    max_tokens: int = 25000
    max_tokens_per_page: int = 1024

    @classmethod
    def from_env(cls) -> "WebSearchConfig":
        """Create configuration from environment variables."""
        api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        max_results = int(os.environ.get("PERPLEXITY_MAX_RESULTS", "3"))
        max_tokens = int(os.environ.get("PERPLEXITY_MAX_TOKENS", "25000"))
        max_tokens_per_page = int(
            os.environ.get("PERPLEXITY_MAX_TOKENS_PER_PAGE", "1024")
        )

        return cls(
            api_key=api_key,
            max_results=max_results,
            max_tokens=max_tokens,
            max_tokens_per_page=max_tokens_per_page,
        )


@dataclass(slots=True)
class WebSearchClient:
    """
    Web search client using Perplexity SDK.

    This client wraps the Perplexity SDK to provide web search capabilities
    with structured result parsing and error handling.
    """

    config: WebSearchConfig
    _client: Optional[Perplexity] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize the Perplexity SDK client."""
        if self.config.api_key:
            object.__setattr__(self, "_client", Perplexity(api_key=self.config.api_key))

    @classmethod
    def from_env(cls) -> "WebSearchClient":
        """Create a client from environment variables."""
        return cls(config=WebSearchConfig.from_env())

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """
        Execute a web search query.

        Args:
            request: The search request with query and context

        Returns:
            WebSearchResponse with results or error information
        """
        if not self.config.api_key or not self._client:
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message="PERPLEXITY_API_KEY not configured",
            )

        try:
            response = self._call_api(request.query)
            return self._parse_response(response, request)
        except Exception as e:
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message=f"Search request failed: {str(e)}",
            )

    def _call_api(self, query: str) -> SearchCreateResponse:
        """Make the actual API call to Perplexity using SDK."""
        if not self._client:
            raise RuntimeError("Perplexity client not initialized")

        # Use the Perplexity SDK search endpoint
        response = self._client.search.create(
            query=query,
            max_results=self.config.max_results,
            max_tokens=self.config.max_tokens,
            max_tokens_per_page=self.config.max_tokens_per_page,
        )
        return response

    def _parse_response(
        self, api_response: SearchCreateResponse, request: WebSearchRequest
    ) -> WebSearchResponse:
        """Parse the API response into a structured format."""
        # Extract search results from the Search API response
        results: List[SearchResultItem] = []
        raw_content = ""
        try:
            # Access results directly from the typed response
            for result in api_response.results:
                # Build snippet from available content
                snippet = result.snippet or ""

                # Add to raw content for context
                if snippet:
                    raw_content += f"{snippet}\n\n"

                # Parse the search result using typed fields
                results.append(
                    SearchResultItem(
                        title=result.title,
                        url=result.url,
                        snippet=snippet,
                        published_date=result.date,  # Uses 'date' field from API
                    )
                )
        except (AttributeError, TypeError) as e:
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message=f"Failed to parse API response: {str(e)}",
            )

        return WebSearchResponse(
            query=request.query,
            node_id=request.node_id,
            results=results,
            raw_content=raw_content.strip(),
            success=True,
            executed_at=datetime.now(),
        )
