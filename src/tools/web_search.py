from __future__ import annotations

"""Web Search tool implementation using Perplexity/Tavily API."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import requests

from .models import (
    SearchResultItem,
    WebSearchRequest,
    WebSearchResponse,
)


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """Configuration for the web search client."""

    api_key: str
    base_url: str = "https://api.perplexity.ai"
    model: str = "llama-3.1-sonar-small-128k-online"
    timeout: int = 30
    max_results: int = 5

    @classmethod
    def from_env(cls) -> "WebSearchConfig":
        """Create configuration from environment variables."""
        api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        base_url = os.environ.get("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
        model = os.environ.get("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")
        timeout = int(os.environ.get("PERPLEXITY_TIMEOUT", "30"))
        max_results = int(os.environ.get("PERPLEXITY_MAX_RESULTS", "5"))

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_results=max_results,
        )


@dataclass(slots=True)
class WebSearchClient:
    """
    Web search client using Perplexity API.

    This client wraps the Perplexity API to provide web search capabilities
    with structured result parsing and error handling.
    """

    config: WebSearchConfig

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
        if not self.config.api_key:
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message="PERPLEXITY_API_KEY not configured",
            )

        try:
            response = self._call_api(request.query)
            return self._parse_response(response, request)
        except requests.exceptions.Timeout:
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message="Search request timed out",
            )
        except requests.exceptions.RequestException as e:
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message=f"Search request failed: {str(e)}",
            )

    def _call_api(self, query: str) -> dict:
        """Make the actual API call to Perplexity."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Provide factual information "
                        "with source citations. Include URLs for all data points."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "return_citations": True,
            "return_related_questions": False,
        }

        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )

        # Handle HTTP errors
        if response.status_code in (403, 404, 429, 500, 502, 503):
            return {
                "error": True,
                "http_status": response.status_code,
                "message": f"HTTP {response.status_code}: {response.reason}",
            }

        response.raise_for_status()
        return response.json()

    def _parse_response(
        self, api_response: dict, request: WebSearchRequest
    ) -> WebSearchResponse:
        """Parse the API response into a structured format."""
        http_errors: List[str] = []

        # Check for API-level errors
        if api_response.get("error"):
            http_errors.append(api_response.get("message", "Unknown error"))
            return WebSearchResponse(
                query=request.query,
                node_id=request.node_id,
                success=False,
                error_message=api_response.get("message"),
                http_errors=http_errors,
            )

        # Extract content and citations
        results: List[SearchResultItem] = []
        raw_content = ""

        try:
            choices = api_response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                raw_content = message.get("content", "")

            # Parse citations if available
            citations = api_response.get("citations", [])
            for i, citation in enumerate(citations[: self.config.max_results]):
                if isinstance(citation, str):
                    # Simple URL citation
                    results.append(
                        SearchResultItem(
                            title=f"Source {i + 1}",
                            url=citation,
                            snippet="",
                        )
                    )
                elif isinstance(citation, dict):
                    # Structured citation
                    results.append(
                        SearchResultItem(
                            title=citation.get("title", f"Source {i + 1}"),
                            url=citation.get("url", ""),
                            snippet=citation.get("snippet", ""),
                            published_date=citation.get("published_date"),
                        )
                    )
        except (KeyError, IndexError, TypeError) as e:
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
            raw_content=raw_content,
            success=True,
            executed_at=datetime.now(),
        )
