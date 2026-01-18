from __future__ import annotations

"""Report generator tool for ICP analysis results."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from src.memory.models import ICPAnalysisOutput
from src.prompting.environment import prompt_environment


class ReportRequest(BaseModel):
    """Request model for report generation tool."""

    analysis_output: ICPAnalysisOutput = Field(
        default_factory=ICPAnalysisOutput,
        description="The completed analysis output to generate report from",
    )
    output_directory: Path = Field(
        default=Path("./reports"), description="Directory to write reports to"
    )
    template_name: str = Field(
        default="report/icp_report.j2",
        description="Jinja2 template to use for HTML generation",
    )
    include_raw_data: bool = Field(
        default=True, description="Whether to include raw JSON data in the report"
    )


class ReportResponse(BaseModel):
    """Response model from report generation tool."""

    html_path: Path = Field(
        default=Path("./reports"), description="Path to generated HTML report"
    )
    json_path: Path = Field(
        default=Path("./reports"), description="Path to exported JSON data"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of report generation"
    )
    success: bool = Field(default=True, description="Whether generation succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if generation failed"
    )


@dataclass(frozen=True, slots=True)
class ReportGeneratorClient:
    """
    Generates HTML reports from ICP analysis results.

    Uses Jinja2 templates to separate data from presentation.
    The AI never writes raw HTML - it populates JSON data that
    the template engine renders.
    """

    @classmethod
    def from_env(cls) -> "ReportGeneratorClient":
        """Create a generator client (no env config needed)."""
        return cls()

    def generate(self, request: ReportRequest) -> ReportResponse:
        """
        Generate HTML and JSON reports from analysis results.

        Args:
            request: The report generation request

        Returns:
            ReportResponse with paths to generated files
        """
        output_dir = request.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self._sanitize_filename(request.analysis_output.input.name)

        json_path = output_dir / f"{base_name}_{timestamp}.json"
        html_path = output_dir / f"{base_name}_{timestamp}.html"

        try:
            # Export JSON data
            self._export_json(request.analysis_output, json_path)

            # Generate HTML report
            self._generate_html(
                request.analysis_output,
                html_path,
                request.template_name,
                request.include_raw_data,
            )

            return ReportResponse(
                html_path=html_path,
                json_path=json_path,
                generated_at=datetime.now(),
                success=True,
            )
        except Exception as e:
            return ReportResponse(
                html_path=html_path,
                json_path=json_path,
                generated_at=datetime.now(),
                success=False,
                error_message=str(e),
            )

    def _export_json(self, output: ICPAnalysisOutput, path: Path) -> None:
        """Export the analysis output as JSON."""
        output.compute_statistics()
        json_data = output.model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    def _generate_html(
        self,
        output: ICPAnalysisOutput,
        path: Path,
        template_name: str,
        include_raw_data: bool,
    ) -> None:
        """Generate HTML report using Jinja2 template."""
        context = self._build_template_context(output, include_raw_data)

        try:
            template = prompt_environment.get_template(template_name)
            html_content = template.render(**context)
        except Exception:
            # Fall back to inline template if custom template not found
            html_content = self._generate_fallback_html(context)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _build_template_context(
        self, output: ICPAnalysisOutput, include_raw_data: bool
    ) -> Dict[str, Any]:
        """Build the context dictionary for template rendering."""
        output.compute_statistics()

        nodes_data = []
        for node in output.input.get_all_nodes():
            node_data = {
                "id": node.id,
                "name": node.name,
                "description": node.description,
                "status": node.status.value,
                "attempt_count": node.attempt_count,
                "extracted_data": [],
                "missing_fields": [],
            }

            if node.result:
                node_data["extracted_data"] = [
                    {
                        "field": dp.field_name,
                        "value": dp.value,
                        "source": dp.source_url,
                        "confidence": dp.confidence,
                        "notes": dp.notes,
                    }
                    for dp in node.result.extracted_data
                ]
                node_data["missing_fields"] = node.result.missing_fields
                node_data["queries_used"] = node.result.search_queries_used

            nodes_data.append(node_data)

        return {
            "report_title": f"ICP Analysis: {output.input.name}",
            "generated_at": output.completed_at.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_name": output.input.name,
            "analysis_description": output.input.description,
            "global_constraints": output.input.global_constraints.model_dump(),
            "statistics": {
                "total": output.total_nodes,
                "successful": output.successful_nodes,
                "partial": output.partial_nodes,
                "failed": output.failed_nodes,
                "skipped": output.skipped_nodes,
            },
            "nodes": nodes_data,
            "include_raw_data": include_raw_data,
            "raw_json": output.model_dump_json(indent=2) if include_raw_data else None,
        }

    def _generate_fallback_html(self, context: Dict[str, Any]) -> str:
        """Generate a simple fallback HTML report if template is not available."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            f"  <title>{context['report_title']}</title>",
            "  <meta charset='utf-8'>",
            "  <style>",
            "    body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "    .node { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }",
            "    .status-completed { border-left: 4px solid #28a745; }",
            "    .status-partial { border-left: 4px solid #ffc107; }",
            "    .status-failed { border-left: 4px solid #dc3545; }",
            "    .status-skipped { border-left: 4px solid #6c757d; }",
            "    .data-point { background: #f8f9fa; padding: 8px; margin: 5px 0; border-radius: 3px; }",
            "    .source { font-size: 0.85em; color: #6c757d; }",
            "    table { width: 100%; border-collapse: collapse; margin: 10px 0; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background: #f4f4f4; }",
            "    .stats { display: flex; gap: 20px; margin: 20px 0; }",
            "    .stat { padding: 15px; background: #f8f9fa; border-radius: 5px; text-align: center; }",
            "    .stat-value { font-size: 2em; font-weight: bold; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{context['report_title']}</h1>",
            f"  <p><strong>Generated:</strong> {context['generated_at']}</p>",
        ]

        if context.get("analysis_description"):
            html_parts.append(f"  <p>{context['analysis_description']}</p>")

        # Statistics
        stats = context["statistics"]
        html_parts.extend(
            [
                "  <div class='stats'>",
                f"    <div class='stat'><div class='stat-value'>{stats['total']}</div>Total Nodes</div>",
                f"    <div class='stat'><div class='stat-value'>{stats['successful']}</div>Successful</div>",
                f"    <div class='stat'><div class='stat-value'>{stats['partial']}</div>Partial</div>",
                f"    <div class='stat'><div class='stat-value'>{stats['failed']}</div>Failed</div>",
                f"    <div class='stat'><div class='stat-value'>{stats['skipped']}</div>Skipped</div>",
                "  </div>",
            ]
        )

        # Nodes
        html_parts.append("  <h2>Analysis Results</h2>")
        for node in context["nodes"]:
            status_class = f"status-{node['status']}"
            html_parts.extend(
                [
                    f"  <div class='node {status_class}'>",
                    f"    <h3>{node['name']}</h3>",
                    f"    <p><strong>Status:</strong> {node['status']} | <strong>Attempts:</strong> {node['attempt_count']}</p>",
                    f"    <p>{node['description']}</p>",
                ]
            )

            if node["extracted_data"]:
                html_parts.append("    <h4>Extracted Data</h4>")
                html_parts.append(
                    "    <table><tr><th>Field</th><th>Value</th><th>Source</th><th>Confidence</th></tr>"
                )
                for dp in node["extracted_data"]:
                    source_link = (
                        f"<a href='{dp['source']}' target='_blank'>Source</a>"
                        if dp.get("source")
                        else "N/A"
                    )
                    confidence = (
                        f"{dp['confidence']:.0%}" if dp.get("confidence") else "N/A"
                    )
                    html_parts.append(
                        f"    <tr><td>{dp['field']}</td><td>{dp['value']}</td>"
                        f"<td>{source_link}</td><td>{confidence}</td></tr>"
                    )
                html_parts.append("    </table>")

            if node["missing_fields"]:
                html_parts.append(
                    f"    <p><strong>Missing Fields:</strong> {', '.join(node['missing_fields'])}</p>"
                )

            html_parts.append("  </div>")

        # Raw JSON if enabled
        if context.get("include_raw_data") and context.get("raw_json"):
            html_parts.extend(
                [
                    "  <h2>Raw JSON Data</h2>",
                    "  <details>",
                    "    <summary>Click to expand</summary>",
                    f"    <pre>{context['raw_json']}</pre>",
                    "  </details>",
                ]
            )

        html_parts.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html_parts)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a string for use as a filename."""
        invalid_chars = '<>:"/\\|?*'
        result = name
        for char in invalid_chars:
            result = result.replace(char, "_")
        return result[:50].strip("_")
