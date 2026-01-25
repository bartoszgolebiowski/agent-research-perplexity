from __future__ import annotations

from dotenv import load_dotenv

"""
ICP Intelligence Agent - Main entry point for the automated research workflow.

This is the sole entry point for the ICP Analysis system. It follows the same
pattern as the standard Agent:
- Coordinator decides next action
- Executor/Tools execute the action  
- State manager updates state
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from src.engine.coordinator import AgentActionCoordinator
from src.engine.executor import LLMExecutor
from src.engine.types import ActionType, AnalysisNodeStatus, WorkflowStage
from src.llm import LLMCallError
from src.logger import get_agent_logger
from src.memory import (
    AgentState,
    AnalysisQueue,
    ICP_STAGE_HANDLERS,
    ICPAnalysisInput,
    ICPAnalysisOutput,
    create_icp_initial_state,
    handle_expand_tree,
    update_state_from_skill,
    update_state_from_tool,
)
from src.memory.state_manager import create_initial_state
from src.tools import (
    InputValidatorClient,
    ReportGeneratorClient,
    ReportRequest,
    ReportResponse,
    ValidationError,
    WebSearchClient,
)
from src.tools.models import ToolName


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ICPAgentConfig:
    """Configuration for the ICP Intelligence Agent."""

    max_iterations: int = 500
    enable_dynamic_expansion: bool = True
    report_output_dir: Path = Path("./reports")
    snapshot_dir: Optional[Path] = None


@dataclass(frozen=True, slots=True)
class ICPAgentResult:
    """Result of an ICP analysis run."""

    state: AgentState
    output: ICPAnalysisOutput
    report: Optional[ReportResponse]
    iterations_executed: int
    success: bool
    error_message: Optional[str] = None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "ICP Analysis Complete",
            f"  Total Nodes: {self.output.total_nodes}",
            f"  Successful: {self.output.successful_nodes}",
            f"  Partial: {self.output.partial_nodes}",
            f"  Failed: {self.output.failed_nodes}",
            f"  Skipped: {self.output.skipped_nodes}",
            f"  Iterations: {self.iterations_executed}",
        ]
        if self.report and self.report.success:
            lines.append(f"  Report: {self.report.html_path}")
        return "\n".join(lines)


class ICPAgent:
    """High-level entry point that orchestrates the ICP research workflow."""

    def __init__(
        self,
        *,
        llm_executor: LLMExecutor,
        search_client: WebSearchClient,
        report_client: ReportGeneratorClient,
        validator_client: InputValidatorClient,
        config: Optional[ICPAgentConfig] = None,
        coordinator: Optional[AgentActionCoordinator] = None,
    ) -> None:
        self._llm_executor = llm_executor
        self._search_client = search_client
        self._report_client = report_client
        self._validator = validator_client
        self._config = config or ICPAgentConfig()
        self._coordinator = coordinator or AgentActionCoordinator()
        self._logger = get_agent_logger()

    @classmethod
    def from_env(cls, config: Optional[ICPAgentConfig] = None) -> "ICPAgent":
        """Create an agent wired to environment-configured dependencies."""
        return cls(
            llm_executor=LLMExecutor.from_env(),
            search_client=WebSearchClient.from_env(),
            report_client=ReportGeneratorClient.from_env(),
            validator_client=InputValidatorClient.from_env(),
            config=config,
        )

    def run_from_snapshot(
        self, snapshot_file: Path, config: Optional[ICPAgentConfig] = None
    ) -> ICPAgentResult:
        """Run the ICP analysis from a saved snapshot."""
        self._logger.info(f"Loading snapshot from: {snapshot_file}")

        state = AgentState.load_snapshot(str(snapshot_file))
        # Recreate the queue from the saved analysis input
        queue = AnalysisQueue.from_input(state.semantic.analysis_input)
        state.icp_queue = queue

        # Update config if provided
        if config:
            self._config = config

        self._logger.info(
            f"Resumed from snapshot with {queue.pending_count()} pending nodes"
        )

        return self._run_from_state(state)

    def run_from_file(self, input_file: Path) -> ICPAgentResult:
        """Run the ICP analysis from a JSON input file."""
        self._logger.info(f"Starting ICP analysis from file: {input_file}")

        try:
            analysis_input = self._validator.validate_file(input_file)
            self._logger.info(f"Validated input: {analysis_input.name}")
        except ValidationError as e:
            self._logger.error(f"Validation failed: {e}")
            return self._error_result(str(e))

        return self.run(analysis_input)

    def run(self, analysis_input: ICPAnalysisInput) -> ICPAgentResult:
        """Execute the ICP analysis workflow."""
        queue = AnalysisQueue.from_input(analysis_input)
        state = create_icp_initial_state(analysis_input, queue)

        self._logger.info(f"Initialized queue with {queue.pending_count()} nodes")

        return self._run_from_state(state)

    def _run_from_state(self, state: AgentState) -> ICPAgentResult:
        """Execute the ICP analysis workflow from a given state."""
        iterations = 0

        try:
            for step in range(self._config.max_iterations):
                iterations = step + 1
                current_stage = state.workflow.current_stage

                # Count node statuses
                all_nodes = state.semantic.analysis_input.get_all_nodes()
                total_nodes = len(all_nodes)
                pending_count = len(
                    [n for n in all_nodes if n.status == AnalysisNodeStatus.PENDING]
                )
                in_progress_count = len(
                    [n for n in all_nodes if n.status == AnalysisNodeStatus.IN_PROGRESS]
                )
                completed_count = len(
                    [n for n in all_nodes if n.status == AnalysisNodeStatus.COMPLETED]
                )
                partial_count = len(
                    [n for n in all_nodes if n.status == AnalysisNodeStatus.PARTIAL]
                )
                failed_count = len(
                    [n for n in all_nodes if n.status == AnalysisNodeStatus.FAILED]
                )
                skipped_count = len(
                    [n for n in all_nodes if n.status == AnalysisNodeStatus.SKIPPED]
                )

                self._logger.info(
                    f"Iteration {iterations}: Total={total_nodes}, Pending={pending_count}, "
                    f"InProgress={in_progress_count}, Completed={completed_count}, "
                    f"Partial={partial_count}, Failed={failed_count}, Skipped={skipped_count} | Stage={current_stage.value}"
                )

                if current_stage == WorkflowStage.ICP_REPORT_COMPLETE:
                    break

                state = self._process_stage(state)

                # Periodic snapshot saving every 10 iterations
                if iterations % 10 == 0 and self._config.snapshot_dir:
                    self._save_snapshot(state, f"periodic_{iterations}")
            else:
                self._logger.warning(
                    f"Reached max iterations ({self._config.max_iterations})"
                )
        except Exception as e:
            self._logger.error(f"Error during processing: {e}")
            return self._error_result(str(e), state, iterations)

        # Generate final output and report
        analysis_input = state.semantic.analysis_input
        output = self._create_output(analysis_input)
        report = self._generate_report(output)

        return ICPAgentResult(
            state=state,
            output=output,
            report=report,
            iterations_executed=iterations,
            success=True,
        )

    def _process_stage(self, state: AgentState) -> AgentState:
        """Process the current workflow stage."""
        stage = state.workflow.current_stage

        # Check for ICP stage handlers (state transitions)
        if stage in ICP_STAGE_HANDLERS:
            handler = ICP_STAGE_HANDLERS[stage]
            # Special case: expand_tree needs config parameter
            if stage == WorkflowStage.ICP_EXPAND_TREE:
                return handle_expand_tree(state, self._config.enable_dynamic_expansion)
            return handler(state)

        # For LLM/Tool stages, use coordinator decision
        decision = self._coordinator.next_action(state)

        if decision.action_type in (ActionType.COMPLETE, ActionType.NOOP):
            return state

        if decision.action_type == ActionType.LLM_SKILL and decision.skill:
            self._logger.info(f"Executing skill: {decision.skill.value}")
            context = self._build_context(state)
            self._logger.debug(f"Skill input context keys: {list(context.keys())}")
            try:
                output = self._llm_executor.execute(decision.skill, context)
                self._logger.info(f"Skill {decision.skill.value} completed")
                self._logger.debug(
                    f"Skill output: {output.model_dump() if hasattr(output, 'model_dump') else output}"
                )
                return update_state_from_skill(state, decision.skill, output)
                return update_state_from_skill(state, decision.skill, output)
            except LLMCallError as e:
                self._logger.error(f"LLM call failed: {e}")
                from copy import deepcopy

                new_state = deepcopy(state)
                new_state.workflow.record_transition(
                    to_stage=WorkflowStage.ICP_NODE_FAILED,
                    reason=f"LLM call failed: {e}",
                )
                return new_state

        if decision.action_type == ActionType.TOOL and decision.tool_type:
            self._logger.info(f"Executing tool: {decision.tool_type.value}")
            output = self._execute_tool(state, decision.tool_type)
            self._logger.info(f"Tool {decision.tool_type.value} completed")
            self._logger.debug(
                f"Tool output: {output.model_dump() if hasattr(output, 'model_dump') else output}"
            )
            new_state = update_state_from_tool(state, decision.tool_type, output)
            # Save snapshot after tool execution
            if self._config.snapshot_dir and decision.tool_type == ToolName.WEB_SEARCH:
                self._save_snapshot(new_state)
            return new_state

        raise RuntimeError(f"Unhandled decision: {decision}")

    def _save_snapshot(self, state: AgentState, suffix: str = "") -> None:
        """Save a snapshot of the current state."""
        if not self._config.snapshot_dir:
            return
        self._config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        node_id = state.working.current_node_id or "unknown"
        suffix_str = f"_{suffix}" if suffix else ""
        filename = f"snapshot_{node_id}_{timestamp}{suffix_str}.json"
        filepath = self._config.snapshot_dir / filename
        state.save_snapshot(str(filepath))
        self._logger.info(f"Snapshot saved: {filepath}")

    def _execute_tool(self, state: AgentState, tool: ToolName) -> BaseModel:
        """Execute a tool and return its output."""
        if tool == ToolName.WEB_SEARCH:
            # Check search limit before executing
            current_node = state.get_current_node()
            if current_node and current_node.search_count >= current_node.max_searches:
                self._logger.warning(
                    f"Search limit reached for node {current_node.id}: "
                    f"{current_node.search_count}/{current_node.max_searches}"
                )
                # Return a failed search response
                from src.tools.models import WebSearchResponse
                from datetime import datetime

                return WebSearchResponse(
                    queries=state.working.current_search_queries,
                    node_id=state.working.current_node_id or "",
                    results=[],
                    raw_content="",
                    success=False,
                    error_message=f"Search limit reached ({current_node.max_searches})",
                    executed_at=datetime.now(),
                    http_errors=[],
                )

            request = state.get_web_search_request()
            self._logger.debug(f"Tool input (web_search): {request.model_dump()}")
            return self._search_client.search(request)
        raise RuntimeError(f"Unknown tool: {tool}")

    def _build_context(self, state: AgentState) -> Dict[str, Any]:
        """Build context for skill prompt rendering."""
        context: Dict[str, Any] = {
            "state": state,
            "core": state.core,
            "semantic": state.semantic,
            "episodic": state.episodic,
            "workflow": state.workflow,
            "working": state.working,
            "procedural": state.procedural,
            "resource": state.resource,
        }

        if state.semantic.global_constraints:
            context["global_constraints"] = state.semantic.global_constraints

        current_node = state.get_current_node()
        if current_node:
            context["current_node"] = current_node

        if state.working.current_search_results:
            results = state.working.current_search_results
            context["search_results"] = results.results
            context["raw_content"] = results.raw_content
        else:
            context["search_results"] = []
            context["raw_content"] = ""

        if state.working.current_extracted_data:
            context["extracted_fields"] = (
                state.working.current_extracted_data.extracted_fields
            )
            context["missing_fields"] = (
                state.working.current_extracted_data.missing_fields
            )
        else:
            context["extracted_fields"] = []
            context["missing_fields"] = []

        context["previous_query"] = (
            state.working.current_search_queries[0]
            if state.working.current_search_queries
            else ""
        )
        context["search_query"] = (
            state.working.current_search_queries[0]
            if state.working.current_search_queries
            else ""
        )
        context["search_queries"] = state.working.current_search_queries
        return context

    def _create_output(self, analysis_input: ICPAnalysisInput) -> ICPAnalysisOutput:
        """Create the final analysis output."""
        output = ICPAnalysisOutput(input=analysis_input, completed_at=datetime.now())
        output.compute_statistics()
        return output

    def _generate_report(self, output: ICPAnalysisOutput) -> ReportResponse:
        """Generate the report using the report tool."""
        request = ReportRequest(
            analysis_output=output,
            output_directory=self._config.report_output_dir,
        )
        return self._report_client.generate(request)

    def _error_result(
        self,
        error_message: str,
        state: Optional[AgentState] = None,
        iterations: int = 0,
    ) -> ICPAgentResult:
        """Create an error result."""
        if state is None:
            state = create_initial_state(goal="Error")

        return ICPAgentResult(
            state=state,
            output=ICPAnalysisOutput(
                input=ICPAnalysisInput(name="Error"),
                completed_at=datetime.now(),
            ),
            report=None,
            iterations_executed=iterations,
            success=False,
            error_message=error_message,
        )


def main() -> None:
    """CLI entry point for running ICP analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="ICP Intelligence Agent")
    parser.add_argument(
        "input_file", type=Path, help="Path to input JSON file or snapshot JSON file"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Treat input_file as a snapshot to resume from",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    parser.add_argument("--snapshot-dir", type=Path, help="Directory to save snapshots")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--no-expansion", action="store_true")

    args = parser.parse_args()

    config = ICPAgentConfig(
        max_iterations=args.max_iterations,
        enable_dynamic_expansion=not args.no_expansion,
        report_output_dir=args.output_dir,
        snapshot_dir=args.snapshot_dir,
    )

    load_dotenv()
    agent = ICPAgent.from_env(config=config)

    if args.snapshot:
        result = agent.run_from_snapshot(args.input_file, config=config)
    else:
        result = agent.run_from_file(args.input_file)

    print(result.summary())

    if not result.success:
        exit(1)


if __name__ == "__main__":
    main()
