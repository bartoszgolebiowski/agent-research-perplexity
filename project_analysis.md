# Detailed Project Analysis: Agent Research Perplexity

## Overview

This project implements a sophisticated **deterministic, state-driven AI agent architecture** designed for automated research and structured data extraction. The system is specifically built for "ICP Intelligence" workflows, where ICP likely stands for "Intelligent Content Processing" or similar research-oriented tasks. The agent orchestrates web searches using Perplexity AI and leverages Large Language Models (LLMs) to extract, validate, and organize data from web sources into structured formats.

The project is written in Python 3.11+ and follows strict engineering guidelines for maintainability, type safety, and separation of concerns.

## Project Structure

### Root Level

- **`pyproject.toml`**: Project configuration using modern Python packaging standards. Defines dependencies (pydantic, jinja2, openai, requests), entry point (`icp-agent` CLI command), and build system.
- **`src/icp_agent.py`**: Main entry point implementing the `ICPAgent` class that orchestrates the entire research workflow.
- **`examples/example_input.json`**: Sample input file demonstrating the analysis specification format for Transition Technologies S.A. financial analysis.
- **`reports/`**: Output directory for generated HTML reports.
- **`licence.md`**, **`README.md`**: Standard project documentation.

### Source Code Organization (`src/`)

#### Core Engine (`src/engine/`)

The engine layer implements the deterministic state machine that drives all agent behavior.

- **`coordinator.py`**: `AgentActionCoordinator` class that deterministically decides the next action based on current workflow stage. Uses a predefined `TRANSITIONS` map rather than LLM decision-making.
- **`executor.py`**: `LLMExecutor` class that handles LLM skill execution using structured prompts and output parsing.
- **`types.py`**: Shared type definitions including `WorkflowStage` enum, `ActionType` enum, and `AnalysisNodeStatus` enum.
- **`workflow_transitions.py`**: Defines the complete state machine transitions for ICP workflows. Maps each `WorkflowStage` to an `ActionType` (LLM_SKILL, TOOL, COMPLETE, NOOP) and associated skill/tool.

#### LLM Integration (`src/llm/`)

Thin wrapper around LLM APIs for structured output generation.

- **`client.py`**: `LLMClient` using OpenAI SDK with OpenRouter base URL. Supports structured outputs with Pydantic model validation.
- **`config.py`**: Configuration management for LLM parameters (API key, model, temperature, etc.).
- **`exceptions.py`**: Custom exceptions for LLM-related errors.

#### Memory System (`src/memory/`)

Implements a 7-layer memory architecture for state management. All state updates are immutable (deepcopy before modification).

- **`models.py`**: Comprehensive data models using Pydantic BaseModel:
  - **ICP Analysis Models**: `ExtractionField`, `ExtractedDataPoint`, `AnalysisNode`, `ICPAnalysisInput`, `ICPAnalysisOutput`
  - **Memory Layers**: `ConstitutionalMemory` (guardrails), `WorkingMemory` (session context), `WorkflowMemory` (state machine), `EpisodicMemory` (event history), `SemanticMemory` (knowledge base), `ProceduralMemory` (capabilities), `ResourceMemory` (limits)
  - **`AgentState`**: Composite model containing all memory layers plus ICP-specific runtime data.

- **`state_manager.py`**: Functions for creating initial states and updating state from skill/tool outputs. Each skill and tool has a dedicated handler function that immutably updates state.

- **`queue.py`**: `AnalysisQueue` class managing the processing order of analysis nodes. Supports tree traversal (parents before children), dynamic node insertion, and retry logic.

#### Prompting System (`src/prompting/`)

Jinja2-based templating for LLM prompts.

- **`environment.py`**: Template environment configuration.
- **`jinja/`**: Template directory
  - **`memory/`**: Templates for rendering different memory layers into prompts.
  - **`skills/`**: Skill-specific prompt templates.
  - **`report/`**: Report generation templates.

#### Skills Layer (`src/skills/`)

Declarative skill definitions for LLM-powered capabilities.

- **`definitions.py`**: Registry of all available skills with their templates and output models.
- **`models.py`**: Pydantic output models for each skill.
- **`base.py`**: Base classes and enums (`SkillName`, `SkillDefinition`).

#### Tools Layer (`src/tools/`)

External tool integrations.

- **`web_search.py`**: `WebSearchClient` using Perplexity SDK for web search with structured result parsing.
- **`input_validator.py`**: Input validation and parsing.
- **`report_generator.py`**: HTML report generation from analysis results.
- **`models.py`**: Shared tool models and request/response schemas.

#### Utilities

- **`logger.py`**: Centralized logging configuration.

## Architecture Principles

### 1. State-Driven Flow

- **No LLM Decision Making**: The coordinator uses a hardcoded `TRANSITIONS` map to determine next actions. LLMs are only used for content processing (extraction, validation, etc.), not control flow.
- **Deterministic Behavior**: Given the same input and state, the system always produces the same sequence of actions.

### 2. Separation of Concerns

- **Engine**: Orchestration and state transitions.
- **Memory**: Immutable state management across 7 layers.
- **Skills**: Declarative LLM prompt definitions.
- **Tools**: External integrations (search, validation, reporting).

### 3. Immutability

- All state updates create deep copies before modification.
- Pattern: `new_state = deepcopy(state)` → apply changes → `return new_state`

### 4. Type Safety

- Full type hinting throughout.
- Pydantic models for all data structures.
- `@dataclass(frozen=True, slots=True)` for configuration classes.

## Workflow Execution

### High-Level Flow

1. **Ingestion**: Load and validate JSON input specification.
2. **Queue Initialization**: Create analysis queue from input nodes.
3. **Node Processing Loop**: Process nodes until completion or iteration limit.
4. **Report Generation**: Create HTML report from results.

### ICP Workflow Stages

Defined in `workflow_transitions.py`:

#### Phase A: Ingestion

- `ICP_INGESTION`: File ingestion (external)
- `ICP_VALIDATION`: Input validation (external)
- `ICP_QUEUE_READY`: Queue initialized
- `ICP_NODE_START`: Starting node processing

#### Phase B: Research Loop (per node)

- `ICP_FORMULATE_QUERY`: Generate optimized search query
- `ICP_EXECUTE_SEARCH`: Execute web search via Perplexity
- `ICP_EXTRACT_DATA`: Extract structured data from search results
- `ICP_VALIDATE_DATA`: Validate extraction completeness
- `ICP_DECISION_GATE`: Route based on validation result
- `ICP_REFINE_QUERY`: Improve query for retry attempts
- `ICP_NODE_COMPLETE`: Node fully successful
- `ICP_NODE_FAILED`: Node exhausted retries
- `ICP_NODE_PARTIAL`: Node partially successful
- `ICP_NODE_SKIPPED`: Node skipped (nice-to-have only)

#### Phase C: Dynamic Expansion

- `ICP_DISCOVER_ENTITIES`: Find new entities for expansion
- `ICP_EXPAND_TREE`: Add discovered nodes to queue

#### Phase D: Completion

- `ICP_ALL_NODES_DONE`: All nodes processed
- `ICP_GENERATE_REPORT`: Create HTML report
- `ICP_REPORT_COMPLETE`: Analysis finished

### Node Processing Details

For each analysis node:

1. **Query Formulation**: LLM analyzes node description, extraction fields, and constraints to create keyword-optimized search query.
2. **Web Search**: Execute query via Perplexity API, retrieve top results with content.
3. **Data Extraction**: LLM parses search results to extract specific data points with citations.
4. **Validation**: LLM assesses completeness against requirements.
5. **Decision Routing**:
   - If complete: Mark node done, proceed to next
   - If partial but acceptable: Mark partial, continue
   - If missing critical data: Retry with refined query (up to max_retries)
   - If exhausted retries: Mark failed

### Dynamic Expansion

After node completion, the system can discover new related entities and dynamically add them to the analysis tree.

## Skills Implementation

All skills use Jinja2 templates and structured Pydantic outputs:

### FORMULATE_QUERY

- **Input**: Node description, extraction fields, global constraints
- **Output**: Optimized search query + target fields list
- **Purpose**: Convert human-readable tasks to web-searchable queries

### EXTRACT_DATA

- **Input**: Search results, target fields, current extractions
- **Output**: Extracted data points with citations + missing fields
- **Purpose**: Parse web content into structured data

### VALIDATE_DATA

- **Input**: Extracted data, requirements, validation criteria
- **Output**: Completeness assessment + recommended action
- **Purpose**: Quality control and routing decisions

### REFINE_QUERY

- **Input**: Previous query, missing fields, failure analysis
- **Output**: Improved query + focus areas
- **Purpose**: Iterative query improvement for retries

### DISCOVER_ENTITIES

- **Input**: Completed node data, expansion criteria
- **Output**: New analysis nodes to add
- **Purpose**: Dynamic workflow expansion

## Tools Implementation

### Web Search Tool

- **Provider**: Perplexity AI SDK
- **Configuration**: API key, result limits, token limits
- **Features**: Structured result parsing, error handling, content extraction

### Input Validator

- **Function**: Parse and validate JSON input files
- **Output**: `ICPAnalysisInput` object or validation errors

### Report Generator

- **Input**: `ICPAnalysisOutput` with all results
- **Output**: HTML report with statistics and data tables
- **Features**: Jinja2 templating, structured data presentation

## Data Models

### Analysis Specification

- **ICPAnalysisInput**: Root specification with global constraints and node tree
- **AnalysisNode**: Hierarchical task definition with extraction fields
- **ExtractionField**: Specific data to extract (name, type, priority, unit)
- **GlobalConstraints**: Currency, language, date format, target years, regions

### Runtime State

- **AgentState**: Complete system state across 7 memory layers
- **AnalysisQueue**: Processing queue with node status tracking
- **WorkflowMemory**: Current stage and transition history

### Results

- **AnalysisNodeResult**: Per-node extraction results
- **ExtractedDataPoint**: Individual data point with citation and confidence
- **ICPAnalysisOutput**: Complete analysis summary with statistics

## Configuration and Environment

### Environment Variables

- `OPENROUTER_API_KEY`: For LLM calls
- `PERPLEXITY_API_KEY`: For web search
- Various limits and model configurations

### CLI Interface

```bash
icp-agent input_file.json [--output-dir reports] [--max-iterations 500] [--no-expansion]
```

### Dependencies

- **Core**: pydantic, jinja2, python-dotenv
- **APIs**: openai, requests
- **Search**: perplexity (external SDK)

## Key Design Patterns

### Factory Methods

- `from_env()` methods for all configurable classes
- Environment-based configuration loading

### Handler Pattern

- Skill and tool outputs route to specific handler functions
- Handlers immutably update state and transition workflow

### Template Pattern

- Jinja2 templates for all LLM prompts
- Consistent prompt structure with memory layer inclusion

### Queue Pattern

- Analysis nodes processed in tree order
- Support for retries and dynamic insertion

## Error Handling and Resilience

### Retry Logic

- Nodes can retry up to `max_retries` (default 3)
- Query refinement on failures
- Exponential backoff not implemented (deterministic)

### Validation

- Input validation before processing
- LLM output schema validation
- Data completeness checking

### Failure Modes

- Node-level failures (marked as FAILED)
- Partial completions (marked as PARTIAL)
- Skipped nodes (nice-to-have fields)
- System-level errors with graceful degradation

## Output and Reporting

### HTML Reports

- Generated from Jinja2 templates
- Include statistics, data tables, citations
- Saved to configurable output directory

### Logging

- Comprehensive logging at INFO/DEBUG levels
- Node status tracking
- Error reporting with context

### Statistics

- Total/pending/completed/failed/skipped node counts
- Success rates and completion metrics
- Execution time tracking

## Example Usage

The system processes JSON specifications like the provided example for analyzing Transition Technologies S.A. financial data. The agent:

1. Loads the specification defining revenue, EBITDA, headcount, etc.
2. Processes each node by searching for financial reports
3. Extracts specific data points with citations
4. Validates completeness and retries if needed
5. Generates a comprehensive HTML report

## Future Extensibility

The architecture supports easy addition of:

- New skills (add definition, template, handler)
- New tools (implement client, add to transitions)
- New workflow stages (extend transitions map)
- New memory layers (add to AgentState)

## Conclusion

This is a production-ready, well-architected AI agent system for automated research and data extraction. The deterministic state machine ensures predictable behavior, while the layered architecture provides maintainability and extensibility. The system successfully bridges web search capabilities with LLM processing to create structured knowledge from unstructured sources.</content>
<parameter name="filePath">c:\Users\golebiowskib\OneDrive - TRANSITION TECHNOLOGIES PSC S.A\Desktop\coding\agents\repos\agent-research-perplexity\project_analysis.md
