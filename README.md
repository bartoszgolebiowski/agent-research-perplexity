# ICP Intelligence Agent

A deterministic, state-driven AI agent for automated research and structured data extraction using web search and LLM processing.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/Mac:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional: Configure search limits
PERPLEXITY_MAX_RESULTS=3
PERPLEXITY_MAX_TOKENS=25000
PERPLEXITY_MAX_TOKENS_PER_PAGE=1024
```

## Usage

### Basic Run

```bash
icp-agent examples/example_input.json
```

### Advanced Options

```bash
# With custom output directory
icp-agent examples/example_input.json --output-dir ./my-reports

# Enable snapshots for resumability
icp-agent examples/example_input.json --snapshot-dir ./snapshots

# Resume from snapshot
icp-agent snapshots/snapshot_node123_20241201_120000.json --snapshot

# Limit iterations
icp-agent examples/example_input.json --max-iterations 100

# Disable dynamic expansion
icp-agent examples/example_input.json --no-expansion
```

## Input Format

The agent expects a JSON file defining analysis tasks. See `examples/example_input.json` for the structure:

- `root_nodes`: Hierarchical analysis tasks
- `extraction_fields`: Data to extract with priorities
- `global_constraints`: Currency, language, date formats
- `parameters`: Task-specific modifiers

## Output

- **Reports**: HTML files in `./reports/` with extracted data and statistics
- **Snapshots**: JSON state files in `--snapshot-dir` for resuming interrupted runs
- **Logs**: Console output with progress and node status

## Architecture

- **State-Driven**: Deterministic workflow using predefined transitions
- **Multi-Query Search**: Executes multiple Perplexity queries per node for better coverage
- **Immutable State**: All updates create deep copies
- **LLM Skills**: Structured prompts for query formulation, data extraction, and validation

## Development

### Run Tests

```bash
# Add test commands as needed
```

### Code Structure

- `src/engine/`: Core orchestration and workflow
- `src/memory/`: State management and models
- `src/skills/`: LLM-powered capabilities
- `src/tools/`: External integrations (search, validation)
- `src/llm/`: Language model client
- `src/prompting/`: Jinja2 templates
