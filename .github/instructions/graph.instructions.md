---
applyTo: "src/agent/**/*.py"
---

# Graph Component Instructions

These guidelines cover authoring LangGraph flows under `src/agent/`. They assume LangGraph 1.x APIs and emphasize deterministic orchestration over implicit choreography.

## Guiding Principles

1. **Graphs Are Orchestrators**

   - Every thread of execution must be driven by the graph edges, not hidden logic inside skills or state updates.
   - Nodes should be pure functions of their input; side effects go through explicit dependencies (agent, tools, etc.).

2. **Multiple Nodes, Explicit Phases**

   - Avoid single-node “catch-all” graphs and resist the temptation to ship a single linear rail.
   - Instead of `ingest -> decide -> execute -> finalize` as a fixed pipeline, build a richer state machine with specialized nodes per workflow phase (e.g., intake, refinement router, persistence handler, summary synthesizer) and explicit edges between them.
   - Each meaningful business phase should have its own node and transitions so the execution path documents how the run evolved.

3. **Deterministic Routing**

   - Use `add_conditional_edges` with enums from `architecture.domain` to express routing decisions.
   - Favor several small routers (e.g., a coordinator per phase) over one mega-conditional. Each router node should map to a single decision concern.

4. **State Contracts**

   - Represent LangGraph state via `TypedDict` (or `typing_extensions.TypedDict` on Python < 3.12).
   - Persist serialized `AgentState` as JSON-safe data at handoff boundaries; convert back to Pydantic objects inside nodes as needed.

5. **Granular Execution Nodes**

   - Separate skill execution from tool execution nodes. This enables independent retries, metrics, and traces.
   - Keep nodes short and composable. If a node grows beyond ~30 lines or gains branching, split it into helper nodes.

6. **Conditional Navigation Patterns**

   - Build loops explicitly: `decide_phase -> router -> specialized_node -> decide_phase`. Keep the routing surface small per node but stitch several routers together to reflect the full state machine.
   - When multiple downstream paths exist, route by `DecisionType` or a dedicated router enum rather than string literals, and prefer chaining smaller routers over one monolithic conditional block.

7. **Context + Runtime Config**

   - Expose runtime settings via a `Context` Pydantic model (e.g., storage locations, feature flags). Inject it once when compiling the graph.
   - Avoid global variables; pass dependencies (agents, tools) into closures that define nodes.

8. **Testing & Observability**
   - Prefer small helper functions (`_ingest`, `_decide`, etc.) so they can be unit-tested without spinning up LangGraph.
   - Emit consistent state keys (`agent_state`, `agent_response`, `decision`, `timestamp`) so LangGraph Studio displays meaningful info.

## Anti-Patterns to Avoid

- Single-node graphs or hardcoded `ingest -> decide -> execute -> finalize` rails that hide the actual business phases.
- Delegating flow control to LLM prompts (“decide what to do next inside the skill”).
- Large router nodes with dozens of nested `if` statements; split them per concern.
- Mutating global state or module-level singletons inside nodes.
- Returning Pydantic objects directly to the runtime without serializing to builtins.

Following these practices keeps the graph deterministic, debuggable, and faithful to an orchestrated architecture.
