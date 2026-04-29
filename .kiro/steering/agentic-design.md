---
inclusion: always
---

# Agentic System Design Principles

This workspace builds LangChain and LangGraph systems. Follow these principles for every agent, graph, and tool you generate.

---

## Memory Hierarchy — Most Important

There are three distinct memory layers. Never conflate them.

| Layer | Mechanism | Scope | Use for |
|---|---|---|---|
| In-context | `messages` in graph state | Current run only | Active conversation turns |
| Short-term persistent | LangGraph checkpointer | Per `thread_id` | Resuming interrupted runs |
| Long-term | External store (vector DB, SQL) | Permanent | Facts, history, knowledge base |

**Rules:**
- Graph state is a scratchpad — not a database
- Long-term facts are queried via a retriever tool, never stored in state
- Never grow `messages` unboundedly — summarize or trim before hitting context limits
- Checkpointer state is scoped to `thread_id` — use distinct IDs per conversation

---

## Exit Conditions — Every Cycle Must Have One

`recursion_limit` is a safety net, not a design. If the only thing stopping your graph from looping is a hard cap, the design is wrong.

**Always model termination explicitly in state:**

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int        # increment in each cycle node
    status: str            # "running" | "done" | "failed"
```

Use `iterations` or `status` in conditional edges to terminate cleanly. Set `recursion_limit` in config as a backstop only.

---

## Irreversible Actions Require interrupt()

Any node that writes to a database, sends a message, calls a mutating external API, or deletes data must be behind an `interrupt()` checkpoint by default.

```python
from langgraph.types import interrupt

def execute_action(state: State):
    decision = interrupt({
        "action": state["pending_action"],
        "question": "Approve this action?",
    })
    if decision == "approved":
        perform_write(state["pending_action"])
    return {"status": "done" if decision == "approved" else "cancelled"}
```

Easier to remove a checkpoint than to recover from an unintended write.

---

## State Design

- Only state that crosses node boundaries belongs in `State` — local variables stay local
- Use `add_messages` reducer for message lists; plain types overwrite on each update
- Adding fields to a state schema is safe — renaming or removing breaks existing checkpoints
- Never put computed or derivable values in state — compute them in the node

---

## Graph Structure

- **One responsibility per node** — routing logic belongs in conditional edges, not inside nodes
- **Name nodes descriptively** — `call_model`, `validate_input`, `fetch_context` not `node_1`
- **Compile once** — do not recompile the graph per request; reuse the compiled instance
- **Retry at the graph level** — conditional edges handle retries, not try/except inside nodes

---

## Tool Design

- Narrow tools over broad — one concern per tool, composable
- **Read before write** — query state before mutating it; separate read and write tools
- **Docstring = prompt engineering** — describe what the tool *returns* precisely; write it before the function body
- Raise `ToolException` for errors — let the LLM decide to retry rather than crashing the graph
- Use `InjectedToolArg` for runtime dependencies (DB connections, auth tokens) the LLM must not see

---

## Multi-Agent Design

- **Supervisor routes, it does not execute** — keep supervisor nodes thin; they only decide who handles what
- **Handoffs via tools** — agents communicate through handoff tools, not shared mutable state
- **Non-overlapping responsibilities** — ambiguous agent boundaries cause infinite supervisor loops
- **Each agent is independently testable** — test agents in isolation with mock LLMs before wiring into a supervisor

---

## Observability

- Set `LANGCHAIN_TRACING_V2=true` at project init — tracing is harder to add retroactively to a running system
- Add `run_name` and `metadata` to every `graph.invoke()` / `graph.stream()` call
- Track token usage per node in multi-agent systems for cost attribution
