# LangChain / LangGraph Workspace

This workspace builds LangChain and LangGraph systems using Anthropic models via AWS Bedrock.

## Stack & Versions

- `langchain >= 1.2`, `langchain-core >= 1.2`
- `langgraph >= 1.1`
- `langchain-aws >= 1.4` — primary LLM and embeddings provider
- `langchain-community >= 0.4`

---

## Default Model

Always use `ChatBedrockConverse`. Never use `ChatBedrock`.

```python
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=4096,
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
```

Pass parameters directly — never use `model_kwargs` on `ChatBedrockConverse`.

---

## Hard Prohibitions

Never generate any of the following:

| Prohibited | Reason |
|---|---|
| `AgentExecutor`, `initialize_agent` | Removed — use LangGraph |
| `LLMChain`, `SequentialChain` | Removed — use LCEL |
| `ConversationalRetrievalChain`, `RetrievalQA` | Removed — use LCEL chain or LangGraph agent |
| `ChatBedrock` | Deprecated — use `ChatBedrockConverse` |
| `model_kwargs` on `ChatBedrockConverse` | Does not work — pass args directly |
| `from langchain.chat_models import ...` | Broken in 1.x — use provider packages |
| `from langchain.llms import ...` | Broken in 1.x — use provider packages |
| `from langchain.embeddings import ...` | Broken in 1.x — use provider packages |
| Manual `tool_calls` parsing | Always use `ToolNode` |
| `MessageGraph` | Removed in langgraph 1.0 — use `StateGraph` |

---

## Decision Rules

**When to use what:**

- Need persistence, HITL, retries, or multi-step reasoning → `StateGraph`
- Simple tool-using agent with no custom routing → `create_react_agent`
- Single-turn stateless transformation → LCEL chain (`prompt | llm | parser`)
- Structured output from LLM → `llm.with_structured_output(PydanticModel)`
- Conversational RAG → LangGraph agent + retriever as a `@tool`, not `ConversationalRetrievalChain`
- Multi-agent → supervisor pattern via `langgraph_supervisor`, not nested `AgentExecutor`

---

## Memory Hierarchy

Three distinct layers — never conflate them:

1. **In-context** — `messages` in graph state. Current run only. Lost when graph ends.
2. **Short-term persistent** — LangGraph checkpointer. Survives interrupts. Scoped to `thread_id`.
3. **Long-term** — External store (vector DB, SQL). Queried via a retriever tool. Never stored in graph state.

Graph state is a scratchpad, not a database. Long-term facts belong in external stores queried via tools.

---

## Agentic System Design Practices

### State Design
- Only put data in state that must cross node boundaries — local variables stay local to nodes
- Use `add_messages` reducer for message lists; plain types for fields that overwrite
- Never grow state unboundedly — summarize or trim message history before it exceeds context limits
- Adding fields to a state schema is safe; renaming or removing breaks existing checkpoints

### Graph Structure
- Each node has one responsibility — routing logic belongs in conditional edges, not inside nodes
- Every cycle must have an explicit exit condition modelled in state (iteration counter, status flag)
- `recursion_limit` is a safety net, not a design — never rely on it as the only exit
- Name nodes after what they do: `call_model`, `validate_input`, `fetch_context` — not `node_1`
- Compile the graph once and reuse the compiled instance — do not recompile per request

### Reliability
- Any irreversible action (write, send, delete, external API mutation) must be behind `interrupt()`
- Retry logic belongs at the graph level via a conditional edge — not inside node functions
- Raise `ToolException` for tool errors — let the LLM decide whether to retry or give up
- Set `recursion_limit` in config for every production graph invocation

### Tool Design
- One concern per tool — narrow and composable over broad and monolithic
- Read tools before write tools — the LLM should query state before mutating it
- The docstring is prompt engineering — describe precisely what the tool returns, not just what it does
- Write the docstring before the function body
- Use `InjectedToolArg` for runtime dependencies (DB connections, user context) the LLM must not see

### Observability
- Set `LANGCHAIN_TRACING_V2=true` from project init — not as an afterthought
- Add `run_name` and `metadata` to every `graph.invoke()` and `graph.stream()` call
- Track per-node token usage for cost attribution in multi-agent systems

### Multi-Agent
- Supervisor routes, it does not execute — keep supervisor nodes thin
- Agent handoffs via tools, not shared mutable state
- Each agent must be independently testable with a mock LLM
- Define non-overlapping agent responsibilities — ambiguous boundaries cause supervisor loops

---

## Canonical Patterns

### Graph skeleton

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_edge(START, "agent")

graph = builder.compile(checkpointer=MemorySaver())
```

### Tool + ToolNode

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def my_tool(query: str) -> str:
    """Returns X given Y — describe the return value precisely."""
    ...

llm_with_tools = llm.bind_tools([my_tool])
tool_node = ToolNode([my_tool])
```

### Invocation with config

```python
config = {
    "configurable": {"thread_id": "session-abc"},
    "recursion_limit": 25,
    "run_name": "my-agent-run",
    "metadata": {"user": "user-id"},
}
result = graph.invoke({"messages": [("human", "...")]}, config=config)
```

### Interrupt before irreversible action

```python
from langgraph.types import interrupt

def confirm_and_execute(state: State):
    decision = interrupt({"action": state["pending_action"], "question": "Approve?"})
    if decision == "approved":
        execute_action(state["pending_action"])
    return {"messages": [("assistant", f"Action {decision}.")]}
```
