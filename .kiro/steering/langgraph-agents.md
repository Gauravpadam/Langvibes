---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# LangGraph Agents — Graphs, State, Checkpointers

Target: `langgraph>=1.1`, `langchain-core>=1.2`, `langchain-aws>=1.4`

---

## Core Imports

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
```

Never use `AgentExecutor`, `initialize_agent`, or `MessageGraph` — all removed.

---

## State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # accumulates via reducer
    iterations: int                                        # overwritten each update
    status: str                                            # "running" | "done" | "failed"

# Shorthand when messages are the only state needed
from langgraph.graph import MessagesState
```

**State rules:**
- Only data crossing node boundaries belongs in state
- `add_messages` accumulates — plain types overwrite
- Model termination in state (`iterations`, `status`) — never rely on `recursion_limit` alone
- Adding fields is safe; renaming or removing breaks existing checkpoints

---

## Graph Construction

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # return only changed keys

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["iterations"] >= 10:
        return END
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())  # compile once, reuse
```

---

## Checkpointers

```python
# Development
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# Lightweight production
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Production
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

Always pass `thread_id` in config — without it, persistence has no effect:

```python
config = {
    "configurable": {"thread_id": "user-session-abc"},
    "recursion_limit": 25,
    "run_name": "agent-run",
    "metadata": {"user_id": "xyz"},
}
result = graph.invoke({"messages": [("human", "Hello")]}, config=config)
```

---

## Streaming

```python
# Full state per step
for event in graph.stream(inputs, config=config):
    for node_name, output in event.items():
        print(f"{node_name}: {output}")

# Token streaming from LLM
for chunk in graph.stream(inputs, config=config, stream_mode="messages"):
    if chunk[1].get("langgraph_node") == "agent":
        print(chunk[0].content, end="", flush=True)
```

---

## Human-in-the-Loop

Use `interrupt()` before any irreversible action. Requires a checkpointer.

```python
from langgraph.types import interrupt

def review_node(state: AgentState):
    decision = interrupt({
        "question": "Approve this action?",
        "action": state["messages"][-1].content,
    })
    return {"messages": [("human", f"Decision: {decision}")]}

# Resume after interrupt
graph.invoke(Command(resume="approved"), config=config)
```

---

## Dynamic Routing

```python
from langgraph.types import Command

def router(state: AgentState) -> Command:
    if "urgent" in state["messages"][-1].content:
        return Command(goto="escalate", update={"status": "urgent"})
    return Command(goto="standard")
```

---

## Prebuilt ReAct Agent

For simple tool-using agents without custom routing:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
    state_modifier="You are a helpful assistant.",
)
result = agent.invoke(
    {"messages": [("human", "Search for X")]},
    config={"configurable": {"thread_id": "1"}},
)
```

---

## Common Mistakes

- **Returning full state from nodes** — return only changed keys: `{"messages": [response]}`
- **No checkpointer + using thread_id** — checkpointer is required for persistence
- **interrupt() without checkpointer** — interrupt only works with a compiled checkpointer
- **Recompiling per request** — compile once at startup, reuse the graph instance
- **No exit condition in cycles** — model termination in state, not just `recursion_limit`
