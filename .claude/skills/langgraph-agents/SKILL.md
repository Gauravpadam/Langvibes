---
name: langgraph-agents
description: Use this skill when building, debugging, or extending LangGraph agents, state machines, or graph-based workflows. Triggers when code imports from langgraph, uses StateGraph, mentions "graph", "node", "edge", "checkpointer", "human-in-the-loop", "interrupt", "supervisor", or "multi-agent". Also triggers when the user asks to "create an agent", "add a node", or "wire up a graph".
version: 1.0.0
---

# LangGraph Agents — v1.0 Reference

Target versions: `langgraph>=1.1`, `langchain-core>=1.2`, `langchain-aws>=1.4`

---

## Core Imports

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated
```

---

## State Schema

Always define state as a `TypedDict`. Use `Annotated` with a reducer for fields that accumulate.

```python
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # accumulates, never overwrites
    context: str                                           # overwritten each update
    iterations: int
```

**Use `MessagesState` shorthand when messages are the only state:**
```python
from langgraph.graph import MessagesState  # equivalent to the above messages field only
```

---

## Building a Graph

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()
```

---

## Persistence (Checkpointers)

```python
# In-memory (development)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SQLite (lightweight production)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Postgres (production)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

graph = builder.compile(checkpointer=checkpointer)

# thread_id scopes memory per conversation
config = {"configurable": {"thread_id": "user-session-abc"}}
result = graph.invoke({"messages": [("human", "Hello")]}, config=config)
```

---

## Invocation & Streaming

```python
# Single invocation
result = graph.invoke({"messages": [("human", "What is 2+2?")]}, config=config)

# Stream all events
for event in graph.stream({"messages": [("human", "Hello")]}, config=config):
    for node_name, node_output in event.items():
        print(f"{node_name}: {node_output}")

# Stream only LLM tokens
for chunk in graph.stream(
    {"messages": [("human", "Hello")]},
    config=config,
    stream_mode="messages",
):
    if chunk[1].get("langgraph_node") == "agent":
        print(chunk[0].content, end="", flush=True)

# Get current state
state = graph.get_state(config)

# Update state externally
graph.update_state(config, {"context": "new context"}, as_node="agent")
```

---

## Human-in-the-Loop

```python
from langgraph.types import interrupt

def human_review_node(state: AgentState):
    # Pauses graph here; graph.invoke() returns control to caller
    decision = interrupt({
        "question": "Approve this action?",
        "action": state["messages"][-1].content,
    })
    return {"messages": [("human", f"Decision: {decision}")]}

builder.add_node("human_review", human_review_node)

# Resume after interrupt:
graph.invoke(Command(resume="approved"), config=config)
```

---

## Dynamic Routing with Command

```python
from langgraph.types import Command

def router_node(state: AgentState) -> Command:
    if "urgent" in state["messages"][-1].content:
        return Command(goto="escalate", update={"priority": "high"})
    return Command(goto="standard")
```

---

## Prebuilt ReAct Agent

For simple tool-using agents, skip building a graph manually:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
    state_modifier="You are a helpful assistant.",  # system prompt
)

result = agent.invoke(
    {"messages": [("human", "Search for X")]},
    config={"configurable": {"thread_id": "1"}},
)
```

---

## Multi-Agent: Supervisor Pattern

```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    agents=[researcher_agent, writer_agent],
    model=llm,
    prompt="Route tasks to the appropriate agent.",
)
graph = supervisor.compile(checkpointer=MemorySaver())
```

**Deprecated / Do Not Use:**
```python
# DEPRECATED — AgentExecutor does not support streaming, HITL, or persistence properly
from langchain.agents import AgentExecutor, initialize_agent, create_react_agent
agent = AgentExecutor(agent=..., tools=...)

# DEPRECATED — MessageGraph removed in langgraph 1.0
from langgraph.graph import MessageGraph
```

---

## Common Mistakes

- **State not updating**: Return a dict from nodes, not the full state. Only return changed keys.
- **Messages duplicating**: `add_messages` appends — don't wrap response in a list twice.
- **Checkpointer missing**: Without a checkpointer, `thread_id` has no effect and memory doesn't persist.
- **Interrupt not suspending**: `interrupt()` only works inside a node when graph is compiled with a checkpointer.
- **Streaming empty**: Use `stream_mode="messages"` for token streaming; default `stream_mode="values"` emits full state per step.
