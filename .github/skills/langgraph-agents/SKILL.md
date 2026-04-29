---
name: langgraph-agents
description: Use when building LangGraph agents, state machines, or graph-based workflows. Covers StateGraph construction, TypedDict state schemas, checkpointers, human-in-the-loop with interrupt(), streaming, Command routing, and the prebuilt ReAct agent. Use when code imports from langgraph, uses StateGraph, or involves agents, nodes, edges, checkpointers, or multi-step reasoning.
argument-hint: "[task e.g. 'agent with tool calling and memory']"
user-invocable: true
disable-model-invocation: false
context: inline
---

# LangGraph Agents — v1.1 Reference

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

Never use `AgentExecutor`, `initialize_agent`, or `MessageGraph`.

---

## State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # accumulates
    iterations: int       # overwritten — used for exit condition
    status: str           # "running" | "done" | "failed"
```

Rules:
- Only data crossing node boundaries belongs in state
- Model termination explicitly in state — never rely on `recursion_limit` alone
- Adding fields is safe; renaming or removing breaks existing checkpoints
- Return only changed keys from nodes, not full state

---

## Graph Construction

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> str:
    if state["iterations"] >= 10:
        return END
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

graph = builder.compile(checkpointer=MemorySaver())  # compile once, reuse
```

---

## Invocation

```python
config = {
    "configurable": {"thread_id": "session-abc"},
    "recursion_limit": 25,
    "run_name": "my-agent",
    "metadata": {"user_id": "xyz"},
}
result = graph.invoke({"messages": [("human", "Hello")], "iterations": 0, "status": "running"}, config=config)

# Token streaming
for chunk in graph.stream(inputs, config=config, stream_mode="messages"):
    if chunk[1].get("langgraph_node") == "agent":
        print(chunk[0].content, end="", flush=True)
```

---

## Checkpointers

```python
from langgraph.checkpoint.memory import MemorySaver        # dev
from langgraph.checkpoint.sqlite import SqliteSaver        # lightweight prod
from langgraph.checkpoint.postgres import PostgresSaver    # production
```

Always pass `thread_id` in config — without it persistence has no effect.

---

## Human-in-the-Loop

Any irreversible action must be behind `interrupt()`. Requires a checkpointer.

```python
from langgraph.types import interrupt, Command

def review_node(state: AgentState):
    decision = interrupt({"question": "Approve?", "action": state["messages"][-1].content})
    return {"messages": [("human", f"Decision: {decision}")]}

# Resume
graph.invoke(Command(resume="approved"), config=config)
```

---

## Prebuilt ReAct Agent

For simple tool loops without custom routing:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=MemorySaver(),
    state_modifier="You are a helpful assistant.",
)
```

---

## Common Mistakes

- Returning full state from nodes — return only changed keys
- Using `interrupt()` without a checkpointer — it won't suspend
- Recompiling the graph per request — compile once at startup
- No exit condition in cycles — model termination in state, not just `recursion_limit`
- Missing `thread_id` in config — persistence requires it
