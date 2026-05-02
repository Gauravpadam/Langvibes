---
name: langchain-mcp
description: Use this skill when connecting LangChain or LangGraph agents to MCP (Model Context Protocol) servers. Triggers when code imports from langchain_mcp_adapters, mentions MultiServerMCPClient, or when the user asks to connect an agent to an MCP server, use MCP tools, or integrate with filesystem, GitHub, database, or other MCP-compatible servers.
version: 1.0.0
---

# LangChain MCP Adapters — v0.2 Reference

Target: `langchain-mcp-adapters==0.2.2`, `langgraph>=1.1`, `langchain-aws>=1.4`

---

## Installation

```bash
pip install langchain-mcp-adapters==0.2.2
```

Node.js is required for `npx`-based MCP servers (filesystem, GitHub, etc.):
```bash
node --version  # must be >= 18
```

---

## Core Imports

```python
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
    load_mcp_tools,
    load_mcp_resources,
    load_mcp_prompt,
)
```

---

## MultiServerMCPClient

The primary interface. Always use as an **async context manager** — it starts MCP server processes on enter and shuts them down on exit.

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

async def main():
    async with MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            },
        }
    ) as client:
        tools = client.get_tools()          # sync — NOT awaited
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke(       # async — must await
            {"messages": [("human", "List files in /tmp")]}
        )
        print(result["messages"][-1].content)

asyncio.run(main())
```

**Critical rules:**
- `client.get_tools()` is **synchronous** — do not `await` it
- All agent/graph invocations inside the context must use `ainvoke` / `astream`
- Never invoke the graph outside the `async with` block — MCP servers won't be running

---

## Transport Types

### stdio — Local process (most common)

```python
"server_name": {
    "transport": "stdio",
    "command": "npx",                    # or "python", "uvx", etc.
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    "env": {"MY_VAR": "value"},          # optional — extra env vars
    "cwd": "/working/directory",         # optional — working directory
}
```

### sse — HTTP Server-Sent Events

```python
"server_name": {
    "transport": "sse",
    "url": "http://localhost:8000/sse",
    "headers": {"Authorization": "Bearer my-token"},   # optional
    "timeout": 30.0,                                   # optional — connect timeout (seconds)
    "sse_read_timeout": 300.0,                         # optional — read timeout (seconds)
}
```

### streamable_http — Streaming HTTP (newer standard)

```python
"server_name": {
    "transport": "streamable_http",
    "url": "http://localhost:8000/mcp",
    "headers": {"Authorization": "Bearer my-token"},
    "terminate_on_close": True,          # optional
}
```

### websocket

```python
"server_name": {
    "transport": "websocket",
    "url": "ws://localhost:8000/ws",
}
```

---

## Tool Naming and Conflicts

By default tools keep their original MCP names. Enable `tool_name_prefix=True` when connecting multiple servers that may have overlapping tool names:

```python
async with MultiServerMCPClient(connections, tool_name_prefix=True) as client:
    tools = client.get_tools()
    # Tools are now prefixed: filesystem__read_file, github__create_issue
```

Get tools from a specific server only:

```python
fs_tools = client.get_tools(server_name="filesystem")
gh_tools = client.get_tools(server_name="github")
```

---

## Integration with StateGraph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    status: str

async def run_agent():
    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state: State):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response], "iterations": state["iterations"] + 1}

        def should_continue(state: State) -> str:
            if state["iterations"] >= 10:
                return END
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(State)
        builder.add_node("agent", call_model)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")

        graph = builder.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "session-1"}, "recursion_limit": 25}
        result = await graph.ainvoke(
            {"messages": [("human", "Help me")], "iterations": 0, "status": "running"},
            config=config,
        )
        return result["messages"][-1].content
```

---

## MCP Resources and Prompts

Load documents and prompt templates directly from MCP servers using `load_mcp_resources` and `load_mcp_prompt`. These require a raw `ClientSession` — use `MultiServerMCPClient` sessions.

```python
from langchain_mcp_adapters.client import load_mcp_resources, load_mcp_prompt

# Resources — returned as langchain_core Blob objects (use as document context)
async with MultiServerMCPClient(connections) as client:
    # Access underlying session for a server
    session = client.sessions["my_server"]
    blobs = await load_mcp_resources(session, uris=["resource://my-doc"])

    # Prompt templates — returned as list[HumanMessage | AIMessage]
    messages = await load_mcp_prompt(session, "my_prompt", arguments={"key": "value"})
```

---

## Common MCP Server Configs

```python
connections = {
    # Local filesystem access
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    },

    # GitHub — requires GITHUB_TOKEN env var
    "github": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
    },

    # Brave Search — requires BRAVE_API_KEY env var
    "brave_search": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": os.environ["BRAVE_API_KEY"]},
    },

    # PostgreSQL
    "postgres": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", os.environ["DATABASE_URL"]],
    },

    # Web fetch
    "fetch": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    },

    # Custom Python MCP server
    "my_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["my_mcp_server.py"],
        "env": {"MY_KEY": os.environ["MY_KEY"]},
    },

    # Remote server over SSE
    "remote": {
        "transport": "sse",
        "url": "http://my-mcp-server.example.com/sse",
        "headers": {"Authorization": f"Bearer {os.environ['API_KEY']}"},
    },
}
```

---

## Writing a Custom MCP Server

```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def get_data(query: str) -> str:
    """Fetches data matching the query string from internal system."""
    return f"Data for: {query}"

if __name__ == "__main__":
    mcp.run()    # stdio transport by default
```

---

## Common Mistakes

- **`await client.get_tools()`** — `get_tools()` is synchronous; do not await it
- **Using `graph.invoke()` instead of `graph.ainvoke()`** — MCP tools are async; sync invocation will hang or error
- **Invoking outside the `async with` block** — MCP server processes have been shut down; tool calls will fail
- **Overlapping tool names across servers** — use `tool_name_prefix=True` to namespace tools
- **Not passing `env` for servers needing API keys** — keys must be passed explicitly in `env` dict or inherited from the process environment
- **`npx` not found** — requires Node.js >= 18 installed and on `PATH`
