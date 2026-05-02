---
name: langchain-mcp
description: Use when connecting LangChain or LangGraph agents to MCP (Model Context Protocol) servers. Triggers when code imports from langchain_mcp_adapters, mentions MultiServerMCPClient, or when the user asks to connect an agent to an MCP server, use MCP tools, or integrate with filesystem, GitHub, database, or other MCP-compatible servers.
argument-hint: "[server name or type e.g. 'filesystem' 'github' 'custom']"
user-invocable: true
disable-model-invocation: false
context: inline
---

# LangChain MCP Adapters — v0.2 Reference

Target: `langchain-mcp-adapters==0.2.2`, `langgraph>=1.1`, `langchain-aws>=1.4`

---

## Installation

```bash
pip install langchain-mcp-adapters==0.2.2
# Node.js >= 18 required for npx-based servers
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

Always use as an **async context manager**. Starts MCP server processes on enter, shuts them down on exit.

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
        tools = client.get_tools()          # sync — do NOT await
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke(       # async — must await
            {"messages": [("human", "List files in /tmp")]}
        )
        print(result["messages"][-1].content)

asyncio.run(main())
```

**Critical rules:**
- `client.get_tools()` is **synchronous** — never `await` it
- All agent/graph invocations must use `ainvoke` / `astream`
- Never invoke the graph outside the `async with` block

---

## Transport Types

### stdio — Local process
```python
"server": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
    "env": {"MY_VAR": "value"},    # optional
    "cwd": "/working/dir",         # optional
}
```

### sse — HTTP Server-Sent Events
```python
"server": {
    "transport": "sse",
    "url": "http://localhost:8000/sse",
    "headers": {"Authorization": "Bearer token"},
    "timeout": 30.0,
    "sse_read_timeout": 300.0,
}
```

### streamable_http
```python
"server": {
    "transport": "streamable_http",
    "url": "http://localhost:8000/mcp",
    "headers": {"Authorization": "Bearer token"},
    "terminate_on_close": True,
}
```

### websocket
```python
"server": {
    "transport": "websocket",
    "url": "ws://localhost:8000/ws",
}
```

---

## Tool Naming

```python
# Prefix tool names with server name to avoid conflicts across servers
async with MultiServerMCPClient(connections, tool_name_prefix=True) as client:
    tools = client.get_tools()
    # filesystem__read_file, github__create_issue, etc.

# Tools from one server only
fs_tools = client.get_tools(server_name="filesystem")
```

---

## StateGraph Integration

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

async def run():
    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        llm_with_tools = llm.bind_tools(tools)

        builder = StateGraph(State)
        builder.add_node("agent", lambda s: {"messages": [llm_with_tools.invoke(s["messages"])], "iterations": s["iterations"] + 1})
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")

        graph = builder.compile(checkpointer=MemorySaver())
        return await graph.ainvoke(state, config=config)
```

---

## Common Server Configs

```python
connections = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
    },
    "github": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
    },
    "brave_search": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": os.environ["BRAVE_API_KEY"]},
    },
    "postgres": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", os.environ["DATABASE_URL"]],
    },
    "fetch": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    },
    "remote": {
        "transport": "sse",
        "url": "http://my-mcp-server.example.com/sse",
        "headers": {"Authorization": f"Bearer {os.environ['API_KEY']}"},
    },
}
```

---

## Custom MCP Server

```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def get_data(query: str) -> str:
    """Returns data matching the query string from the internal system."""
    return f"Data for: {query}"

if __name__ == "__main__":
    mcp.run()
```

```python
# Connect to it
"my_server": {
    "transport": "stdio",
    "command": "python",
    "args": ["my_mcp_server.py"],
}
```

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| `await client.get_tools()` | `get_tools()` is sync — remove `await` |
| `graph.invoke()` with MCP tools | Use `graph.ainvoke()` — MCP tools are async |
| Invoking graph outside `async with` | MCP servers shut down on context exit |
| Overlapping tool names | Set `tool_name_prefix=True` |
| Missing `env` for API-key servers | Pass keys explicitly in `env` dict |
| `npx` not found | Install Node.js >= 18 |
