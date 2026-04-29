---
name: langchain-tools
description: Use this skill when defining tools, binding tools to models, or wiring tool execution in LangChain or LangGraph. Triggers when code uses @tool decorator, StructuredTool, ToolNode, bind_tools, tool_choice, or InjectedToolArg. Also triggers on mentions of "tool calling", "function calling", "tool use", or "tool node".
version: 1.0.0
---

# LangChain Tools — v1.2 Reference

Target versions: `langchain>=1.2`, `langchain-core>=1.2`, `langgraph>=1.1`, `langchain-aws>=1.4`

---

## Defining Tools

### Simple tool with `@tool`

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Args must be documented here."""
    return f"Weather in {city}: sunny, 22°C"

# Access metadata
print(get_weather.name)        # "get_weather"
print(get_weather.description) # docstring
print(get_weather.args_schema) # auto-generated Pydantic schema
```

### Structured tool with Pydantic schema

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")

@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return results."""
    ...
```

### Async tools

```python
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

### Injected arguments (not exposed to LLM)

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

@tool
def query_database(
    sql: str,
    db_conn: Annotated[object, InjectedToolArg],  # LLM never sees this
) -> list:
    """Execute a SQL query and return results."""
    return db_conn.execute(sql).fetchall()

# Pass injected args at runtime
tool_with_conn = query_database.inject(db_conn=my_connection)
```

**Deprecated:**
```python
# DEPRECATED — verbose, no type safety
from langchain.tools import Tool
tool = Tool(name="search", func=search_fn, description="Search the web")

# DEPRECATED
from langchain.tools import StructuredTool
tool = StructuredTool.from_function(func=..., name=..., description=...)
# Migrate to: @tool decorator
```

---

## Binding Tools to Models

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
tools = [get_weather, search_web]

# Bind — model can choose to call any tool
llm_with_tools = llm.bind_tools(tools)

# Force a specific tool
llm_forced = llm.bind_tools(tools, tool_choice="get_weather")

# Force any tool (not no-tool)
llm_any = llm.bind_tools(tools, tool_choice="any")

response = llm_with_tools.invoke("What's the weather in Paris?")
print(response.tool_calls)  # [{"name": "get_weather", "args": {"city": "Paris"}, "id": "..."}]
```

---

## Structured Output (preferred over manual tool parsing)

When you just need structured data back, use `with_structured_output` instead of tools:

```python
from pydantic import BaseModel

class ExtractedData(BaseModel):
    name: str
    age: int
    email: str

structured_llm = llm.with_structured_output(ExtractedData)
result: ExtractedData = structured_llm.invoke("John Doe is 30, email: john@example.com")
```

---

## ToolNode (LangGraph — executes tool calls automatically)

```python
from langgraph.prebuilt import ToolNode

tools = [get_weather, search_web]
tool_node = ToolNode(tools)

# Wire into graph
builder.add_node("tools", tool_node)
builder.add_edge("tools", "agent")  # always return to agent after tool execution
```

**ToolNode handles:**
- Parsing `tool_calls` from `AIMessage`
- Executing each tool
- Returning `ToolMessage` results
- Catching `ToolException` and returning error messages

### Tool error handling

```python
from langchain_core.tools import ToolException

@tool
def risky_operation(param: str) -> str:
    """Do something risky."""
    if not param:
        raise ToolException("param cannot be empty")
    return "success"

# ToolNode catches ToolException and returns it as a ToolMessage to the LLM
# LLM can then decide to retry or give up
```

---

## LCEL Tool Execution (without LangGraph)

```python
from langchain_core.runnables import RunnableLambda

def execute_tools(message):
    tool_map = {t.name: t for t in tools}
    results = []
    for call in message.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
        results.append(result)
    return results

chain = llm_with_tools | RunnableLambda(execute_tools)
```

---

## Built-in Tools

```python
# Web search (requires TAVILY_API_KEY)
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=3)

# Python REPL (sandbox carefully)
from langchain_experimental.tools import PythonREPLTool
repl = PythonREPLTool()

# DuckDuckGo (no API key)
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
```

---

## Common Mistakes

- **Docstring missing**: `@tool` uses the docstring as the tool description sent to the LLM — always write one.
- **Wrong return type**: Return strings or JSON-serializable objects. LLMs receive tool output as text.
- **tool_choice with bind_tools**: `tool_choice` forces usage — don't use in agentic loops or the model can't stop calling tools.
- **Forgetting `InjectedToolArg`**: Any runtime dependency (DB connection, user context) that the LLM shouldn't see must be annotated as injected.
- **Manual tool parsing**: Don't parse `response.tool_calls` manually in LangGraph — use `ToolNode`.
