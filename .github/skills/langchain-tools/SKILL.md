---
name: langchain-tools
description: Use when defining tools, binding tools to a model, or wiring tool execution in LangChain or LangGraph. Covers the @tool decorator, InjectedToolArg, bind_tools, tool_choice, ToolNode, ToolException, and built-in tools. Use when code uses @tool, StructuredTool, ToolNode, bind_tools, or involves tool calling, function calling, or tool use.
argument-hint: "[task e.g. 'define a search tool and wire it into a graph']"
user-invocable: true
disable-model-invocation: false
context: inline
---

# LangChain Tools — v1.2 Reference

Target: `langchain>=1.2`, `langchain-core>=1.2`, `langgraph>=1.1`, `langchain-aws>=1.4`

---

## Tool Design Principles

- One concern per tool — narrow and composable
- Read before write — query state before mutating; separate read and write tools
- Docstring = prompt engineering — describe what the tool *returns*; write it before the function body
- `InjectedToolArg` for runtime deps the LLM must not see
- Raise `ToolException` for errors — let the LLM retry

Never use the deprecated `Tool` class or `StructuredTool.from_function()` — use `@tool`.

---

## Defining Tools

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

@tool
def get_weather(city: str) -> str:
    """Returns current temperature and conditions for the given city."""
    ...

class SearchInput(BaseModel):
    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, description="Maximum results to return")

@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Returns a list of results each with 'title', 'url', and 'snippet'."""
    ...

@tool
async def fetch_data(url: str) -> str:
    """Returns the raw text content of the page at the given URL."""
    ...
```

---

## Injected Arguments

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

@tool
def query_db(
    sql: str,
    db: Annotated[object, InjectedToolArg],  # LLM never sees this
) -> list:
    """Returns rows matching the SQL query."""
    return db.execute(sql).fetchall()

tool_with_db = query_db.inject(db=my_connection)
```

---

## Binding to a Model

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
tools = [get_weather, search_web]

llm_with_tools = llm.bind_tools(tools)
llm_forced = llm.bind_tools(tools, tool_choice="get_weather")  # force specific tool

response = llm_with_tools.invoke("Weather in Paris?")
print(response.tool_calls)
```

Do not use `tool_choice` in agentic loops — the model can't choose to stop.

---

## ToolNode — LangGraph Execution

Never parse `tool_calls` manually — always use `ToolNode`.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
builder.add_node("tools", tool_node)
builder.add_edge("tools", "agent")
```

`ToolNode` handles parsing, execution, `ToolMessage` return, and `ToolException` catching.

---

## Error Handling

```python
from langchain_core.tools import ToolException

@tool
def risky_op(param: str) -> str:
    """Returns a confirmation string on success."""
    if not param:
        raise ToolException("param cannot be empty — provide a non-empty string")
    return "success"
```

---

## Structured Output vs Tools

- `@tool` — when the model needs to *act*
- `llm.with_structured_output(Model)` — when you need *structured data* back

```python
class Extracted(BaseModel):
    name: str
    score: float

result: Extracted = llm.with_structured_output(Extracted).invoke("Extract from: ...")
```

---

## Built-in Tools

```python
from langchain_community.tools.tavily_search import TavilySearchResults  # needs TAVILY_API_KEY
from langchain_community.tools import DuckDuckGoSearchRun                 # no key needed
