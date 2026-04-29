---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# LangChain Tools — Definition, Binding, Execution

Target: `langchain>=1.2`, `langchain-core>=1.2`, `langgraph>=1.1`, `langchain-aws>=1.4`

---

## Tool Design Principles

- **One concern per tool** — narrow and composable over broad and monolithic
- **Read before write** — query state before mutating; separate read tools from write tools
- **Docstring = prompt engineering** — write it before the function body; describe what the tool *returns*, not just what it does
- **Idempotent where possible** — the LLM may call the same tool multiple times
- **`InjectedToolArg`** for any runtime dependency the LLM must not see

---

## Defining Tools

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Simple tool
@tool
def get_weather(city: str) -> str:
    """Returns current temperature and conditions for the given city name."""
    ...

# With explicit Pydantic schema
class SearchInput(BaseModel):
    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, description="Maximum number of results to return")

@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Returns a list of search results, each with 'title', 'url', and 'snippet' keys."""
    ...

# Async tool
@tool
async def fetch_data(url: str) -> str:
    """Returns the raw text content of the page at the given URL."""
    ...
```

**Never use the deprecated `Tool` class or `StructuredTool.from_function()`** — use `@tool`.

---

## Injected Arguments

Runtime dependencies the LLM must not control:

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

@tool
def query_db(
    sql: str,
    db: Annotated[object, InjectedToolArg],  # LLM never sees this argument
) -> list:
    """Returns rows from the database matching the SQL query."""
    return db.execute(sql).fetchall()

# Inject at runtime
tool_with_db = query_db.inject(db=my_connection)
```

---

## Binding Tools to Models

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
tools = [get_weather, search_web]

# Model chooses which tool to call
llm_with_tools = llm.bind_tools(tools)

# Force a specific tool (do not use in agentic loops — model can't stop calling)
llm_forced = llm.bind_tools(tools, tool_choice="get_weather")

response = llm_with_tools.invoke("What's the weather in Paris?")
print(response.tool_calls)
# [{"name": "get_weather", "args": {"city": "Paris"}, "id": "..."}]
```

---

## ToolNode — Automatic Execution in LangGraph

Always use `ToolNode` — never parse `tool_calls` manually.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

# Wire into graph
builder.add_node("tools", tool_node)
builder.add_edge("tools", "agent")
```

`ToolNode` handles: parsing `tool_calls` from `AIMessage`, executing each tool, returning `ToolMessage` results, and catching `ToolException`.

---

## Tool Error Handling

```python
from langchain_core.tools import ToolException

@tool
def risky_operation(param: str) -> str:
    """Performs the operation and returns a confirmation string."""
    if not param:
        raise ToolException("param cannot be empty — provide a non-empty string")
    return "success"
```

`ToolNode` catches `ToolException` and returns the error as a `ToolMessage`. The LLM then decides to retry or give up — do not suppress the error inside the tool.

---

## Structured Output vs Tools

When you only need structured data back from the LLM — not tool execution — use `with_structured_output`:

```python
from pydantic import BaseModel

class Extracted(BaseModel):
    name: str
    confidence: float

structured_llm = llm.with_structured_output(Extracted)
result: Extracted = structured_llm.invoke("Extract the name from: John is confident.")
```

Use `@tool` when the model needs to *act*. Use `with_structured_output` when you need *structured data*.

---

## Built-in Tools

```python
# Web search (requires TAVILY_API_KEY)
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=3)

# DuckDuckGo (no API key)
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
```
