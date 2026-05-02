---
name: scaffold-langchain-project
description: Use when the user wants to scaffold, create, initialise, or set up a new LangChain or LangGraph project. Triggers on requests like "scaffold a new agent project", "create a langgraph project structure", "set up a new langchain system", or "initialise an agentic project". Do not trigger for questions about existing code — only for new project creation.
argument-hint: "[project-name] [type: agent|rag|multi-agent]"
user-invocable: true
disable-model-invocation: false
context: inline
---

# Scaffold LangChain / LangGraph Project

Create a complete, production-ready project structure that enforces good agentic system design from the first file. Every structural decision below is intentional — do not flatten or simplify the layout.

---

## Step 0 — Virtual Environment

Ask the user which virtual environment tool they prefer before creating any files. Create and activate the venv first, then scaffold into the project directory.

### venv (Python standard library — no extra install)
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### uv (recommended — fast, modern)
```bash
pip install uv                   # if not already installed
uv venv .venv --python 3.11
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### conda (Anaconda / Miniconda)
```bash
conda create -n {project_name} python=3.11 -y
conda activate {project_name}
```

### poetry (dependency management + venv)
```bash
pip install poetry               # if not already installed
poetry new {project_name}
cd {project_name}
poetry shell
```

Default to **venv** if the user does not specify.

---

## Step 1 — Determine Project Type

Ask the user which type to scaffold if not already specified:

1. **Agent with tools** — LangGraph agent using `@tool` definitions and `ToolNode`
2. **RAG agent** — LangGraph agent with retrieval pipeline (loaders, embeddings, vector store)
3. **Multi-agent** — Supervisor pattern with specialised sub-agents

Default to **agent with tools** if unclear.

---

## Step 2 — Create Directory Structure

```
{project_name}/
├── src/
│   ├── state/              # State schema definitions — isolated so schema changes are one-place edits
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── read/           # Read-only tools — no side effects
│   │   │   ├── __init__.py
│   │   │   └── search.py
│   │   └── write/          # Mutating tools — always behind interrupt() in the graph
│   │       ├── __init__.py
│   │       └── actions.py
│   ├── graphs/             # Graph construction and compilation — compile once, reuse
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── nodes/              # Individual node functions — one responsibility each
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── memory/             # Checkpointer setup — swap dev↔prod here without touching graph code
│   │   ├── __init__.py
│   │   └── checkpointers.py
│   ├── prompts/            # Prompt templates
│   │   ├── __init__.py
│   │   └── templates.py
│   ├── retrieval/          # RAG components — RAG agent and multi-agent types only
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── embeddings.py
│   │   └── retrievers.py
│   └── config.py           # Model init, region, env vars — single source of truth
├── tests/
│   ├── unit/               # Test nodes and tools with mock state
│   │   ├── __init__.py
│   │   ├── test_tools.py
│   │   └── test_nodes.py
│   └── integration/        # Test full graph runs with MemorySaver
│       ├── __init__.py
│       └── test_graph.py
├── scripts/
│   └── ingest.py           # Document ingestion — RAG type only
├── .env.example
├── main.py
└── requirements.txt
```

---

## Step 3 — Write File Contents

### `src/config.py`
```python
import os
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

def get_llm(temperature: float = 0, max_tokens: int = 4096) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=MODEL_ID, region_name=AWS_REGION,
        temperature=temperature, max_tokens=max_tokens,
    )

def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=AWS_REGION)
```

### `src/state/schemas.py`
```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int    # explicit exit condition — increment per cycle
    status: str        # "running" | "done" | "failed"

DEFAULT_STATE: AgentState = {"messages": [], "iterations": 0, "status": "running"}
```

### `src/memory/checkpointers.py`
```python
import os
from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    env = os.getenv("APP_ENV", "development")
    if env == "production":
        # from langgraph.checkpoint.postgres import PostgresSaver
        # return PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
        pass
    return MemorySaver()
```

### `src/nodes/agent.py`
```python
from src.state.schemas import AgentState
from src.config import get_llm
from src.tools.read.search import search
from src.tools.write.actions import execute_action

READ_TOOLS = [search]
WRITE_TOOLS = [execute_action]
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS

llm = get_llm()
llm_with_tools = llm.bind_tools(ALL_TOOLS)

def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> str:
    from langgraph.graph import END
    if state["iterations"] >= 10:
        return END
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    for call in last.tool_calls:
        if call["name"] in [t.name for t in WRITE_TOOLS]:
            return "review"
    return "tools"
```

### `src/graphs/builder.py`
```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from src.state.schemas import AgentState
from src.nodes.agent import call_model, should_continue, ALL_TOOLS
from src.memory.checkpointers import get_checkpointer

def review_node(state: AgentState) -> dict:
    decision = interrupt({"question": "Approve?", "tool_calls": state["messages"][-1].tool_calls})
    if decision != "approved":
        return {"messages": [("human", "Action cancelled.")], "status": "done"}
    return {}

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("review", review_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "review": "review", END: END})
    builder.add_edge("tools", "agent")
    builder.add_edge("review", "tools")
    return builder.compile(checkpointer=get_checkpointer())

graph = build_graph()  # compile once at import time
```

### `main.py`
```python
import uuid
from src.graphs.builder import graph
from src.state.schemas import DEFAULT_STATE

def run(user_input: str, session_id: str | None = None) -> str:
    thread_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25,
              "run_name": "agent-run", "metadata": {"session_id": thread_id}}
    result = graph.invoke({**DEFAULT_STATE, "messages": [("human", user_input)]}, config=config)
    return result["messages"][-1].content

if __name__ == "__main__":
    print(run("Hello, what can you help me with?"))
```

### `requirements.txt`

Pinned exactly to the LTS versions the langvibes skills are built on.

```
# Core — LTS versions (langvibes v1.0)
langchain==1.2.15
langchain-core==1.3.2
langchain-text-splitters==1.1.2

# LangGraph
langgraph==1.1.10

# Model providers
langchain-aws==1.4.5
langchain-anthropic==1.4.2

# Community integrations
langchain-community==0.4.1

# AWS SDK
boto3==1.42.97

# Vector stores — uncomment as needed
# langchain-chroma==1.1.0
# faiss-cpu==1.13.2

# Production checkpointers — uncomment as needed
# langgraph-checkpoint-sqlite
# langgraph-checkpoint-postgres

# Observability — uncomment to enable LangSmith tracing
# langsmith
```

### `.env.example`
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret-key
MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
APP_ENV=development

# LangSmith tracing — requires langsmith in requirements.txt
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=your-project-name
```

---

## Step 4 — RAG Type Additions

For RAG agent type, additionally create `src/retrieval/retrievers.py`, `src/tools/read/retrieve.py` (using `InjectedToolArg` for the retriever), and `scripts/ingest.py` using `DirectoryLoader` + `RecursiveCharacterTextSplitter` + `FAISS`.

---

## Step 6 — After Scaffolding

Tell the user the exact commands to run, matching whichever venv tool they chose in Step 0:

```bash
# 1. Activate the venv (if not already active)
source .venv/bin/activate          # venv / uv — macOS / Linux
# conda activate {project_name}    # conda
# poetry shell                     # poetry

# 2. Install dependencies
pip install -r requirements.txt    # venv / uv / conda
# poetry install                   # poetry

# 3. Set up environment
cp .env.example .env
# Fill in AWS credentials in .env

# 4. RAG type only — ingest documents
# mkdir docs && cp your-files docs/
# python scripts/ingest.py

# 5. Verify the agent runs
python main.py
```

Before going to production: swap `MemorySaver` → `PostgresSaver` in `src/memory/checkpointers.py` and uncomment `langgraph-checkpoint-postgres` in `requirements.txt`.

---

## Why This Structure

| Decision | Reason |
|---|---|
| `tools/read/` and `tools/write/` split | Write tools are always auditable; enforces read-before-write |
| `state/schemas.py` isolated | Schema changes in one place; avoids circular imports |
| `memory/checkpointers.py` isolated | Dev→prod swap is one file, zero graph changes |
| `iterations` + `status` in state | Explicit exit conditions — `recursion_limit` is backstop only |
| Write tools route through `review` node | All irreversible actions behind `interrupt()` by default |
| `graph = build_graph()` at module level | Compiled once at import, reused across requests |
| `DEFAULT_STATE` constant | State always fully initialised before graph invocation |
