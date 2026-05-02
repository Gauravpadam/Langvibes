---
name: scaffold-langchain-project
description: Use this skill when the user wants to scaffold, create, initialise, or set up a new LangChain or LangGraph project. Triggers on requests like "scaffold a new agent project", "create a langgraph project structure", "set up a new langchain system", or "initialise an agentic project". Do not trigger for questions about existing code — only for new project creation.
version: 1.0.0
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

Scaffold the following layout. Create all directories and `__init__.py` files. Adjust by project type as noted.

```
{project_name}/
├── src/
│   ├── state/              # State schema definitions — isolated so schema changes are one-place edits
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── read/           # Read-only tools — query, fetch, search (no side effects)
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
│   ├── prompts/            # Prompt templates — kept separate so non-engineers can edit them
│   │   ├── __init__.py
│   │   └── templates.py
│   ├── retrieval/          # RAG components — only for RAG agent and multi-agent types
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── embeddings.py
│   │   └── retrievers.py
│   └── config.py           # Model init, region, env vars — single source of truth
├── tests/
│   ├── unit/               # Test nodes and tools in isolation with mock state
│   │   ├── __init__.py
│   │   ├── test_tools.py
│   │   └── test_nodes.py
│   └── integration/        # Test full graph runs with MemorySaver
│       ├── __init__.py
│       └── test_graph.py
├── scripts/
│   └── ingest.py           # Document ingestion script — RAG type only
├── .env.example
├── main.py
└── requirements.txt
```

Omit `src/retrieval/` and `scripts/ingest.py` for agent-with-tools type.
Omit nothing for RAG agent or multi-agent types.

---

## Step 3 — Write File Contents

Write each file with correct boilerplate. Do not leave placeholders — write runnable code.

### `src/config.py`

```python
import os
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

def get_llm(temperature: float = 0, max_tokens: int = 4096) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=MODEL_ID,
        region_name=AWS_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
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
    messages: Annotated[list[BaseMessage], add_messages]  # accumulates via reducer
    iterations: int       # increment per cycle — used for explicit exit condition
    status: str           # "running" | "done" | "failed" — model termination in state

# Initialise state with defaults — never call the graph without these set
DEFAULT_STATE: AgentState = {
    "messages": [],
    "iterations": 0,
    "status": "running",
}
```

### `src/memory/checkpointers.py`

```python
import os
from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    """
    Returns MemorySaver for development.
    Swap to SqliteSaver or PostgresSaver for production without changing graph code.
    """
    env = os.getenv("APP_ENV", "development")

    if env == "production":
        # Uncomment and configure for production:
        # from langgraph.checkpoint.postgres import PostgresSaver
        # return PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
        pass

    return MemorySaver()
```

### `src/prompts/templates.py`

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

AGENT_SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant.\n\n"
        "Rules:\n"
        "- Use read tools to gather information before taking any action.\n"
        "- Use write tools only when the user has explicitly requested an action.\n"
        "- If uncertain, ask for clarification rather than guessing.",
    ),
    MessagesPlaceholder("messages"),
])
```

### `src/tools/read/search.py`

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Searches the knowledge base and returns relevant results for the given query string."""
    # Replace with real implementation
    return f"Results for: {query}"
```

### `src/tools/write/actions.py`

```python
from langchain_core.tools import tool, ToolException

@tool
def execute_action(action: str) -> str:
    """
    Executes the specified action. Returns a confirmation string on success.
    This tool has side effects — it is always placed behind interrupt() in the graph.
    """
    if not action:
        raise ToolException("action cannot be empty")
    # Replace with real implementation
    return f"Executed: {action}"
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
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
    }

def should_continue(state: AgentState) -> str:
    from langgraph.graph import END

    if state["iterations"] >= 10:
        return END

    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END

    # Route write tool calls through interrupt review
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
from src.nodes.agent import call_model, should_continue, ALL_TOOLS, WRITE_TOOLS
from src.memory.checkpointers import get_checkpointer

def review_node(state: AgentState) -> dict:
    """Interrupt before any write tool execution — human approves or rejects."""
    last = state["messages"][-1]
    decision = interrupt({
        "question": "Approve this action?",
        "tool_calls": last.tool_calls,
    })
    if decision != "approved":
        return {"messages": [("human", "Action cancelled.")], "status": "done"}
    return {}

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("review", review_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "review": "review", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("review", "tools")

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)

# Compile once at import time — do not recompile per request
graph = build_graph()
```

### `main.py`

```python
import uuid
from src.graphs.builder import graph
from src.state.schemas import DEFAULT_STATE

def run(user_input: str, session_id: str | None = None) -> str:
    thread_id = session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
        "run_name": "agent-run",
        "metadata": {"session_id": thread_id},
    }

    state = {**DEFAULT_STATE, "messages": [("human", user_input)]}

    result = graph.invoke(state, config=config)
    return result["messages"][-1].content

if __name__ == "__main__":
    print(run("Hello, what can you help me with?"))
```

### `requirements.txt`

Pinned exactly to the LTS versions the langvibes skills are built on. Do not change these pins without updating the skill files.

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

### `tests/unit/test_nodes.py`

```python
from langchain_core.messages import HumanMessage, AIMessage
from src.state.schemas import AgentState
from src.nodes.agent import should_continue

def make_state(messages, iterations=0, status="running") -> AgentState:
    return {"messages": messages, "iterations": iterations, "status": status}

def test_should_continue_no_tool_calls():
    state = make_state([HumanMessage(content="hi"), AIMessage(content="hello")])
    assert should_continue(state) == "__end__"

def test_should_continue_max_iterations():
    state = make_state([HumanMessage(content="hi")], iterations=10)
    assert should_continue(state) == "__end__"
```

### `tests/integration/test_graph.py`

```python
from src.graphs.builder import graph
from src.state.schemas import DEFAULT_STATE

def test_graph_basic_invoke():
    config = {"configurable": {"thread_id": "test-1"}, "recursion_limit": 5}
    state = {**DEFAULT_STATE, "messages": [("human", "Hello")]}
    result = graph.invoke(state, config=config)
    assert "messages" in result
    assert len(result["messages"]) > 0
```

---

## Step 4 — RAG Type Additions

If project type is **RAG agent**, also write:

### `src/retrieval/embeddings.py`

```python
from src.config import get_embeddings

embeddings = get_embeddings()
```

### `src/retrieval/retrievers.py`

```python
from langchain_community.vectorstores import FAISS
from src.retrieval.embeddings import embeddings

def load_retriever(index_path: str, k: int = 6):
    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20})
```

### `src/tools/read/retrieve.py`

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langchain_core.vectorstores import VectorStoreRetriever

@tool
def retrieve(
    query: str,
    retriever: Annotated[VectorStoreRetriever, InjectedToolArg],
) -> str:
    """Searches the knowledge base and returns relevant passages for the given query."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)
```

### `scripts/ingest.py`

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.retrieval.embeddings import embeddings

def ingest(docs_path: str = "./docs", index_path: str = "faiss_index"):
    loader = DirectoryLoader(docs_path, glob="**/*.md")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"Ingested {len(chunks)} chunks into {index_path}")

if __name__ == "__main__":
    ingest()
```

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

## Design Decisions Encoded in This Structure

| Decision | Why |
|---|---|
| `tools/read/` and `tools/write/` are separate directories | Enforces read-before-write at the filesystem level; write tools are always auditable |
| `state/schemas.py` is its own module | State schema changes are isolated; avoids circular imports |
| `memory/checkpointers.py` is its own module | Swapping dev→prod checkpointer is one file, zero graph changes |
| `iterations` and `status` in state | Explicit exit conditions — `recursion_limit` is a backstop only |
| Write tools route through `review` node | All irreversible actions are behind `interrupt()` by default |
| `graph = build_graph()` at module level | Compiled once at import, reused across requests |
| `DEFAULT_STATE` constant | Ensures state is always fully initialised before graph invocation |
