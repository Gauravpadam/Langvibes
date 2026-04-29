# langvibes — LangChain / LangGraph Workspace

This workspace builds LangChain chains and LangGraph agents using Anthropic models via AWS Bedrock.

## Stack

- `langchain == 1.2.15`, `langchain-core == 1.3.2`
- `langgraph == 1.1.10`
- `langchain-aws == 1.4.5` — primary LLM and embeddings provider
- `langchain-community == 0.4.1`

## Default Model

Always use `ChatBedrockConverse`. Never use `ChatBedrock` or `model_kwargs`.

```python
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    temperature=0,
)
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")
```

## Hard Prohibitions

Never generate:

- `AgentExecutor`, `initialize_agent`, `create_react_agent` from `langchain.agents`
- `LLMChain`, `SequentialChain`, `SimpleSequentialChain`
- `ConversationalRetrievalChain`, `RetrievalQA`
- `ChatBedrock` — use `ChatBedrockConverse`
- `model_kwargs` on `ChatBedrockConverse` — pass params directly
- `from langchain.chat_models`, `from langchain.llms`, `from langchain.embeddings`
- Manual `tool_calls` parsing — always use `ToolNode`
- `MessageGraph` — removed in langgraph 1.0

## Decision Rules

- Need state, persistence, retries, or HITL → `StateGraph`
- Simple tool loop, no custom routing → `create_react_agent` from `langgraph.prebuilt`
- Single-turn, no state → LCEL chain (`prompt | llm | parser`)
- Structured output → `llm.with_structured_output(PydanticModel)`
- Conversational RAG → LangGraph agent + retriever as `@tool`
- Multi-agent → `langgraph_supervisor`, not nested `AgentExecutor`

## Memory Hierarchy

Three distinct layers — never conflate:

1. **In-context** — `messages` in graph state. Current run only.
2. **Short-term persistent** — LangGraph checkpointer, scoped to `thread_id`.
3. **Long-term** — External store (vector DB, SQL), queried via retriever tool.

Graph state is a scratchpad. Long-term facts go in external stores.

## Agentic Design

### State
- Only data crossing node boundaries goes in `State` — local vars stay local
- `add_messages` accumulates; plain types overwrite
- Model termination in state (`iterations`, `status`) — `recursion_limit` is a backstop only
- Adding state fields is safe; renaming or removing breaks existing checkpoints

### Graph
- One responsibility per node — routing in conditional edges, not inside nodes
- Every cycle has an explicit exit condition in state
- Name nodes after what they do: `call_model`, `validate_input`
- Compile once, reuse the compiled graph instance

### Reliability
- Irreversible actions (writes, sends, deletes) → always behind `interrupt()`
- Retry logic at graph level via conditional edge, not inside node functions
- Raise `ToolException` for tool errors — let the LLM retry

### Tools
- One concern per tool, narrow and composable
- Read tools before write tools
- Docstring describes what the tool *returns* — write it before the function body
- `InjectedToolArg` for runtime deps the LLM must not see

### Observability
- `LANGCHAIN_TRACING_V2=true` from day one
- Add `run_name` and `metadata` to every `graph.invoke()` / `graph.stream()`

## Skills Available

Deeper reference skills auto-trigger based on imports and context:

| Trigger | Skill |
|---|---|
| `from langchain_core`, `from langchain_aws`, LCEL, prompts | `langchain-core` |
| `from langgraph`, `StateGraph`, graph/agent/checkpoint | `langgraph-agents` |
| `@tool`, `ToolNode`, `bind_tools` | `langchain-tools` |
| Document loaders, vector stores, RAG, embeddings | `langchain-rag` |
