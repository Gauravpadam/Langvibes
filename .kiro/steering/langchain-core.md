---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# LangChain Core — Models, Prompts, LCEL

Target: `langchain>=1.2`, `langchain-core>=1.2`, `langchain-aws>=1.4`, `langchain-community>=0.4`

---

## Default Model — ChatBedrockConverse

```python
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=4096,
)

# Cross-region inference
llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-east-1",
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
```

**Never use `ChatBedrock` or `model_kwargs`** — pass all params directly to `ChatBedrockConverse`.

Legacy (langchain-aws < 1.0): `ChatBedrock(model_id=..., model_kwargs={"temperature": 0})` → migrate to `ChatBedrockConverse`.

---

## Prompts

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context: {context}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
```

Never use `from langchain.prompts` — import from `langchain_core.prompts`.

---

## LCEL — Chain Composition

Use `|` pipe. Never use `LLMChain`, `SequentialChain`, or `SimpleSequentialChain`.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# Basic
chain = prompt | llm | StrOutputParser()

# Parallel inputs
chain = RunnableParallel(context=retriever, question=RunnablePassthrough()) | prompt | llm | StrOutputParser()

# Streaming
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="", flush=True)

# Async
result = await chain.ainvoke({"input": "Hello"})
```

---

## Structured Output

Prefer `with_structured_output` over manual output parsing:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(Answer)
result: Answer = structured_llm.invoke("What is 2+2?")
```

---

## Conversation Memory

Use `RunnableWithMessageHistory` for stateful LCEL chains. Use LangGraph checkpointers for agents.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

result = chain_with_history.invoke(
    {"input": "Hello"},
    config={"configurable": {"session_id": "user-123"}},
)
```

Never use `ConversationBufferMemory` or `ConversationSummaryMemory` — removed in langchain 1.x.

---

## Correct Import Paths

| Component | Import |
|---|---|
| `ChatPromptTemplate`, `MessagesPlaceholder` | `langchain_core.prompts` |
| `StrOutputParser`, `JsonOutputParser` | `langchain_core.output_parsers` |
| `RunnablePassthrough`, `RunnableParallel` | `langchain_core.runnables` |
| `RunnableWithMessageHistory` | `langchain_core.runnables.history` |
| `ChatBedrockConverse`, `BedrockEmbeddings` | `langchain_aws` |
| `ChatAnthropic` | `langchain_anthropic` |
| `ChatMessageHistory` | `langchain_community.chat_message_histories` |

**Broken in 1.x — never use:**
`from langchain.chat_models`, `from langchain.llms`, `from langchain.embeddings`
