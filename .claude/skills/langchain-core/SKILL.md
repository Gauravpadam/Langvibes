---
name: langchain-core
description: Use this skill when working with LangChain chains, prompts, LCEL, output parsers, message history, or model integrations. Triggers when code imports from langchain_core, langchain_aws, langchain_anthropic, langchain_community, or uses the pipe operator to compose chains. Also triggers on mentions of "LCEL", "runnable", "prompt template", "output parser", or "message history".
version: 1.0.0
---

# LangChain Core — v1.2 Reference

Target versions: `langchain>=1.2`, `langchain-core>=1.2`, `langchain-aws>=1.4`, `langchain-community>=0.4`

> **Legacy note**: If working in an environment with `langchain-aws<1.0`, use `ChatBedrock` (see deprecated section). Prefer `ChatBedrockConverse` in all new environments.

---

## Model Initialization

### AWS Bedrock (primary — langchain-aws >= 1.0)

```python
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=4096,
)

# Cross-region inference (use us. prefix)
llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-east-1",
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
```

**Deprecated (langchain-aws 0.2.x):**
```python
# DEPRECATED — uses invoke_model API, does not support streaming/tool use properly
from langchain_aws import ChatBedrock
llm = ChatBedrock(model_id="...", model_kwargs={"temperature": 0})
# Migrate to: ChatBedrockConverse (no model_kwargs — pass args directly)
```

### Anthropic Direct

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
```

---

## Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context: {context}"),
    MessagesPlaceholder("history"),   # for conversation memory
    ("human", "{input}"),
])
```

**Deprecated:**
```python
# DEPRECATED
from langchain.prompts import PromptTemplate          # use langchain_core.prompts
from langchain.prompts import HumanMessagePromptTemplate  # use ChatPromptTemplate.from_messages
```

---

## LCEL Chains (LangChain Expression Language)

Always compose with the `|` pipe operator. Never use `LLMChain`.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Basic chain
chain = prompt | llm | StrOutputParser()

# Parallel inputs
from langchain_core.runnables import RunnableParallel
chain = RunnableParallel(context=retriever, question=RunnablePassthrough()) | prompt | llm | StrOutputParser()

# Branching
chain = prompt | llm | RunnableLambda(lambda x: x.content.upper())

# Streaming
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([{"input": "Q1"}, {"input": "Q2"}])

# Async
result = await chain.ainvoke({"input": "Hello"})
```

**Deprecated:**
```python
# DEPRECATED — do not use
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# DEPRECATED
from langchain.chains import SequentialChain, SimpleSequentialChain
```

---

## Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# Structured output (preferred over manual parsing)
class Answer(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(Answer)
result: Answer = structured_llm.invoke("What is 2+2?")

# JSON parser (when structured output not available)
parser = JsonOutputParser(pydantic_object=Answer)
chain = prompt | llm | parser
```

---

## Conversation Memory / Message History

Use `RunnableWithMessageHistory` for stateful chains. Use LangGraph checkpointers for agents (see langgraph-agents skill).

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

**Deprecated:**
```python
# DEPRECATED — removed in langchain 1.0
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
# Migrate to: RunnableWithMessageHistory (chains) or LangGraph MemorySaver (agents)
```

---

## Runnable Configuration & Callbacks

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs): ...
    def on_llm_end(self, response, **kwargs): ...

result = chain.invoke(
    {"input": "Hello"},
    config={"callbacks": [MyHandler()], "tags": ["prod"], "metadata": {"user": "abc"}},
)
```

---

## Import Path Reference (1.x)

| Component | Correct import |
|---|---|
| `ChatPromptTemplate` | `langchain_core.prompts` |
| `MessagesPlaceholder` | `langchain_core.prompts` |
| `StrOutputParser` | `langchain_core.output_parsers` |
| `RunnablePassthrough` | `langchain_core.runnables` |
| `RunnableParallel` | `langchain_core.runnables` |
| `RunnableLambda` | `langchain_core.runnables` |
| `RunnableWithMessageHistory` | `langchain_core.runnables.history` |
| `BaseCallbackHandler` | `langchain_core.callbacks` |
| `ChatBedrockConverse` | `langchain_aws` |
| `BedrockEmbeddings` | `langchain_aws` |
| `ChatAnthropic` | `langchain_anthropic` |
| `ChatMessageHistory` | `langchain_community.chat_message_histories` |

**Broken imports removed in 1.x (do not use):**
- `from langchain.chat_models import ...` → use provider packages directly
- `from langchain.llms import ...` → use provider packages directly
- `from langchain.embeddings import ...` → use provider packages directly
