---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# LangChain RAG — Loaders, Splitters, Embeddings, Retrievers

Target: `langchain>=1.2`, `langchain-core>=1.2`, `langchain-aws>=1.4`, `langchain-community>=0.4`

---

## Document Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    CSVLoader,
)

docs = PyPDFLoader("report.pdf").load()
docs = WebBaseLoader("https://example.com").load()
docs = DirectoryLoader("./docs", glob="**/*.md").load()
```

Each loader returns `list[Document]` — `.page_content` (str) and `.metadata` (dict).

**Never use `from langchain.document_loaders`** — broken in 1.x; always use `langchain_community.document_loaders`.

---

## Text Splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,           # 10-20% of chunk_size
    separators=["\n\n", "\n", ".", " ", ""],
)
chunks = splitter.split_documents(docs)

# Code-aware splitting
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=100,
)
```

**Never use `from langchain.text_splitter`** — moved to `langchain_text_splitters` package.

---

## Embeddings

```python
# AWS Bedrock (primary)
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",   # 8192 token limit
    region_name="us-east-1",
)

# Cohere on Bedrock (512 token limit — adjust chunk_size accordingly)
embeddings = BedrockEmbeddings(
    model_id="cohere.embed-english-v3",
    region_name="us-east-1",
)
```

**Never use `from langchain.embeddings`** — broken in 1.x; use `langchain_aws` or `langchain_openai`.

Match `chunk_size` to the embedding model's token limit — Titan v2 supports 8192, Cohere v3 supports 512.

---

## Vector Stores

```python
# FAISS (local, no server)
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local(
    "faiss_index", embeddings,
    allow_dangerous_deserialization=True,   # required in 1.x — security acknowledgment
)

# Chroma (local, persistent)
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    chunks, embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs",
)
vectorstore.add_documents(new_chunks)
```

---

## Retrievers

```python
# Basic similarity retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},      # default k=4 is often too few; 6-10 is safer
)

# MMR — reduces redundancy in results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)

# Multi-query — generates query variants to improve recall
from langchain.retrievers import MultiQueryRetriever
from langchain_aws import ChatBedrockConverse

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0"),
)
```

---

## RAG Chain — Single-Turn (LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LangGraph?")
```

**Never use `RetrievalQA` or `ConversationalRetrievalChain`** — removed in langchain 1.x.

---

## Conversational RAG — Multi-Turn (LangGraph)

For multi-turn RAG, make the retriever a `@tool` and wire it into a LangGraph agent. Never use `ConversationalRetrievalChain`.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def retrieve(query: str) -> str:
    """Searches the knowledge base and returns relevant passages for the given query."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

agent = create_react_agent(
    model=llm,
    tools=[retrieve],
    checkpointer=MemorySaver(),
    state_modifier="Use the retrieve tool whenever you need information from the knowledge base.",
)

result = agent.invoke(
    {"messages": [("human", "What did the report say about Q3?")]},
    config={"configurable": {"thread_id": "user-session-1"}},
)
```

---

## Common Mistakes

- **`allow_dangerous_deserialization` missing** — FAISS `load_local` requires this flag in 1.x
- **Chunk size exceeds embedding model limit** — Titan v2 = 8192 tokens, Cohere v3 = 512 tokens
- **No chunk overlap** — missing overlap loses context at boundaries; use 10-20% of chunk size
- **Default k=4 too low** — increase to 6-10 for complex queries; use MMR to reduce redundancy
- **Using deprecated RAG chains** — `RetrievalQA` and `ConversationalRetrievalChain` are removed; use LCEL or LangGraph
