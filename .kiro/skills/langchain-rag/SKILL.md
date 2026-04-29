---
name: langchain-rag
description: Use when building RAG pipelines, working with vector stores, document loaders, text splitters, embeddings, or retrievers. Covers PyPDFLoader, RecursiveCharacterTextSplitter, BedrockEmbeddings, FAISS, Chroma, LCEL RAG chain, and conversational RAG via LangGraph. Use when code involves RAG, retrieval, vector store, embeddings, document loader, or chunking.
metadata:
  version: 1.0.0
  target: "langchain>=1.2, langchain-core>=1.2, langchain-aws>=1.4, langchain-community>=0.4"
---

# LangChain RAG — v1.2 Reference

Target: `langchain>=1.2`, `langchain-core>=1.2`, `langchain-aws>=1.4`, `langchain-community>=0.4`

---

## Document Loaders

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader, TextLoader

docs = PyPDFLoader("report.pdf").load()
docs = WebBaseLoader("https://example.com").load()
docs = DirectoryLoader("./docs", glob="**/*.md").load()
```

Never use `from langchain.document_loaders` — broken in 1.x.

---

## Text Splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=100
)
```

Never use `from langchain.text_splitter` — moved to `langchain_text_splitters`.

---

## Embeddings

```python
from langchain_aws import BedrockEmbeddings

# Titan v2 — 8192 token limit
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")

# Cohere — 512 token limit (set chunk_size accordingly)
embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3", region_name="us-east-1")
```

Never use `from langchain.embeddings` — broken in 1.x.

---

## Vector Stores

```python
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectorstore.add_documents(new_chunks)
```

---

## Retrievers

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

from langchain.retrievers import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
```

Default `k=4` is often too few — use 6-10 for complex queries.

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
    | prompt | llm | StrOutputParser()
)
```

Never use `RetrievalQA` or `ConversationalRetrievalChain` — removed in 1.x.

---

## Conversational RAG — Multi-Turn (LangGraph)

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def retrieve(query: str) -> str:
    """Searches the knowledge base and returns relevant passages for the query."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

agent = create_react_agent(
    model=llm, tools=[retrieve], checkpointer=MemorySaver(),
    state_modifier="Use retrieve when you need information from the knowledge base.",
)
```

---

## Common Mistakes

- `allow_dangerous_deserialization=True` missing on FAISS `load_local` — required in 1.x
- Chunk size exceeding model limit — Titan v2: 8192 tokens, Cohere: 512 tokens
- No chunk overlap — use 10-20% of chunk size
- Default `k=4` too low — use 6-10 for complex queries
- Using removed chains — `RetrievalQA` and `ConversationalRetrievalChain` are gone in 1.x
