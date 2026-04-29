---
name: langchain-rag
description: Use this skill when building retrieval-augmented generation (RAG) pipelines, working with vector stores, document loaders, text splitters, embeddings, or retrievers. Triggers when code imports from langchain_community.document_loaders, langchain_text_splitters, or vector store classes. Also triggers on mentions of "RAG", "retrieval", "vector store", "embeddings", "document loader", "chunking", or "similarity search".
version: 1.0.0
---

# LangChain RAG — v1.2 Reference

Target versions: `langchain>=1.2`, `langchain-core>=1.2`, `langchain-aws>=1.4`, `langchain-community>=0.4`

---

## Document Loaders

```python
# PDF
from langchain_community.document_loaders import PyPDFLoader
docs = PyPDFLoader("report.pdf").load()

# Web page
from langchain_community.document_loaders import WebBaseLoader
docs = WebBaseLoader("https://example.com/page").load()

# Directory of files
from langchain_community.document_loaders import DirectoryLoader
docs = DirectoryLoader("./docs", glob="**/*.md").load()

# Plain text
from langchain_community.document_loaders import TextLoader
docs = TextLoader("notes.txt").load()

# CSV
from langchain_community.document_loaders import CSVLoader
docs = CSVLoader("data.csv", source_column="url").load()
```

Each loader returns `list[Document]` where `Document` has `.page_content` (str) and `.metadata` (dict).

**Deprecated:**
```python
# DEPRECATED — import paths moved to langchain_community in 1.x
from langchain.document_loaders import PyPDFLoader   # broken in 1.x
from langchain.document_loaders import WebBaseLoader # broken in 1.x
```

---

## Text Splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],  # tries each in order
)
chunks = splitter.split_documents(docs)

# For code
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=100,
)
```

**Deprecated:**
```python
# DEPRECATED — moved to langchain_text_splitters package
from langchain.text_splitter import RecursiveCharacterTextSplitter  # broken in 1.x
```

---

## Embeddings

### AWS Bedrock (primary)

```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)

# Cohere on Bedrock
embeddings = BedrockEmbeddings(
    model_id="cohere.embed-english-v3",
    region_name="us-east-1",
)
```

### Other providers

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

**Deprecated:**
```python
# DEPRECATED — removed in 1.x
from langchain.embeddings import BedrockEmbeddings
from langchain.embeddings import OpenAIEmbeddings
```

---

## Vector Stores

```python
# FAISS (local, no server required)
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Chroma (local, persistent)
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs",
)

# Add documents to existing store
vectorstore.add_documents(new_chunks)

# Similarity search
results = vectorstore.similarity_search("query", k=4)
results_with_scores = vectorstore.similarity_search_with_score("query", k=4)
```

---

## Retrievers

```python
# Basic retriever from vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr" (max marginal relevance)
    search_kwargs={"k": 4},
)

# MMR retriever (diverse results, reduces redundancy)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)

# Invoke
docs = retriever.invoke("What is LangGraph?")
```

### Multi-Query Retriever (generates query variants to improve recall)

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)
```

### Contextual Compression Retriever (re-ranks and filters chunks)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(),
)
```

---

## RAG Chain (LCEL)

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

answer = rag_chain.invoke("What is LangGraph used for?")
```

**Deprecated:**
```python
# DEPRECATED — removed in langchain 1.x
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# DEPRECATED
from langchain.chains import ConversationalRetrievalChain
# Migrate to: LangGraph agent with retriever as a tool (see langgraph-agents skill)
```

---

## Conversational RAG (LangGraph pattern)

For conversational RAG, make the retriever a tool and wire it into a LangGraph agent — do NOT use `ConversationalRetrievalChain`.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def retrieve(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

agent = create_react_agent(
    model=llm,
    tools=[retrieve],
    checkpointer=MemorySaver(),
    state_modifier="You are a helpful assistant. Use the retrieve tool when you need information.",
)
```

---

## Common Mistakes

- **Chunk size too large**: Most embedding models have token limits (512–8192 tokens). Titan v2 supports 8192 tokens; Cohere v3 supports 512. Set `chunk_size` accordingly.
- **No overlap**: Without `chunk_overlap`, context at chunk boundaries is lost. 10-20% of chunk size is a good default.
- **allow_dangerous_deserialization missing**: FAISS `load_local` requires this flag in langchain 1.x — it's a security acknowledgment, not a bug.
- **Using deprecated chains**: `RetrievalQA` and `ConversationalRetrievalChain` are removed in 1.x. Use LCEL RAG chain (single-turn) or LangGraph agent (multi-turn).
- **Retriever not returning enough docs**: Default `k=4` is often too few for complex queries. Increase to 8-10 and use MMR to reduce redundancy.
