# langvibes

Instruction files and skills for building LangChain and LangGraph systems with AI coding agents. Drop these files into any LangChain workspace and your vibe coding agent — Claude Code, GitHub Copilot, or Kiro — will follow correct 1.x patterns, use `ChatBedrockConverse` by default, and never reach for deprecated APIs.

---

## What's Included

```
langvibes/
├── CLAUDE.md                          # Claude Code — always-on instructions
├── requirements.txt
├── .claude/
│   └── skills/                        # Claude Code — triggered skills
│       ├── langchain-core/SKILL.md
│       ├── langgraph-agents/SKILL.md
│       ├── langchain-tools/SKILL.md
│       └── langchain-rag/SKILL.md
├── .github/
│   ├── copilot-instructions.md        # Copilot — always-on instructions
│   └── skills/                        # Copilot — agent skills
│       ├── langchain-core/SKILL.md
│       ├── langgraph-agents/SKILL.md
│       ├── langchain-tools/SKILL.md
│       └── langchain-rag/SKILL.md
└── .kiro/
    ├── steering/                      # Kiro — always-on / fileMatch context
    │   ├── agentic-design.md
    │   ├── langchain-core.md
    │   ├── langgraph-agents.md
    │   ├── langchain-tools.md
    │   └── langchain-rag.md
    └── skills/                        # Kiro — invocable skills
        ├── langchain-core/SKILL.md
        ├── langgraph-agents/SKILL.md
        ├── langchain-tools/SKILL.md
        └── langchain-rag/SKILL.md
```

### How each agent uses these files

| Agent | Always-on | Triggered skills |
|---|---|---|
| **Claude Code** | `CLAUDE.md` — always in context | `.claude/skills/` — triggered by imports and keywords |
| **GitHub Copilot** | `.github/copilot-instructions.md` — every chat session | `.github/skills/` — auto-triggered or `/skill-name` |
| **Kiro** | `.kiro/steering/` — always or on Python file open | `.kiro/skills/` — auto-triggered or `/skill-name` |

---

## Stack

| Package | Version |
|---|---|
| `langchain` | `== 1.2.15` |
| `langchain-core` | `== 1.3.2` |
| `langgraph` | `== 1.1.10` |
| `langchain-aws` | `== 1.4.5` |
| `langchain-community` | `== 0.4.1` |
| `langchain-text-splitters` | `== 1.1.2` |

Primary LLM: `ChatBedrockConverse` (Anthropic models via AWS Bedrock).

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Gauravpadam/langvibes.git
cd langvibes
```

### 2. Copy files into your project

```bash
# From inside the langvibes directory
cp -r .claude/ /path/to/your/project/
cp -r .github/ /path/to/your/project/
cp -r .kiro/ /path/to/your/project/
cp requirements.txt /path/to/your/project/
```

Or use the install script:

```bash
./install.sh /path/to/your/project
```

### 3. Install Python dependencies

```bash
cd /path/to/your/project
pip install -r requirements.txt
```

### 4. Configure AWS credentials

```bash
aws configure
```

Your IAM role or user needs Bedrock `InvokeModel` permissions for the Claude model IDs you intend to use.

### 5. Optional — LangSmith tracing

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
export LANGCHAIN_PROJECT=your-project-name
```

---

## Usage by Agent

### Claude Code

Skills auto-trigger — no setup beyond copying `.claude/` into the workspace. Claude detects relevant imports and context and loads the appropriate skill.

| What you're working on | Skill triggered |
|---|---|
| `from langchain_core`, `from langchain_aws`, LCEL pipes | `langchain-core` |
| `from langgraph`, `StateGraph`, graph/agent/checkpoint | `langgraph-agents` |
| `@tool`, `ToolNode`, `bind_tools` | `langchain-tools` |
| Document loaders, vector stores, RAG, embeddings | `langchain-rag` |

Verify it's working: ask Claude to build a LangGraph agent — it should produce `ChatBedrockConverse`, `StateGraph`, `ToolNode`, and `MemorySaver` without being told.

### GitHub Copilot

Copilot reads `.github/copilot-instructions.md` automatically for every Copilot Chat session in the workspace.

Requirements:
- VS Code with GitHub Copilot extension
- Copilot Chat enabled
- Project opened as the workspace root (not a subfolder)

Verify it's working: ask Copilot Chat to set up a LangGraph agent with tool calling — it should default to the correct patterns without prompting.

### Kiro

Kiro reads `.kiro/steering/` files based on each file's `inclusion` frontmatter.

- `agentic-design.md` — loaded every session (design principles)
- All other steering files — loaded when any Python file is open

Verify it's working: open a `.py` file and ask Kiro to build a LangGraph agent — it should model termination in state explicitly, use `ChatBedrockConverse`, and put write operations behind `interrupt()`.

---

## Design Principles Enforced

Every agent guided by these files will follow:

- **Memory hierarchy** — in-context state, checkpointer persistence, and external long-term store are three distinct layers; graph state is never used as a database
- **Explicit exit conditions** — every graph cycle models termination in state; `recursion_limit` is a backstop, not a design
- **Interrupt before irreversible actions** — writes, sends, and deletes are always behind `interrupt()` checkpoints
- **Tool docstrings as prompt engineering** — docstrings describe what the tool returns, written before the function body
- **No deprecated APIs** — `AgentExecutor`, `LLMChain`, `ConversationalRetrievalChain`, `ChatBedrock`, and all `from langchain.*` imports are prohibited

---

## Updating

When upgrading LangChain or LangGraph packages:

1. Update version pins in `requirements.txt`
2. Update the version target line in each `.claude/skills/*/SKILL.md`
3. Update the version target line in each `.kiro/steering/*.md`
4. Update the stack table in `.github/copilot-instructions.md`
5. Add deprecation notes for any APIs removed in the new release
6. Open a PR against this repo so others benefit from the update

---

## Contributing

PRs welcome for:
- Version bumps with updated deprecation notes
- New steering files for additional LangChain integrations
- Corrections to patterns or import paths
