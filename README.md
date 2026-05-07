# 🤖 LangGraph ReAct Agent

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A minimal **ReAct (Reason + Act)** agent built with **LangGraph**, capable of calling external tools — web search via **Tavily** and a custom Python tool — to answer multi-step questions.

---

## ✨ Features

- 🔁 **ReAct loop** with conditional routing
- 🔍 **Web search** via Tavily
- 🧮 **Custom Python tool** (`triple`) for arithmetic
- 📊 **Auto-generated** Mermaid state-graph diagram
- 🪶 **Tiny** — three Python files, easy to read

---

## 🏗️ Architecture

| Component | Description |
| --- | --- |
| **State graph** | Two nodes: `agent_reasoning` and `act` |
| **`agent_reasoning`** | LLM call that decides whether to use a tool or finish |
| **`act`** | Executes the chosen tool(s) via LangGraph's prebuilt `ToolNode` |
| **Routing** | `should_continue` checks for `tool_calls` on the last message — routes to `act`, otherwise to `END` |
| **Model** | OpenAI `gpt-4o-mini`, `temperature=0` |
| **Tools** | `TavilySearch` (web search) + `triple` (returns `n * 3`) |
| **Persistence** | None — single-shot in-memory invocation |

---

## 📊 Diagram

![Agent flow](agent_flow.png)

The diagram is regenerated every time `main.py` runs (via `app.get_graph().draw_mermaid_png(...)`).

---

## 🚀 Getting started

### Prerequisites

- **Python 3.12+** — [download](https://www.python.org/downloads/)
- **Poetry** — [install guide](https://python-poetry.org/docs/#installation):
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

### 1. Clone the repository

```bash
git clone <repo-url>
cd langraph-react-agent
```

### 2. Install dependencies

```bash
poetry install
```

This creates an isolated virtualenv and installs the locked dependencies from `poetry.lock`.

### 3. Get your API keys

You will need two keys (both have free tiers):

| Service | Where to get it | Used for |
| --- | --- | --- |
| 🔑 **OpenAI**  | <https://platform.openai.com/api-keys> | LLM reasoning (`gpt-4o-mini`) |
| 🔍 **Tavily**  | <https://app.tavily.com/home>          | Web search tool |

Optional (for tracing & observability):

| Service | Where to get it | Used for |
| --- | --- | --- |
| 📈 **LangSmith** | <https://smith.langchain.com/settings> | Trace & debug agent runs |

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and paste your keys:

```dotenv
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 5. Run the agent

```bash
poetry run python main.py
```

You should see the agent reason, call the search tool, then call the `triple` tool, and finally print a complete answer. An updated `agent_flow.png` is written to disk on each run.

---

## 🔑 Environment variables reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY`     | ✅       | LLM reasoning |
| `TAVILY_API_KEY`     | ✅       | Web search tool |
| `LANGSMITH_TRACING`  | optional | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY`  | optional | Required if tracing is enabled |
| `LANGSMITH_PROJECT`  | optional | Project name in LangSmith |

---

## 🧠 Notes

This was built as a learning project to explore LangGraph patterns, including:

- `StateGraph` + `MessagesState`
- Conditional edges driven by tool-call presence
- The prebuilt `ToolNode`
- `bind_tools` for OpenAI function-calling
- Mermaid PNG generation from a compiled graph

---

## 📁 Project structure

```
langraph-react-agent/
├── main.py            # graph wiring + entry point
├── nodes.py           # reasoning node + tool node
├── react.py           # LLM + tools definitions
├── pyproject.toml     # Poetry dependencies
├── poetry.lock        # locked dependency versions
├── .env.example       # environment variable template
├── .gitignore
└── agent_flow.png     # generated state-graph diagram
```

---

## 📝 License

[MIT](LICENSE)
