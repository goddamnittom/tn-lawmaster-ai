# ⚖️ TN-LawMaster AI

> **Expert RAG + LangGraph AI Agent for Tennessee Law**  
> Grounded statutory analysis, TCA citations, and compliance tools — powered by any LLM you choose.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧭 What It Does

TN-LawMaster is an autonomous Tennessee law advisory engine that:

- **Retrieves** relevant Tennessee Code Annotated (TCA) statutes from a vector store (ChromaDB) or falls back to domain-profile knowledge
- **Reranks** retrieved context by keyword and citation relevance
- **Analyzes** the query using a grounded LLM prompt that demands TCA-cited responses
- **Extracts** all TCA citations from the generated analysis for audit traceability

### Covered TCA Domains

| Domain | TCA Titles | Coverage |
|--------|-----------|----------|
| Criminal | Title 39 | Offenses, penalties, sentencing |
| Family | Title 36 | Divorce, custody, child support, adoption |
| Property | Title 66 | Real property, deeds, landlord-tenant |
| Business | Title 48 | Corporations, LLCs, partnerships |
| Torts | Title 29 | Civil liability, comparative fault, damages |
| Estates | Titles 30–32 | Wills, trusts, probate |
| Traffic | Title 55 | Motor vehicles, DUI, traffic offenses |
| TIPA | Title 10 | Tennessee Information Practices Act |
| General | All | Cross-domain queries |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│              LangGraph Pipeline                  │
│                                                  │
│  retrieve_law ──► rerank_context ──► legal_analyze ──► generate_citations
│       │                                                        │
│  ChromaDB / TCA                                          TCA § Citations
│  domain fallback                                         Audit trail
└──────────────────────────────────────────────────┘
    │
    ▼
FastAPI REST API  ◄──►  Streamlit UI  ◄──►  CLI (run_agent.py)
```

### Supported LLM Backends

| Backend | Description | Config |
|---------|-------------|--------|
| **Ollama** (default) | Local inference — Gemma, Llama, Mistral | `ACTIVE_BACKEND=ollama` |
| **Groq** | Fast cloud API — free tier available | `ACTIVE_BACKEND=groq` |
| **OpenAI** | GPT-4o, GPT-4o-mini | `ACTIVE_BACKEND=openai` |
| **OpenRouter** | Proxy for 100+ models | `ACTIVE_BACKEND=openrouter` |

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/goddamnittom/tn-lawmaster-ai.git
cd tn-lawmaster-ai
cp .env.example .env
# Edit .env with your preferred backend and API key
```

### 2. Install Dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3a. CLI (fastest way to test)

```bash
# Interactive REPL
python run_agent.py

# One-shot query
python run_agent.py --query "What is the penalty for DUI in Tennessee?" --domain traffic

# Use a specific backend
python run_agent.py --backend groq --domain criminal
```

### 3b. Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501, select your backend, and initialize the agent.

### 3c. FastAPI Server

```bash
uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

API docs at http://localhost:8000/docs

---

## ⚙️ Configuration

All configuration is via environment variables (`.env`):

```bash
# Backend selection
ACTIVE_BACKEND=ollama          # ollama | groq | openai | openrouter

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b

# Groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama3-70b-8192

# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# OpenRouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct

# Storage
DATA_DIR=./data
CHROMA_DB_PATH=./chroma_db
```

See [`.env.example`](.env.example) for the full reference.

---

## 📚 Ingesting TCA Documents

TN-LawMaster works best when you feed it real TCA source documents. Without a populated vector store, it falls back to general TCA knowledge from model training.

### Via CLI / Python

```python
from tn_law_agent.knowledge.ingester import TNLawIngester

ingester = TNLawIngester(data_dir="./data", persist_dir="./chroma_db")
ingester.ingest_directory()          # all PDFs and TXT files in ./data/
ingester.ingest_pdf("tca_title39.pdf")
ingester.ingest_url("https://law.justia.com/codes/tennessee/...")
print(f"Total chunks: {ingester.doc_count}")
```

### Via FastAPI

```bash
# Single PDF
curl -X POST http://localhost:8000/ingest/pdf \
     -F "file=@tca_title39.pdf"

# Raw text
curl -X POST http://localhost:8000/ingest/text \
     -H "Content-Type: application/json" \
     -d '{"text": "TCA § 39-14-105 ...", "source": "TCA § 39-14-105"}'
```

### Via Streamlit UI

Upload PDFs directly in the sidebar "📄 Ingest TCA Documents" panel.

---

## 🔌 API Reference

### `POST /analyze`

```json
{
  "query": "What is the penalty for theft of property worth $1,500?",
  "domain": "criminal"
}
```

**Response:**
```json
{
  "query": "What is the penalty for theft...",
  "domain": "criminal",
  "analysis": "**Summary**: Under TCA § 39-14-105...",
  "citations": ["TCA § 39-14-105"],
  "status": "complete"
}
```

### `POST /analyze/stream`

Server-Sent Events (SSE) streaming variant — same request body as `/analyze`.

### `GET /health`

```json
{"status": "healthy", "agent_ready": true, "model": "Groq / llama3-70b-8192", "version": "1.0.0"}
```

### `GET /domains`

Returns all supported TCA domains with titles and descriptions.

---

## 🐳 Docker

### Single Container

```bash
docker build -t tn-lawmaster .
docker run -p 8000:8000 --env-file .env tn-lawmaster
```

### Docker Compose (with persistent vector store)

```bash
cd tn-lawmaster-production
cp ../.env .env
docker-compose up -d
```

---

## 🧪 Testing

```bash
# Unit tests (no API key needed)
pytest test_production.py -v -m unit

# All tests (integration requires ACTIVE_BACKEND + API key)
pytest test_production.py -v
```

---

## 📁 Project Structure

```
tn-lawmaster-ai/
├── .env.example                    # Configuration template
├── .gitignore                      # Excludes secrets, __pycache__, data/
├── requirements.txt                # All Python dependencies
├── model_config.py                 # Multi-backend LLM switcher
├── main_fastapi.py                 # FastAPI REST API
├── streamlit_app.py                # Streamlit web UI
├── run_agent.py                    # CLI interactive runner
├── test_production.py              # pytest unit + integration tests
├── Dockerfile                      # Single-container build
├── data/                           # Place TCA PDFs/TXT here for ingestion
├── tn_law_agent/
│   ├── core.py                     # TNLawAgent high-level wrapper
│   ├── knowledge/
│   │   └── ingester.py             # TCA document ingestion (PDF, text, URL)
│   └── workflows/
│       └── legal_graph.py          # LangGraph 4-node pipeline
└── tn-lawmaster-production/
    ├── Dockerfile                  # Production-hardened image
    ├── docker-compose.yml          # Multi-service deployment
    └── nginx/                      # Reverse proxy config
```

---

## ⚠️ Legal Disclaimer

TN-LawMaster provides **AI-generated legal information only — not legal advice**.  
Always consult a **licensed Tennessee attorney** for your specific legal situation.  
The authors are not responsible for decisions made based on this tool's output.

---

## 📄 License

MIT © TN-LawMaster Project
