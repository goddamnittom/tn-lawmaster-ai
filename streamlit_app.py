"""
TN-LawMaster — Streamlit UI
============================
Interactive Tennessee law analysis interface powered by LangGraph + any LLM backend.

Features:
  • Multi-backend support (Ollama, Groq, OpenAI, OpenRouter)
  • TCA domain selector
  • Persistent chat history
  • Formatted citations panel
  • Document ingestion sidebar
"""

from __future__ import annotations

import os
import sys
import time
import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="TN-LawMaster AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #30363d;
    }
    /* Chat message bubbles */
    .user-msg {
        background: #1f6feb22;
        border: 1px solid #1f6feb55;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .bot-msg {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    /* Citation badge */
    .citation-chip {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78em;
        font-family: monospace;
        margin: 2px;
        color: #58a6ff;
    }
    /* Header */
    h1 { color: #f0f6fc !important; }
    .stTextArea textarea { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = 0

# ── Header ────────────────────────────────────────────────
col_icon, col_title = st.columns([0.06, 0.94])
with col_icon:
    st.markdown("# ⚖️")
with col_title:
    st.markdown("## TN-LawMaster AI")
    st.caption(
        "Expert Tennessee Code Annotated (TCA) analysis — "
        "Criminal · Family · Property · Business · Torts · Estates · Traffic · TIPA"
    )
st.divider()

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    backend = st.selectbox(
        "LLM Backend",
        ["ollama", "groq", "openai", "openrouter"],
        index=["ollama", "groq", "openai", "openrouter"].index(
            os.getenv("ACTIVE_BACKEND", "ollama")
        ),
        help="Choose your inference provider.",
    )

    # Backend-specific credentials
    api_key_input = None
    model_override = None

    if backend == "ollama":
        model_override = st.text_input(
            "Ollama Model",
            value=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
            help="Must be pulled via `ollama pull <model>`",
        )
        st.caption("💡 Make sure Ollama is running locally.")

    elif backend == "groq":
        api_key_input = st.text_input("Groq API Key", type="password",
                                       value=os.getenv("GROQ_API_KEY", ""))
        model_override = st.selectbox(
            "Groq Model",
            ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
            index=0,
        )

    elif backend == "openai":
        api_key_input = st.text_input("OpenAI API Key", type="password",
                                       value=os.getenv("OPENAI_API_KEY", ""))
        model_override = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=0,
        )

    elif backend == "openrouter":
        api_key_input = st.text_input("OpenRouter API Key", type="password",
                                       value=os.getenv("OPENROUTER_API_KEY", ""))
        model_override = st.text_input(
            "OpenRouter Model",
            value=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
        )

    st.divider()
    st.markdown("### ⚖️ Domain")
    domain = st.selectbox(
        "TCA Domain",
        [
            "general", "criminal", "family", "property",
            "business", "torts", "estates", "traffic", "tipa",
        ],
        help="Narrow the analysis to a specific TCA area.",
    )

    st.divider()
    st.markdown("### 🚀 Initialize")

    if st.button("Initialize Agent", use_container_width=True, type="primary"):
        _env_overrides = {}
        if api_key_input:
            key_map = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
            if backend in key_map:
                os.environ[key_map[backend]] = api_key_input
        if model_override:
            model_env_map = {
                "ollama": "OLLAMA_MODEL",
                "groq": "GROQ_MODEL",
                "openai": "OPENAI_MODEL",
                "openrouter": "OPENROUTER_MODEL",
            }
            if backend in model_env_map:
                os.environ[model_env_map[backend]] = model_override
        os.environ["ACTIVE_BACKEND"] = backend

        with st.spinner("Loading model..."):
            try:
                from model_config import get_llm, get_active_model_name
                from tn_law_agent.core import TNLawAgent
                # Re-import to pick up env changes
                import importlib, model_config
                importlib.reload(model_config)
                llm = model_config.get_llm(backend)
                st.session_state.agent = TNLawAgent(llm=llm)
                st.session_state.chat_history = []
                st.success(f"✅ Ready: {model_config.get_active_model_name(backend)}")
            except Exception as exc:
                st.error(f"❌ {exc}")

    if st.session_state.agent:
        st.markdown("**Status:** 🟢 Agent ready")
    else:
        st.markdown("**Status:** 🔴 Not initialized")

    # ── Document ingestion panel ──────────────────────────
    st.divider()
    st.markdown("### 📄 Ingest TCA Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload TCA statute PDFs to augment the analysis with exact text.",
    )
    if uploaded_files and st.button("Ingest PDFs", use_container_width=True):
        if st.session_state.agent is None:
            st.error("Initialize the agent first.")
        else:
            try:
                from tn_law_agent.knowledge.ingester import TNLawIngester
                import tempfile, pathlib
                ingester = TNLawIngester()
                total = 0
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(f.read())
                        n = ingester.ingest_pdf(pathlib.Path(tmp.name))
                    total += n
                st.session_state.agent.attach_vector_store(ingester)
                st.session_state.ingested_docs = ingester.doc_count
                st.success(f"✅ Ingested {total} chunks from {len(uploaded_files)} file(s)")
            except ImportError as exc:
                st.warning(f"⚠️ {exc}")

    if st.session_state.ingested_docs:
        st.caption(f"📚 {st.session_state.ingested_docs} chunks in vector store")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════

if not st.session_state.agent:
    st.info(
        "👈 **Configure your LLM backend in the sidebar and click Initialize Agent** to get started.\n\n"
        "TN-LawMaster supports **Ollama** (local), **Groq**, **OpenAI**, and **OpenRouter**."
    )
    st.stop()

# ── Chat history display ──────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg["content"])
            if msg.get("citations"):
                citation_html = "".join(
                    f'<span class="citation-chip">{c}</span>'
                    for c in msg["citations"]
                )
                st.markdown(
                    f"📌 **Citations:** {citation_html}",
                    unsafe_allow_html=True,
                )

# ── Input ─────────────────────────────────────────────────
query = st.chat_input(
    "Ask a Tennessee law question… (e.g., 'What is the penalty for felony theft in TN?')"
)

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="⚖️"):
        status_msg = st.empty()
        status_msg.markdown("*🔍 Retrieving statutes…*")

        t0 = time.perf_counter()
        try:
            result = st.session_state.agent.analyze(query, domain=domain)
            elapsed = time.perf_counter() - t0

            status_msg.empty()
            analysis = result.get("analysis", "")
            citations = result.get("citations", [])
            error = result.get("error", "")

            if error or result.get("status") == "error":
                st.error(f"⚠️ Pipeline error: {error or 'Unknown error'}")
            else:
                st.markdown(analysis)
                if citations:
                    citation_html = "".join(
                        f'<span class="citation-chip">{c}</span>'
                        for c in citations
                    )
                    st.markdown(
                        f"📌 **Citations:** {citation_html}",
                        unsafe_allow_html=True,
                    )
                st.caption(f"⏱ Analysis completed in {elapsed:.1f}s · Domain: {domain}")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": analysis or error, "citations": citations}
            )

        except Exception as exc:
            status_msg.error(f"❌ Error: {exc}")
            logging.exception("Streamlit UI error")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Legal Disclaimer**: TN-LawMaster provides AI-generated legal information only — "
    "not legal advice. Always consult a licensed Tennessee attorney for your specific situation."
)