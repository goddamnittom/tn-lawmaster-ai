"""
TN-LawMaster — Streamlit UI (v2)
==================================
Interactive Tennessee law analysis interface powered by LangGraph + any LLM backend.

Features:
  • Multi-backend support (Ollama, Groq, OpenAI, OpenRouter)
  • TCA domain selector
  • Persistent multi-turn chat (session memory)
  • Session history sidebar with past sessions
  • Formatted citation chips
  • Feedback buttons (👍/👎) per response
  • PDF export — single analysis or full chat session
  • Document drafter panel
  • Compliance checker panel
  • Document ingestion sidebar
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import sqlite3
from pathlib import Path

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
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
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
.feedback-row { display: flex; gap: 8px; margin-top: 4px; }
h1 { color: #f0f6fc !important; }
.stTextArea textarea { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; }
.stSelectbox > div { background: #161b22; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────
if "agent"          not in st.session_state: st.session_state.agent = None
if "chat_history"   not in st.session_state: st.session_state.chat_history = []
if "session_id"     not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "ingested_docs"  not in st.session_state: st.session_state.ingested_docs = 0
if "feedback"       not in st.session_state: st.session_state.feedback = {}   # msg_idx -> "up"|"down"
if "active_tab"     not in st.session_state: st.session_state.active_tab = "chat"
if "backend"        not in st.session_state: st.session_state.backend = os.getenv("ACTIVE_BACKEND","ollama")
if "domain"         not in st.session_state: st.session_state.domain = "general"

# ── Feedback DB helper ─────────────────────────────────────
_FEEDBACK_DB = os.getenv("FEEDBACK_DB_PATH", "./feedback.db")

def _init_feedback_db():
    con = sqlite3.connect(_FEEDBACK_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, msg_index INTEGER,
        rating TEXT, query TEXT, analysis_excerpt TEXT,
        created_at TEXT DEFAULT (datetime('now')))""")
    con.commit(); con.close()

def _save_feedback(session_id, msg_idx, rating, query, analysis_excerpt):
    try:
        _init_feedback_db()
        con = sqlite3.connect(_FEEDBACK_DB)
        con.execute("INSERT INTO feedback (session_id,msg_index,rating,query,analysis_excerpt) VALUES (?,?,?,?,?)",
                    (session_id, msg_idx, rating, query, analysis_excerpt[:500]))
        con.commit(); con.close()
    except Exception: pass

# ── Header ────────────────────────────────────────────────
col_icon, col_title = st.columns([0.06, 0.94])
with col_icon:   st.markdown("# ⚖️")
with col_title:
    st.markdown("## TN-LawMaster AI")
    st.caption("Expert Tennessee Code Annotated (TCA) analysis — "
               "Criminal · Family · Property · Business · Torts · Estates · Traffic · TIPA")
st.divider()

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    backend = st.selectbox("LLM Backend", ["ollama","groq","openai","openrouter"],
        index=["ollama","groq","openai","openrouter"].index(
            st.session_state.backend if st.session_state.backend in ["ollama","groq","openai","openrouter"] else "ollama"),
        key="backend_select")

    api_key_input = model_override = None

    if backend == "ollama":
        model_override = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL","gemma3:1b"))
        st.caption("💡 Ollama must be running locally.")
    elif backend == "groq":
        api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY",""))
        model_override = st.selectbox("Groq Model", ["llama3-70b-8192","llama3-8b-8192","mixtral-8x7b-32768"])
    elif backend == "openai":
        api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY",""))
        model_override = st.selectbox("OpenAI Model", ["gpt-4o-mini","gpt-4o","gpt-4-turbo"])
    elif backend == "openrouter":
        api_key_input = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY",""))
        model_override = st.text_input("Model", value=os.getenv("OPENROUTER_MODEL","mistralai/mistral-7b-instruct"))

    use_memory = st.checkbox("Enable conversation memory", value=True,
                             help="Remember prior questions in this session")

    st.divider()
    st.markdown("### ⚖️ Domain")
    domain = st.selectbox("TCA Domain",
        ["general","criminal","family","property","business","torts","estates","traffic","tipa"],
        key="domain_select")

    st.divider()
    st.markdown("### 🚀 Initialize")

    if st.button("Initialize Agent", use_container_width=True, type="primary"):
        key_map = {"groq":"GROQ_API_KEY","openai":"OPENAI_API_KEY","openrouter":"OPENROUTER_API_KEY"}
        if api_key_input and backend in key_map:
            os.environ[key_map[backend]] = api_key_input
        model_env = {"ollama":"OLLAMA_MODEL","groq":"GROQ_MODEL","openai":"OPENAI_MODEL","openrouter":"OPENROUTER_MODEL"}
        if model_override and backend in model_env:
            os.environ[model_env[backend]] = model_override
        os.environ["ACTIVE_BACKEND"] = backend
        st.session_state.backend = backend

        with st.spinner("Loading model..."):
            try:
                import importlib, model_config
                importlib.reload(model_config)
                from tn_law_agent.core import TNLawAgent
                llm = model_config.get_llm(backend)
                st.session_state.agent = TNLawAgent(llm=llm)
                st.session_state.chat_history = []
                st.session_state.feedback = {}
                st.session_state.session_id = str(uuid.uuid4())
                st.success(f"✅ Ready: {model_config.get_active_model_name(backend)}")
            except Exception as exc:
                st.error(f"❌ {exc}")

    status_color = "🟢" if st.session_state.agent else "🔴"
    st.markdown(f"**Status:** {status_color} {'Agent ready' if st.session_state.agent else 'Not initialized'}")
    if st.session_state.agent:
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    # ── History ───────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 🕐 Session History")
        past_queries = [m["content"][:45]+"…" for m in st.session_state.chat_history if m["role"]=="user"]
        for i, q in enumerate(past_queries[-5:], 1):
            st.caption(f"{i}. {q}")

    # ── Ingestion ─────────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Ingest TCA Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"],
                                      accept_multiple_files=True)
    if uploaded_files and st.button("Ingest PDFs", use_container_width=True):
        if not st.session_state.agent:
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
                st.success(f"✅ {total} chunks from {len(uploaded_files)} file(s)")
            except ImportError as exc:
                st.warning(f"⚠️ {exc}")

    if st.session_state.ingested_docs:
        st.caption(f"📚 {st.session_state.ingested_docs} chunks in vector store")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.feedback = {}
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ══════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════
if not st.session_state.agent:
    st.info("👈 **Configure your LLM backend in the sidebar and click Initialize Agent** to get started.\n\n"
            "TN-LawMaster supports **Ollama** (local), **Groq**, **OpenAI**, and **OpenRouter**.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────
tab_chat, tab_draft, tab_compliance, tab_tools = st.tabs(
    ["💬 Chat", "📝 Draft Documents", "🔍 Compliance Check", "🛠️ Legal Tools"]
)

# ══════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════
with tab_chat:
    # Display history
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    chips = "".join(f'<span class="citation-chip">{c}</span>' for c in msg["citations"])
                    st.markdown(f"📌 **Citations:** {chips}", unsafe_allow_html=True)

                # Feedback buttons
                fb_key = f"fb_{i}"
                current_fb = st.session_state.feedback.get(i)
                col_up, col_down, col_pdf, col_space = st.columns([1,1,2,6])
                with col_up:
                    if st.button("👍", key=f"up_{i}",
                                 help="Helpful",
                                 type="primary" if current_fb=="up" else "secondary"):
                        st.session_state.feedback[i] = "up"
                        user_q = st.session_state.chat_history[i-1]["content"] if i > 0 else ""
                        _save_feedback(st.session_state.session_id, i, "up", user_q, msg["content"])
                with col_down:
                    if st.button("👎", key=f"down_{i}",
                                 help="Not helpful",
                                 type="primary" if current_fb=="down" else "secondary"):
                        st.session_state.feedback[i] = "down"
                        user_q = st.session_state.chat_history[i-1]["content"] if i > 0 else ""
                        _save_feedback(st.session_state.session_id, i, "down", user_q, msg["content"])
                with col_pdf:
                    if st.button("📄 Export PDF", key=f"pdf_{i}", help="Download this analysis as PDF"):
                        try:
                            from tn_law_agent.utils.pdf_exporter import export_analysis_pdf
                            user_q = st.session_state.chat_history[i-1]["content"] if i > 0 else "Analysis"
                            result_dict = {"analysis": msg["content"], "citations": msg.get("citations",[])}
                            pdf_bytes = export_analysis_pdf(result_dict, query=user_q, domain=domain)
                            st.download_button("💾 Download", data=pdf_bytes,
                                               file_name="tn-lawmaster-analysis.pdf",
                                               mime="application/pdf",
                                               key=f"dl_{i}")
                        except ImportError:
                            st.warning("Install reportlab for PDF export: `pip install reportlab`")

    # Export full session
    if len(st.session_state.chat_history) >= 2:
        col_export, _ = st.columns([2, 8])
        with col_export:
            if st.button("📋 Export Full Chat as PDF"):
                try:
                    from tn_law_agent.utils.pdf_exporter import export_chat_session_pdf
                    pdf_bytes = export_chat_session_pdf(
                        st.session_state.chat_history,
                        session_id=st.session_state.session_id[:8],
                    )
                    st.download_button("💾 Download Chat PDF", data=pdf_bytes,
                                       file_name="tn-lawmaster-chat.pdf",
                                       mime="application/pdf")
                except ImportError:
                    st.warning("Install reportlab: `pip install reportlab`")

    # Chat input
    query = st.chat_input("Ask a Tennessee law question…")

    if query:
        st.session_state.chat_history.append({"role":"user","content":query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="⚖️"):
            status_ph = st.empty()
            status_ph.markdown("*🔍 Retrieving statutes…*")
            t0 = time.perf_counter()

            try:
                sid = st.session_state.session_id if use_memory else None
                result = st.session_state.agent.analyze(query, domain=domain, session_id=sid)
                elapsed = time.perf_counter() - t0
                status_ph.empty()

                analysis = result.get("analysis","")
                citations = result.get("citations",[])
                error = result.get("error","")

                if error or result.get("status")=="error":
                    st.error(f"⚠️ {error or 'Unknown pipeline error'}")
                else:
                    st.markdown(analysis)
                    if citations:
                        chips = "".join(f'<span class="citation-chip">{c}</span>' for c in citations)
                        st.markdown(f"📌 **Citations:** {chips}", unsafe_allow_html=True)
                    st.caption(f"⏱ {elapsed:.1f}s · Domain: {domain} · Session: {st.session_state.session_id[:8]}…")

                st.session_state.chat_history.append(
                    {"role":"assistant","content":analysis or error,"citations":citations})

            except Exception as exc:
                status_ph.error(f"❌ {exc}")
                logging.exception("Streamlit chat error")

# ══════════════════════════════════════════════════════════
# TAB 2: DOCUMENT DRAFTER
# ══════════════════════════════════════════════════════════
with tab_draft:
    st.markdown("### 📝 Tennessee Legal Document Drafter")
    st.caption("Generate AI-assisted legal document drafts grounded in TCA statutes.")

    from tn_law_agent.drafts.document_drafter import DOCUMENT_TYPES

    doc_type = st.selectbox("Document Type",
        options=list(DOCUMENT_TYPES.keys()),
        format_func=lambda k: DOCUMENT_TYPES[k]["label"])

    defn = DOCUMENT_TYPES[doc_type]
    st.caption(f"**TCA basis:** {', '.join(defn['tca_basis'])}")

    st.markdown("**Required fields:**")
    context = {}
    cols = st.columns(2)
    for i, field in enumerate(defn["required_fields"]):
        with cols[i % 2]:
            context[field] = st.text_input(field.replace("_"," ").title(), key=f"req_{doc_type}_{field}")

    with st.expander("Optional fields"):
        for field in defn.get("optional_fields", []):
            context[field] = st.text_input(field.replace("_"," ").title(), key=f"opt_{doc_type}_{field}")

    if st.button("📝 Generate Document", type="primary", use_container_width=True):
        with st.spinner("Drafting document…"):
            try:
                from tn_law_agent.drafts.document_drafter import TNDocumentDrafter
                drafter = TNDocumentDrafter(llm=st.session_state.agent.llm)
                result = drafter.draft(doc_type, context)
                st.markdown("---")
                st.markdown(result["document"])
                if result.get("missing_fields"):
                    st.warning(f"Missing fields (placeholders used): {result['missing_fields']}")
                # Download
                st.download_button("💾 Download as TXT", data=result["document"],
                                   file_name=f"tn-lawmaster-{doc_type}.txt",
                                   mime="text/plain")
            except Exception as exc:
                st.error(f"❌ {exc}")

# ══════════════════════════════════════════════════════════
# TAB 3: COMPLIANCE CHECKER
# ══════════════════════════════════════════════════════════
with tab_compliance:
    st.markdown("### 🔍 Tennessee Law Compliance Checker")
    st.caption("Paste a document to check it for compliance issues under Tennessee law.")

    from tn_law_agent.compliance.checker import COMPLIANCE_DOMAINS

    comp_domain = st.selectbox("Compliance Domain",
        options=list(COMPLIANCE_DOMAINS.keys()),
        format_func=lambda k: COMPLIANCE_DOMAINS[k]["label"])

    st.caption(f"**TCA refs:** {', '.join(COMPLIANCE_DOMAINS[comp_domain]['tca_refs'])}")
    doc_text = st.text_area("Paste document text here", height=250,
                             placeholder="Paste lease agreement, contract, policy, etc.")

    if st.button("🔍 Check Compliance", type="primary", use_container_width=True):
        if not doc_text.strip():
            st.warning("Please paste document text first.")
        else:
            with st.spinner("Analyzing compliance…"):
                try:
                    from tn_law_agent.compliance.checker import TNComplianceChecker
                    checker = TNComplianceChecker(llm=st.session_state.agent.llm)
                    result = checker.check(doc_text, comp_domain)

                    # Summary
                    st.markdown(f"**{result['summary']}**")
                    counts = result["issue_count"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🚨 High", counts["high"])
                    col2.metric("⚠️ Medium", counts["medium"])
                    col3.metric("ℹ️ Low", counts["low"])

                    # Issues table
                    if result["issues"]:
                        st.markdown("#### Issues Found")
                        for iss in result["issues"]:
                            sev = iss["severity"]
                            icon = {"high":"🚨","medium":"⚠️","low":"ℹ️"}.get(sev,"•")
                            with st.expander(f"{icon} [{sev.upper()}] {iss['description'][:80]}"):
                                st.markdown(f"**TCA Reference:** `{iss['tca_ref']}`")
                                st.markdown(f"**Issue:** {iss['description']}")
                                if iss.get("recommendation"):
                                    st.markdown(f"**Fix:** {iss['recommendation']}")
                    else:
                        st.success("No specific issues detected.")

                except Exception as exc:
                    st.error(f"❌ {exc}")

# ══════════════════════════════════════════════════════════
# TAB 4: LEGAL TOOLS
# ══════════════════════════════════════════════════════════
with tab_tools:
    st.markdown("### 🛠️ Tennessee Legal Tools")

    tool_choice = st.radio("Select tool", [
        "⏰ Statute of Limitations Calculator",
        "⚖️ TN Case Law Search",
        "📋 Legal Forms Finder",
    ], horizontal=True)

    if tool_choice.startswith("⏰"):
        st.markdown("#### Statute of Limitations Calculator (TCA Title 28)")
        from tn_law_agent.tools.legal_tools import _SOL_TABLE
        sol_type = st.selectbox("Offense / Claim Type",
            options=list(_SOL_TABLE.keys()),
            format_func=lambda k: f"{k.replace('_',' ').title()} — {_SOL_TABLE[k]['tca']}")
        sol_date = st.date_input("Date of Incident / Breach")

        if st.button("Calculate Deadline", type="primary"):
            from tn_law_agent.tools.legal_tools import calculate_statute_of_limitations
            result = calculate_statute_of_limitations.invoke(
                {"offense_type": sol_type, "date_of_offense": str(sol_date)})
            st.code(result, language=None)

    elif tool_choice.startswith("⚖️"):
        st.markdown("#### Tennessee Case Law Search (CourtListener)")
        case_query = st.text_input("Search query", placeholder="DUI first offense penalty")
        case_limit = st.slider("Max results", 1, 10, 3)

        if st.button("Search Case Law", type="primary"):
            if not case_query.strip():
                st.warning("Enter a search query.")
            else:
                with st.spinner("Searching CourtListener…"):
                    from tn_law_agent.tools.legal_tools import search_tn_case_law
                    result = search_tn_case_law.invoke(
                        {"query": case_query, "max_results": case_limit})
                    st.code(result, language=None)

    elif tool_choice.startswith("📋"):
        st.markdown("#### Tennessee Legal Forms Finder")
        from tn_law_agent.tools.legal_tools import _FORMS_DIRECTORY
        form_type = st.selectbox("Form Category",
            options=list(_FORMS_DIRECTORY.keys()),
            format_func=lambda k: k.replace("_"," ").title())

        if st.button("Get Form Info", type="primary"):
            from tn_law_agent.tools.legal_tools import get_tennessee_legal_forms
            result = get_tennessee_legal_forms.invoke({"form_type": form_type})
            st.code(result, language=None)

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("⚠️ **Legal Disclaimer**: TN-LawMaster provides AI-generated legal information only — "
           "not legal advice. Always consult a licensed Tennessee attorney for your specific situation.")