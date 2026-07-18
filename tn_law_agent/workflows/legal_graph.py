"""
TN-LawMaster — LangGraph Legal Analysis Workflow
================================================
Production-grade RAG + LangGraph pipeline for Tennessee law analysis.

Pipeline stages:
  1. load_memory       — Load prior conversation turns from SQLite
  2. retrieve_law      — Vector store lookup (or TCA keyword fallback)
  3. rerank_context    — Relevance-score and top-k filter
  4. legal_analyze     — LLM grounded analysis with strict TCA prompt
  5. generate_citations— Audit trail of referenced TCA statutes
  6. save_memory       — Persist Q&A turn to SQLite

Author: TN-LawMaster Project
"""

from __future__ import annotations

import logging
import operator
import os
import re
import uuid
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# ── TCA Domain Profiles ────────────────────────────────────
#
# These are the TCA titles this system covers.  Each entry provides
# a short description used to enrich the retrieval step when a
# vector store is not yet populated.
TCA_DOMAIN_COVERAGE: dict[str, dict] = {
    "criminal": {
        "titles": ["39"],
        "desc": "Tennessee Criminal Law (TCA Title 39): offenses, penalties, sentencing",
    },
    "family": {
        "titles": ["36"],
        "desc": "Tennessee Family Law (TCA Title 36): divorce, custody, child support, adoption",
    },
    "property": {
        "titles": ["66"],
        "desc": "Tennessee Property Law (TCA Title 66): real property, deeds, landlord-tenant",
    },
    "business": {
        "titles": ["48"],
        "desc": "Tennessee Business Law (TCA Title 48): corporations, LLCs, partnerships",
    },
    "torts": {
        "titles": ["29"],
        "desc": "Tennessee Torts (TCA Title 29): civil liability, damages, comparative fault",
    },
    "estates": {
        "titles": ["30", "31", "32"],
        "desc": "Tennessee Estates & Probate (TCA Titles 30-32): wills, trusts, estates",
    },
    "traffic": {
        "titles": ["55"],
        "desc": "Tennessee Traffic Law (TCA Title 55): motor vehicles, DUI, traffic offenses",
    },
    "tipa": {
        "titles": ["10"],
        "desc": "Tennessee Information Practices Act (TCA Title 10): TIPA, public records",
    },
    "general": {
        "titles": [],
        "desc": "General Tennessee law query across all covered TCA titles",
    },
}

# ── System Prompt ─────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are TN-LawMaster, an expert Tennessee legal analysis engine with deep knowledge \
of the Tennessee Code Annotated (TCA).

RULES:
1. Ground every statement in a specific TCA citation (e.g., "TCA § 39-14-105").
2. If the provided context does not contain a relevant TCA reference, clearly state: \
   "No specific TCA reference found in the retrieved context."
3. Never speculate beyond what the TCA and provided context support.
4. Structure your response with: Summary → Applicable Statutes → Analysis → Caveats.
5. Flag if the query requires a licensed Tennessee attorney for advice.

FORMAT example:
**Summary**: [1–2 sentence plain-language answer]

**Applicable Statutes**:
- TCA § XX-XX-XXX — [statute name]

**Analysis**: [detailed grounded analysis]

**⚠️ Legal Disclaimer**: This is AI-generated legal information, not legal advice. \
Consult a licensed Tennessee attorney for your specific situation."""

# ── TCA Citation Pattern ──────────────────────────────────
_TCA_CITATION_RE = re.compile(r"TCA\s*§\s*[\d]+-[\d]+-[\d]+")

# ── Memory configuration ──────────────────────────────────
_MEMORY_DB_PATH: str = os.environ.get("MEMORY_DB_PATH", "./memory.db")
_MEMORY_MAX_TURNS: int = int(os.environ.get("MEMORY_MAX_TURNS", "6"))


# ── LangGraph State ───────────────────────────────────────

class LegalAgentState(TypedDict):
    query: str
    domain: str                                         # e.g. "criminal", "family"
    session_id: str                                     # conversation session identifier
    messages: Annotated[List[BaseMessage], operator.add]  # conversation history messages
    documents: Annotated[List[Dict], operator.add]     # accumulates across retries
    reranked_docs: List[Dict]
    analysis: str
    citations: List[str]
    status: str
    error: str


# ── Graph Class ───────────────────────────────────────────

class TNLawGraph:
    """LangGraph-based Tennessee law analysis pipeline with conversation memory."""

    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store

        # Lazy-init memory: created on first use to avoid DB creation during import
        self._memory = None

        workflow = StateGraph(LegalAgentState)
        workflow.add_node("load_memory", self.load_memory)
        workflow.add_node("retrieve_law", self.retrieve_law)
        workflow.add_node("rerank_context", self.rerank_context)
        workflow.add_node("legal_analyze", self.legal_analyze)
        workflow.add_node("generate_citations", self.generate_citations)
        workflow.add_node("save_memory", self.save_memory)

        workflow.add_edge(START, "load_memory")
        workflow.add_edge("load_memory", "retrieve_law")
        workflow.add_edge("retrieve_law", "rerank_context")
        workflow.add_edge("rerank_context", "legal_analyze")
        workflow.add_edge("legal_analyze", "generate_citations")
        workflow.add_edge("generate_citations", "save_memory")
        workflow.add_edge("save_memory", END)

        self.graph = workflow.compile()

    # ── Memory helpers ────────────────────────────────────

    def _get_memory(self):
        """Lazy-initialise the ConversationMemory instance."""
        if self._memory is None:
            from tn_law_agent.utils.memory import ConversationMemory
            self._memory = ConversationMemory(
                db_path=_MEMORY_DB_PATH,
                max_turns=_MEMORY_MAX_TURNS,
            )
        return self._memory

    # ── Step 0: Load Memory ───────────────────────────────

    def load_memory(self, state: LegalAgentState) -> dict:
        """
        Load prior conversation turns for the current session and convert
        them into LangChain ``BaseMessage`` objects stored in ``messages``.
        """
        session_id = state.get("session_id", "")
        if not session_id:
            logger.debug("[load_memory] no session_id — skipping memory load")
            return {"messages": []}

        logger.info("[load_memory] loading history for session=%s", session_id)
        try:
            turns = self._get_memory().load(session_id)
        except Exception as exc:
            logger.warning("[load_memory] could not load history: %s", exc)
            return {"messages": []}

        messages: List[BaseMessage] = []
        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(SystemMessage(content=content))

        logger.info("[load_memory] loaded %d prior messages", len(messages))
        return {"messages": messages}

    # ── Step 1: Retrieve ─────────────────────────────────

    def retrieve_law(self, state: LegalAgentState) -> dict:
        logger.info("[retrieve_law] query=%s domain=%s", state["query"][:80], state.get("domain"))
        docs: List[Dict] = []

        if self.vector_store is not None:
            try:
                # Prefer hybrid_search when available (BM25 + dense fusion)
                if hasattr(self.vector_store, "hybrid_search"):
                    raw = self.vector_store.hybrid_search(state["query"])
                else:
                    raw = self.vector_store.search(state["query"])
                docs = [
                    {"text": d.get("text", d), "source": d.get("source", "TCA")}
                    if isinstance(d, dict)
                    else {"text": str(d), "source": "TCA"}
                    for d in (raw or [])
                ]
            except Exception as exc:
                logger.warning("[retrieve_law] vector store error: %s", exc)

        # Fallback: inject domain profile knowledge so the LLM still has context
        if not docs:
            domain_info = TCA_DOMAIN_COVERAGE.get(
                state.get("domain", "general"), TCA_DOMAIN_COVERAGE["general"]
            )
            titles = domain_info["titles"]
            desc = domain_info["desc"]
            fallback_text = (
                f"This query relates to {desc}. "
                f"Relevant TCA titles: {', '.join(titles) or 'General'}. "
                "Apply your training knowledge of the Tennessee Code Annotated. "
                "Cite specific TCA sections where applicable."
            )
            docs = [{"text": fallback_text, "source": "TCA Domain Profile (fallback)"}]
            logger.info("[retrieve_law] using domain-profile fallback for domain=%s", state.get("domain"))

        return {"documents": docs, "status": "retrieved"}

    # ── Step 2: Rerank ───────────────────────────────────

    def rerank_context(self, state: LegalAgentState) -> dict:
        logger.info("[rerank_context] ranking %d docs", len(state.get("documents", [])))
        docs = state.get("documents", [])
        if not docs:
            return {"reranked_docs": [], "status": "reranked"}

        keywords = set(
            w.lower()
            for w in state["query"].split()
            if len(w) > 3  # skip short stop words
        )

        scored: List[tuple[float, Dict]] = []
        for doc in docs:
            text_lower = doc.get("text", "").lower()
            # Score: keyword hits + bonus for TCA citation presence
            kw_score = sum(1 for k in keywords if k in text_lower)
            citation_bonus = len(_TCA_CITATION_RE.findall(text_lower)) * 2
            scored.append((kw_score + citation_bonus, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:6]]
        return {"reranked_docs": top_docs, "status": "reranked"}

    # ── Step 3: Analyze ──────────────────────────────────

    def legal_analyze(self, state: LegalAgentState) -> dict:
        logger.info("[legal_analyze] building prompt from %d docs", len(state.get("reranked_docs", [])))
        docs = state.get("reranked_docs", [])
        domain_info = TCA_DOMAIN_COVERAGE.get(
            state.get("domain", "general"), TCA_DOMAIN_COVERAGE["general"]
        )

        context_blocks = "\n\n".join(
            f"[Source: {d.get('source', 'TCA')}]\n{d.get('text', '')}"
            for d in docs
        )

        # Build conversation history prefix if prior messages exist
        prior_messages = state.get("messages", [])
        history_prefix = ""
        if prior_messages:
            history_lines = []
            for msg in prior_messages:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                else:
                    history_lines.append(f"Assistant: {msg.content}")
            history_text = "\n".join(history_lines)
            history_prefix = (
                f"Conversation History (for context):\n{history_text}\n\n"
            )

        human_prompt = (
            f"{history_prefix}"
            f"Domain: {domain_info['desc']}\n\n"
            f"Context:\n{context_blocks or 'No specific context retrieved — use your TCA knowledge.'}\n\n"
            f"Question: {state['query']}"
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )
            analysis = response.content
        except Exception as exc:
            logger.error("[legal_analyze] LLM error: %s", exc)
            analysis = f"⚠️ Analysis failed: {exc}"

        return {"analysis": analysis, "status": "analyzed"}

    # ── Step 4: Citations ────────────────────────────────

    def generate_citations(self, state: LegalAgentState) -> dict:
        logger.info("[generate_citations] extracting citations")
        analysis = state.get("analysis", "")

        # Pull citations from the analysis text
        found = _TCA_CITATION_RE.findall(analysis)
        # Also include sources from retrieved docs that appear in the analysis
        doc_sources = [
            d.get("source", "")
            for d in state.get("reranked_docs", [])
            if d.get("source", "") in analysis
        ]
        all_citations = list(dict.fromkeys(found + doc_sources))  # dedupe, preserve order
        return {"citations": all_citations, "status": "complete"}

    # ── Step 5: Save Memory ──────────────────────────────

    def save_memory(self, state: LegalAgentState) -> dict:
        """
        Persist the current Q&A turn to the conversation memory store.
        A no-op when ``session_id`` is empty or memory save fails.
        """
        session_id = state.get("session_id", "")
        if not session_id:
            logger.debug("[save_memory] no session_id — skipping memory save")
            return {}

        query = state.get("query", "")
        analysis = state.get("analysis", "")

        if not query or not analysis:
            return {}

        logger.info("[save_memory] saving turn for session=%s", session_id)
        try:
            self._get_memory().save(
                session_id=session_id,
                user_msg=query,
                assistant_msg=analysis,
            )
        except Exception as exc:
            logger.warning("[save_memory] could not save history: %s", exc)

        return {}

    # ── Public API ───────────────────────────────────────

    def invoke(self, query: str, domain: str = "general", session_id: Optional[str] = None) -> dict:
        """
        Run the full analysis pipeline.

        Args:
            query:      The legal question.
            domain:     TCA domain hint (e.g. "criminal", "family", "property").
            session_id: Optional session identifier for multi-turn memory.
                        If omitted, no conversation history is loaded or saved.

        Returns:
            dict with keys: query, analysis, citations, status, error
        """
        initial_state: LegalAgentState = {
            "query": query,
            "domain": domain,
            "session_id": session_id or "",
            "messages": [],
            "documents": [],
            "reranked_docs": [],
            "analysis": "",
            "citations": [],
            "status": "init",
            "error": "",
        }
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as exc:
            logger.exception("[TNLawGraph.invoke] pipeline error")
            return {**initial_state, "error": str(exc), "status": "error"}
