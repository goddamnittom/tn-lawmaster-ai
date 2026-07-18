"""
TN-LawMaster — LangGraph Legal Analysis Workflow
================================================
Production-grade RAG + LangGraph pipeline for Tennessee law analysis.

Pipeline stages:
  1. retrieve_law      — Vector store lookup (or TCA keyword fallback)
  2. rerank_context    — Relevance-score and top-k filter
  3. legal_analyze     — LLM grounded analysis with strict TCA prompt
  4. generate_citations— Audit trail of referenced TCA statutes

Author: TN-LawMaster Project
"""

from __future__ import annotations

import re
import logging
import operator
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
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


# ── LangGraph State ───────────────────────────────────────

class LegalAgentState(TypedDict):
    query: str
    domain: str                                         # e.g. "criminal", "family"
    documents: Annotated[List[Dict], operator.add]     # accumulates across retries
    reranked_docs: List[Dict]
    analysis: str
    citations: List[str]
    status: str
    error: str


# ── Graph Class ───────────────────────────────────────────

class TNLawGraph:
    """LangGraph-based Tennessee law analysis pipeline."""

    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store

        workflow = StateGraph(LegalAgentState)
        workflow.add_node("retrieve_law", self.retrieve_law)
        workflow.add_node("rerank_context", self.rerank_context)
        workflow.add_node("legal_analyze", self.legal_analyze)
        workflow.add_node("generate_citations", self.generate_citations)

        workflow.add_edge(START, "retrieve_law")
        workflow.add_edge("retrieve_law", "rerank_context")
        workflow.add_edge("rerank_context", "legal_analyze")
        workflow.add_edge("legal_analyze", "generate_citations")
        workflow.add_edge("generate_citations", END)

        self.graph = workflow.compile()

    # ── Step 1: Retrieve ─────────────────────────────────

    def retrieve_law(self, state: LegalAgentState) -> dict:
        logger.info("[retrieve_law] query=%s domain=%s", state["query"][:80], state.get("domain"))
        docs: List[Dict] = []

        if self.vector_store is not None:
            try:
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
        human_prompt = (
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

    # ── Public API ───────────────────────────────────────

    def invoke(self, query: str, domain: str = "general") -> dict:
        """
        Run the full analysis pipeline.

        Args:
            query:  The legal question.
            domain: TCA domain hint (e.g. "criminal", "family", "property").

        Returns:
            dict with keys: query, analysis, citations, status, error
        """
        initial_state: LegalAgentState = {
            "query": query,
            "domain": domain,
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
