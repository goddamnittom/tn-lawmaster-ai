"""
TN-LawMaster — Core Agent Interface
====================================
High-level wrapper around the LangGraph pipeline, with optional
vector store injection and domain-aware analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

from .workflows.legal_graph import TNLawGraph, TCA_DOMAIN_COVERAGE

logger = logging.getLogger(__name__)

__all__ = ["TNLawAgent"]


class TNLawAgent:
    """
    Main entry point for the TN-LawMaster AI agent.

    Usage::

        from model_config import get_llm
        from tn_law_agent.core import TNLawAgent

        agent = TNLawAgent(llm=get_llm())
        result = agent.analyze("What is the penalty for DUI in Tennessee?", domain="criminal")

        print(result["analysis"])
        print("Citations:", result["citations"])
    """

    def __init__(self, llm, vector_store=None):
        """
        Args:
            llm: Any LangChain-compatible chat model (Ollama, Groq, OpenAI, etc.)
            vector_store: Optional object with a `.search(query) -> List[dict]` method.
                          If None, the agent falls back to domain-profile context.
        """
        self.llm = llm
        self.vector_store = vector_store
        self.graph = TNLawGraph(llm=llm, vector_store=vector_store)
        logger.info("TNLawAgent initialized (vector_store=%s)", vector_store is not None)

    def analyze(self, query: str, domain: str = "general") -> dict:
        """
        Analyze a Tennessee law question.

        Args:
            query:  The legal question (plain English or formal legal language).
            domain: TCA domain hint. One of:
                    criminal, family, property, business, torts,
                    estates, traffic, tipa, general (default)

        Returns:
            A dict with:
                - ``query``     — original query
                - ``analysis``  — LLM-generated grounded legal analysis
                - ``citations`` — list of TCA citations found in analysis
                - ``status``    — pipeline status (``"complete"`` or ``"error"``)
                - ``error``     — error message if status == ``"error"``
        """
        if domain not in TCA_DOMAIN_COVERAGE:
            logger.warning("Unknown domain '%s', falling back to 'general'", domain)
            domain = "general"

        logger.info("Analyzing query (domain=%s): %s", domain, query[:120])
        return self.graph.invoke(query=query, domain=domain)

    @property
    def covered_domains(self) -> list[str]:
        """Return the list of supported TCA domain slugs."""
        return list(TCA_DOMAIN_COVERAGE.keys())

    def attach_vector_store(self, vector_store) -> None:
        """
        Attach (or replace) the vector store after initialization.

        This is useful in the Streamlit UI where the vector store may
        be loaded lazily after the agent is created.
        """
        self.vector_store = vector_store
        self.graph.vector_store = vector_store
        logger.info("Vector store attached: %s", type(vector_store).__name__)