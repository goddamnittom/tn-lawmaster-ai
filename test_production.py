"""
TN-LawMaster — Test Suite
===========================
Unit and integration tests for the LangGraph pipeline.

Designed to run without a real LLM or vector store by using mocks.
Integration tests run when ACTIVE_BACKEND env var is configured and
a real LLM is available.

Run:
    pytest tests/ -v
    pytest tests/ -v -m unit        # only fast unit tests
    pytest tests/ -v -m integration # requires real LLM
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Mock LLM ─────────────────────────────────────────────

class MockLLM:
    """Deterministic fake LLM for unit tests — no API calls."""

    def invoke(self, messages):
        response = MagicMock()
        response.content = (
            "**Summary**: Under TCA § 39-14-105, theft of property valued between $1,000 "
            "and $2,500 is a Class E felony in Tennessee.\n\n"
            "**Applicable Statutes**:\n- TCA § 39-14-105 — Theft of property\n\n"
            "**Analysis**: Tennessee's theft statute grades the offense by the value of "
            "the stolen property. Felony theft begins at $1,000 (Class E), escalating to "
            "Class A felony for amounts exceeding $250,000.\n\n"
            "**⚠️ Legal Disclaimer**: This is AI-generated legal information, not legal advice."
        )
        return response


# ── Mock Vector Store ─────────────────────────────────────

class MockVectorStore:
    """Returns deterministic docs for unit tests."""

    def search(self, query: str) -> List[Dict]:
        return [
            {
                "text": "TCA § 39-14-105 establishes theft of property as Class E felony for values $1,000–$2,500.",
                "source": "TCA § 39-14-105",
                "distance": 0.12,
            },
            {
                "text": "Tennessee criminal penalties for property crimes scale with value under Title 39.",
                "source": "TCA Title 39",
                "distance": 0.25,
            },
        ]


# ════════════════════════════════════════════════════════
# Unit Tests
# ════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTNLawGraph:
    """Unit tests for the LangGraph pipeline nodes."""

    def setup_method(self):
        from tn_law_agent.workflows.legal_graph import TNLawGraph
        self.graph = TNLawGraph(llm=MockLLM(), vector_store=MockVectorStore())

    def test_retrieve_law_with_vector_store(self):
        state = {
            "query": "What is the penalty for theft?",
            "domain": "criminal",
            "documents": [],
            "reranked_docs": [],
            "analysis": "",
            "citations": [],
            "status": "init",
            "error": "",
        }
        result = self.graph.retrieve_law(state)
        assert result["status"] == "retrieved"
        assert len(result["documents"]) >= 1

    def test_retrieve_law_fallback_no_vector_store(self):
        from tn_law_agent.workflows.legal_graph import TNLawGraph
        graph = TNLawGraph(llm=MockLLM(), vector_store=None)
        state = {
            "query": "DUI penalties",
            "domain": "traffic",
            "documents": [],
            "reranked_docs": [],
            "analysis": "",
            "citations": [],
            "status": "init",
            "error": "",
        }
        result = graph.retrieve_law(state)
        assert result["status"] == "retrieved"
        assert len(result["documents"]) == 1  # fallback doc
        assert "TCA" in result["documents"][0]["text"]

    def test_rerank_context_scores_by_keyword(self):
        state = {
            "query": "theft penalty",
            "domain": "criminal",
            "documents": [
                {"text": "Theft is penalized under TCA § 39-14-105", "source": "TCA § 39-14-105"},
                {"text": "Unrelated domestic law provision", "source": "TCA § 36-1-1"},
            ],
            "reranked_docs": [],
            "analysis": "",
            "citations": [],
            "status": "retrieved",
            "error": "",
        }
        result = self.graph.rerank_context(state)
        assert result["status"] == "reranked"
        assert len(result["reranked_docs"]) >= 1
        # theft/penalty doc should rank first
        assert "39-14-105" in result["reranked_docs"][0]["source"]

    def test_legal_analyze_returns_analysis(self):
        state = {
            "query": "What is the penalty for theft?",
            "domain": "criminal",
            "documents": [],
            "reranked_docs": [
                {"text": "TCA § 39-14-105 Class E felony for $1k–$2.5k theft", "source": "TCA § 39-14-105"}
            ],
            "analysis": "",
            "citations": [],
            "status": "reranked",
            "error": "",
        }
        result = self.graph.legal_analyze(state)
        assert result["status"] == "analyzed"
        assert len(result["analysis"]) > 50

    def test_generate_citations_extracts_tca_refs(self):
        state = {
            "query": "theft",
            "domain": "criminal",
            "documents": [],
            "reranked_docs": [{"source": "TCA § 39-14-105", "text": "..."}],
            "analysis": "Under TCA § 39-14-105 the penalty is a Class E felony.",
            "citations": [],
            "status": "analyzed",
            "error": "",
        }
        result = self.graph.generate_citations(state)
        assert "TCA § 39-14-105" in result["citations"]

    def test_full_invoke_pipeline(self):
        result = self.graph.invoke("What is the penalty for theft in Tennessee?", domain="criminal")
        assert result["status"] == "complete"
        assert result["analysis"]
        assert isinstance(result["citations"], list)
        assert result["query"] == "What is the penalty for theft in Tennessee?"


@pytest.mark.unit
class TestTNLawAgent:
    """Unit tests for the high-level agent wrapper."""

    def setup_method(self):
        from tn_law_agent.core import TNLawAgent
        self.agent = TNLawAgent(llm=MockLLM(), vector_store=MockVectorStore())

    def test_analyze_returns_expected_keys(self):
        result = self.agent.analyze("What is DUI penalty in TN?", domain="traffic")
        assert "query" in result
        assert "analysis" in result
        assert "citations" in result
        assert "status" in result

    def test_covered_domains_includes_criminal(self):
        assert "criminal" in self.agent.covered_domains
        assert "family" in self.agent.covered_domains

    def test_unknown_domain_falls_back_to_general(self):
        result = self.agent.analyze("test query", domain="nonexistent_domain_xyz")
        assert result["status"] != "error"

    def test_attach_vector_store(self):
        new_vs = MockVectorStore()
        self.agent.attach_vector_store(new_vs)
        assert self.agent.vector_store is new_vs
        assert self.agent.graph.vector_store is new_vs


@pytest.mark.unit
class TestModelConfig:
    """Unit tests for model_config without real API calls."""

    def test_unknown_backend_raises(self):
        from model_config import get_llm
        with pytest.raises(ValueError, match="Unknown backend"):
            get_llm("invalid_backend_xyz")

    def test_missing_groq_key_raises(self):
        from model_config import _build_groq_llm
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
                _build_groq_llm()

    def test_get_active_model_name(self):
        from model_config import get_active_model_name
        with patch.dict(os.environ, {"ACTIVE_BACKEND": "groq", "GROQ_MODEL": "llama3-70b-8192"}):
            label = get_active_model_name("groq")
            assert "Groq" in label
            assert "llama3-70b-8192" in label


# ════════════════════════════════════════════════════════
# Integration Tests (require real LLM)
# ════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ACTIVE_BACKEND"),
    reason="Set ACTIVE_BACKEND to run integration tests.",
)
class TestIntegration:
    """End-to-end tests against a real LLM."""

    def setup_method(self):
        from model_config import get_llm
        from tn_law_agent.core import TNLawAgent
        self.agent = TNLawAgent(llm=get_llm())

    def test_theft_query(self):
        result = self.agent.analyze(
            "What is the penalty for theft of property valued at $1,500 in Tennessee?",
            domain="criminal",
        )
        assert result["status"] == "complete"
        assert len(result["analysis"]) > 100

    def test_dui_query(self):
        result = self.agent.analyze(
            "What are the DUI penalties in Tennessee for a first offense?",
            domain="traffic",
        )
        assert result["status"] == "complete"

    def test_custody_query(self):
        result = self.agent.analyze(
            "How is child custody determined in Tennessee?",
            domain="family",
        )
        assert result["status"] == "complete"
