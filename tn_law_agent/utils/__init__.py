"""
TN-LawMaster — Utility Package
================================
Shared utilities for the TN-LawMaster AI agent.

Modules:
    citation_verifier — Verifies TCA citations exist in the corpus before surfacing them.
    memory            — SQLite-backed multi-turn conversation memory.
"""

from .citation_verifier import CitationVerifier
from .memory import ConversationMemory

__all__ = ["CitationVerifier", "ConversationMemory"]
