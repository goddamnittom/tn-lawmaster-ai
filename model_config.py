"""
TN-LawMaster — Multi-Backend Model Switcher
==========================================
Supports: Ollama (local) · Groq · OpenAI · OpenRouter

Configure ACTIVE_BACKEND via environment variable or .env file.
All secrets are read from environment — never hardcoded here.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# Backend selection — override via .env or shell export
# ============================================================
ACTIVE_BACKEND: str = os.getenv("ACTIVE_BACKEND", "ollama").lower()

# ──────────────────────────────────────────────────────────
# Ollama (local inference)
# ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")

OLLAMA_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma3:1b": {"num_ctx": 8192, "temperature": 0.1, "num_thread": 6},
    "gemma2:2b": {"num_ctx": 4096, "temperature": 0.1, "num_thread": 4},
    "llama3.2:3b": {"num_ctx": 8192, "temperature": 0.1, "num_thread": 6},
    "mistral:7b": {"num_ctx": 8192, "temperature": 0.1, "num_thread": 8},
    # Fallback for any model not explicitly listed
    "_default": {"num_ctx": 4096, "temperature": 0.1, "num_thread": 4},
}

# ──────────────────────────────────────────────────────────
# Groq (fast cloud inference)
# ──────────────────────────────────────────────────────────
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# ──────────────────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────────────────
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ──────────────────────────────────────────────────────────
# OpenRouter (proxy for many model providers)
# ──────────────────────────────────────────────────────────
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")


def _build_ollama_llm():
    """Build a ChatOllama instance using the configured model."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for the ollama backend. "
            "Install with: pip install langchain-ollama"
        ) from exc

    cfg = OLLAMA_CONFIGS.get(OLLAMA_MODEL, OLLAMA_CONFIGS["_default"])
    logger.info("Loading Ollama model: %s", OLLAMA_MODEL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=cfg["temperature"],
        num_ctx=cfg["num_ctx"],
        num_thread=cfg["num_thread"],
    )


def _build_groq_llm():
    """Build a ChatGroq instance."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "langchain-groq is required for the groq backend. "
            "Install with: pip install langchain-groq"
        ) from exc

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    logger.info("Loading Groq model: %s", GROQ_MODEL)
    return ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=api_key)


def _build_openai_llm():
    """Build a ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for the openai backend. "
            "Install with: pip install langchain-openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    logger.info("Loading OpenAI model: %s", OPENAI_MODEL)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=api_key)


def _build_openrouter_llm():
    """Build a ChatOpenAI instance pointed at OpenRouter's API."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for the openrouter backend. "
            "Install with: pip install langchain-openai"
        ) from exc

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    logger.info("Loading OpenRouter model: %s", OPENROUTER_MODEL)
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=0.1,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/goddamnittom/tn-lawmaster-ai",
            "X-Title": "TN-LawMaster",
        },
    )


_BACKEND_BUILDERS = {
    "ollama": _build_ollama_llm,
    "groq": _build_groq_llm,
    "openai": _build_openai_llm,
    "openrouter": _build_openrouter_llm,
}


def get_llm(backend: str | None = None):
    """
    Return an LLM instance for the specified (or configured) backend.

    Args:
        backend: One of ``ollama``, ``groq``, ``openai``, ``openrouter``.
                 Defaults to the ``ACTIVE_BACKEND`` environment variable.

    Raises:
        ValueError: If the requested backend is unknown.
        EnvironmentError: If a required API key is missing.
        ImportError: If a required package is not installed.
    """
    target = (backend or ACTIVE_BACKEND).lower()
    builder = _BACKEND_BUILDERS.get(target)
    if builder is None:
        raise ValueError(
            f"Unknown backend: '{target}'. "
            f"Choose from: {', '.join(_BACKEND_BUILDERS)}"
        )
    return builder()


def get_active_model_name(backend: str | None = None) -> str:
    """Return a human-readable label for the active model."""
    target = (backend or ACTIVE_BACKEND).lower()
    return {
        "ollama": f"Ollama / {OLLAMA_MODEL}",
        "groq": f"Groq / {GROQ_MODEL}",
        "openai": f"OpenAI / {OPENAI_MODEL}",
        "openrouter": f"OpenRouter / {OPENROUTER_MODEL}",
    }.get(target, target)