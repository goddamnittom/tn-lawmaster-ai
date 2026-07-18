"""
TN-LawMaster — Interactive CLI Runner
=======================================
Quick way to query the Tennessee legal analysis engine from your terminal.

Usage:
    python run_agent.py
    python run_agent.py --backend groq --domain criminal
    python run_agent.py --query "What is the penalty for DUI?" --domain traffic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format="%(levelname)s | %(message)s",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TN-LawMaster CLI — Tennessee legal analysis engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Backends:  ollama (default) | groq | openai | openrouter
            Domains:   general | criminal | family | property | business
                       torts | estates | traffic | tipa

            Set ACTIVE_BACKEND in .env or pass --backend.
            """
        ),
    )
    p.add_argument("--backend", choices=["ollama", "groq", "openai", "openrouter"],
                   default=None, help="LLM backend override.")
    p.add_argument("--domain", default="general", help="TCA domain hint.")
    p.add_argument("--query", default=None, help="Run a single query then exit.")
    return p.parse_args()


def _fmt_analysis(result: dict, width: int = 88) -> str:
    """Pretty-print the analysis result."""
    lines = []
    lines.append("\n" + "═" * width)
    lines.append("  ⚖️  TN-LawMaster Analysis")
    lines.append("═" * width)
    analysis = result.get("analysis", "No analysis returned.")
    for line in analysis.splitlines():
        lines.append(textwrap.fill(line, width=width) if len(line) > width else line)
    citations = result.get("citations", [])
    if citations:
        lines.append("\n📌  Citations:")
        for c in citations:
            lines.append(f"   • {c}")
    status = result.get("status", "")
    error = result.get("error", "")
    if error:
        lines.append(f"\n⚠️  Error: {error}")
    lines.append("═" * width + "\n")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()

    # Apply backend override
    if args.backend:
        os.environ["ACTIVE_BACKEND"] = args.backend

    from model_config import get_llm, get_active_model_name, ACTIVE_BACKEND
    from tn_law_agent.core import TNLawAgent

    backend_label = args.backend or ACTIVE_BACKEND
    print(f"\n✅ TN-LawMaster — backend: {get_active_model_name(backend_label)}")
    print(f"   Domain: {args.domain}")
    print("   Type 'exit' or 'quit' to stop.  Type 'domain <name>' to switch domains.\n")

    try:
        llm = get_llm(backend_label)
        agent = TNLawAgent(llm=llm)
    except Exception as exc:
        print(f"\n❌ Failed to initialize agent: {exc}")
        sys.exit(1)

    current_domain = args.domain

    # One-shot mode
    if args.query:
        result = agent.analyze(args.query, domain=current_domain)
        print(_fmt_analysis(result))
        return

    # Interactive REPL
    while True:
        try:
            q = input(f"[{current_domain}] Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Exiting TN-LawMaster.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("👋  Goodbye.")
            break
        if q.lower().startswith("domain "):
            new_domain = q.split(" ", 1)[1].strip()
            current_domain = new_domain
            print(f"   Domain switched to: {current_domain}\n")
            continue

        result = agent.analyze(q, domain=current_domain)
        print(_fmt_analysis(result))


if __name__ == "__main__":
    main()