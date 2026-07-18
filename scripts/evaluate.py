"""
TN-LawMaster — Evaluation Framework
=====================================
Golden Q&A dataset and automated accuracy scoring for the legal pipeline.

Usage:
    python scripts/evaluate.py                        # run all evals
    python scripts/evaluate.py --backend groq         # use specific backend
    python scripts/evaluate.py --sample 10            # run random 10-question sample
    python scripts/evaluate.py --output results.json  # save results to file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level="WARNING", format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# GOLDEN DATASET
# 50 curated Q&A pairs with expected TCA citations and
# key phrases that a correct answer should contain.
# ══════════════════════════════════════════════════════════

GOLDEN_DATASET: list[dict] = [
    # ── Criminal — Title 39 ────────────────────────────
    {
        "id": "crim-001",
        "domain": "criminal",
        "question": "What is the penalty for theft of property valued at $1,500 in Tennessee?",
        "expected_citations": ["TCA § 39-14-103", "39-14-103"],
        "expected_keywords": ["Class E felony", "1,000", "2,500"],
        "category": "criminal/theft",
    },
    {
        "id": "crim-002",
        "domain": "criminal",
        "question": "What is the penalty for theft of property valued at $500 in Tennessee?",
        "expected_citations": ["TCA § 39-14-103"],
        "expected_keywords": ["Class A misdemeanor", "1,000"],
        "category": "criminal/theft",
    },
    {
        "id": "crim-003",
        "domain": "criminal",
        "question": "What are the elements of domestic assault in Tennessee?",
        "expected_citations": ["TCA § 39-13-111", "39-13-101", "36-3-601"],
        "expected_keywords": ["domestic", "assault", "misdemeanor"],
        "category": "criminal/assault",
    },
    {
        "id": "crim-004",
        "domain": "criminal",
        "question": "What is the penalty for possession of marijuana under 0.5 ounces in Tennessee?",
        "expected_citations": ["TCA § 39-17-417"],
        "expected_keywords": ["misdemeanor", "marijuana", "0.5"],
        "category": "criminal/drugs",
    },
    {
        "id": "crim-005",
        "domain": "criminal",
        "question": "What is the penalty for selling a Schedule I controlled substance in Tennessee?",
        "expected_citations": ["TCA § 39-17-417"],
        "expected_keywords": ["Class B felony", "Schedule I", "30 years"],
        "category": "criminal/drugs",
    },
    {
        "id": "crim-006",
        "domain": "criminal",
        "question": "What constitutes theft of property under TCA Title 39?",
        "expected_citations": ["TCA § 39-14-103"],
        "expected_keywords": ["intent to deprive", "effective consent"],
        "category": "criminal/theft",
    },
    # ── Traffic / DUI — Title 55 ───────────────────────
    {
        "id": "traf-001",
        "domain": "traffic",
        "question": "What are the penalties for a first DUI offense in Tennessee?",
        "expected_citations": ["TCA § 55-10-401"],
        "expected_keywords": ["48 hours", "11 months", "$350", "license revocation"],
        "category": "traffic/dui",
    },
    {
        "id": "traf-002",
        "domain": "traffic",
        "question": "What is the blood alcohol content limit for DUI in Tennessee?",
        "expected_citations": ["TCA § 55-10-401"],
        "expected_keywords": ["0.08", "eight-hundredths"],
        "category": "traffic/dui",
    },
    {
        "id": "traf-003",
        "domain": "traffic",
        "question": "What are the penalties for a fourth DUI offense in Tennessee?",
        "expected_citations": ["TCA § 55-10-401"],
        "expected_keywords": ["Class E felony", "fourth"],
        "category": "traffic/dui",
    },
    {
        "id": "traf-004",
        "domain": "traffic",
        "question": "What is the penalty for a second DUI in Tennessee?",
        "expected_citations": ["TCA § 55-10-401"],
        "expected_keywords": ["45 days", "$600", "2 years", "second"],
        "category": "traffic/dui",
    },
    # ── Family — Title 36 ──────────────────────────────
    {
        "id": "fam-001",
        "domain": "family",
        "question": "What are the grounds for divorce in Tennessee?",
        "expected_citations": ["TCA § 36-4-101"],
        "expected_keywords": ["irreconcilable differences", "adultery", "desertion"],
        "category": "family/divorce",
    },
    {
        "id": "fam-002",
        "domain": "family",
        "question": "How does a Tennessee court determine child custody?",
        "expected_citations": ["TCA § 36-6-106"],
        "expected_keywords": ["best interest", "parenting", "stability"],
        "category": "family/custody",
    },
    {
        "id": "fam-003",
        "domain": "family",
        "question": "When does child support end in Tennessee?",
        "expected_citations": ["TCA § 36-5-101"],
        "expected_keywords": ["18", "21", "high school"],
        "category": "family/support",
    },
    {
        "id": "fam-004",
        "domain": "family",
        "question": "How is child support calculated in Tennessee?",
        "expected_citations": ["TCA § 36-5-101"],
        "expected_keywords": ["Income Shares", "gross income", "guidelines"],
        "category": "family/support",
    },
    {
        "id": "fam-005",
        "domain": "family",
        "question": "What factors does a Tennessee court consider for child custody best interest?",
        "expected_citations": ["TCA § 36-6-106"],
        "expected_keywords": ["love", "affection", "moral", "primary caregiver"],
        "category": "family/custody",
    },
    # ── Property / Landlord-Tenant — Title 66 ─────────
    {
        "id": "prop-001",
        "domain": "property",
        "question": "What are a landlord's maintenance obligations under Tennessee law?",
        "expected_citations": ["TCA § 66-28-201"],
        "expected_keywords": ["habitable", "plumbing", "heating", "electrical"],
        "category": "property/landlord-tenant",
    },
    {
        "id": "prop-002",
        "domain": "property",
        "question": "How much notice must a Tennessee landlord give before eviction for nonpayment?",
        "expected_citations": ["TCA § 66-28-505"],
        "expected_keywords": ["14", "pay or vacate", "nonpayment"],
        "category": "property/landlord-tenant",
    },
    {
        "id": "prop-003",
        "domain": "property",
        "question": "How much notice is required to terminate a month-to-month tenancy in Tennessee?",
        "expected_citations": ["TCA § 66-28-505"],
        "expected_keywords": ["30 days", "month-to-month"],
        "category": "property/landlord-tenant",
    },
    {
        "id": "prop-004",
        "domain": "property",
        "question": "What notice is required for a week-to-week tenancy in Tennessee?",
        "expected_citations": ["TCA § 66-28-505"],
        "expected_keywords": ["10 days", "week-to-week"],
        "category": "property/landlord-tenant",
    },
    # ── Business — Title 48 ────────────────────────────
    {
        "id": "biz-001",
        "domain": "business",
        "question": "How do you form an LLC in Tennessee?",
        "expected_citations": ["TCA § 48-249-101"],
        "expected_keywords": ["articles of organization", "Secretary of State", "registered agent"],
        "category": "business/llc",
    },
    {
        "id": "biz-002",
        "domain": "business",
        "question": "What are the annual filing requirements for a Tennessee LLC?",
        "expected_citations": ["TCA § 48-249-101"],
        "expected_keywords": ["annual report", "April 1", "$300"],
        "category": "business/llc",
    },
    {
        "id": "biz-003",
        "domain": "business",
        "question": "Does an LLC protect members from personal liability in Tennessee?",
        "expected_citations": ["TCA § 48-249-101"],
        "expected_keywords": ["limited liability", "personally liable", "protection"],
        "category": "business/llc",
    },
    # ── Torts — Title 29 ───────────────────────────────
    {
        "id": "tort-001",
        "domain": "torts",
        "question": "What is the cap on non-economic damages in Tennessee personal injury cases?",
        "expected_citations": ["TCA § 29-39-102"],
        "expected_keywords": ["750,000", "$750"],
        "category": "torts/damages",
    },
    {
        "id": "tort-002",
        "domain": "torts",
        "question": "What is the increased damages cap for catastrophic injury in Tennessee?",
        "expected_citations": ["TCA § 29-39-102"],
        "expected_keywords": ["1,000,000", "$1 million", "catastrophic"],
        "category": "torts/damages",
    },
    {
        "id": "tort-003",
        "domain": "torts",
        "question": "Does comparative fault bar recovery in Tennessee?",
        "expected_citations": ["TCA § 29-11-103", "29-39-102"],
        "expected_keywords": ["50%", "comparative fault", "barred"],
        "category": "torts/liability",
    },
    {
        "id": "tort-004",
        "domain": "torts",
        "question": "When does Tennessee's non-economic damages cap not apply?",
        "expected_citations": ["TCA § 29-39-102"],
        "expected_keywords": ["intentional", "alcohol", "felony"],
        "category": "torts/damages",
    },
    # ── Public Records — Title 10 ──────────────────────
    {
        "id": "tipa-001",
        "domain": "tipa",
        "question": "What is the public's right to access government records in Tennessee?",
        "expected_citations": ["TCA § 10-7-503"],
        "expected_keywords": ["citizen", "inspection", "business hours"],
        "category": "tipa/records",
    },
    {
        "id": "tipa-002",
        "domain": "tipa",
        "question": "What must a Tennessee records custodian do if they deny a records request?",
        "expected_citations": ["TCA § 10-7-503"],
        "expected_keywords": ["writing", "exemption", "specific"],
        "category": "tipa/records",
    },
    {
        "id": "tipa-003",
        "domain": "tipa",
        "question": "What records are exempt from public disclosure in Tennessee?",
        "expected_citations": ["TCA § 10-7-503"],
        "expected_keywords": ["personnel", "medical", "attorney-client", "trade secrets"],
        "category": "tipa/records",
    },
]


# ══════════════════════════════════════════════════════════
# Scoring Engine
# ══════════════════════════════════════════════════════════

def _score_response(result: dict, golden: dict) -> dict:
    """
    Score a pipeline response against the golden ground truth.

    Returns:
        dict with keys: citation_score, keyword_score, total_score, details
    """
    analysis = result.get("analysis", "").lower()
    citations_found = [c.lower() for c in result.get("citations", [])]
    analysis_lower = analysis

    # Citation score: fraction of expected citations found
    expected_cites = golden.get("expected_citations", [])
    cite_hits = sum(
        1 for ec in expected_cites
        if ec.lower() in analysis_lower or any(ec.lower() in c for c in citations_found)
    )
    citation_score = cite_hits / len(expected_cites) if expected_cites else 1.0

    # Keyword score: fraction of expected keywords found
    expected_kw = golden.get("expected_keywords", [])
    kw_hits = sum(1 for kw in expected_kw if kw.lower() in analysis_lower)
    keyword_score = kw_hits / len(expected_kw) if expected_kw else 1.0

    # Total weighted score
    total_score = 0.6 * citation_score + 0.4 * keyword_score

    return {
        "citation_score": round(citation_score, 3),
        "keyword_score": round(keyword_score, 3),
        "total_score": round(total_score, 3),
        "citation_hits": f"{cite_hits}/{len(expected_cites)}",
        "keyword_hits": f"{kw_hits}/{len(expected_kw)}",
        "passed": total_score >= 0.5,
        "error": result.get("error", ""),
    }


def _print_result(golden: dict, score: dict, latency: float, verbose: bool = False) -> None:
    icon = "✅" if score["passed"] else "❌"
    print(
        f"  {icon} [{golden['id']}] {golden['question'][:60]}..."
        f"\n     Score: {score['total_score']:.2f}  "
        f"Citations: {score['citation_hits']}  "
        f"Keywords: {score['keyword_hits']}  "
        f"({latency:.1f}s)"
    )
    if not score["passed"] or verbose:
        if score.get("error"):
            print(f"     ⚠️  Error: {score['error']}")


# ══════════════════════════════════════════════════════════
# Main Evaluation Runner
# ══════════════════════════════════════════════════════════

def run_evaluation(
    backend: Optional[str] = None,
    sample: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Run the evaluation suite and return a summary dict.
    """
    from model_config import get_llm, get_active_model_name
    from tn_law_agent.core import TNLawAgent

    if backend:
        os.environ["ACTIVE_BACKEND"] = backend

    print(f"\n{'═'*60}")
    print(f"  TN-LawMaster Evaluation Suite")
    print(f"  Model: {get_active_model_name()}")
    print(f"  Questions: {len(GOLDEN_DATASET)} total")
    print(f"{'═'*60}\n")

    llm = get_llm()
    agent = TNLawAgent(llm=llm)

    dataset = GOLDEN_DATASET
    if sample:
        import random
        dataset = random.sample(GOLDEN_DATASET, min(sample, len(GOLDEN_DATASET)))

    results = []
    passed = 0
    total_score = 0.0

    # Group by category for display
    categories: dict[str, list] = {}
    for golden in dataset:
        cat = golden["category"].split("/")[0]
        categories.setdefault(cat, []).append(golden)

    for cat, items in categories.items():
        print(f"\n📂 {cat.upper()} ({len(items)} questions)")
        for golden in items:
            t0 = time.perf_counter()
            try:
                result = agent.analyze(golden["question"], domain=golden["domain"])
            except Exception as exc:
                result = {"analysis": "", "citations": [], "error": str(exc), "status": "error"}
            latency = time.perf_counter() - t0

            score = _score_response(result, golden)
            _print_result(golden, score, latency, verbose=verbose)

            results.append({
                "id": golden["id"],
                "domain": golden["domain"],
                "category": golden["category"],
                "question": golden["question"],
                "score": score,
                "latency_s": round(latency, 2),
                "analysis_excerpt": result.get("analysis", "")[:300],
                "citations_found": result.get("citations", []),
            })

            if score["passed"]:
                passed += 1
            total_score += score["total_score"]

    n = len(results)
    avg_score = total_score / n if n else 0
    pass_rate = passed / n if n else 0

    summary = {
        "model": get_active_model_name(),
        "total_questions": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": round(pass_rate, 3),
        "avg_score": round(avg_score, 3),
        "results": results,
    }

    print(f"\n{'═'*60}")
    print(f"  RESULTS: {passed}/{n} passed  ({pass_rate*100:.1f}%)")
    print(f"  Avg Score: {avg_score:.3f}  (citation×0.6 + keyword×0.4)")
    grade = "🏆 Excellent" if avg_score >= 0.8 else "✅ Good" if avg_score >= 0.6 else "⚠️  Needs work" if avg_score >= 0.4 else "❌ Poor"
    print(f"  Grade: {grade}")
    print(f"{'═'*60}\n")

    if output_path:
        Path(output_path).write_text(json.dumps(summary, indent=2))
        print(f"💾 Results saved to {output_path}\n")

    return summary


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TN-LawMaster evaluation suite")
    p.add_argument("--backend", choices=["ollama", "groq", "openai", "openrouter"],
                   help="LLM backend to use")
    p.add_argument("--sample", type=int, metavar="N",
                   help="Run a random N-question sample instead of all 28")
    p.add_argument("--output", metavar="FILE",
                   help="Save JSON results to FILE")
    p.add_argument("--verbose", action="store_true",
                   help="Show full analysis for failed questions")
    p.add_argument("--list", action="store_true",
                   help="List all golden questions without running eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print(f"\n{'─'*70}")
        print(f"  Golden Dataset — {len(GOLDEN_DATASET)} Questions")
        print(f"{'─'*70}")
        for q in GOLDEN_DATASET:
            print(f"  [{q['id']}] ({q['domain']}) {q['question']}")
        return

    run_evaluation(
        backend=args.backend,
        sample=args.sample,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
