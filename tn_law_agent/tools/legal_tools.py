"""
TN-LawMaster — Legal Tools for Agentic Use
============================================
LangChain-compatible tools the agent can call to augment analysis with
real-time lookups, deadline calculations, and case law searches.

Tools:
  - calculate_statute_of_limitations  TCA Title 28 deadline calculator
  - lookup_tca_section                Section number → context lookup
  - search_tn_case_law                CourtListener API (TN opinions)
  - get_tennessee_legal_forms         Official TN court form directory
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try importing LangChain tool decorator ────────────────
try:
    from langchain_core.tools import tool as _lc_tool
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    def _lc_tool(fn):          # type: ignore[misc]
        return fn

__all__ = [
    "calculate_statute_of_limitations",
    "lookup_tca_section",
    "search_tn_case_law",
    "get_tennessee_legal_forms",
    "ALL_TOOLS",
]

# ── SOL periods ───────────────────────────────────────────
# Source: TCA Title 28 — Limitations of Actions
_SOL_TABLE: dict[str, dict] = {
    "personal_injury":      {"years": 1,  "tca": "TCA § 28-3-104", "notes": "1 year from injury date"},
    "wrongful_death":       {"years": 1,  "tca": "TCA § 28-3-104", "notes": "1 year from death"},
    "libel_slander":        {"months": 6, "tca": "TCA § 28-3-103", "notes": "6 months from publication"},
    "fraud":                {"years": 3,  "tca": "TCA § 28-3-105", "notes": "3 years from discovery"},
    "property_damage":      {"years": 3,  "tca": "TCA § 28-3-105", "notes": "3 years from damage"},
    "written_contract":     {"years": 6,  "tca": "TCA § 28-3-109", "notes": "6 years from breach"},
    "oral_contract":        {"years": 6,  "tca": "TCA § 28-3-109", "notes": "6 years from breach"},
    "medical_malpractice":  {"years": 1,  "tca": "TCA § 29-26-116", "notes": "1 year from discovery (max 3 years)"},
    "consumer_protection":  {"years": 1,  "tca": "TCA § 47-18-110", "notes": "1 year from violation"},
    "employment":           {"years": 1,  "tca": "TCA § 28-3-104", "notes": "1 year (THRA); federal EEOC 180/300 days"},
}

# ── TCA chapter context ───────────────────────────────────
_TCA_TITLE_MAP: dict[str, str] = {
    "10": "Public Libraries and Archives (TIPA / Public Records)",
    "28": "Limitations of Actions (Statutes of Limitations)",
    "29": "Remedies and Special Proceedings (Torts, Damages)",
    "30": "Administration of Estates",
    "31": "Descent and Distribution",
    "32": "Wills",
    "36": "Domestic Relations (Family Law)",
    "39": "Criminal Offenses",
    "47": "Commercial Instruments and Transactions",
    "48": "Corporations and Associations (Business Entities)",
    "55": "Motor and Other Vehicles (Traffic Law)",
    "66": "Property (Real Estate, Landlord-Tenant)",
}

# ── Legal Forms directory ─────────────────────────────────
_FORMS_DIRECTORY: dict[str, dict] = {
    "divorce": {
        "source": "Tennessee Administrative Office of the Courts (AOC)",
        "url": "https://www.tncourts.gov/programs/self-help-center/forms",
        "forms": [
            "Petition for Divorce (Civil Summons)",
            "Marital Dissolution Agreement (MDA)",
            "Permanent Parenting Plan",
            "Child Support Worksheet",
            "Final Decree of Divorce",
        ],
        "tca_ref": "TCA § 36-4-101",
    },
    "custody": {
        "source": "Tennessee AOC Self-Help Center",
        "url": "https://www.tncourts.gov/programs/self-help-center/forms",
        "forms": [
            "Petition for Custody",
            "Permanent Parenting Plan Order",
            "Agreed Order Modifying Parenting Plan",
        ],
        "tca_ref": "TCA § 36-6-106",
    },
    "small_claims": {
        "source": "Tennessee General Sessions Court",
        "url": "https://www.tncourts.gov/courts/general-sessions-courts",
        "forms": [
            "Civil Warrant / Summons (GS-1)",
            "Judgment Form",
        ],
        "tca_ref": "TCA § 16-15-501 (limit $25,000)",
    },
    "eviction": {
        "source": "Tennessee General Sessions Court",
        "url": "https://www.tncourts.gov/programs/self-help-center/forms",
        "forms": [
            "Detainer Warrant (unlawful detainer)",
            "Notice to Vacate / Notice to Pay or Quit",
        ],
        "tca_ref": "TCA § 66-28-505",
    },
    "llc_formation": {
        "source": "Tennessee Secretary of State",
        "url": "https://sos.tn.gov/businesses/forms",
        "forms": [
            "SS-4270 — Articles of Organization (Domestic LLC)",
            "SS-4271 — Annual Report for LLC",
        ],
        "tca_ref": "TCA § 48-249-101",
        "fee": "$300 filing fee",
    },
    "name_change": {
        "source": "Tennessee Chancery Court",
        "url": "https://www.tncourts.gov/programs/self-help-center/forms",
        "forms": [
            "Petition for Change of Name",
            "Order Changing Name",
        ],
        "tca_ref": "TCA § 29-8-101",
    },
    "expungement": {
        "source": "Tennessee Criminal Court",
        "url": "https://www.tncourts.gov/programs/self-help-center/forms",
        "forms": [
            "Petition for Expungement of Criminal Records",
        ],
        "tca_ref": "TCA § 40-32-101",
    },
    "tipa_request": {
        "source": "Tennessee Attorney General (guidance)",
        "url": "https://www.tn.gov/attorneygeneral/open-government/public-records.html",
        "forms": [
            "Public Records Request Letter (no official form — written request sufficient)",
        ],
        "tca_ref": "TCA § 10-7-503",
    },
}


# ══════════════════════════════════════════════════════════
# Tool Implementations
# ══════════════════════════════════════════════════════════

@_lc_tool
def calculate_statute_of_limitations(offense_type: str, date_of_offense: str) -> str:
    """
    Calculate the Tennessee statute of limitations deadline.

    offense_type: e.g. 'personal_injury', 'fraud', 'written_contract',
                       'medical_malpractice', 'wrongful_death', 'libel_slander',
                       'property_damage', 'oral_contract', 'consumer_protection', 'employment'
    date_of_offense: ISO date string YYYY-MM-DD (date injury/breach/event occurred)
    """
    key = offense_type.lower().replace(" ", "_").replace("-", "_")

    # Fuzzy match: allow partial matches
    matched = None
    for k in _SOL_TABLE:
        if key == k or key in k or k in key:
            matched = k
            break

    if not matched:
        available = ", ".join(_SOL_TABLE.keys())
        return (
            f"Unknown offense type '{offense_type}'. "
            f"Available types: {available}"
        )

    sol = _SOL_TABLE[matched]
    try:
        start = date.fromisoformat(date_of_offense)
    except ValueError:
        return f"Invalid date format '{date_of_offense}'. Use YYYY-MM-DD."

    years = sol.get("years", 0)
    months = sol.get("months", 0)

    if years:
        # Approximate: add years
        try:
            deadline = start.replace(year=start.year + years)
        except ValueError:
            deadline = start + timedelta(days=365 * years)
    else:
        deadline = start + timedelta(days=30 * months)

    today = date.today()
    days_remaining = (deadline - today).days

    if days_remaining < 0:
        status = f"⚠️ EXPIRED {abs(days_remaining)} days ago ({deadline})"
    elif days_remaining == 0:
        status = "⚠️ DEADLINE IS TODAY"
    elif days_remaining <= 30:
        status = f"🔴 URGENT — {days_remaining} days remaining (deadline: {deadline})"
    elif days_remaining <= 90:
        status = f"🟡 {days_remaining} days remaining (deadline: {deadline})"
    else:
        status = f"🟢 {days_remaining} days remaining (deadline: {deadline})"

    return (
        f"Statute of Limitations — {matched.replace('_', ' ').title()}\n"
        f"Applicable statute: {sol['tca']}\n"
        f"Limitation period: {sol['notes']}\n"
        f"Event date: {start}\n"
        f"Filing deadline: {deadline}\n"
        f"Status: {status}\n\n"
        f"⚠️ Note: Tolling rules (minority, discovery rule, fraudulent concealment) "
        f"may extend this deadline. Consult a Tennessee attorney."
    )


@_lc_tool
def lookup_tca_section(section: str) -> str:
    """
    Look up context information for a specific TCA section number.

    section: TCA section number, e.g. '39-14-105' or '§ 39-14-105' or 'TCA § 39-14-105'
    """
    # Normalize: extract digits-digits-digits pattern
    import re
    match = re.search(r"(\d+)-(\d+)-(\d+)", section)
    if not match:
        return f"Could not parse TCA section number from '{section}'. Expected format: XX-XX-XXX"

    title, chapter, sec = match.groups()
    title_desc = _TCA_TITLE_MAP.get(title, f"TCA Title {title} (description not indexed)")

    return (
        f"TCA § {title}-{chapter}-{sec}\n"
        f"Title {title}: {title_desc}\n"
        f"Chapter {chapter}, Section {sec}\n\n"
        f"To read the full text of this section, visit:\n"
        f"  https://law.justia.com/codes/tennessee/title-{title}/chapter-{chapter}/section-{title}-{chapter}-{sec}/\n"
        f"  https://advance.lexis.com/container?config=014CJAA...  (Lexis, subscription)\n\n"
        f"Note: Use the TCA ingestion script to load this section into your local vector store:\n"
        f"  python scripts/ingest_tca.py --titles {title}"
    )


@_lc_tool
def search_tn_case_law(query: str, max_results: int = 3) -> str:
    """
    Search Tennessee case law via the CourtListener public API.

    Searches Tennessee Court of Appeals and Supreme Court opinions.
    query: plain English search terms (e.g. 'DUI first offense penalty')
    max_results: number of results to return (1–10)
    """
    try:
        import requests
    except ImportError:
        return "requests package not available. Install with: pip install requests"

    url = "https://www.courtlistener.com/api/rest/v4/search/"
    params = {
        "q": query,
        "jurisdiction": "tenn",
        "type": "o",           # opinions
        "order_by": "score desc",
        "page_size": min(int(max_results), 10),
    }
    headers = {
        "User-Agent": "TN-LawMaster/1.0 (educational legal research; "
                      "https://github.com/goddamnittom/tn-lawmaster-ai)"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return "CourtListener API timed out. Try again or search manually at courtlistener.com."
    except requests.exceptions.RequestException as exc:
        return f"CourtListener API error: {exc}. Search manually at https://www.courtlistener.com/?jurisdiction=tenn"
    except Exception as exc:
        return f"Unexpected error searching case law: {exc}"

    results_raw = data.get("results", [])
    if not results_raw:
        return (
            f"No Tennessee cases found for '{query}' on CourtListener.\n"
            f"Try searching directly: https://www.courtlistener.com/?q={query.replace(' ', '+')}&jurisdiction=tenn"
        )

    lines = [f"Tennessee Case Law Results for: '{query}'\n{'─'*50}"]
    for i, r in enumerate(results_raw[:max_results], 1):
        case_name = r.get("caseName", r.get("case_name", "Unknown case"))
        court = r.get("court", r.get("court_id", "TN Court"))
        date_filed = r.get("dateFiled", r.get("date_filed", "Unknown date"))
        snippet = r.get("snippet", "")[:300].strip()
        cl_id = r.get("id", "")
        case_url = f"https://www.courtlistener.com/opinion/{cl_id}/" if cl_id else ""

        lines.append(
            f"\n{i}. {case_name}\n"
            f"   Court: {court} | Date: {date_filed}\n"
            + (f"   Excerpt: {snippet}\n" if snippet else "")
            + (f"   URL: {case_url}" if case_url else "")
        )

    lines.append(
        f"\n{'─'*50}\n"
        f"Source: CourtListener (courtlistener.com) — Free public legal database\n"
        f"⚠️ Always verify citations with primary sources before relying on them."
    )
    return "\n".join(lines)


@_lc_tool
def get_tennessee_legal_forms(form_type: str) -> str:
    """
    Get information about official Tennessee court forms and where to find them.

    form_type: e.g. 'divorce', 'custody', 'small_claims', 'eviction',
                     'llc_formation', 'name_change', 'expungement', 'tipa_request'
    """
    key = form_type.lower().replace(" ", "_").replace("-", "_")

    # Fuzzy match
    matched = None
    for k in _FORMS_DIRECTORY:
        if key == k or key in k or k in key:
            matched = k
            break

    if not matched:
        available = ", ".join(_FORMS_DIRECTORY.keys())
        return (
            f"Unknown form type '{form_type}'. "
            f"Available categories: {available}"
        )

    info = _FORMS_DIRECTORY[matched]
    forms_list = "\n".join(f"  • {f}" for f in info["forms"])
    fee = f"\n  Fee: {info['fee']}" if "fee" in info else ""

    return (
        f"Tennessee Legal Forms — {matched.replace('_', ' ').title()}\n"
        f"{'─'*50}\n"
        f"Source: {info['source']}\n"
        f"URL: {info['url']}{fee}\n"
        f"TCA Reference: {info['tca_ref']}\n\n"
        f"Available Forms:\n{forms_list}\n\n"
        f"⚠️ These forms are for informational purposes. Completing legal forms incorrectly "
        f"can harm your case. Consider consulting a licensed Tennessee attorney or using "
        f"Tennessee's Legal Aid Society: https://www.tals.org/"
    )


# ── Tool registry ─────────────────────────────────────────
ALL_TOOLS = [
    calculate_statute_of_limitations,
    lookup_tca_section,
    search_tn_case_law,
    get_tennessee_legal_forms,
]
