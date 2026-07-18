"""
TN-LawMaster — Compliance Checker
===================================
Analyzes documents for Tennessee law compliance issues.

Supported domains:
  - landlord_tenant    TCA Title 66 — Uniform Residential Landlord and Tenant Act
  - llc_operating      TCA Title 48 — LLC operating agreement requirements
  - employment         TN employment law (THRA, wage/hour, non-compete)
  - consumer_protection TCA Title 47-18 — Tennessee Consumer Protection Act

Usage::

    from tn_law_agent.compliance.checker import TNComplianceChecker

    checker = TNComplianceChecker(llm=get_llm())
    result = checker.check(lease_text, domain="landlord_tenant")
    for issue in result["issues"]:
        print(f"[{issue['severity']}] {issue['description']} ({issue['tca_ref']})")
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = ["TNComplianceChecker"]

# ── Domain definitions ────────────────────────────────────
COMPLIANCE_DOMAINS: Dict[str, Dict] = {
    "landlord_tenant": {
        "label": "Tennessee Landlord-Tenant (TCA Title 66)",
        "tca_refs": ["TCA § 66-28-201", "TCA § 66-28-301", "TCA § 66-28-401", "TCA § 66-28-505"],
        "checklist": [
            "Security deposit must not exceed 2 months rent (TCA § 66-28-301)",
            "Security deposit must be returned within 30 days of vacating (TCA § 66-28-301)",
            "Landlord must provide habitable conditions (TCA § 66-28-201)",
            "Landlord must give 14-day notice to pay or vacate before eviction (TCA § 66-28-505)",
            "Landlord must give 24 hours notice before entry except emergencies (TCA § 66-28-403)",
            "Lease cannot waive tenant rights under the URLTA (TCA § 66-28-104)",
            "Late fees must be reasonable and disclosed upfront",
            "No retaliatory eviction within 1 year of tenant complaint (TCA § 66-28-514)",
        ],
        "system_prompt": (
            "You are a Tennessee real estate law expert reviewing a landlord-tenant document "
            "for compliance with Tennessee Code Annotated Title 66 "
            "(Uniform Residential Landlord and Tenant Act — URLTA). "
            "Analyze the provided document for legal compliance issues. "
            "For each issue found, provide:\n"
            "- severity: 'high' (void/illegal clause), 'medium' (potentially unenforceable), "
            "  or 'low' (best practice concern)\n"
            "- description: what the issue is\n"
            "- tca_ref: the specific TCA section violated\n"
            "- recommendation: what to do to fix it\n\n"
            "Common issues: illegal security deposit amounts, illegal self-help eviction clauses, "
            "waiver of tenant URLTA rights, inadequate notice periods, missing landlord disclosure. "
            "Respond in JSON array format: [{\"severity\": ..., \"description\": ..., "
            "\"tca_ref\": ..., \"recommendation\": ...}]"
        ),
    },
    "llc_operating": {
        "label": "Tennessee LLC Operating Agreement (TCA Title 48)",
        "tca_refs": ["TCA § 48-249-101", "TCA § 48-249-202", "TCA § 48-249-401"],
        "checklist": [
            "LLC name must contain 'LLC', 'L.L.C.', or 'Limited Liability Company' (TCA § 48-249-106)",
            "Must designate management structure (member-managed or manager-managed)",
            "Must specify member voting rights",
            "Should address member withdrawal and buyout procedures",
            "Should address dissolution procedures (TCA § 48-249-601)",
            "Must not eliminate duty of loyalty entirely (TCA § 48-249-401)",
        ],
        "system_prompt": (
            "You are a Tennessee business law expert reviewing an LLC operating agreement "
            "for compliance with Tennessee Code Annotated Title 48 (TBCA/LLC Act). "
            "Analyze the document for legal compliance issues and missing provisions. "
            "For each issue, provide severity, description, TCA reference, and recommendation. "
            "Respond in JSON array format: [{\"severity\": ..., \"description\": ..., "
            "\"tca_ref\": ..., \"recommendation\": ...}]"
        ),
    },
    "employment": {
        "label": "Tennessee Employment Law",
        "tca_refs": ["TCA § 4-21-401", "TCA § 50-2-103", "TCA § 47-50-112"],
        "checklist": [
            "Non-compete must be reasonable in scope, geography, and duration (TCA § 47-50-112)",
            "Wage payment must meet TN minimum wage ($7.25/hr federal) (TCA § 50-2-103)",
            "Must not discriminate on basis of protected class (THRA — TCA § 4-21-401)",
            "Drug testing must follow TN Drug-Free Workplace procedures if applicable",
            "Final paycheck due by next regular payday (TCA § 50-2-103)",
        ],
        "system_prompt": (
            "You are a Tennessee employment law expert reviewing an employment document "
            "for compliance with Tennessee employment law, including the Tennessee Human Rights "
            "Act (TCA § 4-21-101 et seq.), Tennessee wage law (TCA Title 50), and "
            "Tennessee non-compete law (TCA § 47-50-112). "
            "Analyze for compliance issues. Respond in JSON array format: "
            "[{\"severity\": ..., \"description\": ..., \"tca_ref\": ..., \"recommendation\": ...}]"
        ),
    },
    "consumer_protection": {
        "label": "Tennessee Consumer Protection Act (TCA Title 47-18)",
        "tca_refs": ["TCA § 47-18-104", "TCA § 47-18-109", "TCA § 47-18-110"],
        "checklist": [
            "No deceptive acts or practices in consumer transactions (TCA § 47-18-104)",
            "No false advertising (TCA § 47-18-104(b)(5))",
            "Refund policy must be clearly disclosed",
            "Automatic renewal clauses must be clearly disclosed (TCA § 47-18-1309)",
            "No pyramid schemes or referral selling (TCA § 47-18-104(b)(21))",
        ],
        "system_prompt": (
            "You are a Tennessee consumer protection law expert reviewing a business document "
            "for compliance with the Tennessee Consumer Protection Act (TCA § 47-18-101 et seq.). "
            "Analyze for deceptive practices, unfair terms, or disclosure failures. "
            "Respond in JSON array format: "
            "[{\"severity\": ..., \"description\": ..., \"tca_ref\": ..., \"recommendation\": ...}]"
        ),
    },
}


class TNComplianceChecker:
    """
    Check documents for Tennessee law compliance issues.

    Args:
        llm:          Any LangChain-compatible chat model.
        vector_store: Optional — for retrieving relevant TCA context.
    """

    def __init__(self, llm, vector_store=None) -> None:
        self.llm = llm
        self.vector_store = vector_store

    def check(self, document_text: str, domain: str) -> dict:
        """
        Analyze a document for TN law compliance issues.

        Args:
            document_text: Full text of the document to check.
            domain:        Compliance domain (see COMPLIANCE_DOMAINS keys).

        Returns:
            dict with keys:
                - ``domain``         — domain checked
                - ``issues``         — list of issue dicts
                - ``compliant``      — True if no high-severity issues
                - ``summary``        — plain-language summary
                - ``recommendations``— top action items
                - ``tca_refs``       — relevant TCA sections
        """
        if domain not in COMPLIANCE_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Available: {list(COMPLIANCE_DOMAINS.keys())}"
            )

        defn = COMPLIANCE_DOMAINS[domain]

        # Truncate very long documents
        max_chars = 8000
        doc_excerpt = document_text[:max_chars]
        if len(document_text) > max_chars:
            doc_excerpt += f"\n\n[Document truncated at {max_chars} chars for analysis]"

        checklist_text = "\n".join(f"  • {item}" for item in defn["checklist"])

        from langchain_core.messages import HumanMessage, SystemMessage
        human_prompt = (
            f"Domain: {defn['label']}\n"
            f"TCA References: {', '.join(defn['tca_refs'])}\n\n"
            f"Key compliance checklist:\n{checklist_text}\n\n"
            f"Document to review:\n{'─'*40}\n{doc_excerpt}\n{'─'*40}\n\n"
            "Review this document for compliance issues. Return ONLY a valid JSON array."
        )

        issues: List[dict] = []
        try:
            response = self.llm.invoke([
                SystemMessage(content=defn["system_prompt"]),
                HumanMessage(content=human_prompt),
            ])
            raw = response.content.strip()
            import json, re
            # Extract JSON array from the response (may be wrapped in markdown)
            json_match = re.search(r"\[[\s\S]*\]", raw)
            if json_match:
                issues = json.loads(json_match.group())
            else:
                logger.warning("[TNComplianceChecker] no JSON array in response")
                issues = [{"severity": "low", "description": raw[:500],
                           "tca_ref": "General", "recommendation": "Manual review needed"}]
        except Exception as exc:
            logger.error("[TNComplianceChecker] LLM error: %s", exc)
            issues = [{"severity": "low",
                       "description": f"Analysis failed: {exc}",
                       "tca_ref": "N/A",
                       "recommendation": "Manual review required"}]

        # Normalize issue structure
        normalized = []
        for iss in issues:
            if isinstance(iss, dict):
                normalized.append({
                    "severity": iss.get("severity", "low"),
                    "description": iss.get("description", ""),
                    "tca_ref": iss.get("tca_ref", "TCA"),
                    "recommendation": iss.get("recommendation", ""),
                })

        high_issues = [i for i in normalized if i["severity"] == "high"]
        compliant = len(high_issues) == 0

        # Build summary
        n_high = len(high_issues)
        n_med = sum(1 for i in normalized if i["severity"] == "medium")
        n_low = sum(1 for i in normalized if i["severity"] == "low")

        if compliant and not normalized:
            summary = f"✅ No compliance issues found under {defn['label']}."
        elif compliant:
            summary = (
                f"⚠️ No critical issues, but found {n_med} medium and {n_low} low concerns "
                f"under {defn['label']}. Review recommended."
            )
        else:
            summary = (
                f"🚨 Found {n_high} high-severity compliance issue(s) under {defn['label']}. "
                f"These clauses may be void or illegal under Tennessee law. "
                f"Also found {n_med} medium and {n_low} low concerns."
            )

        recommendations = [i["recommendation"] for i in normalized if i["recommendation"]][:5]

        return {
            "domain": domain,
            "domain_label": defn["label"],
            "issues": normalized,
            "issue_count": {"high": n_high, "medium": n_med, "low": n_low},
            "compliant": compliant,
            "summary": summary,
            "recommendations": recommendations,
            "tca_refs": defn["tca_refs"],
        }

    def list_domains(self) -> List[dict]:
        """Return metadata for all supported compliance domains."""
        return [
            {
                "domain": k,
                "label": v["label"],
                "tca_refs": v["tca_refs"],
                "checklist_items": len(v["checklist"]),
            }
            for k, v in COMPLIANCE_DOMAINS.items()
        ]
