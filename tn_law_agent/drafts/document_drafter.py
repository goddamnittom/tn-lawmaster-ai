"""
TN-LawMaster — Document Drafter
================================
Generates Tennessee-specific legal document templates using LLM + TCA grounding.

Supported document types:
  - demand_letter      Pre-litigation demand letter
  - tipa_request       TCA § 10-7-503 public records request
  - cease_desist       Cease and desist letter
  - lease_notice       TCA § 66-28-505 notice to pay or vacate
  - custody_summary    Custody arrangement term summary
  - llc_checklist      TCA § 48-249-101 LLC formation checklist

Usage::

    from model_config import get_llm
    from tn_law_agent.drafts.document_drafter import TNDocumentDrafter

    drafter = TNDocumentDrafter(llm=get_llm())
    result = drafter.draft("tipa_request", context={
        "requester_name": "Jane Smith",
        "agency_name": "Nashville Police Department",
        "records_description": "All incident reports from January 2024",
    })
    print(result["document"])
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["TNDocumentDrafter"]

_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Legal Disclaimer**: This document is AI-generated and provided for informational "
    "purposes only. It does not constitute legal advice. Have this document reviewed by a "
    "licensed Tennessee attorney before sending or filing it."
)

# ── Document type definitions ─────────────────────────────
DOCUMENT_TYPES: Dict[str, Dict] = {
    "demand_letter": {
        "label": "Pre-Litigation Demand Letter",
        "tca_basis": ["TCA § 28-3-104", "TCA § 29-39-102"],
        "required_fields": [
            "sender_name", "sender_address",
            "recipient_name", "recipient_address",
            "amount_demanded", "incident_date", "incident_description",
            "deadline_days",
        ],
        "optional_fields": ["sender_phone", "sender_email", "attorney_name"],
        "system_prompt": (
            "You are a Tennessee legal document drafter. Draft a professional pre-litigation "
            "demand letter based on the provided context. "
            "The letter must: (1) clearly state the facts of the dispute, "
            "(2) specify the exact dollar amount demanded, "
            "(3) cite the applicable TCA statute(s) supporting the claim, "
            "(4) set a clear deadline for response (typically 30 days), "
            "(5) state consequences if demand is not met (filing suit). "
            "Use formal legal letter format with date, addresses, subject line, body, and signature block. "
            "Keep tone firm but professional — not threatening."
        ),
    },
    "tipa_request": {
        "label": "Tennessee Public Records Request (TIPA)",
        "tca_basis": ["TCA § 10-7-503"],
        "required_fields": [
            "requester_name", "agency_name", "records_description",
        ],
        "optional_fields": [
            "requester_address", "requester_email", "requester_phone",
            "date_range", "preferred_format",
        ],
        "system_prompt": (
            "You are a Tennessee legal document drafter. Draft a formal Tennessee public records "
            "request letter under the Tennessee Public Records Act (TCA § 10-7-503). "
            "The letter must: (1) clearly identify the requester, "
            "(2) specifically identify the records requested, "
            "(3) cite TCA § 10-7-503 as the legal authority, "
            "(4) request response within a reasonable time, "
            "(5) request written denial with specific exemption cited if denied. "
            "Keep the request specific and professional."
        ),
    },
    "cease_desist": {
        "label": "Cease and Desist Letter",
        "tca_basis": ["TCA § 47-25-1101", "TCA § 39-14-134"],
        "required_fields": [
            "sender_name", "recipient_name",
            "violation_description", "demanded_action",
        ],
        "optional_fields": [
            "sender_address", "sender_attorney", "deadline_days",
        ],
        "system_prompt": (
            "You are a Tennessee legal document drafter. Draft a formal cease and desist letter "
            "based on the provided context. "
            "The letter must: (1) identify the specific conduct that must stop, "
            "(2) cite applicable Tennessee law (TCA sections), "
            "(3) demand specific actions by a clear deadline, "
            "(4) state legal consequences of non-compliance, "
            "(5) preserve the sender's legal rights. "
            "Use formal letter format. Keep tone firm and legally precise."
        ),
    },
    "lease_notice": {
        "label": "Tennessee Notice to Pay or Vacate (TCA § 66-28-505)",
        "tca_basis": ["TCA § 66-28-505"],
        "required_fields": [
            "landlord_name", "tenant_name", "property_address",
            "amount_owed", "due_date",
        ],
        "optional_fields": ["landlord_address", "landlord_phone"],
        "system_prompt": (
            "You are a Tennessee legal document drafter. Draft a formal Notice to Pay Rent or "
            "Vacate under TCA § 66-28-505. "
            "The notice must: (1) identify the landlord and tenant by name, "
            "(2) specify the property address, "
            "(3) state the exact amount of unpaid rent owed, "
            "(4) give exactly 14 days to pay or vacate (required by TCA § 66-28-505), "
            "(5) cite TCA § 66-28-505 as legal authority, "
            "(6) state that failure to comply will result in a detainer warrant (eviction). "
            "Date the notice clearly. Include a signature block for the landlord."
        ),
    },
    "custody_summary": {
        "label": "Custody Arrangement Summary",
        "tca_basis": ["TCA § 36-6-106", "TCA § 36-6-404"],
        "required_fields": [
            "child_name", "child_dob",
            "parent1_name", "parent2_name",
            "primary_residential_parent", "parenting_schedule_description",
        ],
        "optional_fields": [
            "decision_making", "holiday_schedule", "child_support_amount",
        ],
        "system_prompt": (
            "You are a Tennessee legal document drafter. Draft a clear summary of a proposed "
            "custody and parenting arrangement under TCA § 36-6-106 (best interest factors) "
            "and TCA § 36-6-404 (permanent parenting plan requirements). "
            "The summary must: (1) identify both parents and the child(ren), "
            "(2) specify the primary residential parent, "
            "(3) describe the parenting schedule clearly (weekdays, weekends, school breaks), "
            "(4) address decision-making authority (joint vs. sole), "
            "(5) note this is a summary — a formal Permanent Parenting Plan must be filed with the court. "
            "Reference TCA § 36-6-106 best interest factors where applicable."
        ),
    },
    "llc_checklist": {
        "label": "Tennessee LLC Formation Checklist",
        "tca_basis": ["TCA § 48-249-101", "TCA § 48-249-202"],
        "required_fields": ["llc_name", "business_purpose"],
        "optional_fields": [
            "registered_agent_name", "management_type", "member_names",
        ],
        "system_prompt": (
            "You are a Tennessee business law expert. Generate a comprehensive Tennessee LLC "
            "formation checklist based on TCA § 48-249-101 et seq. "
            "The checklist must include: "
            "(1) Name availability check (TN SOS), "
            "(2) Articles of Organization (SS-4270) filing with required provisions, "
            "(3) Registered agent requirement, "
            "(4) Operating Agreement preparation (recommended), "
            "(5) EIN application (IRS Form SS-4), "
            "(6) Tennessee business licenses (if applicable), "
            "(7) Annual Report requirement (due April 1, $300 fee), "
            "(8) Opening a business bank account. "
            "Cite specific TCA sections and link to Tennessee SOS (sos.tn.gov)."
        ),
    },
}


class TNDocumentDrafter:
    """
    Generate Tennessee legal document drafts using LLM + TCA grounding.

    Args:
        llm: Any LangChain-compatible chat model.
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def draft(self, doc_type: str, context: dict) -> dict:
        """
        Draft a legal document.

        Args:
            doc_type: One of the keys in DOCUMENT_TYPES.
            context:  Dict of variables for the document (see required_fields).

        Returns:
            dict with keys:
                - ``document``    — the drafted text
                - ``type``        — doc_type used
                - ``label``       — human-readable label
                - ``tca_basis``   — list of TCA sections cited
                - ``disclaimer``  — standard legal disclaimer
                - ``missing_fields`` — any required fields not supplied
        """
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(
                f"Unknown document type '{doc_type}'. "
                f"Available: {list(DOCUMENT_TYPES.keys())}"
            )

        defn = DOCUMENT_TYPES[doc_type]
        missing = [f for f in defn["required_fields"] if not context.get(f)]

        # Build human prompt from context
        today = date.today().strftime("%B %d, %Y")
        context_lines = "\n".join(
            f"  {k}: {v}" for k, v in context.items() if v
        )
        human_prompt = (
            f"Today's date: {today}\n\n"
            f"Document type: {defn['label']}\n"
            f"TCA basis: {', '.join(defn['tca_basis'])}\n\n"
            f"Context provided:\n{context_lines}\n\n"
            + (f"Missing fields (use [UNKNOWN] placeholder): {missing}\n\n" if missing else "")
            + "Please draft the document now."
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = self.llm.invoke([
                SystemMessage(content=defn["system_prompt"]),
                HumanMessage(content=human_prompt),
            ])
            document = response.content
        except Exception as exc:
            logger.error("[TNDocumentDrafter] LLM error: %s", exc)
            document = f"⚠️ Document generation failed: {exc}"

        return {
            "document": document + _DISCLAIMER,
            "type": doc_type,
            "label": defn["label"],
            "tca_basis": defn["tca_basis"],
            "disclaimer": _DISCLAIMER.strip(),
            "missing_fields": missing,
        }

    def list_document_types(self) -> List[dict]:
        """Return metadata for all supported document types."""
        return [
            {
                "type": k,
                "label": v["label"],
                "tca_basis": v["tca_basis"],
                "required_fields": v["required_fields"],
                "optional_fields": v.get("optional_fields", []),
            }
            for k, v in DOCUMENT_TYPES.items()
        ]
