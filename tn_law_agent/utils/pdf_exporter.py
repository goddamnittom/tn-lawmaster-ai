"""
TN-LawMaster — PDF Report Exporter
=====================================
Generates formatted PDF reports from legal analysis results.

Usage:
    from tn_law_agent.utils.pdf_exporter import export_analysis_pdf

    result = agent.analyze("DUI penalty?", domain="traffic")
    pdf_bytes = export_analysis_pdf(result, query="DUI penalty?", domain="traffic")
    with open("report.pdf", "wb") as f:
        f.write(pdf_bytes)
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


def export_analysis_pdf(
    result: dict,
    query: str,
    domain: str = "general",
    include_disclaimer: bool = True,
) -> bytes:
    """
    Export a legal analysis result to a formatted PDF.

    Args:
        result: The dict returned by TNLawAgent.analyze()
        query: The original user query
        domain: The TCA domain used
        include_disclaimer: Whether to include the legal disclaimer footer

    Returns:
        PDF as bytes (can be written to file or served via HTTP)

    Requires:
        reportlab (pip install reportlab)
        Falls back to plain-text PDF if reportlab is not available.
    """
    try:
        return _export_with_reportlab(result, query, domain, include_disclaimer)
    except ImportError:
        logger.warning("reportlab not installed — generating plain text PDF fallback")
        return _export_plain_text(result, query, domain, include_disclaimer)


def _export_with_reportlab(
    result: dict,
    query: str,
    domain: str,
    include_disclaimer: bool,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ──────────────────────────────────
    title_style = ParagraphStyle(
        "TNTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0d1117"),
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "TNSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#57606a"),
        spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "TNSection",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1f6feb"),
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "TNBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=8,
    )
    citation_style = ParagraphStyle(
        "TNCitation",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Courier",
        textColor=colors.HexColor("#1f6feb"),
        leftIndent=16,
        spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "TNDisclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#57606a"),
        leftIndent=8,
        rightIndent=8,
        leading=12,
    )

    story = []

    # ── Header ─────────────────────────────────────────
    story.append(Paragraph("⚖️ TN-LawMaster AI — Legal Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')} · Domain: {domain.title()}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d0d7de")))
    story.append(Spacer(1, 10))

    # ── Query ──────────────────────────────────────────
    story.append(Paragraph("📋 Question", section_style))
    story.append(Paragraph(query, body_style))
    story.append(Spacer(1, 8))

    # ── Analysis ───────────────────────────────────────
    story.append(Paragraph("⚖️ Legal Analysis", section_style))
    analysis_text = result.get("analysis", "No analysis available.")
    # Convert markdown-ish bold (**text**) to reportlab bold (<b>text</b>)
    import re
    analysis_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", analysis_text)
    for para in analysis_text.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    # ── Citations ──────────────────────────────────────
    citations = result.get("citations", [])
    if citations:
        story.append(Paragraph("📌 TCA Citations", section_style))
        for cite in citations:
            story.append(Paragraph(f"• {cite}", citation_style))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d0d7de")))
    story.append(Spacer(1, 8))

    # ── Disclaimer ─────────────────────────────────────
    if include_disclaimer:
        story.append(Paragraph(
            "⚠️ Legal Disclaimer: This report is generated by TN-LawMaster AI and provides "
            "legal information only — not legal advice. The analysis is based on Tennessee "
            "Code Annotated (TCA) statutes and AI-generated content. Always consult a "
            "licensed Tennessee attorney before making legal decisions.",
            disclaimer_style,
        ))

    doc.build(story)
    return buf.getvalue()


def _export_plain_text(
    result: dict,
    query: str,
    domain: str,
    include_disclaimer: bool,
) -> bytes:
    """Minimal fallback: wraps content in a plain-text 'PDF' via fpdf2 or raw bytes."""
    lines = [
        "TN-LawMaster AI — Legal Analysis Report",
        f"Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}",
        f"Domain: {domain.title()}",
        "=" * 60,
        "",
        "QUESTION",
        query,
        "",
        "ANALYSIS",
        result.get("analysis", "No analysis available."),
        "",
    ]
    citations = result.get("citations", [])
    if citations:
        lines.append("TCA CITATIONS")
        for c in citations:
            lines.append(f"  • {c}")
        lines.append("")

    if include_disclaimer:
        lines += [
            "=" * 60,
            "DISCLAIMER: This report provides legal information only — not legal advice.",
            "Consult a licensed Tennessee attorney for your specific situation.",
        ]

    text = "\n".join(lines)
    return text.encode("utf-8")


def export_chat_session_pdf(
    history: list[dict],
    session_id: str = "",
    include_disclaimer: bool = True,
) -> bytes:
    """
    Export an entire chat session to PDF.

    Args:
        history: List of {"role": "user"|"assistant", "content": str, "citations": [...]}
        session_id: Optional session identifier for the header
        include_disclaimer: Include legal disclaimer

    Returns:
        PDF bytes
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=LETTER,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=1.0*inch, bottomMargin=1.0*inch)
        styles = getSampleStyleSheet()

        user_style = ParagraphStyle("User", parent=styles["Normal"], fontSize=10,
                                    backColor=colors.HexColor("#f6f8fa"),
                                    borderPad=6, fontName="Helvetica-Bold",
                                    spaceAfter=4)
        bot_style = ParagraphStyle("Bot", parent=styles["Normal"], fontSize=10,
                                   leading=15, spaceAfter=12)

        story = []
        story.append(Paragraph("⚖️ TN-LawMaster AI — Chat Session Export", styles["Heading1"]))
        story.append(Paragraph(
            f"Exported: {datetime.now().strftime('%B %d, %Y %I:%M %p')}"
            + (f" · Session: {session_id}" if session_id else ""),
            styles["Normal"],
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d0d7de")))
        story.append(Spacer(1, 12))

        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                story.append(Paragraph(f"👤 You: {content}", user_style))
            else:
                import re
                content_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
                story.append(Paragraph(f"⚖️ TN-LawMaster: {content_html}", bot_style))
                citations = msg.get("citations", [])
                if citations:
                    cite_str = " · ".join(citations)
                    story.append(Paragraph(
                        f"📌 {cite_str}",
                        ParagraphStyle("Cite", parent=styles["Normal"],
                                       fontSize=8, fontName="Courier",
                                       textColor=colors.HexColor("#1f6feb"),
                                       spaceAfter=10),
                    ))
            story.append(Spacer(1, 4))

        if include_disclaimer:
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#d0d7de")))
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "⚠️ Disclaimer: TN-LawMaster provides AI-generated legal information only — "
                "not legal advice. Consult a licensed Tennessee attorney.",
                ParagraphStyle("Disc", parent=styles["Normal"], fontSize=8,
                               textColor=colors.HexColor("#57606a")),
            ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        lines = [f"TN-LawMaster Chat Export — {datetime.now().strftime('%B %d, %Y')}",
                 "=" * 60, ""]
        for msg in history:
            prefix = "YOU" if msg["role"] == "user" else "TN-LAWMASTER"
            lines.append(f"[{prefix}]")
            lines.append(msg.get("content", ""))
            if msg.get("citations"):
                lines.append(f"Citations: {', '.join(msg['citations'])}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")
