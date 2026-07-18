#!/usr/bin/env python3
"""
TN-LawMaster — TCA Ingestion Script
=====================================
Downloads and ingests Tennessee Code Annotated (TCA) statutes
into the local ChromaDB vector store.

Sources used (all public domain / public law):
  • Justia US Law — Tennessee Code: https://law.justia.com/codes/tennessee/
  • Tennessee General Assembly: https://www.tn.gov/content/tn/laws.html

Usage:
    # Interactive mode — choose which titles to fetch
    python scripts/ingest_tca.py

    # Ingest specific titles
    python scripts/ingest_tca.py --titles 39 36 66 48

    # Just seed the synthetic fallback corpus (no network needed)
    python scripts/ingest_tca.py --seed-only

    # Ingest all local PDFs in data/
    python scripts/ingest_tca.py --local-only

    # Full run: seed + scrape + local files
    python scripts/ingest_tca.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── TCA Justia URLs ───────────────────────────────────────
#
# Each entry: (title_number, label, justia_url)
# These are the top-level chapter listing pages — the scraper
# then follows individual section links.
#
TCA_SOURCES: list[tuple[str, str, str]] = [
    ("39", "Criminal Offenses",
     "https://law.justia.com/codes/tennessee/title-39/"),
    ("36", "Domestic Relations",
     "https://law.justia.com/codes/tennessee/title-36/"),
    ("66", "Property",
     "https://law.justia.com/codes/tennessee/title-66/"),
    ("48", "Corporations and Associations",
     "https://law.justia.com/codes/tennessee/title-48/"),
    ("29", "Remedies and Special Proceedings",
     "https://law.justia.com/codes/tennessee/title-29/"),
    ("30", "Administration of Estates",
     "https://law.justia.com/codes/tennessee/title-30/"),
    ("31", "Descent and Distribution",
     "https://law.justia.com/codes/tennessee/title-31/"),
    ("32", "Wills",
     "https://law.justia.com/codes/tennessee/title-32/"),
    ("55", "Motor and Other Vehicles",
     "https://law.justia.com/codes/tennessee/title-55/"),
    ("10", "Public Libraries and Archives",
     "https://law.justia.com/codes/tennessee/title-10/"),
]

# ── Synthetic seed corpus ──────────────────────────────────
#
# Representative TCA text for each domain.
# Used as a fallback when no vector store is populated and for
# quick-start without network access.
#
SEED_CORPUS: list[dict] = [
    # Criminal — Title 39
    {
        "source": "TCA § 39-14-103",
        "text": (
            "TCA § 39-14-103. Theft of property. "
            "(a) A person commits theft of property if, with intent to deprive the owner of "
            "property, the person knowingly obtains or exercises control over the property "
            "without the owner's effective consent. "
            "(b) Theft of property is: "
            "(1) A Class A misdemeanor if the value of the property obtained is less than $1,000; "
            "(2) A Class E felony if the value of the property obtained is $1,000 or more but "
            "less than $2,500; "
            "(3) A Class D felony if the value of the property obtained is $2,500 or more but "
            "less than $10,000; "
            "(4) A Class C felony if the value of the property obtained is $10,000 or more but "
            "less than $60,000; "
            "(5) A Class B felony if the value of the property obtained is $60,000 or more but "
            "less than $250,000; "
            "(6) A Class A felony if the value of the property obtained is $250,000 or more."
        ),
    },
    {
        "source": "TCA § 39-13-111",
        "text": (
            "TCA § 39-13-111. Domestic assault. "
            "(a) A person commits domestic assault who commits an assault as defined in "
            "§ 39-13-101 against a domestic abuse victim as defined in § 36-3-601. "
            "(b) A violation of this section is a Class A misdemeanor; "
            "provided, however, that, if the defendant has a prior conviction for a violation "
            "of this section or § 39-13-101, the defendant shall be required to serve at least "
            "forty-eight (48) consecutive hours of the sentence imposed by the court. "
            "(c) On the second offense, the offense is a Class A misdemeanor with mandatory "
            "30-day sentence. On the third or subsequent offense, it is a Class E felony."
        ),
    },
    {
        "source": "TCA § 39-17-417",
        "text": (
            "TCA § 39-17-417. Criminal penalties for controlled substance violations. "
            "It is an offense for a defendant to knowingly: "
            "(1) Manufacture a controlled substance; "
            "(2) Deliver a controlled substance; "
            "(3) Sell a controlled substance; or "
            "(4) Possess a controlled substance with intent to manufacture, deliver, or sell. "
            "Penalties: Schedule I or II: Class B felony (not less than 8 nor more than 30 years). "
            "Schedule III: Class D felony. "
            "Schedule IV: Class D felony. "
            "Schedule V: Class E felony. "
            "Schedule VI (marijuana ≥0.5 oz): Class E felony. "
            "Marijuana less than 0.5 oz possession: Class A misdemeanor."
        ),
    },
    {
        "source": "TCA § 55-10-401",
        "text": (
            "TCA § 55-10-401. Driving under the influence (DUI). "
            "It is unlawful for any person to drive or to be in physical control of any automobile "
            "or other motor-driven vehicle on any of the public roads and highways of the state, "
            "or on any streets or alleys, or while on the premises of any shopping center, trailer "
            "park or any apartment house complex, or any other premises that is generally "
            "frequented by the public at large, while: "
            "(1) Under the influence of any intoxicant, marijuana, controlled substance, "
            "controlled substance analogue, drug, substance affecting the central nervous system, "
            "or combination thereof that impairs the driver's ability to safely operate a motor "
            "vehicle by depriving the driver of the clearness of mind and control of oneself that "
            "the driver would otherwise possess; "
            "(2) The alcohol concentration in the person's blood or breath is eight-hundredths "
            "of one percent (0.08%) or more. "
            "First DUI offense: Class A misdemeanor — not less than 48 hours nor more than "
            "11 months 29 days confinement; fine of $350–$1,500; license revocation 1 year. "
            "Second DUI offense: Class A misdemeanor — not less than 45 days nor more than "
            "11 months 29 days confinement; fine of $600–$3,500; license revocation 2 years. "
            "Third DUI offense: Class A misdemeanor — not less than 120 days confinement; "
            "fine of $1,100–$10,000; license revocation 3–10 years. "
            "Fourth or subsequent offense: Class E felony."
        ),
    },
    # Family — Title 36
    {
        "source": "TCA § 36-4-101",
        "text": (
            "TCA § 36-4-101. Grounds for absolute divorce. "
            "The following are grounds for divorce from the bonds of matrimony: "
            "(1) Either party, at the time of the contract, was and still is naturally impotent "
            "and incapable of procreation; "
            "(2) Either party has knowingly entered into a second marriage, in violation of a "
            "previous marriage, still subsisting; "
            "(3) Either party has committed adultery; "
            "(4) Willful or malicious desertion or absence of either party, without a reasonable "
            "cause, for one (1) whole year; "
            "(5) Conviction of either party of any crime that, by the laws of the state, renders "
            "the party infamous; "
            "(11) Irreconcilable differences between the parties (no-fault ground, requires "
            "marital dissolution agreement or parenting plan)."
        ),
    },
    {
        "source": "TCA § 36-6-106",
        "text": (
            "TCA § 36-6-106. Child custody — best interest factors. "
            "In a suit for annulment, divorce, separate maintenance, or in any other proceeding "
            "requiring the court to make a custody determination regarding a minor child, "
            "the determination shall be made upon the basis of the best interest of the child. "
            "The court shall consider all relevant factors, including: "
            "(1) The strength, nature, and stability of the child's relationship with each parent; "
            "(2) Each parent's or caregiver's past and potential for future performance of "
            "parenting responsibilities; "
            "(3) Refusal to attend a court-ordered parent education seminar; "
            "(4) The disposition of each parent to provide the child with food, clothing, medical "
            "care, education, and other necessary care; "
            "(5) The degree to which a parent has been the primary caregiver; "
            "(6) The love, affection, and emotional ties existing between each parent and the child; "
            "(7) The emotional needs and developmental level of the child; "
            "(8) The moral, physical, mental and emotional fitness of each parent; "
            "(9) The child's interaction and interrelationships with siblings and with significant "
            "others, as well as the child's adjustment to the child's home, school, and community."
        ),
    },
    {
        "source": "TCA § 36-5-101",
        "text": (
            "TCA § 36-5-101. Child support. "
            "In any proceeding for divorce, legal separation, or separate maintenance, or in any "
            "proceeding for the dissolution of a marriage, the court shall, upon request of either "
            "party, make an order requiring either or both parents to make payments for the support "
            "and maintenance of the child or children. "
            "Child support in Tennessee is calculated using the Income Shares Model. "
            "The basic child support obligation is based on the combined gross income of both parents "
            "and the number of children. Deviation from the guidelines may be allowed based on: "
            "extraordinary educational expenses, extraordinary medical expenses, primary residential "
            "parent's low income, or other extraordinary circumstances. "
            "Child support continues until age 18, or age 21 if the child is still in high school "
            "and living with the primary residential parent."
        ),
    },
    # Property — Title 66
    {
        "source": "TCA § 66-28-201",
        "text": (
            "TCA § 66-28-201. Landlord obligations — residential tenancy. "
            "The landlord shall: "
            "(1) Comply with requirements of applicable building and housing codes materially "
            "affecting health and safety; "
            "(2) Make all repairs and do whatever is necessary to put and keep the premises "
            "in a fit and habitable condition; "
            "(3) Keep all common areas of the premises in a reasonably clean and safe condition; "
            "(4) Maintain in good and safe working order and condition all electrical, plumbing, "
            "sanitary, heating, ventilating, air-conditioning, and other facilities and appliances, "
            "including elevators, supplied or required to be supplied by the landlord; "
            "(5) Supply running water and reasonable amounts of hot water at all times and reasonable "
            "heat, except where the building is not required by law to be equipped for that purpose, "
            "or where the heat or hot water is generated by an installation within the exclusive "
            "control of the tenant and supplied by a direct public utility connection."
        ),
    },
    {
        "source": "TCA § 66-28-505",
        "text": (
            "TCA § 66-28-505. Termination of tenancy — notice requirements. "
            "(a) The landlord or the tenant may terminate a week-to-week tenancy by a written "
            "notice given to the other at least ten (10) days before the termination date specified "
            "in the notice. "
            "(b) The landlord or the tenant may terminate a month-to-month tenancy by a written "
            "notice given to the other at least thirty (30) days before the periodic rental date "
            "specified in the notice. "
            "Nonpayment of rent: Landlord must give a written 14-day notice to pay or vacate "
            "before filing for detainer warrant (eviction). "
            "Lease violation (other than nonpayment): Landlord must give a 14-day notice to "
            "remedy or vacate."
        ),
    },
    # Business — Title 48
    {
        "source": "TCA § 48-249-101",
        "text": (
            "TCA § 48-249-101. Tennessee Limited Liability Company Act — Formation. "
            "One or more persons may organize a limited liability company (LLC) by delivering "
            "articles of organization to the Secretary of State for filing. "
            "The articles of organization must set forth: "
            "(1) The name of the LLC (must contain 'LLC', 'L.L.C.', 'Limited Liability Company', "
            "or similar designation); "
            "(2) The address of the LLC's principal office; "
            "(3) The name and address of the LLC's registered agent; "
            "(4) Whether the LLC is member-managed or manager-managed. "
            "Annual report: Tennessee LLCs must file an annual report with the Secretary of State "
            "by April 1 each year. Filing fee is $300. "
            "An LLC provides limited liability protection — members are generally not personally "
            "liable for the LLC's debts and obligations."
        ),
    },
    # Torts — Title 29
    {
        "source": "TCA § 29-39-102",
        "text": (
            "TCA § 29-39-102. Caps on non-economic damages — Tennessee Civil Justice Act. "
            "(a) In a civil action, each injured plaintiff may be awarded: "
            "(1) Noneconomic damages up to seven hundred fifty thousand dollars ($750,000) "
            "for all injuries and losses per occurrence; "
            "(2) In cases involving catastrophic loss or injury (e.g., spinal cord injury, "
            "amputation, severe burns, wrongful death), the cap is one million dollars ($1,000,000). "
            "(b) These caps do not apply to: "
            "(1) Actions where the defendant acted intentionally; "
            "(2) Actions where the defendant was under the influence of alcohol or drugs; "
            "(3) Actions against a defendant convicted of a felony arising from the same incident. "
            "Tennessee follows a modified comparative fault rule: a plaintiff is barred from "
            "recovery if 50% or more at fault (TCA § 29-11-103)."
        ),
    },
    # TIPA — Title 10
    {
        "source": "TCA § 10-7-503",
        "text": (
            "TCA § 10-7-503. Tennessee Public Records Act — right of inspection. "
            "(a)(1)(A) All state, county and municipal records shall at all times, during business "
            "hours, be open for personal inspection by any citizen of Tennessee, and those "
            "in charge of the records shall not refuse such right of inspection to any citizen, "
            "unless the records are exempt from disclosure as provided by law. "
            "(b) A records custodian shall promptly make available for inspection any public record "
            "not specifically exempt from disclosure. "
            "(c) If a records request is denied, the custodian must state in writing the specific "
            "exemption claimed. "
            "Common exemptions: personnel files, criminal investigation records, "
            "attorney-client privileged communications, medical records, "
            "trade secrets, and security plans."
        ),
    },
]


# ── Scraper ───────────────────────────────────────────────

def _scrape_justia_title(title_url: str, max_sections: int = 40) -> list[dict]:
    """
    Scrape TCA statute text from a Justia title page.

    Returns list of dicts: {source, text}
    Limited to `max_sections` to stay polite to the server.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests and beautifulsoup4 are required for web scraping.")
        return []

    headers = {
        "User-Agent": "TN-LawMaster/1.0 (educational legal research tool; "
                      "https://github.com/goddamnittom/tn-lawmaster-ai)"
    }

    try:
        resp = requests.get(title_url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", title_url, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find section links on the title page
    section_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/codes/tennessee/title-" in href and "section-" in href:
            full = href if href.startswith("http") else f"https://law.justia.com{href}"
            section_links.append(full)

    section_links = list(dict.fromkeys(section_links))[:max_sections]
    logger.info("  Found %d section links on %s", len(section_links), title_url)

    docs = []
    for url in section_links:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            s = BeautifulSoup(r.text, "html.parser")

            # Justia puts statute text in <div class="primary-content"> or similar
            content_div = (
                s.find("div", class_="primary-content")
                or s.find("div", {"id": "statute-text"})
                or s.find("article")
            )
            if not content_div:
                continue

            text = content_div.get_text(separator="\n", strip=True)
            if len(text) < 100:
                continue

            # Extract TCA section number from URL slug
            slug = url.rstrip("/").split("/")[-1]
            source = f"TCA § {slug.replace('section-', '').replace('-', '.')}"

            docs.append({"source": source, "text": text[:5000]})
            time.sleep(0.8)  # be polite
        except Exception as exc:
            logger.debug("  Skip %s: %s", url, exc)

    return docs


# ── Ingest helpers ────────────────────────────────────────

def _get_ingester(persist_dir: str, data_dir: str):
    from tn_law_agent.knowledge.ingester import TNLawIngester
    return TNLawIngester(data_dir=data_dir, persist_dir=persist_dir)


def seed_corpus(ingester) -> int:
    """Ingest the built-in synthetic TCA seed corpus."""
    logger.info("🌱 Seeding with %d synthetic TCA entries...", len(SEED_CORPUS))
    total = 0
    for entry in SEED_CORPUS:
        n = ingester.ingest_text(entry["text"], source=entry["source"])
        total += n
        logger.info("  ✅ %s → %d chunk(s)", entry["source"], n)
    return total


def scrape_titles(ingester, title_numbers: list[str], max_sections: int = 40) -> int:
    """Scrape and ingest TCA titles from Justia."""
    sources = {t: (lbl, url) for t, lbl, url in TCA_SOURCES}
    total = 0
    for num in title_numbers:
        if num not in sources:
            logger.warning("Unknown title: %s  (available: %s)", num, list(sources))
            continue
        label, url = sources[num]
        logger.info("🌐 Scraping Title %s — %s", num, label)
        docs = _scrape_justia_title(url, max_sections=max_sections)
        for doc in docs:
            n = ingester.ingest_text(doc["text"], source=doc["source"])
            total += n
        logger.info("  → %d chunks from Title %s", total, num)
    return total


def ingest_local(ingester) -> dict:
    """Ingest all local files from data/."""
    logger.info("📂 Ingesting local files from data/ ...")
    return ingester.ingest_directory()


# ── CLI ───────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TN-LawMaster TCA ingestion script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--titles", nargs="+", metavar="N",
        default=[],
        help="TCA title numbers to scrape from Justia (e.g. --titles 39 36 66)",
    )
    p.add_argument(
        "--max-sections", type=int, default=40,
        help="Max sections to scrape per title (default: 40, be polite to Justia)",
    )
    p.add_argument(
        "--seed-only", action="store_true",
        help="Only ingest the built-in synthetic seed corpus (no network)",
    )
    p.add_argument(
        "--local-only", action="store_true",
        help="Only ingest local files from data/ (no network)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Seed + scrape all titles + ingest local files",
    )
    p.add_argument(
        "--persist-dir", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        help="ChromaDB persistence directory",
    )
    p.add_argument(
        "--data-dir", default=os.getenv("DATA_DIR", "./data"),
        help="Local data directory containing PDFs/TXT files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ingester = _get_ingester(args.persist_dir, args.data_dir)
    logger.info("Vector store: %s  (existing docs: %d)", args.persist_dir, ingester.doc_count)

    total_new = 0

    # ── Seed corpus (always on --seed-only or --all, or if nothing else specified)
    if args.seed_only or args.all or (not args.titles and not args.local_only):
        total_new += seed_corpus(ingester)

    # ── Local files
    if args.local_only or args.all:
        summary = ingest_local(ingester)
        total_new += sum(summary.values())

    # ── Justia scrape
    if args.titles or args.all:
        titles = [str(t) for t, _, _ in TCA_SOURCES] if args.all else args.titles
        total_new += scrape_titles(ingester, titles, max_sections=args.max_sections)

    print(f"\n{'─'*50}")
    print(f"✅ Done!  New chunks added: {total_new}")
    print(f"📚 Total docs in store:    {ingester.doc_count}")
    print(f"📂 Persist path:           {args.persist_dir}")
    print(f"{'─'*50}\n")
    print("You can now start the agent:")
    print("  streamlit run streamlit_app.py")
    print("  python run_agent.py --domain criminal")
    print("  uvicorn main_fastapi:app --reload\n")


if __name__ == "__main__":
    main()
