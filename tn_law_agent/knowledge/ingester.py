"""
TN-LawMaster — TCA Document Ingester
=====================================
Loads Tennessee Code Annotated (TCA) documents from PDFs, plain-text,
or web scrapes, chunks them appropriately, and upserts into a vector
store (ChromaDB by default).

Usage::

    from tn_law_agent.knowledge.ingester import TNLawIngester

    ingester = TNLawIngester(data_dir="./data", persist_dir="./chroma_db")
    ingester.ingest_directory()          # batch ingest all files in data/
    ingester.ingest_pdf(Path("doc.pdf")) # single PDF
    ingester.ingest_url("https://...") # ingest from URL
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _load_chromadb():
    """Lazy import of ChromaDB to avoid hard dependency at import time."""
    try:
        import chromadb
        from chromadb.config import Settings
        return chromadb, Settings
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for document ingestion. "
            "Install with: pip install chromadb"
        ) from exc


def _load_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install with: pip install sentence-transformers"
        ) from exc


# ── Simple chunker ────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for better retrieval recall."""
    words = text.split()
    chunks: List[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── Ingester Class ────────────────────────────────────────

class TNLawIngester:
    """
    Ingest TCA documents into a ChromaDB vector store.

    The default embedding model is ``all-MiniLM-L6-v2`` (fast, free, local).
    Replace with any ``langchain_community.embeddings`` class if preferred.

    Args:
        data_dir:    Directory containing PDF / TXT source files.
        persist_dir: Directory where ChromaDB will persist the index.
        collection:  ChromaDB collection name.
        chunk_size:  Token window per chunk (in words, approx.).
        overlap:     Overlap between consecutive chunks (in words).
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        data_dir: str | Path = "./data",
        persist_dir: str | Path = "./chroma_db",
        collection: str = "tn_law",
        chunk_size: int = 1000,
        overlap: int = 150,
    ):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection
        self.chunk_size = chunk_size
        self.overlap = overlap

        chromadb, Settings = _load_chromadb()
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        SentenceTransformer = _load_sentence_transformers()
        self._embedder = SentenceTransformer(self.EMBED_MODEL)
        logger.info(
            "TNLawIngester ready  collection=%s  persist_dir=%s",
            self.collection_name,
            self.persist_dir,
        )

    # ── Ingestion methods ─────────────────────────────────

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Chunk and upsert raw text. Returns number of chunks added."""
        chunks = _chunk_text(text, self.chunk_size, self.overlap)
        return self._upsert_chunks(chunks, source)

    def ingest_pdf(self, path: Path | str) -> int:
        """Ingest a single PDF file. Returns number of chunks added."""
        path = Path(path)
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("pymupdf is required. Install with: pip install pymupdf")

        text_parts: List[str] = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        full_text = "\n".join(text_parts)
        logger.info("Ingesting PDF: %s (%d chars)", path.name, len(full_text))
        return self.ingest_text(full_text, source=path.name)

    def ingest_url(self, url: str) -> int:
        """Scrape plain text from a URL and ingest it."""
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        logger.info("Ingesting URL: %s (%d chars)", url, len(text))
        return self.ingest_text(text, source=url)

    def ingest_directory(self, exts: tuple[str, ...] = (".pdf", ".txt")) -> dict[str, int]:
        """
        Recursively ingest all matching files in ``data_dir``.

        Returns:
            A dict mapping filename → chunks_added.
        """
        summary: dict[str, int] = {}
        files = [p for p in self.data_dir.rglob("*") if p.suffix.lower() in exts]
        if not files:
            logger.warning("No files found in %s with extensions %s", self.data_dir, exts)
            return summary

        for f in files:
            try:
                if f.suffix.lower() == ".pdf":
                    n = self.ingest_pdf(f)
                else:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    n = self.ingest_text(text, source=f.name)
                summary[f.name] = n
                logger.info("  ✅ %s → %d chunks", f.name, n)
            except Exception as exc:
                logger.error("  ❌ %s — %s", f.name, exc)
                summary[f.name] = 0
        return summary

    # ── Search ────────────────────────────────────────────

    def search(self, query: str, top_k: int = 6) -> List[dict]:
        """
        Semantic similarity search against the ingested TCA corpus.

        Returns:
            List of dicts with keys: ``text``, ``source``, ``distance``
        """
        embedding = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [
            {"text": d, "source": m.get("source", "TCA"), "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

    @property
    def doc_count(self) -> int:
        """Number of chunks currently in the collection."""
        return self._collection.count()

    # ── Internal ──────────────────────────────────────────

    def _upsert_chunks(self, chunks: List[str], source: str) -> int:
        if not chunks:
            return 0
        embeddings = self._embedder.encode(chunks).tolist()
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )
        logger.debug("Upserted %d chunks from source '%s'", len(chunks), source)
        return len(chunks)
