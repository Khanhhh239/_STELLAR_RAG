"""
STELLAR-RAG v4 (improve_RAG) — Ingestion entry point.

Usage:
    python ingest.py

Reads PDFs from data/raw/, builds the GraphRAG index (dense FAISS + BM25 +
knowledge graph + entity index + EHRAG hypergraph), and saves everything to
storage/.
"""
from __future__ import annotations

import io
import json
import os
import sys

# ── Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import settings
from graphrag import GraphRAG
from pdf_pipeline import PdfPipeline


def main() -> None:
    settings.ensure_dirs()
    pipeline = PdfPipeline()
    chunks   = pipeline.ingest_folder(settings.data_raw)
    if not chunks:
        print("No PDF found in data/raw/")
        return

    gr = GraphRAG()
    gr.build(chunks)

    serializable = [chunk.__dict__ for chunk in chunks]
    (settings.data_processed / "chunks.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nIngestion complete: {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    main()
