"""
ingest_data.py — Multi-PDF ingestion pipeline.
Lives at project_root/rag/ingest_data.py

Drop any number of SEC 10-K PDFs into project_root/rag/data/raw/ and run:
    cd project_root/rag
    python ingest_data.py

Directory layout this script creates/uses:
    project_root/rag/
        data/
            raw/        <- drop your PDFs here
        chroma_db/      <- vector DB is written here
        ingest_data.py  <- this file

data_gatherer.py in project_root/agents/ reads from project_root/rag/chroma_db/
so both files must agree on that path. If you ever move this script, update
CHROMA_PATH in data_gatherer.py to match.

IMPORTANT — Apple PDF naming:
    The Apple 10-K may be named '_10-K-2025-As-Filed.pdf', which contains no
    recognisable company fragment. Rename it before ingesting:
        mv '_10-K-2025-As-Filed.pdf' 'aapl-10k-2025.pdf'
    The COMPANY_MAP below also includes the original name as a safety fallback.

Usage:
    python ingest_data.py                        # ingest all PDFs in data/raw/
    python ingest_data.py --reset                # wipe DB first, then ingest
    python ingest_data.py --file aapl-10k-2025.pdf  # ingest a single file
"""

import os
import re
import argparse
import shutil
import pdfplumber
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ── Paths ─────────────────────────────────────────────────────────────────────
# This script lives at project_root/rag/ingest_data.py
# dirname(__file__) = project_root/rag/
_RAG_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(_RAG_DIR, "data", "raw")
CHROMA_PATH = os.path.join(_RAG_DIR, "chroma_db")

# Must match model name in agents/data_gatherer.py exactly.
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Company detection ─────────────────────────────────────────────────────────
# Maps lowercase filename fragments -> (company_name, ticker).
# Keep this in sync with TICKER_MAP in agents/data_gatherer.py.
COMPANY_MAP = {
    # Tesla
    "tsla":                ("Tesla, Inc.",           "TSLA"),
    "tesla":               ("Tesla, Inc.",           "TSLA"),
    # Microsoft
    "msft":                ("Microsoft Corporation", "MSFT"),
    "microsoft":           ("Microsoft Corporation", "MSFT"),
    # Apple — canonical fragment AND original SEC filename fallback
    "aapl":                ("Apple Inc.",            "AAPL"),
    "apple":               ("Apple Inc.",            "AAPL"),
    "10-k-2025-as-filed":  ("Apple Inc.",            "AAPL"),  # original filename safety net
    # Amazon
    "amzn":                ("Amazon.com, Inc.",      "AMZN"),
    "amazon":              ("Amazon.com, Inc.",      "AMZN"),
    # Alphabet / Google
    "googl":               ("Alphabet Inc.",         "GOOGL"),
    "alphabet":            ("Alphabet Inc.",         "GOOGL"),
    # Meta
    "meta":                ("Meta Platforms, Inc.",  "META"),
    # NVIDIA
    "nvda":                ("NVIDIA Corporation",    "NVDA"),
    "nvidia":              ("NVIDIA Corporation",    "NVDA"),
}


def detect_company(filename: str) -> tuple:
    """
    Detect company name and ticker from a PDF filename.
    Sorts fragments longest-first so specific patterns match before short ones.
    Falls back to uppercased filename stem with a warning.
    """
    lower = filename.lower()
    for fragment in sorted(COMPANY_MAP.keys(), key=len, reverse=True):
        if fragment in lower:
            return COMPANY_MAP[fragment]

    stem = os.path.splitext(filename)[0].upper()
    print(f"  [WARNING] Unknown company for '{filename}'. Storing as ticker='{stem}'.")
    print(f"  [WARNING] Add a fragment for this file to COMPANY_MAP in ingest_data.py")
    print(f"  [WARNING] AND to TICKER_MAP in agents/data_gatherer.py.")
    return stem, stem


def detect_fiscal_year(filename: str, pdf_path: str = None) -> str:
    """Detect fiscal year from filename, then from first 3 PDF pages as fallback."""
    match = re.search(r'(20\d{2})', filename)
    if match:
        return match.group(1)

    if pdf_path:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:3]:
                    text = page.extract_text() or ""
                    m = re.search(r'fiscal year ended[^\d]*(20\d{2})', text, re.IGNORECASE)
                    if m:
                        return m.group(1)
                    m = re.search(r'for the year ended[^\d]*(20\d{2})', text, re.IGNORECASE)
                    if m:
                        return m.group(1)
        except Exception as e:
            print(f"  [WARNING] Could not scan PDF for fiscal year: {e}")

    return "unknown"


# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_documents(pdf_path: str) -> list:
    """
    Extract text and table chunks from a PDF.
    Company name and ticker are prepended to every chunk so similarity search
    always retrieves the right company even on generic financial queries.
    """
    filename = os.path.basename(pdf_path)
    company_name, ticker = detect_company(filename)
    fiscal_year = detect_fiscal_year(filename, pdf_path)

    print(f"  Company      : {company_name} ({ticker})")
    print(f"  Fiscal year  : {fiscal_year}")
    print(f"  Extracting   : {filename} ...")

    base_meta = {
        "company":     company_name,
        "ticker":      ticker,
        "fiscal_year": fiscal_year,
        "source_file": filename,
    }

    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):

            # Text chunk
            text = page.extract_text()
            if text and text.strip():
                content = (
                    f"[Company: {company_name} | Ticker: {ticker} | "
                    f"Fiscal Year: {fiscal_year} | Page: {i+1}]\n"
                    f"{text.strip()}"
                )
                docs.append(Document(
                    page_content=content,
                    metadata={**base_meta, "page": i + 1, "chunk_type": "text"}
                ))

            # Table chunks
            for j, table in enumerate(page.extract_tables()):
                rows = [
                    " | ".join(
                        str(cell).replace('\n', ' ') if cell else ""
                        for cell in row
                    )
                    for row in table
                ]
                table_str = "\n".join(rows)
                if not table_str.strip():
                    continue

                content = (
                    f"[FINANCIAL TABLE | Company: {company_name} | Ticker: {ticker} | "
                    f"Fiscal Year: {fiscal_year} | Page: {i+1} | Table: {j+1}]\n"
                    f"{table_str}"
                )
                docs.append(Document(
                    page_content=content,
                    metadata={**base_meta, "page": i + 1, "chunk_type": "table", "table_index": j}
                ))

    print(f"  Extracted {len(docs)} chunks.")
    return docs


# ── DB helpers ────────────────────────────────────────────────────────────────
def get_already_ingested(db: Chroma) -> set:
    try:
        result = db.get(include=["metadatas"])
        return {m.get("source_file", "") for m in result["metadatas"]}
    except Exception:
        return set()


def ingest_pdf(pdf_path: str, db: Chroma) -> int:
    docs = extract_documents(pdf_path)
    if not docs:
        print("  No content extracted — skipping.")
        return 0
    batch_size = 100
    for start in range(0, len(docs), batch_size):
        db.add_documents(docs[start : start + batch_size])
    return len(docs)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(target_file=None, reset=False):
    print(f"[INFO] RAW_DIR     : {RAW_DIR}")
    print(f"[INFO] CHROMA_PATH : {CHROMA_PATH}")

    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Wiped existing DB at {CHROMA_PATH}")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
    already_done = get_already_ingested(db)

    if already_done:
        print(f"Already in DB: {', '.join(sorted(already_done))}")

    if target_file:
        pdf_paths = [
            target_file if os.path.isabs(target_file)
            else os.path.join(RAW_DIR, target_file)
        ]
    else:
        os.makedirs(RAW_DIR, exist_ok=True)
        pdf_paths = [
            os.path.join(RAW_DIR, f)
            for f in sorted(os.listdir(RAW_DIR))
            if f.lower().endswith(".pdf")
        ]

    if not pdf_paths:
        print(f"\nNo PDFs found in {RAW_DIR}/")
        print("Drop your 10-K PDFs there and re-run.")
        print("\nNaming convention:")
        print("  Tesla   -> tsla-10k-2023.pdf")
        print("  Apple   -> aapl-10k-2025.pdf   (rename from '_10-K-2025-As-Filed.pdf')")
        print("  MSFT    -> msft-10k-2025.pdf")
        return

    total_chunks = 0
    skipped      = []
    processed    = []

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")

        if filename in already_done and not reset:
            print(f"  Already ingested — skipping. (Use --reset to re-ingest.)")
            skipped.append(filename)
            continue

        if not os.path.exists(pdf_path):
            print(f"  File not found: {pdf_path}")
            continue

        n = ingest_pdf(pdf_path, db)
        total_chunks += n
        processed.append((filename, n))

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    for fname, n in processed:
        _, ticker = detect_company(fname)
        print(f"  {ticker:6s} | {fname:50s} | {n:4d} chunks")
    if skipped:
        print(f"\n  Skipped (already in DB): {', '.join(skipped)}")
    print(f"\n  Total new chunks : {total_chunks}")
    print(f"  DB location      : {CHROMA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K PDFs into ChromaDB.")
    parser.add_argument("--file",  type=str, default=None,
                        help="Ingest a single PDF by name or absolute path.")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the DB before ingesting.")
    args = parser.parse_args()
    run(target_file=args.file, reset=args.reset)
