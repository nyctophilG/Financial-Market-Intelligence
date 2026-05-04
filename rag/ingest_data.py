"""
ingest_data.py — Multi-PDF ingestion pipeline.

Drop any number of SEC 10-K PDFs into data/raw/ and run this script once.
Each chunk is stored with metadata (company, ticker, fiscal_year, source_file)
so the retriever can filter by company when needed.

Usage:
    python ingest_data.py                  # ingest all PDFs in data/raw/
    python ingest_data.py --reset          # wipe DB first, then ingest
    python ingest_data.py --file tsla.pdf  # ingest a single file only
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
# ROOT_DIR always points to the project root regardless of where this script
# lives (e.g. rag/, scripts/, or the root itself).
# We walk UP from the script's location until we find the folder that already
# contains data/raw/, or fall back to the script's own directory.
def _find_project_root() -> str:
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(candidate, "data", "raw")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:   # reached filesystem root
            break
        candidate = parent
    # Fallback: use the script's own directory (data/raw/ will be created there)
    return os.path.dirname(os.path.abspath(__file__))

ROOT_DIR    = _find_project_root()
RAW_DIR     = os.path.join(ROOT_DIR, "data", "raw")
CHROMA_PATH = os.path.join(ROOT_DIR, "chroma_db")

# Must match model name in data_gatherer.py exactly.
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Company detection ─────────────────────────────────────────────────────────
# Maps lowercase filename fragments → (company_name, ticker).
# Add a row here whenever you add a new company.
COMPANY_MAP = {
    "tsla":      ("Tesla, Inc.",           "TSLA"),
    "tesla":     ("Tesla, Inc.",           "TSLA"),
    "msft":      ("Microsoft Corporation", "MSFT"),
    "microsoft": ("Microsoft Corporation", "MSFT"),
    "aapl":      ("Apple Inc.",            "AAPL"),
    "apple":     ("Apple Inc.",            "AAPL"),
    "amzn":      ("Amazon.com, Inc.",      "AMZN"),
    "amazon":    ("Amazon.com, Inc.",      "AMZN"),
    "googl":     ("Alphabet Inc.",         "GOOGL"),
    "alphabet":  ("Alphabet Inc.",         "GOOGL"),
    "meta":      ("Meta Platforms, Inc.",  "META"),
    "nvda":      ("NVIDIA Corporation",    "NVDA"),
    "nvidia":    ("NVIDIA Corporation",    "NVDA"),
}

def detect_company(filename: str) -> tuple:
    lower = filename.lower()
    for fragment, (name, ticker) in COMPANY_MAP.items():
        if fragment in lower:
            return name, ticker
    stem = os.path.splitext(filename)[0].upper()
    print(f"  Warning: Unknown company for '{filename}'. Storing as ticker='{stem}'.")
    return stem, stem

def detect_fiscal_year(filename: str, pdf_path: str = None) -> str:
    # 1. Try filename first (fastest)
    match = re.search(r'(20\d{2})', filename)
    if match:
        return match.group(1)
    # 2. Scan first 3 pages of the PDF as fallback
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
        except Exception:
            pass
    return "unknown"

# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_documents(pdf_path: str) -> list:
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

            # Text chunk — company header prepended so similarity search
            # always retrieves the right company even on generic queries.
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
                    " | ".join(str(cell).replace('\n', ' ') if cell else "" for cell in row)
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
    # Batch to avoid memory spikes on large PDFs
    batch_size = 100
    for start in range(0, len(docs), batch_size):
        db.add_documents(docs[start : start + batch_size])
    return len(docs)

# ── Main ──────────────────────────────────────────────────────────────────────
def run(target_file=None, reset=False):
    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Wiped existing DB at {CHROMA_PATH}")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
    already_done = get_already_ingested(db)

    if already_done:
        print(f"Already in DB: {', '.join(sorted(already_done))}")

    # Collect PDFs to process
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
        print(f"No PDFs found in {RAW_DIR}/  — drop your 10-K PDFs there and re-run.")
        return

    total_chunks = 0
    skipped = []
    processed = []

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

    # Summary
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    for fname, n in processed:
        _, ticker = detect_company(fname)
        print(f"  {ticker:6s} | {fname:45s} | {n:4d} chunks")
    if skipped:
        print(f"\n  Skipped (already in DB): {', '.join(skipped)}")
    print(f"\n  Total new chunks : {total_chunks}")
    print(f"  DB location      : {CHROMA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K PDFs into ChromaDB.")
    parser.add_argument("--file",  type=str, default=None, help="Ingest a single PDF by name or path.")
    parser.add_argument("--reset", action="store_true",    help="Wipe the DB before ingesting.")
    args = parser.parse_args()
    run(target_file=args.file, reset=args.reset)
