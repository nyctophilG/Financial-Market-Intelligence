"""
main.py  --  lives at <project_root>/app/main.py
"""

import os
import sys
import threading
import warnings
import requests

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ──────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHROMA_DIR    = os.path.join(PROJECT_ROOT, "chroma_db")

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR,    exist_ok=True)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# ── Singletons -- loaded once per process, thread-safe ─────────
_lock        = threading.Lock()
_embeddings  = None
_chroma_db   = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from langchain_huggingface import HuggingFaceEmbeddings
                    print("[Init] Loading embedding model...")
                    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    print("[Init] Embedding model ready.")
    return _embeddings

def _get_chroma():
    global _chroma_db
    if _chroma_db is None:
        with _lock:
            if _chroma_db is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from langchain_chroma import Chroma
                    print("[Init] Connecting to ChromaDB...")
                    _chroma_db = Chroma(
                        persist_directory=CHROMA_DIR,
                        embedding_function=_get_embeddings(),
                    )
                    print("[Init] ChromaDB ready.")
    return _chroma_db


# ══════════════════════════════════════════════════════════════
# Routing triggers
# ══════════════════════════════════════════════════════════════
CREWAI_TRIGGERS = {
    "analyz", "analyse", "analysis",
    "run analysis", "run the analysis",
    "use crewai", "use the agents", "use agents",
    "run pipeline", "pipeline",
    "financial report", "generate report", "create report",
    "the file", "this file", "the document", "this document",
    "the filing", "the report", "from the",
    "according to", "in the document", "in the file",
    "summarize the", "summarize this",
    "read the file", "check the file",
    "what does the file", "what does the document", "what is in the",
    "sec", "10-k", "10k", "10-q", "10q", "8-k", "8k",
    "annual report", "quarterly report", "earnings report", "earnings call",
    "net sales", "gross profit", "gross margin",
    "operating income", "operating profit",
    "net income", "net profit", "net loss",
    "revenue", "revenues",
    "ebitda", "ebit",
    "eps", "earnings per share",
    "cash flow", "free cash flow",
    "balance sheet", "income statement", "financial statement",
    "profit and loss", "p&l",
    "total assets", "total liabilities", "shareholders equity",
    "dividend", "dividend yield",
    "market cap", "market capitalization",
    "pe ratio", "price to earnings",
    "ipo", "initial public offering",
    "risk factor", "risk factors",
    "fiscal year", "fiscal quarter",
    "q1 20", "q2 20", "q3 20", "q4 20",
    "quarter", "quarterly", "annual", "yearly",
    "aapl", "goog", "googl", "amzn", "msft", "tsla", "meta",
    "nvda", "nflx", "baba", "jpm",
}

def is_crewai_query(query: str) -> bool:
    lower = query.lower()
    for trigger in sorted(CREWAI_TRIGGERS, key=len, reverse=True):
        if trigger in lower:
            return True
    if _has_files() and len(query.split()) >= 4:
        question_words = {"what", "how", "show", "tell", "find",
                          "get", "list", "explain", "describe",
                          "compare", "which", "when", "where"}
        if set(lower.split()) & question_words:
            return True
    return False

def _has_files() -> bool:
    for d in [RAW_DIR, PROCESSED_DIR]:
        if os.path.isdir(d) and any(
            os.path.isfile(os.path.join(d, f)) for f in os.listdir(d)
        ):
            return True
    return False


# ══════════════════════════════════════════════════════════════
# File reading
# ══════════════════════════════════════════════════════════════

def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import pypdf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = pypdf.PdfReader(path)
                text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
            if text:
                print(f"[PDF] pypdf: {len(text)} chars from {os.path.basename(path)}")
                return text
        except ImportError:
            pass
        except Exception as e:
            print(f"[PDF] pypdf failed: {e}")

        try:
            import pypdfium2 as pdfium
            doc   = pdfium.PdfDocument(path)
            pages = []
            for page in doc:
                tp = page.get_textpage()
                pages.append(tp.get_text_range())
                tp.close()
                page.close()
            doc.close()
            text = "\n".join(pages).strip()
            if text:
                print(f"[PDF] pypdfium2: {len(text)} chars from {os.path.basename(path)}")
                return text
        except ImportError:
            pass
        except Exception as e:
            print(f"[PDF] pypdfium2 failed: {e}")

        try:
            import pdfminer.high_level as pm
            text = pm.extract_text(path).strip()
            if text:
                print(f"[PDF] pdfminer: {len(text)} chars from {os.path.basename(path)}")
                return text
        except ImportError:
            pass
        except Exception as e:
            print(f"[PDF] pdfminer failed: {e}")

        print(f"[PDF] All readers failed for {os.path.basename(path)} -- install pypdf")
        return ""
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[File] Read error: {e}")
            return ""


# ══════════════════════════════════════════════════════════════
# ChromaDB ingestion
# ══════════════════════════════════════════════════════════════

def ingest_to_chroma(file_path: str) -> str:
    fname = os.path.basename(file_path)
    print(f"[Ingest] Reading {fname}...")
    content = _read_file(file_path)

    if not content.strip():
        msg = f"Could not extract text from {fname}. Install pypdf: pip install pypdf"
        print(f"[Ingest] {msg}")
        return f"[Ingest] {msg}"

    print(f"[Ingest] {fname}: {len(content)} chars, chunking...")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_text(content)
        if not chunks:
            return f"No chunks from {fname}"

        docs = [
            Document(page_content=c, metadata={"source": file_path, "filename": fname})
            for c in chunks
        ]

        print(f"[Ingest] {fname}: {len(chunks)} chunks, embedding...")
        db = _get_chroma()
        db.add_documents(docs)

        global _chroma_count_cache
        _chroma_count_cache = None

        msg = f"[SUCCESS] {fname} -- {len(chunks)} chunks added to ChromaDB"
        print(f"[Ingest] {msg}")
        return msg

    except Exception as e:
        import traceback
        print(f"[Ingest] ERROR:\n{traceback.format_exc()}")
        return f"[ERROR] {fname}: {e}"


# ── ChromaDB count (cached) ────────────────────────────────────
_chroma_count_cache = None

def get_chroma_doc_count() -> int:
    global _chroma_count_cache
    if _chroma_count_cache is not None:
        return _chroma_count_cache
    try:
        db = _get_chroma()
        _chroma_count_cache = db._collection.count()
        return _chroma_count_cache
    except Exception as e:
        print(f"[ChromaDB] count error: {e}")
        return -1

def refresh_chroma_count() -> int:
    global _chroma_count_cache
    _chroma_count_cache = None
    return get_chroma_doc_count()


# ══════════════════════════════════════════════════════════════
# Ollama
# ══════════════════════════════════════════════════════════════

def call_ollama(prompt: str, system: str = "") -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or "_(Ollama returned empty)_"
    except requests.exceptions.ConnectionError:
        return "Ollama is not running.\n\nStart it:\n```\nollama serve\nollama pull llama3.1\n```"
    except Exception as e:
        return f"Ollama error: {e}"


# ══════════════════════════════════════════════════════════════
# CrewAI
# ══════════════════════════════════════════════════════════════

def run_crewai(query: str) -> dict:
    result = {"response": "", "error": ""}
    try:
        from agents.orchestrator import run_financial_analysis

        n = get_chroma_doc_count()
        print(f"\n[CrewAI] ChromaDB chunks: {n}")

        if n == 0:
            result["error"] = "no_docs"
            result["response"] = (
                "ChromaDB is empty.\n\n"
                "Upload your documents in the sidebar and click 'Save & Ingest' first."
            )
            return result

        crew_result = run_financial_analysis(query)
        raw = getattr(crew_result, "raw", None) or str(crew_result or "")
        result["response"] = raw.strip()

        if not result["response"]:
            result["error"] = "empty_output"
            result["response"] = "Pipeline ran but returned no output -- check terminal."

    except ImportError as e:
        result["error"] = str(e)
        result["response"] = (
            f"Import error: {e}\n\n"
            f"Run from project root:\n```\ncd <project_root>\nstreamlit run app/ui.py\n```"
        )
        print(f"[CrewAI ImportError] {e}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result["error"] = tb
        result["response"] = f"Pipeline error: {e}\n\n```\n{tb[:800]}\n```"
        print(f"[CrewAI ERROR]\n{tb}")

    return result


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def process_query(query: str) -> dict:
    if is_crewai_query(query):
        r = run_crewai(query)
        r["route"] = "crewai"
        return r
    return {
        "response": call_ollama(
            query,
            system=(
                "You are a helpful, friendly AI assistant with financial expertise. "
                "Answer naturally and conversationally. "
                "To analyze an uploaded document, use keywords like "
                "'analyze', 'from the file', or 'revenue'."
            ),
        ),
        "route": "ollama",
        "error": "",
    }


def save_uploaded_file(file_bytes: bytes, filename: str) -> tuple:
    ext      = os.path.splitext(filename)[1].lower()
    dest_dir = RAW_DIR if ext == ".pdf" else PROCESSED_DIR
    dest     = os.path.join(dest_dir, filename)
    with open(dest, "wb") as f:
        f.write(file_bytes)
    print(f"[Save] Written to {dest}")
    return dest, ""


def list_uploaded_files() -> dict:
    def ls(d):
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    return {"raw": ls(RAW_DIR), "processed": ls(PROCESSED_DIR)}


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"ChromaDB:     {CHROMA_DIR} ({get_chroma_doc_count()} chunks)")
    q = input("\nQuery: ").strip()
    if q:
        r = process_query(q)
        print(f"\n[{r['route']}]\n{r['response']}")
