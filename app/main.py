"""
main.py  --  <project_root>/app/main.py
"""

import os
import sys
import warnings
import requests

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Must match data_gatherer.py exactly:
# CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")
# __file__ there = <project_root>/agents/data_gatherer.py
# so CHROMA_PATH = <project_root>/agents/../chroma_db = <project_root>/chroma_db
CHROMA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "agents", "..", "chroma_db"))

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR,    exist_ok=True)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"


# =============================================================
# Routing
# =============================================================
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
        q_words = {"what", "how", "show", "tell", "find",
                   "get", "list", "explain", "describe",
                   "compare", "which", "when", "where"}
        if set(lower.split()) & q_words:
            return True
    return False

def _has_files() -> bool:
    for d in [RAW_DIR, PROCESSED_DIR]:
        if os.path.isdir(d) and any(
            os.path.isfile(os.path.join(d, f)) for f in os.listdir(d)
        ):
            return True
    return False


# =============================================================
# ChromaDB count -- raw sqlite, no embedding model needed
# =============================================================

def get_chroma_doc_count() -> int:
    """
    Count docs by reading ChromaDB's sqlite directly.
    Uses same CHROMA_DIR as data_gatherer.py.
    No embedding model load -- instant.
    """
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.isfile(sqlite_path):
        return 0
    try:
        import sqlite3
        conn = sqlite3.connect(sqlite_path)
        # ChromaDB stores embeddings in 'embeddings' table
        cur = conn.execute("SELECT COUNT(*) FROM embeddings")
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception as e:
        print(f"[ChromaDB count] {e}")
        return -1

def refresh_chroma_count() -> int:
    return get_chroma_doc_count()


# =============================================================
# File save (bytes to disk only, no embedding here)
# =============================================================

def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    ext      = os.path.splitext(filename)[1].lower()
    dest_dir = RAW_DIR if ext == ".pdf" else PROCESSED_DIR
    dest     = os.path.join(dest_dir, filename)
    with open(dest, "wb") as f:
        f.write(file_bytes)
    print(f"[Save] Written to {dest}")
    return dest

def list_uploaded_files() -> dict:
    def ls(d):
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    return {"raw": ls(RAW_DIR), "processed": ls(PROCESSED_DIR)}


# =============================================================
# Ollama
# =============================================================

def call_ollama(prompt: str, system: str = "") -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or "_(Ollama returned empty)_"
    except requests.exceptions.ConnectionError:
        return "Ollama is not running.\n\n```\nollama serve\nollama pull llama3.1\n```"
    except Exception as e:
        return f"Ollama error: {e}"


# =============================================================
# CrewAI
# =============================================================

def run_crewai(query: str) -> dict:
    result = {"response": "", "error": ""}
    try:
        from agents.orchestrator import run_financial_analysis

        n = get_chroma_doc_count()
        print(f"\n[CrewAI] ChromaDB chunks: {n}")
        print(f"[CrewAI] ChromaDB path:   {CHROMA_DIR}")

        if n == 0:
            result["error"] = "no_docs"
            result["response"] = (
                "ChromaDB is empty.\n\n"
                "Upload your documents and click 'Save & Ingest' first."
            )
            return result

        crew_result = run_financial_analysis(query)
        raw = getattr(crew_result, "raw", None) or str(crew_result or "")
        result["response"] = raw.strip()

        if not result["response"]:
            result["error"]    = "empty_output"
            result["response"] = "Pipeline ran but returned no output -- check terminal."

    except ImportError as e:
        result["error"]    = str(e)
        result["response"] = (
            f"Import error: {e}\n\n"
            "Run from project root:\n```\nstreamlit run app/ui.py\n```"
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result["error"]    = tb
        result["response"] = f"Pipeline error: {e}\n\n```\n{tb[:800]}\n```"
        print(f"[CrewAI ERROR]\n{tb}")
    return result


# =============================================================
# Public API
# =============================================================

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
                "To analyze an uploaded document use keywords like "
                "'analyze', 'from the file', or 'revenue'."
            ),
        ),
        "route": "ollama",
        "error": "",
    }


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"ChromaDB     : {CHROMA_DIR}")
    print(f"Count        : {get_chroma_doc_count()} chunks")
    q = input("\nQuery: ").strip()
    if q:
        r = process_query(q)
        print(f"\n[{r['route']}]\n{r['response']}")
