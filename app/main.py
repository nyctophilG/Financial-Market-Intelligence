"""
main.py

Routing:
  General questions  → Ollama llama3.1 directly
  Financial queries  → CrewAI 3-agent pipeline

Key fixes vs previous version:
  - sys.path patched BEFORE any agents import so CrewAI finds agents/
  - Uploaded files are ingested into ChromaDB so Data Gatherer can find them
  - stdout is NO LONGER swallowed — CrewAI prints go to terminal as normal
  - Errors are surfaced, not silently caught
"""

import os
import sys
import requests

# ── Path setup — MUST happen before any agents import ────────
# Assumes structure:  <project_root>/
#                       agents/
#                       data/raw, data/processed
#                       chroma_db/
#                       ui.py  (or app/ui.py)
#                       main.py (or app/main.py)

# The file that contains main.py
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
# Walk up until we find the 'agents' folder — that is the project root
_check = THIS_DIR
PROJECT_ROOT = THIS_DIR
for _ in range(4):
    if os.path.isdir(os.path.join(_check, "agents")):
        PROJECT_ROOT = _check
        break
    _check = os.path.dirname(_check)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# ── Data directories (absolute) ───────────────────────────────
RAW_DATA_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHROMA_DB_DIR      = os.path.join(PROJECT_ROOT, "chroma_db")

os.makedirs(RAW_DATA_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR,      exist_ok=True)

# ── Ollama ────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# ── Keywords that route to CrewAI ─────────────────────────────
FINANCIAL_KEYWORDS = {
    "sec", "10-k", "10k", "10q", "8-k", "earnings", "revenue", "profit",
    "loss", "stock", "equity", "market cap", "portfolio", "risk", "asset",
    "liability", "balance sheet", "income statement", "cash flow", "dividend",
    "ipo", "quarterly", "annual report", "fiscal", "ebitda", "eps",
    "pe ratio", "nasdaq", "nyse", "hedge", "derivative", "bond", "yield",
    "inflation", "interest rate", "gdp", "financial report", "investment",
    "shares", "ticker", "valuation", "audit", "filing", "q1", "q2", "q3", "q4",
    "net sales", "gross margin", "operating income", "net income",
}

DOC_TRIGGERS = {
    "summarize", "analyze", "what does", "according to",
    "from the document", "in the file", "the report", "the filing",
    "what is in", "read the", "check the",
}


# ══════════════════════════════════════════════════════════════
# ChromaDB ingestion
# ══════════════════════════════════════════════════════════════

def ingest_file_to_chroma(file_path: str) -> str:
    """
    Read a file, chunk it, embed it, and store it in ChromaDB.
    This makes it searchable by the Data Gatherer agent.
    Returns a status message.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
# If the old paths fail, these are the modern locations:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        # Read content
        content = _read_file(file_path)
        if not content or content.startswith("["):
            return f"⚠️ Could not read {os.path.basename(file_path)}"

        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        docs = [
            Document(
                page_content=chunk,
                metadata={"source": file_path, "filename": os.path.basename(file_path)}
            )
            for chunk in chunks
        ]

        # Embed and store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        db.add_documents(docs)

        return f"✅ Ingested {os.path.basename(file_path)} — {len(chunks)} chunks added to ChromaDB"

    except ImportError as e:
        return f"⚠️ Missing library for ingestion: {e}\nRun: pip install langchain-chroma langchain-huggingface sentence-transformers"
    except Exception as e:
        return f"❌ Ingestion error for {os.path.basename(file_path)}: {e}"


def _read_file(path: str) -> str:
    """Read any file to text."""
    try:
        if path.lower().endswith(".pdf"):
            try:
                from pypdfium2 import PdfReader
                return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
            except ImportError:
                pass
            try:
                import pdfminer.high_level as pm
                return pm.extract_text(path)
            except ImportError:
                return f"[Install pypdf: pip install pypdf]"
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"[Read error: {e}]"


# ══════════════════════════════════════════════════════════════
# Query classifier
# ══════════════════════════════════════════════════════════════

def is_financial_query(query: str) -> bool:
    lower = query.lower()
    for kw in FINANCIAL_KEYWORDS:
        if kw in lower:
            return True
    # If files are in data dirs, also trigger on doc-question words
    if list_uploaded_files()["raw"] or list_uploaded_files()["processed"]:
        for t in DOC_TRIGGERS:
            if t in lower:
                return True
    return False


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
        answer = resp.json().get("response", "").strip()
        return answer or "_(Ollama returned empty)_"
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ **Ollama is not running.**\n\n"
            "Start it:\n```bash\nollama serve\n```\n"
            "Pull the model:\n```bash\nollama pull llama3.1\n```"
        )
    except Exception as e:
        return f"⚠️ Ollama error: `{e}`"


# ══════════════════════════════════════════════════════════════
# CrewAI runner
# ══════════════════════════════════════════════════════════════

def run_crewai(query: str) -> dict:
    """
    Run the CrewAI pipeline.
    stdout is NOT redirected — CrewAI verbose output prints to terminal normally.
    Returns dict with 'response' and 'error'.
    """
    result = {"response": "", "error": ""}

    try:
        # Import here so path fix above takes effect first
        from agents.orchestrator import run_financial_analysis

        print(f"\n{'='*60}")
        print(f"[CrewAI] Starting pipeline for: {query[:80]}")
        print(f"{'='*60}\n")

        crew_result = run_financial_analysis(query)

        # CrewAI returns CrewOutput object — get .raw string
        raw = getattr(crew_result, "raw", None)
        if raw is None:
            raw = str(crew_result) if crew_result is not None else ""

        result["response"] = raw.strip()

        if not result["response"]:
            result["error"] = "CrewAI pipeline ran but returned empty output."

    except ImportError as e:
        result["error"] = f"ImportError: {e}"
        result["response"] = (
            f"🚨 **Cannot import CrewAI agents.**\n\n"
            f"`{e}`\n\n"
            f"**Make sure you run Streamlit from the project root:**\n"
            f"```bash\ncd <project_root>\nstreamlit run ui.py\n```"
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result["error"] = tb
        result["response"] = f"🚨 **Pipeline error:** `{e}`\n\n```\n{tb}\n```"
        print(f"\n[CrewAI ERROR]\n{tb}")

    return result


# ══════════════════════════════════════════════════════════════
# Public API  (called by ui.py)
# ══════════════════════════════════════════════════════════════

def process_query(query: str) -> dict:
    """
    Returns:
      response  — answer string for the UI
      route     — 'ollama' | 'crewai'
      error     — error message if something went wrong
    """
    if not is_financial_query(query):
        return {
            "response": call_ollama(
                query,
                system=(
                    "You are a helpful, friendly assistant with financial expertise. "
                    "Answer naturally and conversationally."
                ),
            ),
            "route": "ollama",
            "error": "",
        }

    result = run_crewai(query)
    result["route"] = "crewai"
    return result


# ══════════════════════════════════════════════════════════════
# File management
# ══════════════════════════════════════════════════════════════

def save_uploaded_file(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Save file to correct directory and ingest into ChromaDB.
    Returns (saved_path, ingest_status_message).
    """
    ext = os.path.splitext(filename)[1].lower()
    dest_dir = RAW_DATA_DIR if ext == ".pdf" else PROCESSED_DATA_DIR
    dest = os.path.join(dest_dir, filename)

    with open(dest, "wb") as f:
        f.write(file_bytes)

    # Ingest into ChromaDB so agents can search it
    ingest_msg = ingest_file_to_chroma(dest)
    print(f"[Ingestion] {ingest_msg}")

    return dest, ingest_msg


def list_uploaded_files() -> dict:
    def ls(d):
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    return {"raw": ls(RAW_DATA_DIR), "processed": ls(PROCESSED_DATA_DIR)}


def get_chroma_doc_count() -> int:
    """Return number of documents currently in ChromaDB."""
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return db._collection.count()
    except Exception:
        return -1


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─── Financial Market Intelligence CLI ───")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"ChromaDB:     {CHROMA_DB_DIR}")
    q = input("\nQuery: ").strip()
    if q:
        r = process_query(q)
        print(f"\n[Route: {r['route']}]")
        print(r["response"])
        if r["error"]:
            print(f"\n[Error]\n{r['error']}")
