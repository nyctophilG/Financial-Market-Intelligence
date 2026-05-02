"""
main.py  —  lives at <project_root>/app/main.py

Project layout:
  <project_root>/
    app/
      main.py   ← this file
      ui.py
    agents/
      orchestrator.py
      data_gatherer.py
      financial_analyst.py
      risk_monitor.py
    data/
      raw/        ← PDFs saved here
      processed/  ← CSV / TXT saved here
    chroma_db/    ← ChromaDB vector store (data_gatherer.py points here)
    prompts/
"""

import os
import sys
import requests
import warnings

# Suppress the noisy transformers __path__ warning
warnings.filterwarnings("ignore", message=".*__path__.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ─────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))   # .../app
PROJECT_ROOT = os.path.dirname(APP_DIR)                      # .../

# Add project root so `from agents.x import y` works
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Data dirs live at project root, NOT inside app/
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHROMA_DIR    = os.path.join(PROJECT_ROOT, "chroma_db")   # matches data_gatherer.py

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR,    exist_ok=True)

# ── Ollama ─────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# ── Routing keywords ───────────────────────────────────────────
# These FORCE the CrewAI pipeline regardless of phrasing
CREWAI_HARD_TRIGGERS = {
    # Explicit pipeline requests
    "analyze the file", "analyse the file",
    "analyze the document", "analyse the document",
    "analyze this document", "analyse this document",
    "run analysis", "run financial analysis",
    "use crewai", "use the agents",
    "search the database", "search database",
    "check the filing", "check the file",
    "summarize the file", "summarize the document",
    "summarize this file", "summarize this document",
    "what does the file say", "what does the document say",
    "from the file", "from the document", "from the filing",
    "in the file", "in the document",
    "according to the", "the report says",
    "read the file", "read the document",
    # SEC / regulatory
    "sec filing", "sec report", "10-k", "10k", "10-q", "10q", "8-k", "8k",
    "annual report", "quarterly report", "earnings report",
    # Financial statement terms
    "net sales", "gross profit", "operating income", "net income",
    "revenue", "ebitda", "eps", "pe ratio", "cash flow",
    "balance sheet", "income statement", "earnings per share",
    # Market terms that imply data lookup
    "stock price", "market cap", "dividend yield", "ipo",
    "risk factor", "risk factors",
    "q1 20", "q2 20", "q3 20", "q4 20",   # e.g. "Q3 2023"
}

# These alone are NOT enough — Ollama handles general finance questions
OLLAMA_FINANCE_WORDS = {
    "financial", "finance", "investment", "market", "economy",
    "inflation", "interest rate", "stock", "equity", "bond",
    "asset", "portfolio", "hedge", "fund",
}


def is_crewai_query(query: str) -> bool:
    """
    Returns True only when the query clearly asks for document/pipeline analysis.
    General financial chat → Ollama. Document/data queries → CrewAI.
    """
    lower = query.lower()

    # Check hard triggers (multi-word phrases first for accuracy)
    sorted_triggers = sorted(CREWAI_HARD_TRIGGERS, key=len, reverse=True)
    for trigger in sorted_triggers:
        if trigger in lower:
            return True

    # If files are uploaded AND user asks a question about content
    if _has_uploaded_files():
        doc_question_words = {"what", "how much", "how many", "show", "tell",
                              "find", "get", "list", "explain", "describe",
                              "compare", "which", "when", "where", "who"}
        words = set(lower.split())
        # Has question intent AND is not a casual greeting/general statement
        if words & doc_question_words and len(query.split()) >= 4:
            # Extra confirmation: mentions a financial noun
            financial_nouns = {
                "apple", "google", "amazon", "microsoft", "tesla", "meta",
                "aapl", "goog", "amzn", "msft", "tsla",
                "profit", "loss", "sales", "revenue", "income",
                "liability", "asset", "debt", "equity", "capital",
            }
            if words & financial_nouns:
                return True

    return False


def _has_uploaded_files() -> bool:
    for d in [RAW_DIR, PROCESSED_DIR]:
        if os.path.isdir(d) and os.listdir(d):
            return True
    return False


# ══════════════════════════════════════════════════════════════
# ChromaDB ingestion
# ══════════════════════════════════════════════════════════════

def ingest_to_chroma(file_path: str) -> str:
    """
    Chunk a file and embed it into ChromaDB at CHROMA_DIR.
    Uses the exact same embedding model as data_gatherer.py (all-MiniLM-L6-v2).
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document

        fname = os.path.basename(file_path)
        content = _read_file(file_path)

        if not content or not content.strip():
            return f"⚠️ {fname} — could not extract text"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_text(content)
        if not chunks:
            return f"⚠️ {fname} — no chunks produced"

        docs = [
            Document(
                page_content=chunk,
                metadata={"source": file_path, "filename": fname},
            )
            for chunk in chunks
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            db.add_documents(docs)

        return f"✅ {fname} — {len(chunks)} chunks → ChromaDB"

    except ImportError as e:
        return (
            f"❌ Missing package: {e}\n"
            f"Run: pip install langchain-chroma langchain-huggingface sentence-transformers"
        )
    except Exception as e:
        return f"❌ {os.path.basename(file_path)}: {e}"


def _read_file(path: str) -> str:
    try:
        if path.lower().endswith(".pdf"):
            try:
                from pypdfium2 import PdfReader
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
            except ImportError:
                pass
            try:
                import pdfminer.high_level as pm
                return pm.extract_text(path)
            except ImportError:
                return ""
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""


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
        return resp.json().get("response", "").strip() or "_(empty response from Ollama)_"
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ **Ollama is not running.**\n\n"
            "```bash\nollama serve\nollama pull llama3.1\n```"
        )
    except Exception as e:
        return f"⚠️ Ollama error: `{e}`"


# ══════════════════════════════════════════════════════════════
# CrewAI
# ══════════════════════════════════════════════════════════════

def run_crewai(query: str) -> dict:
    """
    Run the 3-agent CrewAI pipeline.
    stdout NOT redirected — verbose output prints to terminal as normal.
    """
    result = {"response": "", "error": ""}
    try:
        from agents.orchestrator import run_financial_analysis

        print(f"\n{'─'*60}")
        print(f"[CrewAI] Pipeline start: {query[:100]}")
        print(f"[CrewAI] ChromaDB path:  {CHROMA_DIR}")
        chroma_count = get_chroma_doc_count()
        print(f"[CrewAI] ChromaDB docs:  {chroma_count}")
        print(f"{'─'*60}\n")

        if chroma_count == 0:
            result["error"] = "no_docs"
            result["response"] = (
                "⚠️ **ChromaDB is empty — no documents have been ingested yet.**\n\n"
                "Upload your files using the sidebar and click **'💾 Save & Ingest'** first.\n"
                "The Data Gatherer agent searches ChromaDB, so files must be ingested before analysis."
            )
            return result

        crew_result = run_financial_analysis(query)
        raw = getattr(crew_result, "raw", None) or str(crew_result or "")
        result["response"] = raw.strip()

        if not result["response"]:
            result["error"] = "empty_output"
            result["response"] = "_(CrewAI pipeline ran but returned no output — check terminal)_"

    except ImportError as e:
        result["error"] = str(e)
        result["response"] = (
            f"🚨 **Import error:** `{e}`\n\n"
            f"Run Streamlit from the **project root**:\n"
            f"```bash\ncd <project_root>\nstreamlit run app/ui.py\n```"
        )
        print(f"[CrewAI ImportError] {e}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result["error"] = tb
        result["response"] = f"🚨 **Pipeline error:** `{e}`\n\n```\n{tb[:800]}\n```"
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
                "You are a helpful, friendly AI assistant with financial knowledge. "
                "Answer naturally and conversationally. "
                "If the user wants to analyze a specific document or file, "
                "tell them to use keywords like 'analyze the file' or 'run analysis'."
            ),
        ),
        "route": "ollama",
        "error": "",
    }


def save_uploaded_file(file_bytes: bytes, filename: str) -> tuple:
    """Save to correct data dir and ingest into ChromaDB. Returns (path, status_msg)."""
    ext = os.path.splitext(filename)[1].lower()
    dest_dir = RAW_DIR if ext == ".pdf" else PROCESSED_DIR
    dest = os.path.join(dest_dir, filename)

    with open(dest, "wb") as f:
        f.write(file_bytes)

    msg = ingest_to_chroma(dest)
    print(f"[Ingest] {msg}")
    return dest, msg


def list_uploaded_files() -> dict:
    def ls(d):
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    return {"raw": ls(RAW_DIR), "processed": ls(PROCESSED_DIR)}


def get_chroma_doc_count() -> int:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            return db._collection.count()
    except Exception:
        return -1


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"ChromaDB     : {CHROMA_DIR}")
    print(f"ChromaDB docs: {get_chroma_doc_count()}")
    q = input("\nQuery: ").strip()
    if q:
        r = process_query(q)
        print(f"\n[{r['route']}]\n{r['response']}")
