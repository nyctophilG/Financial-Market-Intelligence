"""
ingest_worker.py  --  <project_root>/app/ingest_worker.py

Spawned as a subprocess by ui.py for each uploaded file.
Uses the EXACT same ChromaDB path and embedding model as data_gatherer.py
so the data gatherer agent can actually find what was ingested.

Usage:
    python ingest_worker.py /absolute/path/to/file.pdf
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    if len(sys.argv) < 2:
        print("[ERROR] No file path given")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    # ── Resolve project root (app/../  ==  project_root) ──────
    app_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # ── Use the EXACT same path as data_gatherer.py ───────────
    # data_gatherer.py:  CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")
    # os.path.dirname(__file__) there = <project_root>/agents/
    # so CHROMA_PATH = <project_root>/agents/../chroma_db = <project_root>/chroma_db
    agents_dir = os.path.join(project_root, "agents")
    chroma_dir = os.path.normpath(os.path.join(agents_dir, "..", "chroma_db"))

    print(f"[Worker] Project root : {project_root}")
    print(f"[Worker] ChromaDB path: {chroma_dir}")
    print(f"[Worker] File         : {file_path}")

    # ── Read file ─────────────────────────────────────────────
    fname   = os.path.basename(file_path)
    content = read_file(file_path)

    if not content.strip():
        print(f"[Worker] ERROR: could not extract text from {fname}")
        print(f"[Worker] Install a PDF reader:  pip install pypdf")
        sys.exit(1)

    print(f"[Worker] {fname}: {len(content)} chars extracted")

    # ── Chunk ─────────────────────────────────────────────────
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(content)

    if not chunks:
        print(f"[Worker] ERROR: no chunks from {fname}")
        sys.exit(1)

    print(f"[Worker] {fname}: {len(chunks)} chunks created")

    docs = [
        Document(
            page_content=c,
            metadata={"source": file_path, "filename": fname},
        )
        for c in chunks
    ]

    # ── Embed + store (same model as data_gatherer.py) ────────
    print(f"[Worker] Loading embedding model all-MiniLM-L6-v2...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"[Worker] Embedding model ready. Writing to ChromaDB...")

    db = Chroma(persist_directory=chroma_dir, embedding_function=emb)
    db.add_documents(docs)

    total = db._collection.count()
    print(f"[SUCCESS] {fname} -- {len(chunks)} chunks added. ChromaDB total: {total}")
    sys.exit(0)


def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        # pypdf
        try:
            import pypdf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = pypdf.PdfReader(path)
                text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
            if text:
                print(f"[Worker] pypdf: {len(text)} chars")
                return text
        except ImportError:
            print("[Worker] pypdf not installed, trying pdfminer...")
        except Exception as e:
            print(f"[Worker] pypdf error: {e}")

        # pdfminer fallback
        try:
            import pdfminer.high_level as pm
            text = pm.extract_text(path).strip()
            if text:
                print(f"[Worker] pdfminer: {len(text)} chars")
                return text
        except ImportError:
            print("[Worker] pdfminer not installed")
        except Exception as e:
            print(f"[Worker] pdfminer error: {e}")

        return ""
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[Worker] read error: {e}")
            return ""


if __name__ == "__main__":
    main()
