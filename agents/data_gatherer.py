import os
import re
from crewai import Agent, LLM
from crewai.tools import BaseTool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Type, Optional


def load_prompt(filename: str) -> str:
    """
    Loads a prompt file from project_root/prompts/<filename>.
    This file lives at project_root/agents/data_gatherer.py
      dirname(__file__)     = project_root/agents/
      dirname(dirname(...)) = project_root/
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(root_dir, 'prompts', filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[WARNING] Prompt file not found: {prompt_path}")
        print(f"[WARNING] Check that '{filename}' exists in project_root/prompts/")
        return "You are a helpful AI assistant."


local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── CHROMA_PATH FIX ───────────────────────────────────────────────────────────
# ingest_data.py lives at project_root/rag/ingest_data.py and writes its DB to
# project_root/rag/chroma_db/ (one level below project_root, inside rag/).
#
# This file lives at project_root/agents/data_gatherer.py, so:
#   project_root = dirname(dirname(__file__))
#   rag/chroma_db = project_root/rag/chroma_db
#
# WRONG (old):  project_root/chroma_db  <- empty, never written to
# CORRECT:      project_root/rag/chroma_db  <- where ingest_data.py writes
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH   = os.path.join(_PROJECT_ROOT, "rag", "chroma_db")

# Quick sanity check on startup so you know immediately if the path is wrong.
if not os.path.exists(CHROMA_PATH):
    print(f"[WARNING] chroma_db not found at: {CHROMA_PATH}")
    print(f"[WARNING] Run: cd project_root/rag && python ingest_data.py")
else:
    print(f"[INFO] Using chroma_db at: {CHROMA_PATH}")


# Maps company name fragments (lowercase) to the canonical ticker stored in DB metadata.
# Keep this in sync with ingest_data.py's COMPANY_MAP.
TICKER_MAP = {
    "tesla":     "TSLA", "tsla":    "TSLA",
    "microsoft": "MSFT", "msft":    "MSFT",
    "apple":     "AAPL", "aapl":    "AAPL",
    "amazon":    "AMZN", "amzn":    "AMZN",
    "alphabet":  "GOOGL","google":  "GOOGL", "googl": "GOOGL",
    "meta":      "META",
    "nvidia":    "NVDA", "nvda":    "NVDA",
}

def resolve_ticker(query: str) -> Optional[str]:
    """Detect which company the query is about and return its canonical ticker."""
    lower = query.lower()
    for fragment, ticker in TICKER_MAP.items():
        if fragment in lower:
            return ticker
    return None


class SearchQuerySchema(BaseModel):
    query: str = Field(
        ...,
        description=(
            "The exact financial search string. Always include the company name "
            "AND fiscal year, e.g. 'Tesla 2023 total automotive revenues' or "
            "'Apple fiscal 2025 iPhone net sales'."
        )
    )


class SearchFinancialDatabaseTool(BaseTool):
    name: str = "Search Financial Database"
    description: str = (
        "Search the local vector database of SEC 10-K filings. "
        "The database contains filings from MULTIPLE companies. "
        "Always include the company name AND the fiscal year in your query "
        "so the correct filing is returned."
    )
    args_schema: Type[BaseModel] = SearchQuerySchema

    def _run(self, query: str) -> str:
        print(f"\n[TOOL] Searching DB for: '{query}'")
        print(f"[TOOL] DB path: {CHROMA_PATH}")

        if not os.path.exists(CHROMA_PATH):
            return (
                f"ERROR: chroma_db not found at {CHROMA_PATH}. "
                f"Run ingest_data.py first: cd project_root/rag && python ingest_data.py"
            )

        vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=GLOBAL_EMBEDDINGS
        )

        ticker = resolve_ticker(query)

        if ticker:
            print(f"[TOOL] Company filter applied: ticker={ticker}")
            results = vector_db.similarity_search(
                query,
                k=6,
                filter={"ticker": ticker}
            )
        else:
            print("[TOOL] Warning: no company detected in query — running unfiltered search.")
            results = vector_db.similarity_search(query, k=6)

        if not results:
            return (
                "No relevant documents found. "
                f"DB path used: {CHROMA_PATH}. "
                "Check that the correct PDF was ingested via ingest_data.py "
                "and that the PDF filename contains the company name or ticker."
            )

        chunks = []
        for doc in results:
            meta = doc.metadata
            header = (
                f"[Source: {meta.get('company', '?')} | "
                f"Ticker: {meta.get('ticker', '?')} | "
                f"FY: {meta.get('fiscal_year', '?')} | "
                f"Page: {meta.get('page', '?')} | "
                f"Type: {meta.get('chunk_type', '?')}]"
            )
            chunks.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(chunks)


def create_data_gatherer():
    gatherer_backstory = load_prompt('gatherer_prompt.txt')
    return Agent(
        role='Senior Data Gatherer',
        goal=(
            'Search the multi-company Vector DB to extract precise financial '
            'numbers and statements for the specific company mentioned in the query.'
        ),
        backstory=gatherer_backstory,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        tools=[SearchFinancialDatabaseTool()],
        llm=local_llm
    )


if __name__ == "__main__":
    from crewai import Task, Crew

    print("Solo test: multi-company retrieval")
    gatherer = create_data_gatherer()

    for test_query in [
        "What was Tesla's total automotive revenues for the year 2023?",
        "What was Apple's total net sales for fiscal year 2025?",
        "What was Microsoft's net income for fiscal year 2025?",
    ]:
        print(f"\n{'='*60}\nQuery: {test_query}\n{'='*60}")
        task = Task(
            description=f"Search the database for: {test_query}",
            expected_output="Exact figures from the database.",
            agent=gatherer
        )
        crew = Crew(agents=[gatherer], tasks=[task], verbose=False)
        result = crew.kickoff()
        print(f"Result: {result}\n")
