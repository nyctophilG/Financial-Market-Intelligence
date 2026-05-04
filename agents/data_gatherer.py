import os
import re
from crewai import Agent, LLM
from crewai.tools import BaseTool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Type, Optional


def load_prompt(filename: str) -> str:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(root_dir, 'prompts', filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default.")
        return "You are a helpful AI assistant."


local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
)

# Maps company name fragments to the canonical ticker stored in DB metadata.
TICKER_MAP = {
    "tesla":     "TSLA", "tsla":   "TSLA",
    "microsoft": "MSFT", "msft":   "MSFT",
    "apple":     "AAPL", "aapl":   "AAPL",
    "amazon":    "AMZN", "amzn":   "AMZN",
    "alphabet":  "GOOGL","google": "GOOGL", "googl": "GOOGL",
    "meta":      "META",
    "nvidia":    "NVDA", "nvda":   "NVDA",
}

def resolve_ticker(query: str) -> Optional[str]:
    """Detect which company the query is about and return its ticker."""
    lower = query.lower()
    for fragment, ticker in TICKER_MAP.items():
        if fragment in lower:
            return ticker
    return None


class SearchQuerySchema(BaseModel):
    query: str = Field(
        ...,
        description=(
            "The exact financial search string. Always include company name "
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

        vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=GLOBAL_EMBEDDINGS
        )

        ticker = resolve_ticker(query)

        if ticker:
            # Filtered search — only chunks from the correct company.
            # This prevents cross-company contamination (e.g. Apple revenue
            # bleeding into a Tesla query).
            print(f"[TOOL] Company filter applied: ticker={ticker}")
            results = vector_db.similarity_search(
                query,
                k=6,
                filter={"ticker": ticker}
            )
        else:
            print("[TOOL] Warning: no company detected — running unfiltered search.")
            results = vector_db.similarity_search(query, k=6)

        if not results:
            return (
                "No relevant documents found. "
                "Check that the correct PDF was ingested via ingest_data.py."
            )

        chunks = []
        for doc in results:
            meta = doc.metadata
            header = (
                f"[Source: {meta.get('company','?')} | "
                f"Ticker: {meta.get('ticker','?')} | "
                f"FY: {meta.get('fiscal_year','?')} | "
                f"Page: {meta.get('page','?')} | "
                f"Type: {meta.get('chunk_type','?')}]"
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
