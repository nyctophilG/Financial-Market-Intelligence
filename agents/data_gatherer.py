import os
from crewai import Agent, LLM
from crewai.tools import BaseTool # swapp 'tool' for 'BaseTool'
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Type

# 1. Define the LLM
local_llm = LLM(
    model="ollama/llama3.1", # this is our upgraded LLM 3 to 3.1
    base_url="http://localhost:11434"
)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")

# 2. Define the Strict Schema
class SearchQuerySchema(BaseModel):
    query: str = Field(..., description="The exact text string to search the database with, such as 'Apple Q3 net sales'")

# 3. Define the Tool as a strict Class (The modern CrewAI way)
class SearchFinancialDatabaseTool(BaseTool):
    name: str = "Search Financial Database"
    description: str = "Useful for searching the local vector database for SEC filings, risk factors, and financial summaries."
    args_schema: Type[BaseModel] = SearchQuerySchema

    def _run(self, query: str) -> str:
        print(f"\n[TOOL EXECUTION] Searching Vector DB for: {query}")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        results = vector_db.similarity_search(query, k=3)
        
        if not results:
            return "No relevant financial documents found in the database."
        
        return "\n\n---\n\n".join([doc.page_content for doc in results])

# 4. Define the Agent
def create_data_gatherer():
    return Agent(
        role='Senior Data Gatherer',
        goal='Accurately retrieve the most relevant financial data and SEC filings based on the user query.',
        backstory=(
            "You are a highly skilled financial archivist. Your only job is to use your "
            "'Search Financial Database' tool to find exact quotes, numbers, and risk factors "
            "from the company's official documents. You never invent numbers; you only provide "
            "what the database returns."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[SearchFinancialDatabaseTool()], # <-- Notice we instantiate the class here!
        llm=local_llm
    )

if __name__ == "__main__":
    from crewai import Task, Crew

    print("🧪 Initiating Solo Test Run for Data Gatherer...")
    
    gatherer = create_data_gatherer()

    test_task = Task(
        description="Search the database for Apple's (AAPL) Q3 2023 financial summary. What were the exact net sales, and what are the listed risk factors?",
        expected_output="A short summary containing the exact net sales figure and listed risk factors, pulled directly from the database.",
        agent=gatherer
    )

    test_crew = Crew(
        agents=[gatherer],
        tasks=[test_task],
        verbose=True
    )

    result = test_crew.kickoff()

    print("\n================================================")
    print(" FINAL AGENT OUTPUT:")
    print(result)
    print("================================================")