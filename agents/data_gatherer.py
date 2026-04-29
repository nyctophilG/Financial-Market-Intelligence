import os
from crewai import Agent, LLM
from crewai.tools import BaseTool # swapp 'tool' for 'BaseTool'
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Type

def load_prompt(filename):
    """Dynamically loads prompt text from the prompts directory."""
    # Find the root of the project, then point to the prompts folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(root_dir, 'prompts', 'gatherer_prompt.txt')
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f" Warning: {'gatherer_prompt.txt'} not found. Falling back to default.")
        return "You are a helpful AI assistant." # Safe fallback

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
    # Load Emir's specific prompt for this agent
    gatherer_backstory = load_prompt('gatherer_prompt.txt')

    return Agent(
        role='Senior Data Gatherer',
        goal='Search the Vector DB to extract precise financial numbers and statements.',
        backstory=gatherer_backstory,  # <--- HERE IS THE MAGIC
        verbose=True,
        allow_delegation=False,
        tools=[SearchFinancialDatabaseTool()],
        llm=local_llm
    )

if __name__ == "__main__":
    from crewai import Task, Crew

    print(" Initiating Solo Test Run for Data Gatherer...")
    
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