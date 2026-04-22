import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define where Chroma will save the database locally, keeping it out of the rag folder
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")

def ingest_financial_data():
    print("🚀 Initializing Local Data Ingestion Pipeline...")

    # 1. Load the FREE, local embedding model
    # all-MiniLM-L6-v2 is fast, lightweight, and perfect for your 6GB VRAM constraint
    print("Loading HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Simulate raw SEC 10-K Data (Later, you can read this from actual PDFs in data/raw/)
    raw_documents = [
        Document(
            page_content="Apple Inc. Q3 2023 Financial Summary: Total net sales were $81.8 billion, down 1% year over year. Services revenue reached an all-time high of $21.2 billion.",
            metadata={"source": "AAPL_Q3_2023", "type": "financial_summary"}
        ),
        Document(
            page_content="Risk Factors Apple Inc: Macroeconomic conditions, including inflation and fluctuations in interest rates, could negatively impact consumer spending and our future hardware margins.",
            metadata={"source": "AAPL_Q3_2023", "type": "risk_factor"}
        ),
        Document(
            page_content="Tesla Q4 Earnings Call: We achieved a record production of 494,989 vehicles. However, supply chain disruptions and battery material costs remain a key risk for Q1 2024.",
            metadata={"source": "TSLA_Q4_2023", "type": "earnings_call"}
        )
    ]

    # 3. Chunk the data
    # Overlap ensures context isn't lost between chunks
    print("Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} chunks.")

    # 4. Build and persist the Vector Database
    print(f"Embedding and saving to ChromaDB at: {CHROMA_PATH}")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("✅ Ingestion Complete! Your local Vector DB is ready for the Data Gatherer Agent.")

if __name__ == "__main__":
    ingest_financial_data()