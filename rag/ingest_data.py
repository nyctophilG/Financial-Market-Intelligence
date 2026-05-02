import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Define our paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../chroma_db")

def ingest_pdfs():
    print(" Initializing Production Data Ingestion Pipeline...")
    
    # 2. Load all PDFs from the data/raw directory
    print(f" Scanning '{RAW_DATA_DIR}' for PDFs...")
    loader = PyPDFDirectoryLoader(RAW_DATA_DIR)
    documents = loader.load()

    if not documents:
        print(" No PDFs found! Please place SEC 10-K PDF files in the 'data/raw/' folder.")
        return

    print(f" Successfully loaded {len(documents)} pages of data.")

    # 3. Chunking: Break the massive PDFs into small, semantic overlapping chunks
    print("✂️ Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Each chunk is about 1000 characters
        chunk_overlap=200, # Overlap by 200 chars so we don't cut sentences in half
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f" Created {len(chunks)} individual chunks.")

    # 4. Clear the old dummy database (Optional, but good for a fresh start)
    if os.path.exists(CHROMA_PATH):
        print(" Clearing old database...")
        shutil.rmtree(CHROMA_PATH)

    # 5. Embed the chunks and save them to the Vector Database
    print(" Embedding chunks using HuggingFace and saving to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(" Ingestion Complete! Your Vector DB is now populated with real PDF data.")

if __name__ == "__main__":
    ingest_pdfs()