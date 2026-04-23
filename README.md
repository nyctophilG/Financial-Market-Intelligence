# Financial-Market-Intelligence
# Multi-Agent Financial Analysis & Risk Monitoring System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-Agents-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)

## Project Overview
This project is a fully local, privacy-first **Agentic Large Language Model (LLM) application** designed for financial market intelligence. By leveraging modern Transformer architectures and Retrieval-Augmented Generation (RAG), the system ingests complex financial documents (SEC filings, earnings call transcripts) and synthesizes professional analytical reports. 

Crucially, the architecture features a dedicated **Risk Monitor Agent** to mitigate LLM hallucinations, enforce groundedness, and monitor for allocational biases—ensuring highly reliable and safe outputs.

## System Architecture & Agent Flow

Our multi-agent pipeline operates sequentially to guarantee data integrity:

1. **User Query:** User inputs a financial question via the Streamlit UI.
2. **Data Gatherer Agent:** Queries the local ChromaDB vector database to retrieve the most relevant semantic chunks of financial data.
3. **Financial Analyst Agent:** Analyzes the retrieved context and drafts a comprehensive, data-driven financial summary.
4. **Risk Monitor Agent:** Acts as the strict guardrail. It cross-references the Analyst's draft against the original retrieved documents to flag unsupported claims, hallucinated numbers, or biased statements.
5. **UI Delivery:** The final, verified output and risk flags are displayed to the user.

## Tech Stack
This project runs entirely on local, open-source infrastructure:
* **Agent Framework:** CrewAI + LangChain
* **LLM Engine:** Llama 3.1 (8B) running locally via Ollama
* **Vector Database:** ChromaDB (Local Persistent)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **User Interface:** Streamlit

## Team & Roles
* **Lara (Project Manager):** System overview, timeline, final integration, and comprehensive report architecture.
* **Göktuğ (Lead Developer):** Core backend architecture, CrewAI orchestration, and RAG pipeline integration.
* **Kutay (UI/UX Designer):** Streamlit interface, user flow, and agent output visualization.
* **Emir (Prompt Engineer):** Agent behavioral tuning, strict output formatting (JSON), and guardrail prompt design.
* **Zehra (Risk & Evaluation Lead):** End-to-end evaluation pipeline (Precision@k, factual consistency metrics), and risk taxonomy management.

## Repository Structure

```text
project/
│
├── app/                  # Streamlit UI and main entry point
│   ├── ui.py
│   └── main.py
│
├── agents/               # CrewAI logic and agent definitions
│   ├── data_gatherer.py
│   ├── financial_analyst.py
│   ├── risk_monitor.py
│   └── orchestrator.py
│
├── rag/                  # Data ingestion and ChromaDB integration
│   ├── ingest_data.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── evaluation/           # Metric scripts and test sets
│   ├── eval_queries.json
│   ├── retrieval_eval.py
│   └── answer_eval.py
│
├── data/                 # Financial PDFs and SEC transcripts
│   ├── raw/
│   └── processed/
│
├── prompts/              # System instructions for LLMs
│   ├── gatherer_prompt.txt
│   ├── analyst_prompt.txt
│   └── risk_prompt.txt
│
└── requirements.txt      # Python dependencies
