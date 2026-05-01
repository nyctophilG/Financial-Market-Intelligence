import json
import sys
import os
import re
from crewai import LLM

# Add the root directory to Python path so we can import orchestrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_financial_analysis

# 1. Initialize the Judge Model
judge_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.0 # Strict, deterministic grading
)

# 2. Define the Evaluation Prompt
JUDGE_PROMPT = """
You are an impartial, strict Financial AI Evaluator.
Your job is to compare an Agent's generated report against the Ground Truth facts.

[Criteria]
1. Factual Consistency: Does the Agent Report contain any numbers or claims that contradict the Ground Truth? (Score 0 if yes, 1 if no). NOTE: You MUST account for standard mathematical rounding (e.g., $82.419 billion and $82.42 billion are factually consistent).
2. Hallucination: Did the Agent invent any financial data or claims not present in the Ground Truth? (Score 0 if yes, 1 if no). IMPORTANT EXCEPTION: Ignore the "Confidence Score" and "Risk Flag" labels; these are internal system metrics and do NOT count as hallucinations.
3. Formatting: Did the Agent provide a professional summary? (Score 0 if no, 1 if yes).
4. Factual Consistency: Does the Agent Report contain any numbers or claims that contradict the Ground Truth? 
CRITICAL RULE: You MUST evaluate mathematical equivalencies as correct. For example, $81,545 million is mathematically equivalent to $81.545 billion, which rounds to $81.55 billion. Do NOT penalize for standard rounding or unit conversions (millions to billions). (Score 0 if yes, 1 if no)

[Ground Truth Data]
{ground_truth}

[Agent Report]
{agent_report}

Output your evaluation strictly in the following JSON format:
{{"factual_score": 1, "hallucination_score": 1, "format_score": 1, "reasoning": "brief explanation"}}
"""

def evaluate_pipeline_output(query: str, ground_truth: str, agent_report: str):
    prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, agent_report=agent_report)
    response = judge_llm.call(prompt)
    
    try:
        # Find everything between the first { and the last }
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            clean_json = match.group(0)
            scores = json.loads(clean_json)
            return scores
        else:
            raise ValueError("No JSON object found in judge output.")
    except Exception as e:
        print(f"Judge failed to parse JSON. \nRaw output: {response}\nError: {e}")
        return {"factual_score": 0, "hallucination_score": 0, "format_score": 0, "reasoning": "JSON Parsing Error"}

def run_full_evaluation():
    with open("evaluation/eval_queries.json", "r") as f:
        test_cases = json.load(f)

    results_table = []

    for test in test_cases:
        query = test["query"]
        expected = test["expected_truth"]
        
        print(f"\n============================================================")
        print(f" RUNNING EVAL FOR: {query}")
        print(f"============================================================")
        
        # 1. Run your pipeline
        final_report, retrieved_chunks = run_financial_analysis(query)

        # Strip out everything from "# Confidence Score" to the end of the string
        safe_report = str(final_report).split("# Confidence Score")[0].strip()

        # CLEAN THE MESSY CHROMADB TEXT
        clean_retrieved_text = " ".join(retrieved_chunks.split())
        
        # 2. Intrinsic Eval: Precision@K 
        hits = sum(1 for keyword in test["required_context_keywords"] if keyword.lower() in clean_retrieved_text.lower())
        precision_at_k = hits / len(test["required_context_keywords"])
        
        # 3. Extrinsic Eval: LLM Judge
        scores = evaluate_pipeline_output(query, clean_retrieved_text, safe_report)
        
        # 4. Save for PDF Table
        results_table.append({
            "Query": query[:40] + "...", # Truncate for clean table view
            "Precision@K": precision_at_k,
            "Factual Score": scores["factual_score"],
            "Hallucination Score": scores["hallucination_score"],
            "Reasoning": scores["reasoning"]
        })

    print("\n\n=== FINAL EVALUATION METRICS FOR PDF REPORT ===")
    for res in results_table:
        print(f"Query: {res['Query']} | Precision@K: {res['Precision@K']} | Factual: {res['Factual Score']} | Hallucination: {res['Hallucination Score']}")
        print(f"Judge Notes: {res['Reasoning']}\n")

if __name__ == "__main__":
    run_full_evaluation()