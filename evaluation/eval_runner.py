import json
import sys
import os
import re
import litellm  # FIX #3: Use litellm directly instead of judge_llm.call()
               # CrewAI's LLM wrapper does not expose a public .call() method
               # in newer versions — it's an internal detail. litellm.completion()
               # is the correct, stable way to call Ollama from outside a CrewAI crew.

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_financial_analysis


# FIX #1: Removed duplicate Criterion #4 from the judge prompt.
# The original prompt had "Factual Consistency" listed twice (as #1 and #4), which
# confused the LLM judge and caused inconsistent JSON output. Now each criterion is unique.
JUDGE_PROMPT = """
You are an impartial, strict Financial AI Evaluator.
Your job is to compare an Agent's generated report against the Ground Truth facts.

[Criteria]
1. Factual Consistency: Does the Agent Report contain any numbers or claims that contradict 
   the Ground Truth? IMPORTANT: Mathematical equivalencies are always correct — e.g., 
   $82,419 million and $82.42 billion are the same number. Do NOT penalize for unit 
   conversions or standard rounding. (Score 1 if consistent, 0 if contradictory)

2. Hallucination: Did the Agent invent any financial data or claims not present in the 
   Ground Truth? EXCEPTION: "Confidence Score" and "Risk Flag" are internal system labels, 
   NOT hallucinations. (Score 1 if clean, 0 if hallucinated)

3. Formatting: Did the Agent provide a professional, structured summary with headings 
   and bullet points? (Score 1 if yes, 0 if it's a wall of unformatted text)

4. Refusal Accuracy: If the query was out-of-scope (e.g. absurd geography, personal data), 
   did the Agent correctly refuse instead of making something up? 
   For normal in-scope queries, score 1 automatically. (Score 1 if correct, 0 if wrong)

[Ground Truth Data]
{ground_truth}

[Agent Report]
{agent_report}

Output ONLY a JSON object — no preamble, no markdown fences:
{{"factual_score": 1, "hallucination_score": 1, "format_score": 1, "refusal_score": 1, "reasoning": "brief explanation"}}
"""


def call_judge(prompt: str) -> str:
    """Call the local Ollama judge model via litellm."""
    response = litellm.completion(
        model="ollama/llama3.1",
        messages=[{"role": "user", "content": prompt}],
        api_base="http://localhost:11434",
        temperature=0.0
    )
    return response.choices[0].message.content


def evaluate_pipeline_output(ground_truth: str, agent_report: str) -> dict:
    prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, agent_report=agent_report)
    response = call_judge(prompt)

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON object found in judge output.")
    except Exception as e:
        print(f"Judge failed to parse JSON.\nRaw output: {response}\nError: {e}")
        return {
            "factual_score": 0,
            "hallucination_score": 0,
            "format_score": 0,
            "refusal_score": 0,
            "reasoning": "JSON Parsing Error"
        }


def run_full_evaluation():
    # FIX #2: Use __file__-relative path so this works regardless of CWD.
    # The old hardcoded "evaluation/eval_queries.json" only worked if you ran
    # the script from the project root — it would crash from any other directory.
    eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_queries.json")
    with open(eval_path, "r") as f:
        test_cases = json.load(f)

    results_table = []

    for test in test_cases:
        query = test["query"]
        expected_truth = test["expected_truth"]
        is_refusal_case = test.get("is_refusal_case", False)

        print(f"\n{'='*60}")
        print(f"RUNNING EVAL FOR: {query}")
        print(f"{'='*60}")

        final_report, retrieved_text = run_financial_analysis(query)

        # Strip internal system section before sending to judge
        safe_report = str(final_report).split("# Confidence Score")[0].strip()
        clean_retrieved_text = " ".join(retrieved_text.split())

        # --- Intrinsic Eval: Precision@K (keyword hit rate) ---
        keywords = test["required_context_keywords"]

        # FIX #4: For refusal cases the "retrieved text" is the pipeline's guardrail
        # output string, not a DB result. Match against the full pipeline output instead
        # so that "DATA_UNAVAILABLE" and similar refusal signals are caught correctly.
        search_space = (clean_retrieved_text + " " + safe_report).lower() if is_refusal_case else clean_retrieved_text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in search_space)
        precision_at_k = hits / len(keywords) if keywords else 0.0

        # --- Extrinsic Eval: LLM Judge ---
        # Pass expected_truth as the ground truth so the judge has the right reference.
        scores = evaluate_pipeline_output(expected_truth, safe_report)

        results_table.append({
            "Query": query[:50] + ("..." if len(query) > 50 else ""),
            "Is Refusal Case": is_refusal_case,
            "Precision@K": round(precision_at_k, 2),
            "Factual Score": scores.get("factual_score", 0),
            "Hallucination Score": scores.get("hallucination_score", 0),
            "Format Score": scores.get("format_score", 0),
            "Refusal Score": scores.get("refusal_score", 0),
            "Reasoning": scores.get("reasoning", "N/A")
        })

    print("\n\n=== FINAL EVALUATION METRICS ===")
    for res in results_table:
        print(
            f"\nQuery       : {res['Query']}\n"
            f"Precision@K : {res['Precision@K']} | "
            f"Factual: {res['Factual Score']} | "
            f"Hallucination: {res['Hallucination Score']} | "
            f"Format: {res['Format Score']} | "
            f"Refusal: {res['Refusal Score']}\n"
            f"Judge Notes : {res['Reasoning']}"
        )


if __name__ == "__main__":
    run_full_evaluation()
