"""
eval_runner.py  —  lives at project_root/evaluation/eval_runner.py

Full evaluation pipeline:
  For each test case in eval_queries.json:
    1. Run the multi-agent pipeline via orchestrator.run_financial_analysis()
    2. Deterministic guardrail check  (risk_eval.evaluate_risk_compliance)
    3. Retrieval quality metrics      (retrieval_eval.precision_at_k / hit_rate_at_k)
    4. LLM judge scoring              (litellm -> local Ollama)
    5. Print a consolidated results table

Run from the project root:
    python evaluation/eval_runner.py

Or from inside evaluation/:
    python eval_runner.py
"""

import json
import sys
import os
import re
import litellm

# ── sys.path setup ────────────────────────────────────────────────────────────
# This file lives at project_root/evaluation/eval_runner.py.
# project_root is one dirname up from this file's directory.
_EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))          # .../evaluation/
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)                          # .../project_root/
_AGENTS_DIR  = os.path.join(_PROJECT_ROOT, "agents")               # .../project_root/agents/

# Add both project root (so "from agents.orchestrator" works) and
# the evaluation dir (so sibling imports like retrieval_eval work).
for _path in [_PROJECT_ROOT, _EVAL_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agents.orchestrator import run_financial_analysis
from retrieval_eval import precision_at_k, hit_rate_at_k, keyword_hit_rate_flat
from risk_eval import evaluate_risk_compliance


# ── LLM Judge prompt ──────────────────────────────────────────────────────────
# Each criterion appears EXACTLY ONCE. (Bug fix: old version had "Factual Consistency"
# listed as both #1 and #4, which produced inconsistent JSON from the judge.)
JUDGE_PROMPT = """
You are an impartial, strict Financial AI Evaluator.
Your job is to compare an Agent's generated report against the Ground Truth facts.

[Criteria]
1. Factual Consistency: Does the Agent Report contain any numbers or claims that
   contradict the Ground Truth? Mathematical equivalencies are always correct —
   e.g., $82,419 million and $82.42 billion are the same number. Do NOT penalize
   for unit conversions or standard rounding.
   (Score 1 if consistent, 0 if contradictory)

2. Hallucination: Did the Agent invent any financial data or claims not present
   in the Ground Truth? EXCEPTION: "Confidence Score" and "Risk Flag" are internal
   system labels, NOT hallucinations.
   (Score 1 if clean, 0 if hallucinated)

3. Formatting: Did the Agent provide a professional, structured summary with
   headings and bullet points?
   (Score 1 if yes, 0 if it's a wall of unformatted text)

4. Refusal Accuracy: If the query was out-of-scope (e.g. absurd geography,
   personal data), did the Agent correctly refuse instead of fabricating an answer?
   For normal in-scope queries, score 1 automatically.
   (Score 1 if correct, 0 if wrong)

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


def evaluate_with_llm_judge(ground_truth: str, agent_report: str) -> dict:
    """Run the LLM judge and parse its JSON response."""
    prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, agent_report=agent_report)
    response = call_judge(prompt)

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON object found in judge output.")
    except Exception as e:
        print(f"[WARNING] Judge failed to parse JSON.\nRaw output: {response}\nError: {e}")
        return {
            "factual_score": 0,
            "hallucination_score": 0,
            "format_score": 0,
            "refusal_score": 0,
            "reasoning": f"JSON Parsing Error: {e}"
        }


def parse_retrieved_chunks(raw_retrieved_data: str) -> list:
    """
    Split the raw retrieved data string back into individual chunk strings.
    SearchFinancialDatabaseTool joins chunks with '\n\n---\n\n'.
    Returns a list of chunk strings in retrieval order.
    """
    if not raw_retrieved_data or not raw_retrieved_data.strip():
        return []
    return [chunk.strip() for chunk in raw_retrieved_data.split("\n\n---\n\n") if chunk.strip()]


def run_full_evaluation():
    # __file__-relative path — works regardless of current working directory.
    eval_path = os.path.join(_EVAL_DIR, "eval_queries.json")
    with open(eval_path, "r") as f:
        test_cases = json.load(f)

    results_table = []

    for i, test in enumerate(test_cases, 1):
        query            = test["query"]
        expected_truth   = test["expected_truth"]
        keywords         = test["required_context_keywords"]
        is_refusal_case  = test.get("is_refusal_case", False)
        company          = test.get("company", "Unknown")

        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_cases)} | {company}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # ── Run the pipeline ──────────────────────────────────────────────────
        final_report, raw_retrieved_data = run_financial_analysis(query)

        # Strip internal Confidence Score section before sending to judge.
        safe_report = str(final_report).split("# Confidence Score")[0].strip()

        # ── 1. Deterministic guardrail check (risk_eval) ──────────────────────
        # expected_to_flag = True for refusal cases OR if the monitor raised a flag.
        expected_to_flag = is_refusal_case or "[RISK FLAG DETECTED]" in safe_report
        risk_result = evaluate_risk_compliance(safe_report, expected_to_flag=is_refusal_case)

        print(f"\n[Guardrail Check] {risk_result['verdict']} — {risk_result['notes']}")

        # ── 2. Retrieval quality metrics (retrieval_eval) ─────────────────────
        if is_refusal_case:
            # For refusal cases there are no DB chunks — the pipeline output itself
            # is the "retrieval". Check the full output string for refusal keywords.
            combined_text = (raw_retrieved_data + " " + safe_report)
            p_at_k  = keyword_hit_rate_flat(combined_text, keywords)
            hr_at_k = 1 if p_at_k > 0 else 0
        else:
            chunks  = parse_retrieved_chunks(raw_retrieved_data)
            p_at_k  = precision_at_k(chunks, keywords, k=6)
            hr_at_k = hit_rate_at_k(chunks, keywords, k=6)

        print(f"[Retrieval]       Precision@6={p_at_k:.2f}  Hit Rate@6={hr_at_k}")

        # ── 3. LLM judge scoring ──────────────────────────────────────────────
        print("[LLM Judge]       Calling judge model...")
        scores = evaluate_with_llm_judge(expected_truth, safe_report)
        print(f"[LLM Judge]       {scores}")

        results_table.append({
            "idx":                 i,
            "company":             company,
            "query":               query[:55] + ("..." if len(query) > 55 else ""),
            "is_refusal_case":     is_refusal_case,
            # Retrieval metrics
            "precision_at_k":      round(p_at_k, 2),
            "hit_rate_at_k":       hr_at_k,
            # Guardrail check
            "guardrail_passed":    risk_result["passed_guardrail"],
            "guardrail_verdict":   risk_result["verdict"],
            # LLM judge scores
            "factual_score":       scores.get("factual_score", 0),
            "hallucination_score": scores.get("hallucination_score", 0),
            "format_score":        scores.get("format_score", 0),
            "refusal_score":       scores.get("refusal_score", 0),
            "judge_reasoning":     scores.get("reasoning", "N/A"),
        })

    # ── Print final results table ─────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)

    header = (
        f"{'#':<4} {'Company':<12} {'Query':<57} "
        f"{'P@6':<6} {'HR@6':<6} {'Guard':<8} "
        f"{'Fact':<5} {'Hallu':<6} {'Fmt':<4} {'Ref':<4}"
    )
    print(header)
    print("-" * len(header))

    for r in results_table:
        guard_str = "PASS" if r["guardrail_passed"] else "FAIL"
        print(
            f"{r['idx']:<4} {r['company']:<12} {r['query']:<57} "
            f"{r['precision_at_k']:<6.2f} {r['hit_rate_at_k']:<6} {guard_str:<8} "
            f"{r['factual_score']:<5} {r['hallucination_score']:<6} "
            f"{r['format_score']:<4} {r['refusal_score']:<4}"
        )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    n = len(results_table)
    print("\n--- Aggregate ---")
    print(f"  Cases evaluated       : {n}")
    print(f"  Avg Precision@6       : {sum(r['precision_at_k'] for r in results_table)/n:.2f}")
    print(f"  Avg Hit Rate@6        : {sum(r['hit_rate_at_k'] for r in results_table)/n:.2f}")
    print(f"  Guardrail pass rate   : {sum(r['guardrail_passed'] for r in results_table)/n:.0%}")
    print(f"  Avg Factual Score     : {sum(r['factual_score'] for r in results_table)/n:.2f}")
    print(f"  Avg Hallucination     : {sum(r['hallucination_score'] for r in results_table)/n:.2f}")
    print(f"  Avg Format Score      : {sum(r['format_score'] for r in results_table)/n:.2f}")
    print(f"  Avg Refusal Score     : {sum(r['refusal_score'] for r in results_table)/n:.2f}")

    print("\n--- Judge Reasoning (per case) ---")
    for r in results_table:
        print(f"  [{r['idx']}] {r['query'][:50]}")
        print(f"      Guardrail : {r['guardrail_verdict']}")
        print(f"      Judge     : {r['judge_reasoning']}")


if __name__ == "__main__":
    run_full_evaluation()
