"""
orchestrator.py  —  lives at project_root/agents/orchestrator.py

Architecture: THREE separate single-agent Crews.
  Stage 1: Gatherer Crew  ->  Python extracts RAW TOOL OUTPUT (not LLM final answer)
  Stage 1b: Python guardrail  ->  halt or continue
  Stage 2: Analyst Crew   ->  receives raw tool output injected into task description
  Stage 3: Monitor Crew   ->  receives raw tool output + analyst report injected into task

Key insight — why we intercept tool output instead of using LLM final answer:
  llama3.1 (and most small local LLMs) do NOT reliably pass raw retrieved text
  downstream as instructed. Instead they summarize, rephrase, or hallucinate.
  The tool output in gather_task.output.raw contains exactly what the DB returned.
  We extract it directly in Python and inject it into Stage 2, completely bypassing
  the Gatherer's "Final Answer" which cannot be trusted.
"""

import re
from crewai import Crew, Task

from data_gatherer import create_data_gatherer, SearchFinancialDatabaseTool
from financial_analyst import create_financial_analyst
from risk_monitor import create_risk_monitor


def extract_tool_output(task_output) -> str:
    """
    Extract the raw tool call result from a CrewAI TaskOutput object.

    CrewAI stores the full agent scratchpad (thought + tool input + tool output +
    final answer) in task_output.raw. We parse out the tool output block — the
    section between 'Observation:' and the next 'Thought:' or end of string.
    This gives us the actual DB chunks, not the LLM's summary of them.

    Falls back to task_output.raw if no tool output block is found (e.g. when
    the agent skipped the tool call and answered directly from training data —
    which itself is a sign the guardrail should fire).
    """
    raw = task_output.raw if hasattr(task_output, 'raw') else str(task_output)

    # Pattern 1: CrewAI verbose format — "Observation:\n<tool output>\nThought:"
    match = re.search(r'Observation:\s*(.*?)(?=\nThought:|\nFinal Answer:|$)', raw, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            print(f"[ORCHESTRATOR] Extracted tool output via 'Observation:' pattern ({len(extracted)} chars).")
            return extracted

    # Pattern 2: Look for our tool's header format as a reliable anchor
    match = re.search(r'(\[Source:.*?)(?=\nThought:|\nFinal Answer:|$)', raw, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            print(f"[ORCHESTRATOR] Extracted tool output via '[Source:]' anchor ({len(extracted)} chars).")
            return extracted

    # Fallback: return the full raw output and let the guardrail decide
    print("[ORCHESTRATOR] Warning: could not isolate tool output — using full raw LLM output.")
    print("[ORCHESTRATOR] This usually means the Gatherer skipped the tool and answered from memory.")
    return raw


def run_financial_analysis(query: str):
    print(f"\nInitiating Multi-Agent Pipeline for query: '{query}'\n")

    gatherer = create_data_gatherer()
    analyst  = create_financial_analyst()
    monitor  = create_risk_monitor()

    # ── STAGE 1: DATA GATHERING ───────────────────────────────────────────────
    print("\n--- STAGE 1: DATA GATHERING ---")

    gather_task = Task(
        description=(
            f"Search the database for: {query}\n"
            f"Extract the exact numbers and statements. "
            f"Your Final Answer MUST be the exact, unmodified text returned by the "
            f"Search Financial Database tool. Do NOT summarize, rephrase, or add anything."
        ),
        expected_output="The exact raw text returned by the Search Financial Database tool. No summarization.",
        agent=gatherer
    )

    gatherer_crew = Crew(agents=[gatherer], tasks=[gather_task], verbose=True)
    gatherer_crew.kickoff()

    # KEY FIX: extract the raw tool output directly from the task, NOT from the
    # LLM's final answer. Local LLMs like llama3.1 routinely ignore "pass this
    # through verbatim" instructions and write their own summary instead.
    raw_retrieved_data = extract_tool_output(gather_task.output)

    print(f"\n[ORCHESTRATOR] Raw data captured ({len(raw_retrieved_data)} chars).")
    print(f"[ORCHESTRATOR] Preview: {raw_retrieved_data[:300]}...")

    # ── PYTHON GUARDRAIL / KILL-SWITCH ────────────────────────────────────────
    check_text = raw_retrieved_data.lower()
    refusal_triggers = [
        "cannot provide", "fraud", "sensitive", "out of scope", "out_of_scope",
        "apologize", "as an ai", "unfortunately", "couldn't find",
        "could not find", "no relevant", "no documents found",
        "error: chroma_db not found"
    ]

    if any(trigger in check_text for trigger in refusal_triggers):
        print("\n[GUARDRAIL TRIGGERED] Gatherer refused or returned empty. Halting pipeline.")
        return "DATA_UNAVAILABLE: Query triggered safety filters or returned empty.", raw_retrieved_data

    # If the extracted text is suspiciously short, it likely means the tool was
    # never called and the LLM answered from training data (hallucination).
    if len(raw_retrieved_data.strip()) < 100:
        print("\n[GUARDRAIL TRIGGERED] Retrieved data too short — tool may not have been called.")
        return "DATA_UNAVAILABLE: Insufficient data retrieved from database.", raw_retrieved_data

    # ── STAGE 2: ANALYSIS ─────────────────────────────────────────────────────
    print("\n--- STAGE 2: ANALYSIS ---")

    analyze_task = Task(
        description=(
            f"The user asked: '{query}'\n\n"
            f"Answer the user's specific question using ONLY the data below. "
            f"Do not summarize the entire document — find the exact metric asked for.\n\n"
            f"--- RAW DATA FROM DATABASE ---\n{raw_retrieved_data}"
        ),
        expected_output=(
            "A structured financial summary with bullet points and clear headings "
            "that directly answers the user's question using exact figures from the raw data."
        ),
        agent=analyst
    )

    analyst_crew = Crew(agents=[analyst], tasks=[analyze_task], verbose=True)
    analyst_result = analyst_crew.kickoff()
    analyst_report = analyst_result.raw.strip()

    # ── STAGE 3: RISK AUDIT ───────────────────────────────────────────────────
    print("\n--- STAGE 3: RISK AUDIT ---")

    monitor_task = Task(
        description=(
            f"You are auditing a Financial Analyst's report for factual accuracy.\n\n"
            f"--- ORIGINAL RAW DATA (source of truth) ---\n{raw_retrieved_data}\n\n"
            f"--- ANALYST REPORT (to be audited) ---\n{analyst_report}\n\n"
            f"Cross-check every number and claim in the Analyst Report against "
            f"the Original Raw Data above.\n"
            f"Rules:\n"
            f"- Mathematical conversions (millions <-> billions) are always valid.\n"
            f"- Table headers stating 'in millions' apply to all numbers in that table.\n"
            f"- Do NOT flag internal labels like 'Confidence Score' or 'Risk Flag'.\n"
            f"- Do NOT flag 'Data unavailable' if the raw data genuinely does not contain the answer.\n\n"
            f"Output [APPROVED] if all claims are grounded in the raw data, or "
            f"[RISK FLAG DETECTED] followed by a detailed explanation of the discrepancy. "
            f"Then paste the final verified report text."
        ),
        expected_output=(
            "The final risk-assessed financial report beginning with "
            "[APPROVED] or [RISK FLAG DETECTED], followed by audit reasoning "
            "and the verified report text."
        ),
        agent=monitor
    )

    monitor_crew = Crew(agents=[monitor], tasks=[monitor_task], verbose=True)
    final_result = monitor_crew.kickoff()

    return str(final_result), raw_retrieved_data


if __name__ == "__main__":
    final_output, raw = run_financial_analysis(
        "Can you give me revenue of Microsoft?"
    )

    print("\n================================================")
    print("FINAL PIPELINE OUTPUT:")
    print("================================================")
    print(final_output)
