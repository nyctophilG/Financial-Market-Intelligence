from crewai import Process, Crew, Task

# FIX #1: Import paths now match the agents/ subfolder structure that eval_runner.py expects.
# The old flat imports (from data_gatherer import ...) conflicted with eval_runner.py's
# (from agents.orchestrator import ...) — they can't both be right at the same time.
from agents.data_gatherer import create_data_gatherer
from agents.financial_analyst import create_financial_analyst
from agents.risk_monitor import create_risk_monitor


def run_financial_analysis(query: str):
    print(f"Initiating Multi-Agent Pipeline for query: '{query}'\n")

    gatherer = create_data_gatherer()
    analyst = create_financial_analyst()
    monitor = create_risk_monitor()

    # --- STAGE 1: DATA GATHERING ---
    print("\n--- STAGE 1: DATA GATHERING ---")

    gather_task = Task(
        description=f"Search the database for: {query}. Extract the exact numbers and statements.",
        expected_output="Raw financial data and risk factors directly from the Vector DB.",
        agent=gatherer
    )

    gatherer_crew = Crew(agents=[gatherer], tasks=[gather_task], verbose=True)
    raw_data_result = gatherer_crew.kickoff()
    raw_retrieved_data = str(raw_data_result).strip()

    # --- PYTHON GUARDRAIL / KILL-SWITCH ---
    check_text = raw_retrieved_data.lower()
    refusal_triggers = [
        "cannot provide", "fraud", "sensitive", "out of scope", "out_of_scope",
        "apologize", "as an ai", "unfortunately", "couldn't find",
        "could not find", "no relevant"
    ]

    if any(trigger in check_text for trigger in refusal_triggers):
        print("GUARDRAIL TRIGGERED: Gatherer refused or failed. Halting pipeline.")
        return "DATA_UNAVAILABLE: Query triggered safety filters or returned empty.", raw_retrieved_data

    # --- STAGE 2: ANALYSIS & AUDIT ---
    # FIX #2: Tasks are now created AFTER we have raw_retrieved_data, and the gathered
    # text is injected directly into each task description. This replaces the broken
    # .context = [gather_task] pattern which was set after gather_task already ran
    # inside a separate Crew object — CrewAI does not share task outputs across Crew instances.
    print("\n--- STAGE 2: ANALYSIS & AUDIT ---")

    analyze_task = Task(
        description=(
            f"Format the following raw financial data into a professional summary "
            f"with bullet points and clear headings:\n\n"
            f"--- RAW DATA ---\n{raw_retrieved_data}"
        ),
        expected_output="A structured, professional financial summary report.",
        agent=analyst
    )

    monitor_task = Task(
        description=(
            f"Review the Financial Analyst's draft report. Cross-check it against "
            f"the original raw data below. Look for hallucinated numbers or unsupported claims.\n\n"
            f"--- ORIGINAL RAW DATA ---\n{raw_retrieved_data}\n\n"
            f"Output [APPROVED] if safe, or [RISK FLAG DETECTED] if there are discrepancies, "
            f"followed by the final verified text."
        ),
        expected_output="The final risk-assessed financial report with an approval or warning flag.",
        agent=monitor,
        # FIX #3: context here is valid because both tasks belong to the SAME downstream_crew.
        # analyze_task's output will be available to monitor_task within this crew run.
        context=[analyze_task]
    )

    downstream_crew = Crew(
        agents=[analyst, monitor],
        tasks=[analyze_task, monitor_task],
        process=Process.sequential,
        verbose=True
    )

    final_result = downstream_crew.kickoff()

    return str(final_result), raw_retrieved_data


if __name__ == "__main__":
    final_output, raw = run_financial_analysis(
        "What was Tesla's future plans?"
    )

    print("\n================================================")
    print("FINAL PIPELINE OUTPUT:")
    print("================================================")
    print(final_output)
