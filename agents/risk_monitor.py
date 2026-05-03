import os
from crewai import Agent, LLM


# FIX #1: load_prompt now uses the `filename` parameter.
# The old version hardcoded 'risk_prompt.txt' and ignored whatever was passed in.
def load_prompt(filename: str) -> str:
    """Dynamically loads prompt text from the prompts directory."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(root_dir, 'prompts', filename)  # <-- FIXED: use param
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Falling back to default.")
        return "You are a helpful AI assistant."


local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)


def create_risk_monitor():
    risk_backstory = load_prompt('risk_prompt.txt')

    return Agent(
        role='Chief Risk & Compliance Monitor',
        goal=(
            "Scrutinize the Financial Analyst's report to ensure absolute factual accuracy "
            "and flag any hallucinations."
        ),
        backstory=risk_backstory,
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=local_llm
    )


if __name__ == "__main__":
    from crewai import Task, Crew

    print("Initiating Solo Test Run for Risk Monitor...")

    monitor = create_risk_monitor()

    raw_gathered_data = "Tesla FY2023: Total automotive revenues were $82,419 million."
    hallucinated_analyst_draft = (
        "Tesla had a phenomenal 2023, generating $95.0 billion in total automotive revenue, "
        "showing massive year-over-year growth."
    )

    test_task = Task(
        description=(
            f"Review the following draft report against the original raw data. Cross-check the numbers.\n\n"
            f"--- ORIGINAL RAW DATA ---\n{raw_gathered_data}\n\n"
            f"--- ANALYST DRAFT REPORT ---\n{hallucinated_analyst_draft}\n\n"
            f"Did the analyst hallucinate? Issue your final verdict."
        ),
        expected_output=(
            "A safety verdict starting with either [APPROVED] or [RISK FLAG DETECTED], "
            "followed by the explanation."
        ),
        agent=monitor
    )

    test_crew = Crew(agents=[monitor], tasks=[test_task], verbose=True)
    result = test_crew.kickoff()

    print("\n================================================")
    print("FINAL RISK MONITOR VERDICT:")
    print(result)
    print("================================================")
