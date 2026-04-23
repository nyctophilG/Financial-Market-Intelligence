from crewai import Agent, LLM

# 1. Define our trusty local engine
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# 2. Define the Risk Monitor Agent (Zehra's Engine)
def create_risk_monitor():
    return Agent(
        role='Chief Risk & Compliance Monitor',
        goal='Scrutinize the Financial Analyst\'s report to ensure absolute factual accuracy and flag any hallucinations, unsupported claims, or risky projections.',
        backstory=(
            "You are a ruthless and strict Chief Risk Officer. Your job is to review financial summaries "
            "before they are shown to the user. You are actively looking for 'hallucinations'—numbers or "
            "claims that sound confident but are actually made up. "
            "CRITICAL INSTRUCTION: If you detect a claim or number that seems unsupported, you must prepend "
            "your response with [ RISK FLAG DETECTED] and explain the issue. If the report is safe and grounded, "
            "prepend your response with [ APPROVED] and output the final verified text."
        ),
        verbose=True,
        allow_delegation=False,
        llm=local_llm
    )

# ==========================================
#  Execution Block for Solo Testing
# ==========================================
if __name__ == "__main__":
    from crewai import Task, Crew

    print(" Initiating Solo Test Run for Risk Monitor...")
    
    # 1. Instantiate Zehra's agent
    monitor = create_risk_monitor()

    # 2. We simulate a "Bad" Analyst who hallucinated fake numbers
    # Remember how Llama 3 made up "$123 billion" earlier? We will use that to test the monitor!
    raw_gathered_data = "Apple Q3 2023: Total net sales were $81.8 billion."
    hallucinated_analyst_draft = "Apple had an incredible Q3, generating a massive $123.0 billion in total net sales, showing huge growth."

    # 3. Create a task that forces the monitor to compare the draft against the raw data
    test_task = Task(
        description=(
            f"Review the following draft report against the original raw data. Cross-check the numbers.\n\n"
            f"--- ORIGINAL RAW DATA ---\n{raw_gathered_data}\n\n"
            f"--- ANALYST DRAFT REPORT ---\n{hallucinated_analyst_draft}\n\n"
            f"Did the analyst hallucinate? Issue your final verdict."
        ),
        expected_output="A safety verdict starting with either [ APPROVED] or [ RISK FLAG DETECTED], followed by the explanation.",
        agent=monitor
    )

    # 4. Run the mini-crew
    test_crew = Crew(
        agents=[monitor],
        tasks=[test_task],
        verbose=True
    )

    result = test_crew.kickoff()

    print("\n================================================")
    print(" FINAL RISK MONITOR VERDICT:")
    print(result)
    print("================================================")