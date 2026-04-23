from crewai import Agent, LLM

# 1. Define the exact same local LLM engine
# We use Llama 3.1 because of its superior reasoning and formatting capabilities
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# 2. Define the Agent
def create_financial_analyst():
    return Agent(
        role='Senior Financial Analyst',
        goal='Analyze raw financial data and synthesize it into a highly structured, professional report.',
        backstory=(
            "You are a meticulous financial analyst working for a top-tier investment firm. "
            "Your job is to take the raw text and numbers provided by the Data Gatherer and "
            "turn them into a clean, readable summary with clear headings. "
            "CRITICAL INSTRUCTION: You must NEVER invent or hallucinate numbers. You only use "
            "the exact data provided to you in the context. If specific data is missing, you "
            "explicitly state 'Data not provided'."
        ),
        verbose=True,
        allow_delegation=False, # The analyst works alone, no delegating
        tools=[], # Notice this is empty! It relies entirely on the Gatherer's output
        llm=local_llm
    )

# ==========================================
#  Execution Block for Solo Testing
# ==========================================
if __name__ == "__main__":
    from crewai import Task, Crew

    print(" Initiating Solo Test Run for Financial Analyst...")
    
    # 1. Instantiate the agent
    analyst = create_financial_analyst()

    # 2. Simulate the exact output your Gatherer successfully produced in the last test
    simulated_gatherer_output = """
    Apple Inc. Q3 2023 Financial Summary: Total net sales were $81.8 billion, down 1% year over year. 
    Services revenue reached an all-time high of $21.2 billion. 
    Risk Factors Apple Inc: Macroeconomic conditions, including inflation and fluctuations in interest rates, 
    could negatively impact consumer spending and our future hardware margins.
    """

    # 3. Create a task that feeds the simulated data to the Analyst
    test_task = Task(
        description=(
            f"Format the following raw data into a professional financial summary with "
            f"bullet points and clear headings:\n\n{simulated_gatherer_output}"
        ),
        expected_output="A well-formatted financial summary with headings like 'Revenue Overview' and 'Risk Factors'.",
        agent=analyst
    )

    # 4. Run the mini-crew
    test_crew = Crew(
        agents=[analyst],
        tasks=[test_task],
        verbose=True
    )

    result = test_crew.kickoff()

    print("\n================================================")
    print(" FINAL ANALYST OUTPUT:")
    print(result)
    print("================================================")