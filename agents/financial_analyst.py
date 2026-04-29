from crewai import Agent, LLM
import os

def load_prompt(filename):
    """Dynamically loads prompt text from the prompts directory."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(root_dir, 'prompts', 'analyst_prompt.txt')
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f" Warning: {'analyst_prompt.txt'} not found. Falling back to default.")
        return "You are a helpful AI assistant."

# 1. Define the exact same local LLM engine
# We use Llama 3.1 because of its superior reasoning and formatting capabilities
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# 2. Define the Agent
def create_financial_analyst():
    # Load Emir's Analyst prompt
    analyst_backstory = load_prompt('analyst_prompt.txt')

    return Agent(
        role='Senior Financial Analyst',
        goal='Analyze raw financial data and synthesize it into a highly structured, professional report.',
        backstory=analyst_backstory,
        verbose=True,
        allow_delegation=False,
        tools=[], # Keep this empty! The analyst only reads/writes.
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