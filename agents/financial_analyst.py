import os
from crewai import Agent, LLM


def load_prompt(filename: str) -> str:
    """
    Loads a prompt file from project_root/prompts/<filename>.
    project_root is two levels up from this file:
        project_root/agents/financial_analyst.py  ->  project_root/
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(root_dir, 'prompts', filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"[WARNING] Prompt file not found: {prompt_path}")
        print(f"[WARNING] Falling back to default. Check that '{filename}' exists in project_root/prompts/")
        return "You are a helpful AI assistant."


local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)


def create_financial_analyst():
    analyst_backstory = load_prompt('analyst_prompt.txt')

    return Agent(
        role='Senior Financial Analyst',
        goal='Analyze raw financial data and synthesize it into a highly structured, professional report.',
        backstory=analyst_backstory,
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=local_llm
    )


if __name__ == "__main__":
    from crewai import Task, Crew

    print("Initiating Solo Test Run for Financial Analyst...")

    analyst = create_financial_analyst()

    simulated_gatherer_output = """
    Tesla Inc. FY2023 Financial Summary: Total automotive revenues were $82,419 million.
    Services and other revenues were $8,319 million.
    Risk Factors: Tesla faces supply and pricing risks for raw materials including
    lithium, nickel, and copper due to market fluctuations and industry-wide shortages.
    """

    test_task = Task(
        description=(
            f"Format the following raw data into a professional financial summary with "
            f"bullet points and clear headings:\n\n{simulated_gatherer_output}"
        ),
        expected_output="A well-formatted financial summary with headings like 'Revenue Overview' and 'Risk Factors'.",
        agent=analyst
    )

    test_crew = Crew(agents=[analyst], tasks=[test_task], verbose=True)
    result = test_crew.kickoff()

    print("\n================================================")
    print("FINAL ANALYST OUTPUT:")
    print(result)
    print("================================================")
