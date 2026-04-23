from crewai import Process, Crew, Task

# 1. Import your modular agents
# (Assuming you run this script from inside the 'agents' folder)
from data_gatherer import create_data_gatherer
from financial_analyst import create_financial_analyst
from risk_monitor import create_risk_monitor

def run_financial_analysis(query: str):
    print(f" Initiating Multi-Agent Pipeline for query: '{query}'\n")

    # 2. Instantiate the engines
    gatherer = create_data_gatherer()
    analyst = create_financial_analyst()
    monitor = create_risk_monitor()

    # 3. Define the sequential tasks
    # Task 1: Fetch the data
    gather_task = Task(
        description=f"Search the database for: {query}. Extract the exact numbers and statements.",
        expected_output="Raw financial data and risk factors directly from the Vector DB.",
        agent=gatherer
    )

    # Task 2: Format the report
    # CrewAI automatically passes the output of Task 1 into the context of Task 2
    analyze_task = Task(
        description="Take the raw data provided by the Data Gatherer and format it into a professional financial summary with bullet points and clear headings.",
        expected_output="A structured, professional financial summary report.",
        agent=analyst
    )

    # Task 3: The Guardrail
    monitor_task = Task(
        description=(
            "Review the Financial Analyst's draft report. Check it against the original raw data "
            "provided by the Data Gatherer. Look for hallucinated numbers or unsupported claims. "
            "Output [ APPROVED] if safe, or [ RISK FLAG DETECTED] if there are discrepancies, "
            "followed by the final verified text."
        ),
        expected_output="The final risk-assessed financial report with an approval or warning flag.",
        agent=monitor
    )

    # 4. Assemble the Crew
    financial_crew = Crew(
        agents=[gatherer, analyst, monitor],
        tasks=[gather_task, analyze_task, monitor_task],
        process=Process.sequential, # CRITICAL: Forces them to run in exact order
        verbose=True
    )

    # 5. Kickoff the pipeline
    result = financial_crew.kickoff()
    return result

if __name__ == "__main__":
    # The ultimate end-to-end test
    final_output = run_financial_analysis("Apple AAPL Q3 2023 net sales")
    
    print("\n================================================")
    print(" FINAL PIPELINE OUTPUT:")
    print("================================================")
    print(final_output)